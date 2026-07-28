#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CP Decomposition + MLP for Data Center Temperature Field Reconstruction
-----------------------------------------------------------------------
- CP / PARAFAC decomposition using training set only
- MLP maps 14-dimensional operating parameters to CP coefficients
- Train-only fitting for all scalers
- Per-case MAE, RMSE, rRMSE
- Y-slice temperature and error contour maps
- Error map title displays current maximum absolute error
- Optional unified error colormap with POD if global_error_max.npy exists
"""

import os
import sys
import math
import json
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import tensorly as tl
    from tensorly.decomposition import parafac
    TL_OK = True
except Exception:
    TL_OK = False


# ========================== User Config ==========================

PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"

NX, NY, NZ = 75, 51, 103
N_CASES = 89
Y_SLICE = 1.26

CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

CP_RANK = 16
CP_N_ITER_MAX = 2000
CP_TOL = 1e-7

REGRESSOR = "MLP_TORCH"

HIDDEN_LAYERS = [128, 256, 128]
USE_LAYERNORM = True
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 2000
BATCH_SIZE = 16

USE_ENERGY_WEIGHTS = True

USE_PCA_BOTTLENECK = False
PCA_LATENT_Q = 16

GPR_KERNEL = "Matern"
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = "scale"
SVR_CACHE_MB = 500

RANDOM_STATE = 42

PLOT_METHOD = "contourf"
LEVELS = 80
UPSAMPLE_FX = 1
UPSAMPLE_FZ = 1
SAVE_ERROR_MAP = True

FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved_CP"
os.makedirs(FIG_DIR, exist_ok=True)

POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"


# ========================== Utils ==========================

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")

    if df.shape[0] == N_CASES + 1:
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[0] != N_CASES:
        warnings.warn(
            f"[WARN] Parameter CSV has {df.shape[0]} rows, expected {N_CASES}. Proceeding."
        )

    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain at least 14 columns.")

    return df.iloc[:, :14]


def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    required_cols = {"X (m)", "Y (m)", "Z (m)", "Temperature"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"Snapshot {path} missing columns {required_cols}")

    return df


def build_grid_from_df(df: pd.DataFrame):
    xs = np.sort(df["X (m)"].unique())
    ys = np.sort(df["Y (m)"].unique())
    zs = np.sort(df["Z (m)"].unique())

    if (len(xs), len(ys), len(zs)) != (NX, NY, NZ):
        raise ValueError(
            f"Grid size mismatch: {(len(xs), len(ys), len(zs))} vs {(NX, NY, NZ)}"
        )

    return xs, ys, zs


def df_to_grid_values(df, xs, ys, zs):
    ix = np.searchsorted(xs, df["X (m)"].to_numpy())
    iy = np.searchsorted(ys, df["Y (m)"].to_numpy())
    iz = np.searchsorted(zs, df["Z (m)"].to_numpy())

    temp = df["Temperature"].to_numpy()

    grid = np.empty((NX, NY, NZ), dtype=np.float64)
    grid[ix, iy, iz] = temp

    return grid


def upsample_if_needed(xs, zs, Txz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, Txz

    xi = np.linspace(xs.min(), xs.max(), len(xs) * UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs) * UPSAMPLE_FZ)

    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)

    return xi, zi, Txz_hi


# ========================== MLP and Regressors ==========================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers,
        use_layernorm=USE_LAYERNORM,
        dropout=DROPOUT
    ):
        super().__init__()

        def block(d_in, d_out):
            layers = [nn.Linear(d_in, d_out)]

            if use_layernorm:
                layers += [nn.LayerNorm(d_out)]

            layers += [nn.ReLU(), nn.Dropout(dropout)]

            return nn.Sequential(*layers)

        layers = []
        prev = in_dim

        for h in hidden_layers:
            layers.append(block(prev, h))
            prev = h

        layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp_torch(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    weight_per_dim: Optional[np.ndarray] = None,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    hidden_layers=HIDDEN_LAYERS,
    seed=RANDOM_STATE,
    use_layernorm=USE_LAYERNORM,
    dropout=DROPOUT,
    weight_decay=WEIGHT_DECAY
) -> nn.Module:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(
        X_train.shape[1],
        Y_train.shape[1],
        hidden_layers,
        use_layernorm=use_layernorm,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if weight_per_dim is None:
        weight = torch.ones(Y_train.shape[1], dtype=torch.float32, device=device)
    else:
        w = weight_per_dim.astype(np.float32)
        w = w / (np.mean(w) + 1e-12)
        weight = torch.tensor(w, dtype=torch.float32, device=device)

    model.train()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            diff = pred - yb

            loss = torch.mean(torch.sum((diff ** 2) * (weight ** 2), dim=1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"[MLP] Epoch {epoch + 1}/{epochs} | Loss={loss.item():.6f}")

    return model


def predict_mlp_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device

    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y_pred = model(X_t).cpu().numpy()

    return Y_pred


def choose_regressor(name: str, y_dim: int):
    name = name.upper()

    if name == "GPR":
        if GPR_KERNEL.upper() == "RBF":
            base_kernel = RBF(
                length_scale=np.ones(14),
                length_scale_bounds=(1e-2, 1e3)
            )
        else:
            base_kernel = Matern(
                length_scale=np.ones(14),
                length_scale_bounds=(1e-2, 1e3),
                nu=1.5
            )

        kernel = (
            C(1.0, (1e-3, 1e3))
            * base_kernel
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        )

        base = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=0.0
        )

        model = MultiOutputRegressor(base, n_jobs=None)

    elif name == "SVR":
        base = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            cache_size=SVR_CACHE_MB
        )

        model = MultiOutputRegressor(base, n_jobs=None)

    elif name == "MLP_TORCH":
        return None

    else:
        raise ValueError("Unknown REGRESSOR. Supported: 'MLP_TORCH', 'SVR', 'GPR'.")

    return model


# ========================== CP / PARAFAC Core ==========================

def cp_fit_train_only(
    T_centered: np.ndarray,
    idx_train: np.ndarray,
    R: int,
    n_iter_max: int = 2000,
    tol: float = 1e-7,
    random_state: int = 42
):
    if not TL_OK:
        raise RuntimeError("Please install tensorly first: pip install tensorly")

    tl.set_backend("numpy")

    T_train = T_centered[..., idx_train]

    cp = parafac(
        T_train,
        rank=R,
        init="svd",
        tol=tol,
        n_iter_max=n_iter_max,
        random_state=random_state,
        normalize_factors=True
    )

    lambdas = cp.weights
    Ax, Ay, Az, An_train = cp.factors

    return lambdas, Ax, Ay, Az, An_train


def cp_build_projection_matrix(
    Ax: np.ndarray,
    Ay: np.ndarray,
    Az: np.ndarray,
    lambdas: np.ndarray
) -> np.ndarray:
    NX_local, R = Ax.shape
    NY_local = Ay.shape[0]
    NZ_local = Az.shape[0]

    M = np.empty((NX_local * NY_local * NZ_local, R), dtype=np.float64)

    for r in range(R):
        outer3 = np.multiply.outer(
            np.multiply.outer(Ax[:, r], Ay[:, r]),
            Az[:, r]
        )

        M[:, r] = float(lambdas[r]) * outer3.reshape(-1)

    return M


def project_case_to_cp_coeff(T_case: np.ndarray, M: np.ndarray) -> np.ndarray:
    b = T_case.reshape(-1)

    d, *_ = np.linalg.lstsq(M, b, rcond=None)

    return d


def cp_reconstruct_from_d(
    d: np.ndarray,
    Ax,
    Ay,
    Az,
    lambdas,
    T_mean_train=None
):
    NX_local, R = Ax.shape
    NY_local = Ay.shape[0]
    NZ_local = Az.shape[0]

    T_rec = np.zeros((NX_local, NY_local, NZ_local), dtype=np.float64)

    for r in range(R):
        T_rec += (
            lambdas[r]
            * d[r]
            * np.multiply.outer(
                np.multiply.outer(Ax[:, r], Ay[:, r]),
                Az[:, r]
            )
        )

    if T_mean_train is not None:
        T_rec = T_rec + T_mean_train[..., 0]

    return T_rec


def cp_component_energy_weights(lambdas, Ax, Ay, Az):
    R = Ax.shape[1]

    w = np.zeros(R, dtype=np.float64)

    for r in range(R):
        w[r] = (
            abs(float(lambdas[r]))
            * np.linalg.norm(Ax[:, r])
            * np.linalg.norm(Ay[:, r])
            * np.linalg.norm(Az[:, r])
        )

    w = w / (np.max(w) + 1e-12)
    w = np.clip(w, 1e-3, None)

    return w


# ========================== Main ==========================

def main():
    np.random.seed(RANDOM_STATE)
    start_main_time = time.time()

    if not TL_OK:
        raise RuntimeError("Tensorly is not installed. Please run: pip install tensorly")

    print("=== Load parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read first snapshot and build grid ===")
    df0 = read_one_snapshot_csv(os.path.join(SNAPSHOT_DIR, "1.csv"))
    xs, ys, zs = build_grid_from_df(df0)

    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))

    print(
        f"Y slice target = {Y_SLICE} m -> "
        f"index {y_slice_index}, actual y = {ys[y_slice_index]:.4f} m"
    )

    print("=== Build tensor T ===")

    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    T[..., 0] = df_to_grid_values(df0, xs, ys, zs)

    for i in range(2, N_CASES + 1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        dfi = read_one_snapshot_csv(p)

        T[..., i - 1] = df_to_grid_values(dfi, xs, ys, zs)

        if i % 20 == 0 or i == N_CASES:
            print(f"Loaded {i}/{N_CASES}")

    print("=== Train/Test split ===")

    TEST_IDX_ONE_BASED = [8, 9, 20, 21, 58, 68, 72, 76, 84]

    idx_test = np.array([i - 1 for i in TEST_IDX_ONE_BASED], dtype=int)
    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(idx_all, idx_test, assume_unique=False)

    print(f"Test indices 1-based: {TEST_IDX_ONE_BASED}")
    print(f"Train size = {len(idx_train)}, Test size = {len(idx_test)}")

    if CENTER_ALONG_N:
        T_mean_train = T[..., idx_train].mean(axis=3, keepdims=True)
        T_centered = T - T_mean_train
        print("Applied N-mode mean centering using training set mean.")
    else:
        T_mean_train = None
        T_centered = T

    print(f"=== CP decomposition on training set, rank = {CP_RANK} ===")

    t0 = time.time()

    lambdas, Ax, Ay, Az, An_train = cp_fit_train_only(
        T_centered,
        idx_train,
        CP_RANK,
        n_iter_max=CP_N_ITER_MAX,
        tol=CP_TOL,
        random_state=RANDOM_STATE
    )

    print(
        f"[CP] Done. Shapes: "
        f"Ax={Ax.shape}, Ay={Ay.shape}, Az={Az.shape}, An_train={An_train.shape}"
    )
    print(f"CP fit time: {time.time() - t0:.2f} s")

    print("=== Build CP coefficients for all cases by projection ===")

    M_cp = cp_build_projection_matrix(Ax, Ay, Az, lambdas)
    R = Ax.shape[1]

    Y_coeff_all = np.zeros((N_CASES, R), dtype=np.float64)

    for n in range(N_CASES):
        Tc = T_centered[..., n]
        Y_coeff_all[n, :] = project_case_to_cp_coeff(Tc, M_cp)

    X_train_raw = X_params[idx_train]
    X_test_raw = X_params[idx_test]

    Y_train_raw = Y_coeff_all[idx_train]
    Y_test_raw = Y_coeff_all[idx_test]

    if USE_ENERGY_WEIGHTS:
        weights = cp_component_energy_weights(lambdas, Ax, Ay, Az)
    else:
        weights = None

    if USE_PCA_BOTTLENECK:
        raise NotImplementedError("PCA bottleneck with CP is not enabled in this version.")

    else:
        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
        else:
            X_train = X_train_raw
            X_test = X_test_raw

        if SCALE_OUTPUT_COEFF:
            y_scaler = StandardScaler().fit(Y_train_raw)
            Y_train = y_scaler.transform(Y_train_raw)
        else:
            y_scaler = None
            Y_train = Y_train_raw

        print(
            f"=== Fit regressor: {REGRESSOR} "
            f"(energy-weighted loss = {USE_ENERGY_WEIGHTS}) ==="
        )

        t0 = time.time()

        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(
                X_train,
                Y_train,
                weight_per_dim=weights,
                lr=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                hidden_layers=HIDDEN_LAYERS,
                seed=RANDOM_STATE,
                use_layernorm=USE_LAYERNORM,
                dropout=DROPOUT,
                weight_decay=WEIGHT_DECAY
            )

            Y_pred_scaled = predict_mlp_torch(model, X_test)

        else:
            base = choose_regressor(REGRESSOR, Y_train.shape[1])
            base.fit(X_train, Y_train)
            Y_pred_scaled = base.predict(X_test)

        print(f"Regressor fit time: {time.time() - t0:.2f} s")

        if SCALE_OUTPUT_COEFF:
            Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
        else:
            Y_pred = Y_pred_scaled

    print("=== Reconstruct test 3D fields using CP ===")

    preds = []
    gts = []

    for j, case_idx in enumerate(idx_test):
        d_pred = Y_pred[j]

        T_pred = cp_reconstruct_from_d(
            d_pred,
            Ax,
            Ay,
            Az,
            lambdas,
            T_mean_train
        )

        preds.append(T_pred)
        gts.append(T[..., case_idx])

    preds = np.stack(preds, axis=-1)
    gts = np.stack(gts, axis=-1)

    print("=== Metrics ===")

    mae_list = []
    rmse_list = []
    rRMSE_list = []

    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_pred = preds[..., k].ravel()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        err_vector = y_true - y_pred
        rRMSE_case = np.linalg.norm(err_vector) / (np.linalg.norm(y_true) + 1e-9)

        mae_list.append(mae)
        rmse_list.append(rmse)
        rRMSE_list.append(rRMSE_case)

        print(
            f"Case {idx_test[k] + 1:2d}: "
            f"MAE = {mae:.4f} °C, "
            f"RMSE = {rmse:.4f} °C, "
            f"rRMSE = {rRMSE_case:.4f}"
        )

    print(
        f"\n[Full 3D Mean] "
        f"MAE = {np.mean(mae_list):.4f} °C, "
        f"RMSE = {np.mean(rmse_list):.4f} °C, "
        f"rRMSE = {np.mean(rRMSE_list):.4f}"
    )

    print("\n=== Calculating rRMSE_n for top 10 hottest unique points ===")

    if T_mean_train is not None:
        mean_temp_field = T_mean_train.squeeze()
    else:
        mean_temp_field = T[..., idx_train].mean(axis=3)

    mean_temp_flat = mean_temp_field.ravel()
    unique_temps_sorted = np.unique(mean_temp_flat)
    num_points_to_select = min(10, len(unique_temps_sorted))
    top_10_unique_temps = unique_temps_sorted[-num_points_to_select:]

    top_10_indices = []

    for temp_val in reversed(top_10_unique_temps):
        idx = np.where(mean_temp_flat == temp_val)[0][0]

        if idx not in top_10_indices:
            top_10_indices.append(idx)

    rRMSE_n_results = {}

    for flat_idx in top_10_indices:
        i, j, k_dim = np.unravel_index(flat_idx, (NX, NY, NZ))

        true_series = gts[i, j, k_dim, :]
        pred_series = preds[i, j, k_dim, :]

        rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (
            np.linalg.norm(true_series) + 1e-9
        )

        avg_temp_at_point = mean_temp_field[i, j, k_dim]
        rRMSE_n_results[int(flat_idx)] = float(rRMSE_n_value)

        print(
            f"Grid Point Index {flat_idx:<7} "
            f"(Avg T ≈ {avg_temp_at_point:.1f} °C): "
            f"rRMSE_n = {rRMSE_n_value:.4f}"
        )

    df_metrics = pd.DataFrame({
        "Case": [int(i) + 1 for i in idx_test],
        "MAE": [float(x) for x in mae_list],
        "RMSE": [float(x) for x in rmse_list],
        "rRMSE": [float(x) for x in rRMSE_list]
    })

    df_metrics = df_metrics.sort_values("Case")

    df_metrics.to_csv(
        os.path.join(FIG_DIR, "per_case_metrics_cp.csv"),
        index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(df_metrics["Case"], df_metrics["MAE"], "-o", label="MAE (°C)")
    plt.plot(df_metrics["Case"], df_metrics["RMSE"], "-s", label="RMSE (°C)")
    plt.plot(df_metrics["Case"], df_metrics["rRMSE"], "-^", label="rRMSE")
    plt.xlabel("Case")
    plt.ylabel("Error Value")
    plt.title("CP + MLP Per-case Error Metrics")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "error_metrics_compare_cp.png"),
        dpi=200
    )

    plt.close()

    if os.path.exists(POD_ERROR_MAX_PATH):
        try:
            shared_error_max = float(np.load(POD_ERROR_MAX_PATH))

            if not np.isfinite(shared_error_max) or shared_error_max <= 0:
                shared_error_max = None
                print("[Warn] Loaded POD error max is invalid. CP error maps will use per-case vmax.")
            else:
                print(f"[Info] Unified error colormap vmax from POD: {shared_error_max:.4f} °C")

        except Exception:
            shared_error_max = None
            print("[Warn] Failed to load POD error max. CP error maps will use per-case vmax.")
    else:
        shared_error_max = None
        print("[Info] POD global_error_max.npy not found. CP error maps will use per-case vmax.")

    print("=== Plot Y-slice maps ===")

    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]
        err_slice = np.abs(T_true_slice - T_pred_slice)

        err_max_current = float(np.nanmax(err_slice))

        vmax = shared_error_max if shared_error_max is not None else err_max_current

        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice)
        _, _, T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice)
        _, _, err_plot = upsample_if_needed(xs, zs, err_slice)

        Xg, Zg = np.meshgrid(xs_plot, zs_plot, indexing="ij")

        fig = plt.figure(figsize=(18, 5), dpi=150)

        plt.suptitle(
            f"Y = {ys[y_slice_index]:.3f} m | "
            f"Case {case_idx + 1} | "
            f"Max Error = {err_max_current:.2f} °C",
            fontsize=16
        )

        ax1 = fig.add_subplot(1, 3, 1)

        c1 = ax1.contourf(
            Xg,
            Zg,
            T_true_plot,
            levels=LEVELS,
            cmap="jet"
        )

        ax1.set_title("True Temperature")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_aspect("equal", adjustable="box")
        fig.colorbar(c1, ax=ax1)

        ax2 = fig.add_subplot(1, 3, 2)

        c2 = ax2.contourf(
            Xg,
            Zg,
            T_pred_plot,
            levels=LEVELS,
            cmap="jet",
            vmin=c1.get_clim()[0],
            vmax=c1.get_clim()[1]
        )

        ax2.set_title("Reconstructed Temperature")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax2.set_aspect("equal", adjustable="box")
        fig.colorbar(c2, ax=ax2)

        if SAVE_ERROR_MAP:
            ax3 = fig.add_subplot(1, 3, 3)

            c3 = ax3.contourf(
                Xg,
                Zg,
                err_plot,
                levels=LEVELS,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax
            )

            ax3.set_title(
                f"Absolute Error\nMax = {err_max_current:.2f} °C"
            )
            ax3.set_xlabel("X (m)")
            ax3.set_ylabel("Z (m)")
            ax3.set_aspect("equal", adjustable="box")
            fig.colorbar(c3, ax=ax3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(
            FIG_DIR,
            f"yslice_case_{case_idx + 1}_cp.png"
        )

        plt.savefig(
            out_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"Saved {out_path} | "
            f"Current max error = {err_max_current:.4f} °C"
        )

    summary = {
        "model_type": "CP_PARAFAC",
        "decomposition_set": "TRAIN",
        "cp_rank_R": int(CP_RANK),
        "config": {
            "center_along_N": bool(CENTER_ALONG_N),
            "scale_inputs": bool(SCALE_INPUTS),
            "scale_output_coeff": bool(SCALE_OUTPUT_COEFF)
        },
        "regressor": REGRESSOR,
        "mlp_params": {
            "hidden_layers": HIDDEN_LAYERS,
            "use_layernorm": USE_LAYERNORM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "energy_weighted_loss": USE_ENERGY_WEIGHTS,
            "pca_bottleneck": USE_PCA_BOTTLENECK,
            "pca_latent_q": PCA_LATENT_Q if USE_PCA_BOTTLENECK else None
        },
        "random_state": RANDOM_STATE,
        "test_indices_one_based": [int(x) for x in (idx_test + 1).tolist()],
        "metrics_summary": {
            "full3d_mae_mean": float(np.mean(mae_list)),
            "full3d_rmse_mean": float(np.mean(rmse_list)),
            "full3d_rRMSE_mean": float(np.mean(rRMSE_list)),
            "full3d_mae_each": [float(x) for x in mae_list],
            "full3d_rmse_each": [float(x) for x in rmse_list],
            "full3d_rRMSE_each": [float(x) for x in rRMSE_list],
            "rRMSE_n_top10_points": rRMSE_n_results
        },
        "y_slice_visualization": {
            "y_value": float(ys[y_slice_index]),
            "unified_error_vmax_from_pod": float(shared_error_max)
            if shared_error_max is not None else None
        },
        "cp_component_weights_preview": [
            float(x) for x in cp_component_energy_weights(lambdas, Ax, Ay, Az).tolist()
        ],
        "execution_time_seconds": time.time() - start_main_time
    }

    with open(
        os.path.join(FIG_DIR, "metrics_summary_cp.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Outputs saved in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    try:
        main()

    except Exception as e:
        print(f"ERROR: {e}")

        import traceback
        traceback.print_exc()

        sys.exit(1)
