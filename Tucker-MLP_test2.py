#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker Decomposition + MLP (improved, unified error colormap, per-case metrics & line plots)
--------------------------------------------------------------------------------------------
- Energy-weighted loss along N-mode (default: ON)
- Optional PCA bottleneck on N-mode coefficients (default: OFF)
- LayerNorm + mild Dropout + weight decay
- Train-only fitting for all scalers (no leakage)
- NumPy norm(axis=...) compatibility fix
- Unified error colormap with POD (reads global_error_max.npy if available)
- Per-case MAE/RMSE saved & line chart

Author: (your name)
Date: 2025-10-19
"""

import os
import sys
import math
import json
import time
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# (Baselines kept for optional switching)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from sklearn.svm import SVR

# PyTorch (MLP)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optional interpolation
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ========================== User Config ==========================

# Paths
PARAMS_PATH   = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR  = r"C:\Users\Lenovo\Desktop\insert"                   # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES    = 89
Y_SLICE    = 1.26  # meters (用于切片可视化)

# Adaptive rank selection
USE_ADAPTIVE_RANKS = True
ENERGY_THRESH = 0.995
FALLBACK_RANKS = (15, 15, 15, 20)

# Centering and scaling
CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

# Regressor choice
# 'MLP_TORCH'（默认，改进版） | 'SVR' | 'GPR'
REGRESSOR = "MLP_TORCH"

# MLP (Torch) hyperparameters
HIDDEN_LAYERS = [128, 256, 128]
USE_LAYERNORM = True
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 2000
BATCH_SIZE = 16

# Energy-weighted loss
USE_ENERGY_WEIGHTS = True  # 推荐开启

# PCA bottleneck on N-mode coefficients
USE_PCA_BOTTLENECK = False
PCA_LATENT_Q = 16

# Baseline options (if you switch REGRESSOR)
GPR_KERNEL = "Matern"      # "RBF" or "Matern"
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = "scale"
SVR_CACHE_MB = 500

RANDOM_STATE = 42

# Visualization
PLOT_METHOD = "contourf"   # "contourf" | "imshow"
LEVELS = 80
UPSAMPLE_FX = 1
UPSAMPLE_FZ = 1
SAVE_ERROR_MAP = True

# Output directory
FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved"
os.makedirs(FIG_DIR, exist_ok=True)

# Unified error colormap with POD
POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"


# ========================== Utils ==========================

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    if df.shape[0] == N_CASES + 1:
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[0] != N_CASES:
        warnings.warn(f"[WARN] Parameter CSV has {df.shape[0]} rows, expected {N_CASES}. Proceeding.")
    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain 14 columns.")
    return df.iloc[:, :14]


def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    need = {"X (m)", "Y (m)", "Z (m)", "Temperature"}
    if not need.issubset(df.columns):
        raise ValueError(f"Snapshot {path} missing columns {need}")
    return df


def build_grid_from_df(df: pd.DataFrame):
    xs = np.sort(df["X (m)"].unique())
    ys = np.sort(df["Y (m)"].unique())
    zs = np.sort(df["Z (m)"].unique())
    if (len(xs), len(ys), len(zs)) != (NX, NY, NZ):
        raise ValueError(f"Grid size mismatch: {(len(xs), len(ys), len(zs))} vs {(NX,NY,NZ)}")
    return xs, ys, zs


def df_to_grid_values(df, xs, ys, zs):
    ix = np.searchsorted(xs, df["X (m)"].to_numpy())
    iy = np.searchsorted(ys, df["Y (m)"].to_numpy())
    iz = np.searchsorted(zs, df["Z (m)"].to_numpy())
    temp = df["Temperature"].to_numpy()
    grid = np.empty((NX, NY, NZ), dtype=np.float64)
    grid[ix, iy, iz] = temp
    return grid


def mode_n_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def n_mode_product(T: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    T_perm = np.moveaxis(T, mode, 0)
    out = U @ T_perm.reshape(T_perm.shape[0], -1)
    new_shape = (U.shape[0],) + T_perm.shape[1:]
    out = out.reshape(new_shape)
    return np.moveaxis(out, 0, mode)


def select_rank_by_energy(X: np.ndarray, thr: float, max_rank: Optional[int]=None):
    I = X.shape[0]
    n_comp = I if max_rank is None else min(I, max_rank)
    U, S, VT = randomized_svd(X, n_components=n_comp, random_state=RANDOM_STATE)
    energy = (S**2)
    cum = np.cumsum(energy) / np.sum(energy)
    r = int(np.searchsorted(cum, thr) + 1)
    r = max(1, min(r, n_comp))
    return U[:, :r], r, S, VT


def hosvd_adaptive(T: np.ndarray, energy_thr: float, fallback_ranks: Tuple[int,int,int,int]):
    X0 = mode_n_unfold(T, 0); Ux, rx, *_ = select_rank_by_energy(X0, energy_thr)
    X1 = mode_n_unfold(T, 1); Uy, ry, *_ = select_rank_by_energy(X1, energy_thr)
    X2 = mode_n_unfold(T, 2); Uz, rz, *_ = select_rank_by_energy(X2, energy_thr)
    X3 = mode_n_unfold(T, 3); Un, rn, *_ = select_rank_by_energy(X3, energy_thr)
    ranks = (rx, ry, rz, rn)
    if any(r < 2 for r in ranks):
        warnings.warn(f"[WARN] Very small adaptive ranks {ranks}; falling back to {fallback_ranks}.")
        return hosvd_fixed(T, fallback_ranks), fallback_ranks
    G = T.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)
    return (G, [Ux, Uy, Uz, Un]), ranks


def hosvd_fixed(T: np.ndarray, ranks: Tuple[int,int,int,int]):
    rx, ry, rz, rn = ranks
    Ux, *_ = randomized_svd(mode_n_unfold(T, 0), n_components=rx, random_state=RANDOM_STATE)
    Uy, *_ = randomized_svd(mode_n_unfold(T, 1), n_components=ry, random_state=RANDOM_STATE)
    Uz, *_ = randomized_svd(mode_n_unfold(T, 2), n_components=rz, random_state=RANDOM_STATE)
    Un, *_ = randomized_svd(mode_n_unfold(T, 3), n_components=rn, random_state=RANDOM_STATE)
    G = T.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)
    return (G, [Ux, Uy, Uz, Un])


def reconstruct_from_coeff(G, Ux, Uy, Uz, coeff):
    M = np.tensordot(G, coeff, axes=([3], [0]))
    T1 = n_mode_product(M, Ux, 0)
    T2 = n_mode_product(T1, Uy, 1)
    T3 = n_mode_product(T2, Uz, 2)
    return T3


def choose_regressor(name: str, y_dim: int):
    """Baseline switch helper."""
    name = name.upper()
    if name == "GPR":
        if GPR_KERNEL.upper() == "RBF":
            base_kernel = RBF(length_scale=np.ones(14), length_scale_bounds=(1e-2, 1e3))
        else:
            base_kernel = Matern(length_scale=np.ones(14), length_scale_bounds=(1e-2, 1e3), nu=1.5)
        kernel = C(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        base = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE, alpha=0.0)
        model = MultiOutputRegressor(base, n_jobs=None)
    elif name == "SVR":
        base = SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, cache_size=SVR_CACHE_MB)
        model = MultiOutputRegressor(base, n_jobs=None)
    elif name == "MLP_TORCH":
        return None
    else:
        raise ValueError("Unknown REGRESSOR (supported: 'MLP_TORCH', 'SVR', 'GPR').")
    return model


def upsample_if_needed(xs, zs, Txz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, Txz
    xi = np.linspace(xs.min(), xs.max(), len(xs)*UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs)*UPSAMPLE_FZ)
    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)
    return xi, zi, Txz_hi


# ========================== MLP (Torch) ==========================

class MLP(nn.Module):
    """Linear -> (LayerNorm?) -> ReLU -> Dropout blocks + Linear out"""
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_layers, use_layernorm=USE_LAYERNORM, dropout=DROPOUT):
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
    lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
    hidden_layers=HIDDEN_LAYERS, seed=RANDOM_STATE,
    use_layernorm=USE_LAYERNORM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY
) -> nn.Module:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_train.shape[1], Y_train.shape[1],
                hidden_layers, use_layernorm=use_layernorm, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # per-dimension weights for weighted MSE
    if weight_per_dim is None:
        weight = torch.ones(Y_train.shape[1], dtype=torch.float32, device=device)
    else:
        weight = torch.tensor(weight_per_dim, dtype=torch.float32, device=device)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            diff = pred - yb
            loss = torch.mean(torch.sum((diff ** 2) * (weight ** 2), dim=1))
            loss.backward()
            opt.step()
        if (ep + 1) % 100 == 0:
            print(f"[MLP] Epoch {ep+1}/{epochs} | Loss={loss.item():.6f}")
    return model


def predict_mlp_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Yp = model(X_t).cpu().numpy()
    return Yp


# ========================== Main ==========================

def main():
    np.random.seed(RANDOM_STATE)

    print("=== Load parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read first snapshot, build grid ===")
    df0 = read_one_snapshot_csv(os.path.join(SNAPSHOT_DIR, "1.csv"))
    xs, ys, zs = build_grid_from_df(df0)
    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))
    print(f"Y slice ≈ {Y_SLICE} m -> index {y_slice_index}, y={ys[y_slice_index]:.4f}")

    print("=== Build tensor T ===")
    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    T[..., 0] = df_to_grid_values(df0, xs, ys, zs)
    for i in range(2, N_CASES + 1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        dfi = read_one_snapshot_csv(p)
        T[..., i - 1] = df_to_grid_values(dfi, xs, ys, zs)
        if i % 10 == 0 or i == N_CASES:
            print(f"  Loaded {i}/{N_CASES}")

    # Optional centering along N (cases)
    T_mean = None
    if CENTER_ALONG_N:
        T_mean = T.mean(axis=3, keepdims=True)
        T_used = T - T_mean
        print("Applied N-mode mean centering.")
    else:
        T_used = T

    print("=== HOSVD / Tucker ===")
    if USE_ADAPTIVE_RANKS:
        (G, [Ux, Uy, Uz, Un]), ranks = hosvd_adaptive(T_used, ENERGY_THRESH, FALLBACK_RANKS)
    else:
        (G, [Ux, Uy, Uz, Un]) = hosvd_fixed(T_used, FALLBACK_RANKS)
        ranks = FALLBACK_RANKS
    rx, ry, rz, rn = ranks
    print(f"Ranks: rx={rx}, ry={ry}, rz={rz}, rn={rn}")
    print(f"Core shape: {G.shape}")

    # Regression targets (per case N-mode coefficients)
    Y_coeff = Un  # (N_CASES, rn)

    # ---- Energy weights along N-mode (length rn) ----
    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        # 兼容性修复：将空间维度展平后按列求范数 (rx*ry*rz, rn) -> (rn,)
        G_mat = G.reshape(-1, G.shape[-1])
        Gn = np.linalg.norm(G_mat, axis=0)
        w = Gn / (Gn.max() + 1e-12)
        w = np.clip(w, 1e-3, None)           # avoid zeros
    else:
        w = None

    # ----------------- Fixed test indices (1-based -> 0-based) -----------------
    print("=== Train/Test split (fixed indices) ===")
    TEST_IDX_ONE_BASED = [8, 9, 20, 21, 58, 68, 72, 76, 84]
    idx_test = np.array([i - 1 for i in TEST_IDX_ONE_BASED], dtype=int)
    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(idx_all, idx_test, assume_unique=False)
    print(f"  Test (1-based): {TEST_IDX_ONE_BASED}")
    print(f"  Test (0-based): {idx_test.tolist()}")
    print(f"  Train size = {len(idx_train)}, Test size = {len(idx_test)}")
    # ---------------------------------------------------------------------------

    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff[idx_train], Y_coeff[idx_test]

    # ---- Branch A: PCA bottleneck on outputs ----
    if USE_PCA_BOTTLENECK:
        print(f"=== PCA bottleneck on Y (q={PCA_LATENT_Q}) ===")
        pca = PCA(n_components=PCA_LATENT_Q, random_state=RANDOM_STATE)
        Z_train = pca.fit_transform(Y_train_raw)     # (n_train, q)
        Z_test  = pca.transform(Y_test_raw)

        # Scale X (fit on train)
        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test  = x_scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # Scale Z (fit on train)
        if SCALE_OUTPUT_COEFF:
            z_scaler = StandardScaler().fit(Z_train)
            Z_train_s = z_scaler.transform(Z_train)
        else:
            z_scaler = None
            Z_train_s = Z_train

        # Fit regressor
        print(f"=== Fit regressor: {REGRESSOR} (on PCA scores) ===")
        t0 = time.time()
        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(
                X_train, Z_train_s,
                weight_per_dim=None,           # q 维等权即可
                lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                hidden_layers=HIDDEN_LAYERS, seed=RANDOM_STATE,
                use_layernorm=USE_LAYERNORM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY
            )
            Z_pred_s = predict_mlp_torch(model, X_test)
        else:
            base = choose_regressor(REGRESSOR, PCA_LATENT_Q)
            base.fit(X_train, Z_train_s)
            Z_pred_s = base.predict(X_test)
        print(f"Regressor fit time: {time.time()-t0:.2f}s")

        # Inverse scaling & inverse PCA
        Z_pred = z_scaler.inverse_transform(Z_pred_s) if SCALE_OUTPUT_COEFF else Z_pred_s
        Y_pred = pca.inverse_transform(Z_pred)         # (n_test, rn)

    # ---- Branch B: No PCA; use energy-weighted loss ----
    else:
        # Scale X
        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test  = x_scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # Scale Y (rn dims)
        if SCALE_OUTPUT_COEFF:
            y_scaler = StandardScaler().fit(Y_train_raw)
            Y_train = y_scaler.transform(Y_train_raw)
        else:
            y_scaler = None
            Y_train = Y_train_raw

        print(f"=== Fit regressor: {REGRESSOR} (energy-weighted={USE_ENERGY_WEIGHTS}) ===")
        t0 = time.time()
        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(
                X_train, Y_train,
                weight_per_dim=w,              # << 能量加权
                lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                hidden_layers=HIDDEN_LAYERS, seed=RANDOM_STATE,
                use_layernorm=USE_LAYERNORM, dropout=DROPOUT, weight_decay=WEIGHT_DECAY
            )
            Y_pred_s = predict_mlp_torch(model, X_test)
        else:
            base = choose_regressor(REGRESSOR, Y_train.shape[1])
            base.fit(X_train, Y_train)
            Y_pred_s = base.predict(X_test)
        print(f"Regressor fit time: {time.time()-t0:.2f}s")

        Y_pred = y_scaler.inverse_transform(Y_pred_s) if SCALE_OUTPUT_COEFF else Y_pred_s

    # ---- Reconstruct test 3D fields ----
    print("=== Reconstruct test 3D fields ===")
    preds, gts = [], []
    for j, case_idx in enumerate(idx_test):
        coeff = Y_pred[j]
        T_pred = reconstruct_from_coeff(G, Ux, Uy, Uz, coeff)
        if CENTER_ALONG_N:
            T_pred = T_pred + T_mean[..., 0]
        preds.append(T_pred)
        gts.append(T[..., case_idx])
    preds = np.stack(preds, axis=-1)
    gts   = np.stack(gts, axis=-1)

    # ---- Load shared error max from POD (for unified colormap) ----
    if os.path.exists(POD_ERROR_MAX_PATH):
        shared_error_max = float(np.load(POD_ERROR_MAX_PATH))
        if not np.isfinite(shared_error_max) or shared_error_max <= 0:
            shared_error_max = None
            print("[Warn] Loaded POD error max is non-positive; fallback to per-case vmax.")
        else:
            print(f"[Info] Unified error colormap vmax from POD: {shared_error_max:.4f}")
    else:
        shared_error_max = None
        print("[Info] POD global_error_max.npy not found; Tucker error maps will use per-case vmax.")

    # ---- Metrics ----
    print("=== Metrics (per-case & mean) ===")
    mae_list, rmse_list = [], []
    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_hat  = preds[..., k].ravel()
        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        mae_list.append(mae)
        rmse_list.append(rmse)
        print(f"  Case {idx_test[k]+1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C")

    print(f"\n[Full 3D Mean] MAE={np.mean(mae_list):.4f} °C, RMSE={np.mean(rmse_list):.4f} °C")

    # ---- Save per-case CSV ----
    df = pd.DataFrame({"Case": [int(i)+1 for i in idx_test],
                       "MAE": [float(x) for x in mae_list],
                       "RMSE": [float(x) for x in rmse_list]})
    df = df.sort_values("Case")
    df.to_csv(os.path.join(FIG_DIR, "per_case_metrics.csv"), index=False)

    # ---- Line plot (MAE/RMSE) ----
    plt.figure(figsize=(8, 4))
    plt.plot(df["Case"], df["MAE"], '-o', label="MAE")
    plt.plot(df["Case"], df["RMSE"], '-s', label="RMSE")
    plt.xlabel("Case")
    plt.ylabel("Error (°C)")
    plt.title("Tucker+MLP Per-case Error Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mae_rmse_compare.png"), dpi=200)
    plt.close()

    # ---- Unified visualization on Y-slice (GT vs Pred; +Error) ----
    print("=== Plot Y-slice maps (GT vs Pred; +Error, unified colormap if available) ===")
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]
        err_slice    = np.abs(T_true_slice - T_pred_slice)
        vmax = shared_error_max if (shared_error_max is not None) else float(np.nanmax(err_slice))

        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice)
        _,       _,       T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice)
        Xg, Zg = np.meshgrid(xs_plot, zs_plot, indexing="ij")

        fig = plt.figure(figsize=(18, 5), dpi=150)
        plt.suptitle(f"Y={ys[y_slice_index]:.3f} m | Unified Error Colormap")

        ax1 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 1)
        c1 = ax1.contourf(Xg, Zg, T_true_plot, levels=50, cmap='jet')
        ax1.set_title("True Temperature"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
        ax1.set_aspect('equal', adjustable='box'); fig.colorbar(c1, ax=ax1)

        ax2 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 2)
        c2 = ax2.contourf(Xg, Zg, T_pred_plot, levels=50, cmap='jet')
        ax2.set_title("Reconstructed Temperature"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)")
        ax2.set_aspect('equal', adjustable='box'); fig.colorbar(c2, ax=ax2)

        if SAVE_ERROR_MAP:
            ax3 = fig.add_subplot(1, 3, 3)
            c3 = ax3.contourf(Xg, Zg, err_slice, levels=50, cmap='YlOrRd', vmin=0, vmax=vmax)
            ax3.set_title("Absolute Error (Unified Scale)")
            ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
            ax3.set_aspect('equal', adjustable='box'); fig.colorbar(c3, ax=ax3)

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx+1}_improved.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")

    # ---- Save summary ----
    summary = {
        "ranks": {"rx": int(rx), "ry": int(ry), "rz": int(rz), "rn": int(rn)},
        "adaptive": bool(USE_ADAPTIVE_RANKS),
        "energy_threshold": float(ENERGY_THRESH),
        "center_along_N": bool(CENTER_ALONG_N),
        "scale_inputs": bool(SCALE_INPUTS),
        "scale_output_coeff": bool(SCALE_OUTPUT_COEFF),
        "regressor": REGRESSOR,
        "mlp": {
            "hidden_layers": HIDDEN_LAYERS,
            "use_layernorm": USE_LAYERNORM,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "energy_weighted_loss": USE_ENERGY_WEIGHTS,
            "pca_bottleneck": USE_PCA_BOTTLENECK,
            "pca_latent_q": PCA_LATENT_Q if USE_PCA_BOTTLENECK else None
        },
        "random_state": RANDOM_STATE,
        "test_indices": [int(x) for x in idx_test.tolist()],
        "test_indices_one_based": [int(x) for x in (idx_test + 1).tolist()],
        "full3d_mae_each": [float(x) for x in mae_list],
        "full3d_rmse_each": [float(x) for x in rmse_list],
        "full3d_mae_mean": float(np.mean(mae_list)),
        "full3d_rmse_mean": float(np.mean(rmse_list)),
        "y_slice": float(ys[y_slice_index]),
        "unified_error_vmax_from_pod": float(shared_error_max) if shared_error_max is not None else None
    }
    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Outputs in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
