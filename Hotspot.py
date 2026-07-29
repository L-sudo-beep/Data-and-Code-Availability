#!/usr/bin/env python
# -*- coding: utf-8 -*-


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

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, Matern
)
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




PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"

NX, NY, NZ = 75, 51, 103
N_CASES = 89

Y_SLICE = 1.53
Y_PLANE_FOR_RRMSE = 1.53
RELATIVE_ERROR_THRESHOLD = 0.10

ENERGY_THRESH = 0.99

CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

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

FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved"
os.makedirs(FIG_DIR, exist_ok=True)

POD_ERROR_MAX_PATH = (
    r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"
)



LEFT_OUTLET_X_RANGE = (1.69, 1.90)
LEFT_OUTLET_Y_RANGE = (0.67, 2.77)
LEFT_OUTLET_Z_RANGE = (3.06, 6.03)

RIGHT_OUTLET_X_RANGE = (4.65, 4.95)
RIGHT_OUTLET_Y_RANGE = (0.67, 2.77)
RIGHT_OUTLET_Z_RANGE = (3.06, 6.03)

HOTSPOT_TOP_FRACTION = 0.30
COORDINATE_TOLERANCE = 1e-9




def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    if df.shape[0] == N_CASES + 1:
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[0] != N_CASES:
        warnings.warn(
            f"[WARN] Parameter CSV has {df.shape[0]} rows, expected {N_CASES}."
        )
    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain at least 14 columns.")
    return df.iloc[:, :14]


def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required = {"X (m)", "Y (m)", "Z (m)", "Temperature"}
    if not required.issubset(df.columns):
        raise ValueError(f"Snapshot {path} is missing columns: {required}")
    return df


def build_grid_from_df(df: pd.DataFrame):
    xs = np.sort(df["X (m)"].unique())
    ys = np.sort(df["Y (m)"].unique())
    zs = np.sort(df["Z (m)"].unique())
    if (len(xs), len(ys), len(zs)) != (NX, NY, NZ):
        raise ValueError(
            f"Grid size mismatch: {(len(xs), len(ys), len(zs))} "
            f"vs {(NX, NY, NZ)}"
        )
    return xs, ys, zs


def df_to_grid_values(df, xs, ys, zs):
    ix = np.searchsorted(xs, df["X (m)"].to_numpy())
    iy = np.searchsorted(ys, df["Y (m)"].to_numpy())
    iz = np.searchsorted(zs, df["Z (m)"].to_numpy())
    temperature = df["Temperature"].to_numpy()
    grid = np.empty((NX, NY, NZ), dtype=np.float64)
    grid[ix, iy, iz] = temperature
    return grid


def mode_n_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def n_mode_product(T: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    T_permuted = np.moveaxis(T, mode, 0)
    output = U @ T_permuted.reshape(T_permuted.shape[0], -1)
    new_shape = (U.shape[0],) + T_permuted.shape[1:]
    output = output.reshape(new_shape)
    return np.moveaxis(output, 0, mode)


def upsample_if_needed(xs, zs, field_xz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, field_xz
    xi = np.linspace(xs.min(), xs.max(), len(xs) * UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs) * UPSAMPLE_FZ)
    spline = RectBivariateSpline(xs, zs, field_xz)
    return xi, zi, spline(xi, zi)




def interval_mask(
        coordinates: np.ndarray,
        lower: float,
        upper: float,
        tolerance: float = COORDINATE_TOLERANCE,
) -> np.ndarray:
    return (
        (coordinates >= lower - tolerance)
        & (coordinates <= upper + tolerance)
    )


def build_outlet_region_masks(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
):

    left_mask = (
        interval_mask(xs, *LEFT_OUTLET_X_RANGE)[:, None, None]
        & interval_mask(ys, *LEFT_OUTLET_Y_RANGE)[None, :, None]
        & interval_mask(zs, *LEFT_OUTLET_Z_RANGE)[None, None, :]
    )

    right_mask = (
        interval_mask(xs, *RIGHT_OUTLET_X_RANGE)[:, None, None]
        & interval_mask(ys, *RIGHT_OUTLET_Y_RANGE)[None, :, None]
        & interval_mask(zs, *RIGHT_OUTLET_Z_RANGE)[None, None, :]
    )

    combined_mask = left_mask | right_mask
    return left_mask, right_mask, combined_mask


def select_top_outlet_hotspots(
        mean_temperature_field: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        top_fraction: float = HOTSPOT_TOP_FRACTION,
):

    if mean_temperature_field.shape != (NX, NY, NZ):
        raise ValueError(
            f"Expected mean-temperature shape {(NX, NY, NZ)}, "
            f"got {mean_temperature_field.shape}."
        )

    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("HOTSPOT_TOP_FRACTION must be in (0, 1].")

    left_mask, right_mask, combined_mask = build_outlet_region_masks(
        xs, ys, zs
    )

    left_count = int(np.count_nonzero(left_mask))
    right_count = int(np.count_nonzero(right_mask))
    total_count = int(np.count_nonzero(combined_mask))

    if total_count == 0:
        raise RuntimeError(
            "No grid points were found in the configured outlet regions. "
            "Please check the coordinate ranges and grid."
        )

    candidate_flat_indices = np.flatnonzero(combined_mask.ravel())
    i_indices, j_indices, k_indices = np.unravel_index(
        candidate_flat_indices,
        (NX, NY, NZ),
    )

    candidate_temperatures = mean_temperature_field.ravel()[
        candidate_flat_indices
    ]

    region_names = np.where(
        left_mask.ravel()[candidate_flat_indices],
        "Left outlet",
        "Right outlet",
    )

    candidate_df = pd.DataFrame({
        "Point_Index": candidate_flat_indices.astype(int),
        "Outlet_Region": region_names,
        "X (m)": xs[i_indices],
        "Y (m)": ys[j_indices],
        "Z (m)": zs[k_indices],
        "Training_Mean_Temperature (°C)": candidate_temperatures,
    })

    candidate_df = (
        candidate_df
        .sort_values(
            by=["Training_Mean_Temperature (°C)", "Point_Index"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    candidate_df.insert(
        0,
        "Temperature_Rank",
        np.arange(1, len(candidate_df) + 1, dtype=int),
    )

    selected_count = max(
        1,
        int(math.ceil(total_count * top_fraction)),
    )

    selected_df = (
        candidate_df
        .iloc[:selected_count]
        .copy()
        .reset_index(drop=True)
    )

    selected_df.insert(
        1,
        "Selected_Top_Fraction",
        float(top_fraction),
    )

    cutoff_temperature = float(
        selected_df["Training_Mean_Temperature (°C)"].iloc[-1]
    )

    selection_info = {
        "left_outlet_candidate_count": left_count,
        "right_outlet_candidate_count": right_count,
        "total_candidate_count": total_count,
        "top_fraction": float(top_fraction),
        "selected_count": selected_count,
        "selection_count_rule": "ceil(total_candidate_count * top_fraction)",
        "cutoff_training_mean_temperature_C": cutoff_temperature,
    }

    return candidate_df, selected_df, selection_info


def append_hotspot_prediction_metrics(
        selected_df: pd.DataFrame,
        preds: np.ndarray,
        gts: np.ndarray,
        idx_test: np.ndarray,
) -> pd.DataFrame:
    detailed_rows = []

    for _, row in selected_df.iterrows():
        flat_index = int(row["Point_Index"])
        i_index, j_index, k_index = np.unravel_index(
            flat_index,
            (NX, NY, NZ),
        )

        true_series = gts[i_index, j_index, k_index, :]
        predicted_series = preds[i_index, j_index, k_index, :]
        absolute_error_series = np.abs(true_series - predicted_series)

        rrmse_n = float(
            np.linalg.norm(true_series - predicted_series)
            / (np.linalg.norm(true_series) + 1e-9)
        )

        point_data = row.to_dict()
        point_data["rRMSE_n"] = rrmse_n

        for test_position, original_case_index in enumerate(idx_test):
            case_number = int(original_case_index + 1)
            point_data[f"Case_{case_number}_True_T"] = float(
                true_series[test_position]
            )
            point_data[f"Case_{case_number}_Pred_T"] = float(
                predicted_series[test_position]
            )
            point_data[f"Case_{case_number}_AbsError_T"] = float(
                absolute_error_series[test_position]
            )

        detailed_rows.append(point_data)

    return pd.DataFrame(detailed_rows)



def select_rank_by_energy_unfold(
        X: np.ndarray,
        threshold: float,
        max_rank: Optional[int] = None,
):
    mode_size = X.shape[0]
    n_components = (
        mode_size
        if max_rank is None
        else min(mode_size, max_rank)
    )

    U, singular_values, VT = randomized_svd(
        X,
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    energy = singular_values ** 2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)

    rank = int(
        np.searchsorted(cumulative_energy, threshold) + 1
    )
    rank = max(1, min(rank, n_components))

    return U[:, :rank], rank, singular_values


def hosvd_per_mode_energy_train_only(
        T_centered: np.ndarray,
        idx_train: np.ndarray,
        threshold: float,
):
    T_train = T_centered[..., idx_train]

    Ux, rx, Sx = select_rank_by_energy_unfold(
        mode_n_unfold(T_train, 0),
        threshold,
    )
    Uy, ry, Sy = select_rank_by_energy_unfold(
        mode_n_unfold(T_train, 1),
        threshold,
    )
    Uz, rz, Sz = select_rank_by_energy_unfold(
        mode_n_unfold(T_train, 2),
        threshold,
    )
    Un, rn, Sn = select_rank_by_energy_unfold(
        mode_n_unfold(T_train, 3),
        threshold,
    )

    core = T_train.copy()
    core = n_mode_product(core, Ux.T, 0)
    core = n_mode_product(core, Uy.T, 1)
    core = n_mode_product(core, Uz.T, 2)
    core = n_mode_product(core, Un.T, 3)

    ranks = (rx, ry, rz, rn)

    singular_values = {
        "Sx": Sx,
        "Sy": Sy,
        "Sz": Sz,
        "Sn": Sn,
    }

    return (
        core,
        [Ux, Uy, Uz, Un],
        ranks,
        singular_values,
    )


def project_case_to_coeff(
        T_case: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        core: np.ndarray,
) -> np.ndarray:
    reduced_field = n_mode_product(
        n_mode_product(
            n_mode_product(
                T_case.copy(),
                Ux.T,
                0,
            ),
            Uy.T,
            1,
        ),
        Uz.T,
        2,
    )

    b = reduced_field.reshape(-1)
    A = core.reshape(-1, core.shape[-1])

    coefficient, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    return coefficient


class MLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers,
            use_layernorm=USE_LAYERNORM,
            dropout=DROPOUT,
    ):
        super().__init__()

        def block(input_dim, output_dim):
            layers = [
                nn.Linear(input_dim, output_dim)
            ]

            if use_layernorm:
                layers.append(
                    nn.LayerNorm(output_dim)
                )

            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

            return nn.Sequential(*layers)

        layers = []
        previous_dim = in_dim

        for hidden_dim in hidden_layers:
            layers.append(
                block(previous_dim, hidden_dim)
            )
            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, out_dim)
        )

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
        weight_decay=WEIGHT_DECAY,
) -> nn.Module:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MLP(
        X_train.shape[1],
        Y_train.shape[1],
        hidden_layers,
        use_layernorm=use_layernorm,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )
    Y_tensor = torch.tensor(
        Y_train,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        Y_tensor,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    if weight_per_dim is None:
        weight = torch.ones(
            Y_train.shape[1],
            dtype=torch.float32,
            device=device,
        )
    else:
        normalized_weight = (
            weight_per_dim.astype(np.float32)
        )
        normalized_weight = normalized_weight / (
            np.mean(normalized_weight) + 1e-12
        )
        weight = torch.tensor(
            normalized_weight,
            dtype=torch.float32,
            device=device,
        )

    model.train()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            prediction = model(xb)
            difference = prediction - yb

            loss = torch.mean(
                torch.sum(
                    (difference ** 2) * (weight ** 2),
                    dim=1,
                )
            )

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"[MLP] Epoch {epoch + 1}/{epochs} | "
                f"Loss={loss.item():.6f}"
            )

    return model


def predict_mlp_torch(
        model: nn.Module,
        X: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=device,
        )
        prediction = model(X_tensor).cpu().numpy()

    return prediction


def choose_regressor(name: str, output_dim: int):
    name = name.upper()

    if name == "GPR":
        if GPR_KERNEL.upper() == "RBF":
            base_kernel = RBF(
                length_scale=np.ones(14),
                length_scale_bounds=(1e-2, 1e3),
            )
        else:
            base_kernel = Matern(
                length_scale=np.ones(14),
                length_scale_bounds=(1e-2, 1e3),
                nu=1.5,
            )

        kernel = (
            C(1.0, (1e-3, 1e3)) * base_kernel
            + WhiteKernel(
                noise_level=1e-4,
                noise_level_bounds=(1e-10, 1e-1),
            )
        )

        base = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=0.0,
        )

        return MultiOutputRegressor(
            base,
            n_jobs=None,
        )

    if name == "SVR":
        base = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            cache_size=SVR_CACHE_MB,
        )

        return MultiOutputRegressor(
            base,
            n_jobs=None,
        )

    if name == "MLP_TORCH":
        return None

    raise ValueError(
        "Unknown REGRESSOR. Supported: MLP_TORCH, SVR, GPR."
    )




def main():
    np.random.seed(RANDOM_STATE)
    start_main_time = time.time()

    print("=== Load boundary-condition parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read the first snapshot and build the grid ===")
    first_snapshot = read_one_snapshot_csv(
        os.path.join(SNAPSHOT_DIR, "1.csv")
    )
    xs, ys, zs = build_grid_from_df(first_snapshot)

    y_slice_index = int(
        np.argmin(np.abs(ys - Y_SLICE))
    )
    print(
        f"Visualization Y slice: requested={Y_SLICE:.4f} m, "
        f"actual={ys[y_slice_index]:.4f} m, index={y_slice_index}"
    )

    y_plane_rrmse_index = int(
        np.argmin(np.abs(ys - Y_PLANE_FOR_RRMSE))
    )
    y_plane_rrmse_value = float(
        ys[y_plane_rrmse_index]
    )
    print(
        f"rRMSE Y plane: requested={Y_PLANE_FOR_RRMSE:.4f} m, "
        f"actual={y_plane_rrmse_value:.4f} m, "
        f"index={y_plane_rrmse_index}"
    )

    print("=== Build the four-dimensional temperature tensor ===")
    T = np.empty(
        (NX, NY, NZ, N_CASES),
        dtype=np.float64,
    )

    T[..., 0] = df_to_grid_values(
        first_snapshot,
        xs,
        ys,
        zs,
    )

    for case_number in range(2, N_CASES + 1):
        snapshot_path = os.path.join(
            SNAPSHOT_DIR,
            f"{case_number}.csv",
        )
        snapshot_df = read_one_snapshot_csv(
            snapshot_path
        )
        T[..., case_number - 1] = df_to_grid_values(
            snapshot_df,
            xs,
            ys,
            zs,
        )

        if case_number % 20 == 0 or case_number == N_CASES:
            print(
                f"  Loaded {case_number}/{N_CASES}"
            )

    print("=== Fixed train/test split ===")
    TEST_IDX_ONE_BASED = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    idx_test = np.asarray(
        [case_number - 1 for case_number in TEST_IDX_ONE_BASED],
        dtype=int,
    )
    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(
        idx_all,
        idx_test,
        assume_unique=False,
    )

    print(
        f"  Train size={len(idx_train)}, "
        f"test size={len(idx_test)}"
    )
    print(
        f"  Test cases (1-based)={TEST_IDX_ONE_BASED}"
    )

    # Training-set mean only
    if CENTER_ALONG_N:
        T_mean_train = T[..., idx_train].mean(
            axis=3,
            keepdims=True,
        )
        T_centered = T - T_mean_train
    else:
        T_mean_train = None
        T_centered = T.copy()

    print(
        f"=== Tucker decomposition on training data only; "
        f"energy threshold={ENERGY_THRESH} ==="
    )

    (
        G_train,
        [Ux, Uy, Uz, Un_train],
        ranks,
        singular_values,
    ) = hosvd_per_mode_energy_train_only(
        T_centered,
        idx_train,
        ENERGY_THRESH,
    )

    rx, ry, rz, rn = ranks

    print(
        f"Selected ranks: "
        f"rx={rx}, ry={ry}, rz={rz}, rn={rn}"
    )

    print("=== Project training fields to Tucker coefficients ===")
    Y_train_raw = np.zeros(
        (len(idx_train), rn),
        dtype=np.float64,
    )

    for local_index, original_case_index in enumerate(idx_train):
        Y_train_raw[local_index, :] = project_case_to_coeff(
            T_centered[..., original_case_index],
            Ux,
            Uy,
            Uz,
            G_train,
        )

    X_train_raw = X_params[idx_train]
    X_test_raw = X_params[idx_test]

    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        core_mode_norm = np.linalg.norm(
            G_train.reshape(-1, G_train.shape[-1]),
            axis=0,
        )
        energy_weights = np.clip(
            core_mode_norm
            / (core_mode_norm.max() + 1e-12),
            1e-3,
            None,
        )
    else:
        energy_weights = None

    if SCALE_INPUTS:
        x_scaler = StandardScaler().fit(
            X_train_raw
        )
        X_train = x_scaler.transform(
            X_train_raw
        )
        X_test = x_scaler.transform(
            X_test_raw
        )
    else:
        x_scaler = None
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    pca_model = None
    output_scaler = None

    if USE_PCA_BOTTLENECK:
        actual_pca_components = min(
            PCA_LATENT_Q,
            Y_train_raw.shape[0],
            Y_train_raw.shape[1],
        )

        pca_model = PCA(
            n_components=actual_pca_components,
            random_state=RANDOM_STATE,
        )
        Z_train = pca_model.fit_transform(
            Y_train_raw
        )

        if SCALE_OUTPUT_COEFF:
            output_scaler = StandardScaler().fit(
                Z_train
            )
            regressor_target = output_scaler.transform(
                Z_train
            )
        else:
            regressor_target = Z_train

        print(
            f"=== Train regressor {REGRESSOR} "
            f"on PCA scores, q={actual_pca_components} ==="
        )

        training_start = time.time()

        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(
                X_train,
                regressor_target,
            )
            predicted_scaled_output = predict_mlp_torch(
                model,
                X_test,
            )
        else:
            model = choose_regressor(
                REGRESSOR,
                regressor_target.shape[1],
            )
            model.fit(
                X_train,
                regressor_target,
            )
            predicted_scaled_output = model.predict(
                X_test
            )

        print(
            f"Regressor training and batch prediction time: "
            f"{time.time() - training_start:.2f} s"
        )

        predicted_pca_scores = (
            output_scaler.inverse_transform(
                predicted_scaled_output
            )
            if output_scaler is not None
            else predicted_scaled_output
        )

        Y_pred = pca_model.inverse_transform(
            predicted_pca_scores
        )

    else:
        if SCALE_OUTPUT_COEFF:
            output_scaler = StandardScaler().fit(
                Y_train_raw
            )
            regressor_target = output_scaler.transform(
                Y_train_raw
            )
        else:
            regressor_target = Y_train_raw

        print(
            f"=== Train regressor {REGRESSOR}; "
            f"energy-weighted={USE_ENERGY_WEIGHTS} ==="
        )

        training_start = time.time()

        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(
                X_train,
                regressor_target,
                weight_per_dim=energy_weights,
            )
            predicted_scaled_output = predict_mlp_torch(
                model,
                X_test,
            )
        else:
            model = choose_regressor(
                REGRESSOR,
                regressor_target.shape[1],
            )
            model.fit(
                X_train,
                regressor_target,
            )
            predicted_scaled_output = model.predict(
                X_test
            )

        print(
            f"Regressor training and batch prediction time: "
            f"{time.time() - training_start:.2f} s"
        )

        Y_pred = (
            output_scaler.inverse_transform(
                predicted_scaled_output
            )
            if output_scaler is not None
            else predicted_scaled_output
        )

    print("=== Reconstruct test three-dimensional temperature fields ===")
    predicted_fields = []
    ground_truth_fields = []

    for test_position, original_case_index in enumerate(idx_test):
        coefficient = Y_pred[test_position]

        reduced_spatial_core = np.tensordot(
            G_train,
            coefficient,
            axes=([3], [0]),
        )

        reconstructed_x = n_mode_product(
            reduced_spatial_core,
            Ux,
            0,
        )
        reconstructed_xy = n_mode_product(
            reconstructed_x,
            Uy,
            1,
        )
        reconstructed_xyz = n_mode_product(
            reconstructed_xy,
            Uz,
            2,
        )

        if CENTER_ALONG_N:
            predicted_field = (
                reconstructed_xyz
                + T_mean_train[..., 0]
            )
        else:
            predicted_field = reconstructed_xyz

        predicted_fields.append(predicted_field)
        ground_truth_fields.append(
            T[..., original_case_index]
        )

    preds = np.stack(
        predicted_fields,
        axis=-1,
    )
    gts = np.stack(
        ground_truth_fields,
        axis=-1,
    )

    print("=== Calculate accuracy metrics ===")

    mae_list = []
    rmse_list = []
    rrmse_list = []
    rrmse_plane_list = []
    percentage_below_threshold_list = []

    for test_position in range(preds.shape[-1]):
        true_full = gts[..., test_position].ravel()
        predicted_full = preds[..., test_position].ravel()

        mae = mean_absolute_error(
            true_full,
            predicted_full,
        )
        rmse = math.sqrt(
            mean_squared_error(
                true_full,
                predicted_full,
            )
        )
        rrmse = float(
            np.linalg.norm(true_full - predicted_full)
            / (np.linalg.norm(true_full) + 1e-9)
        )

        mae_list.append(float(mae))
        rmse_list.append(float(rmse))
        rrmse_list.append(rrmse)

        true_plane = gts[
            :,
            y_plane_rrmse_index,
            :,
            test_position,
        ].ravel()
        predicted_plane = preds[
            :,
            y_plane_rrmse_index,
            :,
            test_position,
        ].ravel()

        plane_rrmse = float(
            np.linalg.norm(true_plane - predicted_plane)
            / (np.linalg.norm(true_plane) + 1e-9)
        )
        rrmse_plane_list.append(plane_rrmse)

        relative_error = (
            np.abs(true_full - predicted_full)
            / (np.abs(true_full) + 1e-9)
        )

        percentage_below_threshold = float(
            np.mean(
                relative_error
                < RELATIVE_ERROR_THRESHOLD
            )
            * 100.0
        )

        percentage_below_threshold_list.append(
            percentage_below_threshold
        )

        print(
            f"  Case {idx_test[test_position] + 1:2d}: "
            f"MAE={mae:.4f} °C, "
            f"RMSE={rmse:.4f} °C, "
            f"rRMSE={rrmse:.4f}, "
            f"plane rRMSE={plane_rrmse:.4f}, "
            f"points below {RELATIVE_ERROR_THRESHOLD * 100:.0f}% "
            f"relative error={percentage_below_threshold:.2f}%"
        )

    print(
        f"\nMean full-field MAE={np.mean(mae_list):.4f} °C, "
        f"RMSE={np.mean(rmse_list):.4f} °C, "
        f"rRMSE={np.mean(rrmse_list):.4f}"
    )


    print("\n=== Select hottest top 30% points in rack-outlet regions ===")

    if T_mean_train is not None:
        training_mean_temperature_field = (
            T_mean_train[..., 0]
        )
    else:
        training_mean_temperature_field = (
            T[..., idx_train].mean(axis=3)
        )

    (
        outlet_candidate_df,
        outlet_top30_df,
        outlet_selection_info,
    ) = select_top_outlet_hotspots(
        mean_temperature_field=(
            training_mean_temperature_field
        ),
        xs=xs,
        ys=ys,
        zs=zs,
        top_fraction=HOTSPOT_TOP_FRACTION,
    )

    selected_left_count = int(
        np.sum(
            outlet_top30_df["Outlet_Region"]
            == "Left outlet"
        )
    )
    selected_right_count = int(
        np.sum(
            outlet_top30_df["Outlet_Region"]
            == "Right outlet"
        )
    )

    outlet_selection_info[
        "selected_left_outlet_count"
    ] = selected_left_count
    outlet_selection_info[
        "selected_right_outlet_count"
    ] = selected_right_count

    print(
        "Left outlet candidate points: "
        f"{outlet_selection_info['left_outlet_candidate_count']}"
    )
    print(
        "Right outlet candidate points: "
        f"{outlet_selection_info['right_outlet_candidate_count']}"
    )
    print(
        "Total outlet candidate points: "
        f"{outlet_selection_info['total_candidate_count']}"
    )
    print(
        f"Selected top fraction: "
        f"{HOTSPOT_TOP_FRACTION * 100:.1f}%"
    )
    print(
        "Selected point count: "
        f"{outlet_selection_info['selected_count']} "
        "(ceil rule)"
    )
    print(
        "Selected points by region: "
        f"left={selected_left_count}, "
        f"right={selected_right_count}"
    )
    print(
        "Cutoff training mean temperature: "
        f"{outlet_selection_info['cutoff_training_mean_temperature_C']:.6f} °C"
    )

    outlet_candidate_path = os.path.join(
        FIG_DIR,
        "outlet_candidate_points_all_tucker.csv",
    )
    outlet_top30_path = os.path.join(
        FIG_DIR,
        "outlet_hotspots_top30_percent_tucker.csv",
    )

    outlet_candidate_df.to_csv(
        outlet_candidate_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    outlet_top30_df.to_csv(
        outlet_top30_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print(
        f"Saved all outlet candidate points to: "
        f"{outlet_candidate_path}"
    )
    print(
        f"Saved hottest top 30% points to: "
        f"{outlet_top30_path}"
    )

    outlet_top30_detailed_df = (
        append_hotspot_prediction_metrics(
            selected_df=outlet_top30_df,
            preds=preds,
            gts=gts,
            idx_test=idx_test,
        )
    )

    outlet_top30_detailed_path = os.path.join(
        FIG_DIR,
        "outlet_hotspots_top30_percent_detailed_tucker.csv",
    )

    outlet_top30_detailed_df.to_csv(
        outlet_top30_detailed_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print(
        "Saved top-30% outlet hotspot coordinates, "
        "temperatures and prediction errors to: "
        f"{outlet_top30_detailed_path}"
    )


    metrics_df = pd.DataFrame({
        "Case": [
            int(case_index) + 1
            for case_index in idx_test
        ],
        "MAE": mae_list,
        "RMSE": rmse_list,
        "rRMSE": rrmse_list,
        f"rRMSE_plane_Y={y_plane_rrmse_value:.2f}m": (
            rrmse_plane_list
        ),
        (
            "Points_Percentage_RelErr_"
            f"<{int(RELATIVE_ERROR_THRESHOLD * 100)}%"
        ): percentage_below_threshold_list,
    }).sort_values("Case")

    per_case_metrics_path = os.path.join(
        FIG_DIR,
        "per_case_metrics.csv",
    )

    metrics_df.to_csv(
        per_case_metrics_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        metrics_df["Case"],
        metrics_df["MAE"],
        "-o",
        label="MAE (°C), full field",
    )
    plt.plot(
        metrics_df["Case"],
        metrics_df["RMSE"],
        "-s",
        label="RMSE (°C), full field",
    )
    plt.plot(
        metrics_df["Case"],
        metrics_df["rRMSE"],
        "-^",
        label="rRMSE, full field",
    )
    plt.plot(
        metrics_df["Case"],
        metrics_df[
            f"rRMSE_plane_Y={y_plane_rrmse_value:.2f}m"
        ],
        "-x",
        label=(
            f"rRMSE, Y={y_plane_rrmse_value:.2f} m"
        ),
    )
    plt.xlabel("Case")
    plt.ylabel("Error metric")
    plt.title(
        "Tucker+MLP Error Metrics by Test Case"
    )
    plt.legend()
    plt.grid(
        True,
        which="both",
        linestyle="--",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIG_DIR,
            "error_metrics_compare.png",
        ),
        dpi=200,
    )
    plt.close()


    if os.path.exists(POD_ERROR_MAX_PATH):
        try:
            shared_error_max = float(
                np.load(POD_ERROR_MAX_PATH)
            )
            if (
                not np.isfinite(shared_error_max)
                or shared_error_max <= 0
            ):
                shared_error_max = None
            else:
                print(
                    "[Info] Unified error-map vmax from POD: "
                    f"{shared_error_max:.4f}"
                )
        except Exception:
            shared_error_max = None
    else:
        shared_error_max = None

    print(
        "=== Plot Y-slice maps "
        "(ground truth, reconstruction and absolute error) ==="
    )

    for test_position, original_case_index in enumerate(idx_test):
        true_slice = gts[
            :,
            y_slice_index,
            :,
            test_position,
        ]
        predicted_slice = preds[
            :,
            y_slice_index,
            :,
            test_position,
        ]
        error_slice = np.abs(
            true_slice - predicted_slice
        )

        error_vmax = (
            shared_error_max
            if shared_error_max is not None
            else float(np.nanmax(error_slice))
        )

        xs_plot, zs_plot, true_plot = upsample_if_needed(
            xs,
            zs,
            true_slice.T,
        )
        _, _, predicted_plot = upsample_if_needed(
            xs,
            zs,
            predicted_slice.T,
        )
        _, _, error_plot = upsample_if_needed(
            xs,
            zs,
            error_slice.T,
        )

        X_grid, Z_grid = np.meshgrid(
            xs_plot,
            zs_plot,
            indexing="ij",
        )

        figure = plt.figure(figsize=(18, 5))
        plt.suptitle(
            f"Y-slice Reconstruction | "
            f"Case {original_case_index + 1}",
            fontsize=16,
        )

        axis_true = figure.add_subplot(1, 3, 1)
        contour_true = axis_true.contourf(
            X_grid,
            Z_grid,
            true_plot.T,
            levels=LEVELS,
            cmap="jet",
        )
        axis_true.set_title("True Temperature")
        axis_true.set_xlabel("X (m)")
        axis_true.set_ylabel("Z (m)")
        axis_true.set_aspect(
            "equal",
            adjustable="box",
        )
        figure.colorbar(
            contour_true,
            ax=axis_true,
        )

        axis_predicted = figure.add_subplot(1, 3, 2)
        contour_predicted = axis_predicted.contourf(
            X_grid,
            Z_grid,
            predicted_plot.T,
            levels=LEVELS,
            cmap="jet",
            vmin=contour_true.get_clim()[0],
            vmax=contour_true.get_clim()[1],
        )
        axis_predicted.set_title(
            "Reconstructed Temperature"
        )
        axis_predicted.set_xlabel("X (m)")
        axis_predicted.set_ylabel("Z (m)")
        axis_predicted.set_aspect(
            "equal",
            adjustable="box",
        )
        figure.colorbar(
            contour_predicted,
            ax=axis_predicted,
        )

        if SAVE_ERROR_MAP:
            axis_error = figure.add_subplot(1, 3, 3)
            contour_error = axis_error.contourf(
                X_grid,
                Z_grid,
                error_plot.T,
                levels=LEVELS,
                cmap="YlOrRd",
                vmin=0,
                vmax=error_vmax,
            )
            axis_error.set_title(
                f"Absolute Error (vmax={error_vmax:.2f})"
            )
            axis_error.set_xlabel("X (m)")
            axis_error.set_ylabel("Z (m)")
            axis_error.set_aspect(
                "equal",
                adjustable="box",
            )
            figure.colorbar(
                contour_error,
                ax=axis_error,
            )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        output_path = os.path.join(
            FIG_DIR,
            f"yslice_case_{original_case_index + 1}.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)


    def cumulative_energy_preview(singular_values):
        energy = singular_values ** 2
        cumulative = np.cumsum(energy) / np.sum(energy)
        preview_length = min(50, len(singular_values))
        return cumulative[:preview_length].tolist()

    summary = {
        "model_type": "Tucker_HOSVD",
        "decomposition_set": "TRAIN",
        "ranks": {
            "rx": int(rx),
            "ry": int(ry),
            "rz": int(rz),
            "rn": int(rn),
        },
        "energy_threshold_per_mode": float(
            ENERGY_THRESH
        ),
        "config": {
            "center_along_N": bool(
                CENTER_ALONG_N
            ),
            "scale_inputs": bool(
                SCALE_INPUTS
            ),
            "scale_output_coeff": bool(
                SCALE_OUTPUT_COEFF
            ),
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
            "pca_latent_q": (
                PCA_LATENT_Q
                if USE_PCA_BOTTLENECK
                else None
            ),
        },
        "random_state": RANDOM_STATE,
        "test_indices_one_based": [
            int(value)
            for value in (idx_test + 1).tolist()
        ],
        "metrics_summary": {
            "full3d_mae_mean": float(
                np.mean(mae_list)
            ),
            "full3d_rmse_mean": float(
                np.mean(rmse_list)
            ),
            "full3d_rRMSE_mean": float(
                np.mean(rrmse_list)
            ),
            "plane_rRMSE_mean": {
                "y_value": float(
                    y_plane_rrmse_value
                ),
                "value": float(
                    np.mean(rrmse_plane_list)
                ),
            },
            "points_percentage_below_threshold": {
                "threshold": float(
                    RELATIVE_ERROR_THRESHOLD
                ),
                "mean_percentage": float(
                    np.mean(
                        percentage_below_threshold_list
                    )
                ),
                "each_case_percentage": [
                    float(value)
                    for value in (
                        percentage_below_threshold_list
                    )
                ],
            },
            "full3d_mae_each": [
                float(value)
                for value in mae_list
            ],
            "full3d_rmse_each": [
                float(value)
                for value in rmse_list
            ],
            "full3d_rRMSE_each": [
                float(value)
                for value in rrmse_list
            ],
            "plane_rRMSE_each": [
                float(value)
                for value in rrmse_plane_list
            ],
        },
        "outlet_hotspot_selection": {
            "selection_temperature_field": (
                "training-set mean temperature"
            ),
            "left_outlet_region": {
                "x_range_m": list(
                    LEFT_OUTLET_X_RANGE
                ),
                "y_range_m": list(
                    LEFT_OUTLET_Y_RANGE
                ),
                "z_range_m": list(
                    LEFT_OUTLET_Z_RANGE
                ),
            },
            "right_outlet_region": {
                "x_range_m": list(
                    RIGHT_OUTLET_X_RANGE
                ),
                "y_range_m": list(
                    RIGHT_OUTLET_Y_RANGE
                ),
                "z_range_m": list(
                    RIGHT_OUTLET_Z_RANGE
                ),
            },
            **outlet_selection_info,
        },
        "y_slice_visualization": {
            "y_value": float(
                ys[y_slice_index]
            ),
            "unified_error_vmax_from_pod": (
                float(shared_error_max)
                if shared_error_max is not None
                else None
            ),
        },
        "cumulative_energy_curves": {
            "mode_x": cumulative_energy_preview(
                singular_values["Sx"]
            ),
            "mode_y": cumulative_energy_preview(
                singular_values["Sy"]
            ),
            "mode_z": cumulative_energy_preview(
                singular_values["Sz"]
            ),
            "mode_n": cumulative_energy_preview(
                singular_values["Sn"]
            ),
        },
        "output_paths": {
            "per_case_metrics_csv": os.path.abspath(
                per_case_metrics_path
            ),
            "outlet_candidate_points_all_csv": os.path.abspath(
                outlet_candidate_path
            ),
            "outlet_hotspots_top30_percent_csv": os.path.abspath(
                outlet_top30_path
            ),
            "outlet_hotspots_top30_percent_detailed_csv": os.path.abspath(
                outlet_top30_detailed_path
            ),
            "metrics_summary_json": os.path.abspath(
                os.path.join(
                    FIG_DIR,
                    "metrics_summary.json",
                )
            ),
        },
        "execution_time_seconds": float(
            time.time() - start_main_time
        ),
    }

    summary_path = os.path.join(
        FIG_DIR,
        "metrics_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Final outlet hotspot summary ===")
    print(
        f"Candidate points: "
        f"{outlet_selection_info['total_candidate_count']}"
    )
    print(
        f"Selected hottest top "
        f"{HOTSPOT_TOP_FRACTION * 100:.1f}%: "
        f"{outlet_selection_info['selected_count']} points"
    )
    print(
        f"Left/right selected counts: "
        f"{selected_left_count}/{selected_right_count}"
    )
    print(
        f"Total execution time: "
        f"{time.time() - start_main_time:.2f} s"
    )
    print(
        f"Outputs saved in: "
        f"{os.path.abspath(FIG_DIR)}"
    )


if __name__ == "__main__":
    np.set_printoptions(
        precision=4,
        suppress=True,
    )
    pd.set_option(
        "display.width",
        160,
    )
    pd.set_option(
        "display.max_columns",
        50,
    )

    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
