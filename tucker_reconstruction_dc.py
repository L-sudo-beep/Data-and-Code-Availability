#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker Decomposition + Data-Driven Regression to Reconstruct Data Center Temperature Field
-----------------------------------------------------------------------------------------
Author: (your name)
Date: 2025-10-15

• Tensor construction: T ∈ R^{X,Y,Z,N} with shape (75, 51, 103, 89)
• Tucker (HOSVD) with target ranks (rx, ry, rz, rn) = (15, 15, 15, 20)
• Learn mapping: 14D operating parameters  ->  N-mode factor (size rn)
  - Default regressor: Gaussian Process Regression (Kriging-like). 
  - Optional: MLP neural network.
• Train / Test split: random 9 test, 80 train (random_state fixed)
• Output:
  - Metrics (MAE / RMSE) on full 3D field and Y=1.71 m slice
  - Comparison plots (real vs predicted) on Y=1.71 m for test samples
  - Saved figures into ./figures/

Note:
- Modify the PARAMS_PATH and SNAPSHOT_DIR to your actual Windows paths before running.
- The script assumes all 89 snapshots share the same structured grid coordinates.
- Snapshot CSV encoding is UTF-8; parameter CSV encoding is UTF-16 little endian.
"""

import os
import sys
import math
import json
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy.typing import NDArray


# -------------------------- User Config --------------------------
# !!! Update these paths to your environment (Windows paths are okay here).

PARAMS_PATH   = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR  = r"C:\Users\Lenovo\Desktop\insert"                   # contains 1.csv ... 89.csv (UTF-8)

# Grid info (given by user)
NX, NY, NZ = 75, 51, 103
N_CASES    = 89
Y_SLICE    = 1.71  # meters

# Tucker target ranks
RANKS = (20, 20, 20, 25)  # (rx, ry, rz, rn)

# Regression model: "GPR" (Kriging-like) or "MLP"
REGRESSOR = "GPR"
RANDOM_STATE = 42

# Figures output folder
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------


def read_params_csv(path: str) -> pd.DataFrame:
    """Read the 14-parameter CSV (UTF-16 LE, unknown delimiter allowed)."""
    # sep=None lets pandas try to infer delimiter (comma/tab)
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    # Basic validation
    if df.shape[0] != N_CASES + 1:
        warnings.warn(f"[WARN] Expected 90 rows (1 header + 89 cases), got {df.shape[0]}.")
    if df.shape[1] != 14:
        warnings.warn(f"[WARN] Expected 14 columns of parameters, got {df.shape[1]}.")
    return df


def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    """Read a single snapshot CSV (UTF-8) with columns: X (m), Y (m), Z (m), Temperature"""
    df = pd.read_csv(path, encoding="utf-8")
    expected_cols = {"X (m)", "Y (m)", "Z (m)", "Temperature"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Snapshot file {path} missing required columns: {expected_cols}")
    return df


def build_grid_from_df(df: pd.DataFrame) -> Tuple[NDArray, NDArray, NDArray]:
    """Extract sorted unique coordinates from a snapshot and validate counts."""
    xs = np.sort(df["X (m)"].unique())
    ys = np.sort(df["Y (m)"].unique())
    zs = np.sort(df["Z (m)"].unique())
    if (len(xs), len(ys), len(zs)) != (NX, NY, NZ):
        raise ValueError(f"Grid size mismatch: {(len(xs), len(ys), len(zs))} vs expected {(NX, NY, NZ)}")
    return xs, ys, zs


def df_to_grid_values(df: pd.DataFrame, xs: NDArray, ys: NDArray, zs: NDArray) -> NDArray:
    """Vectorized mapping from scattered rows to structured 3D array [NX, NY, NZ]."""
    # Find integer indices along each axis
    ix = np.searchsorted(xs, df["X (m)"].to_numpy())
    iy = np.searchsorted(ys, df["Y (m)"].to_numpy())
    iz = np.searchsorted(zs, df["Z (m)"].to_numpy())
    temp = df["Temperature"].to_numpy()

    # Sanity checks
    if not (np.all(xs[ix] == df["X (m)"]) and np.all(ys[iy] == df["Y (m)"]) and np.all(zs[iz] == df["Z (m)"])):
        warnings.warn("Some coordinates did not match the structured grid exactly (precision issues?).")

    grid = np.empty((NX, NY, NZ), dtype=np.float64)
    grid[ix, iy, iz] = temp
    return grid


def mode_n_unfold(T: NDArray, mode: int) -> NDArray:
    """Unfold tensor T along mode (0-based)."""
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def n_mode_product(T: NDArray, U: NDArray, mode: int) -> NDArray:
    """
    Compute n-mode product: T ×_mode U, where U shape is (J, I_mode).
    Result shape is (d0,...,d_{mode-1}, J, d_{mode+1},...,d_{N-1}).
    """
    T_perm = np.moveaxis(T, mode, 0)          # (I_mode, ...)
    out = U @ T_perm.reshape(T_perm.shape[0], -1)  # (J, prod(others))
    new_shape = (U.shape[0],) + T_perm.shape[1:]
    out = out.reshape(new_shape)
    return np.moveaxis(out, 0, mode)


def hosvd(T: NDArray, ranks: Tuple[int, int, int, int]) -> Tuple[NDArray, List[NDArray]]:
    """
    Higher-Order SVD (Tucker) using randomized_svd for efficiency.
    Returns: core tensor G and factor matrices [Ux, Uy, Uz, Un]
    """
    rx, ry, rz, rn = ranks
    Us = []

    # Mode-0 (X)
    X0 = mode_n_unfold(T, 0)
    Ux, Sx, VTx = randomized_svd(X0, n_components=rx, random_state=RANDOM_STATE)
    Us.append(Ux)

    # Mode-1 (Y)
    X1 = mode_n_unfold(T, 1)
    Uy, Sy, VTy = randomized_svd(X1, n_components=ry, random_state=RANDOM_STATE)
    Us.append(Uy)

    # Mode-2 (Z)
    X2 = mode_n_unfold(T, 2)
    Uz, Sz, VTz = randomized_svd(X2, n_components=rz, random_state=RANDOM_STATE)
    Us.append(Uz)

    # Mode-3 (N)
    X3 = mode_n_unfold(T, 3)
    Un, Sn, VTn = randomized_svd(X3, n_components=rn, random_state=RANDOM_STATE)
    Us.append(Un)

    # Core tensor: G = T ×1 Ux^T ×2 Uy^T ×3 Uz^T ×4 Un^T
    G = T.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)

    return G, Us


def reconstruct_from_coeff(G: NDArray, Ux: NDArray, Uy: NDArray, Uz: NDArray, coeff: NDArray) -> NDArray:
    """
    Reconstruct full field from a single N-mode coefficient vector of length rn.
    Steps: 
      M = tensordot(G, coeff, axes=([3],[0])) -> (rx, ry, rz)
      T = M ×1 Ux ×2 Uy ×3 Uz -> (NX, NY, NZ)
    """
    # Contract along N-mode of core
    M = np.tensordot(G, coeff, axes=([3], [0]))  # (rx, ry, rz)
    # Back-project spatially
    T1 = n_mode_product(M, Ux, 0)  # (NX, ry, rz)
    T2 = n_mode_product(T1, Uy, 1) # (NX, NY, rz)
    T3 = n_mode_product(T2, Uz, 2) # (NX, NY, NZ)
    return T3


def choose_regressor(model_name: str):
    if model_name.upper() == "GPR":
        # RBF length_scale set heuristically; noise handled by WhiteKernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(14), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        base = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE, alpha=0.0)
        model = MultiOutputRegressor(base, n_jobs=None)
    elif model_name.upper() == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
                             alpha=1e-4, batch_size=16, learning_rate_init=1e-3,
                             max_iter=5000, random_state=RANDOM_STATE, verbose=True)
    else:
        raise ValueError("Unknown REGRESSOR. Use 'GPR' or 'MLP'.")
    return model


def main():
    print("=== Loading parameters CSV ===")
    params_df = read_params_csv(PARAMS_PATH)
    # Expect 89 rows for cases after header, but if read_csv already parsed headers, we have 89 rows.
    # Just ensure we have 89 rows: if extra header line counted, drop first if needed.
    if params_df.shape[0] == N_CASES + 1:
        # Likely the first line is header duplicated as data; try drop first row.
        params_df = params_df.iloc[1:].reset_index(drop=True)

    # Extract features matrix X (89 x 14)
    X_params = params_df.iloc[:, :14].to_numpy(dtype=np.float64)
    if X_params.shape != (N_CASES, 14):
        warnings.warn(f"[WARN] Parameter matrix shape {X_params.shape} != (89,14). Proceeding anyway.")

    print("=== Reading first snapshot to build grid ===")
    first_path = os.path.join(SNAPSHOT_DIR, "1.csv")
    df0 = read_one_snapshot_csv(first_path)
    xs, ys, zs = build_grid_from_df(df0)
    print(f"Grid sizes: NX={len(xs)}, NY={len(ys)}, NZ={len(zs)}")

    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))
    print(f"Y slice target = {Y_SLICE} m, using nearest grid index = {y_slice_index}, y={ys[y_slice_index]:.4f} m")

    print("=== Building 4D tensor T (this may take a while) ===")
    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    # Fill case 1 from df0
    T[..., 0] = df_to_grid_values(df0, xs, ys, zs)

    # Remaining cases
    for i in range(2, N_CASES + 1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Snapshot file not found: {p}")
        dfi = read_one_snapshot_csv(p)
        T[..., i - 1] = df_to_grid_values(dfi, xs, ys, zs)
        if i % 10 == 0 or i == N_CASES:
            print(f"  Loaded {i}/{N_CASES} snapshots")

    # Optional: mean-centering across N (could help SVD); here we keep original data.
    # If needed, uncomment:
    # T_mean = T.mean(axis=3, keepdims=True)
    # T_centered = T - T_mean
    # (Use T_centered for HOSVD and add T_mean back after reconstruction)

    print("=== HOSVD / Tucker decomposition ===")
    G, [Ux, Uy, Uz, Un] = hosvd(T, RANKS)
    rx, ry, rz, rn = RANKS
    print(f"Core shape: {G.shape}, factor shapes: Ux{Ux.shape}, Uy{Uy.shape}, Uz{Uz.shape}, Un{Un.shape}")

    # Targets for regression: rows of Un correspond to each case's N-mode coefficients
    Y_coeff = Un  # shape (89, rn)

    print("=== Train/Test split ===")
    idx_all = np.arange(N_CASES)
    idx_train, idx_test = train_test_split(idx_all, test_size=9, random_state=RANDOM_STATE, shuffle=True)
    X_train, X_test = X_params[idx_train], X_params[idx_test]
    Y_train, Y_test = Y_coeff[idx_train], Y_coeff[idx_test]

    print(f"Train size: {len(idx_train)}, Test size: {len(idx_test)}")

    print(f"=== Fitting regressor: {REGRESSOR} ===")
    model = choose_regressor(REGRESSOR)
    model.fit(X_train, Y_train)
    print("Regressor fitted.")

    print("=== Predicting coefficients for test set ===")
    Y_pred = model.predict(X_test)  # (9, rn)

    print("=== Reconstructing full 3D fields for test set ===")
    # Ground-truth and predicted reconstructions
    # Ground-truth: we can directly take T[..., idx]
    preds = []
    gts = []
    for j, case_idx in enumerate(idx_test):
        coeff = Y_pred[j]  # (rn,)
        T_pred = reconstruct_from_coeff(G, Ux, Uy, Uz, coeff)  # (NX, NY, NZ)
        preds.append(T_pred)
        gts.append(T[..., case_idx])
    preds = np.stack(preds, axis=-1)  # (NX, NY, NZ, 9)
    gts   = np.stack(gts,   axis=-1)

    print("=== Computing metrics (MAE / RMSE) ===")
    # Full-volume metrics
    mae_list, rmse_list = [], []
    mae_slice_list, rmse_slice_list = [], []

    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_hat  = preds[..., k].ravel()
        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        mae_list.append(mae); rmse_list.append(rmse)

        # Slice metrics at Y ≈ 1.71 m
        y_true_s = gts[:, y_slice_index, :, k].ravel()
        y_hat_s  = preds[:, y_slice_index, :, k].ravel()
        mae_s = mean_absolute_error(y_true_s, y_hat_s)
        rmse_s = math.sqrt(mean_squared_error(y_true_s, y_hat_s))
        mae_slice_list.append(mae_s); rmse_slice_list.append(rmse_s)

    print(f"[Full 3D]   MAE(mean over tests) = {np.mean(mae_list):.4f}, RMSE = {np.mean(rmse_list):.4f}")
    print(f"[Y={ys[y_slice_index]:.3f} m] MAE(mean) = {np.mean(mae_slice_list):.4f}, RMSE = {np.mean(rmse_slice_list):.4f}")

    # Save metrics summary
    summary = {
        "regressor": REGRESSOR,
        "ranks": RANKS,
        "random_state": RANDOM_STATE,
        "test_indices": idx_test.tolist(),
        "full3d_mae_each": mae_list,
        "full3d_rmse_each": rmse_list,
        "slice_mae_each": mae_slice_list,
        "slice_rmse_each": rmse_slice_list,
        "full3d_mae_mean": float(np.mean(mae_list)),
        "full3d_rmse_mean": float(np.mean(rmse_list)),
        "slice_mae_mean": float(np.mean(mae_slice_list)),
        "slice_rmse_mean": float(np.mean(rmse_slice_list)),
    }
    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Plotting Y-slice comparison maps for test set ===")
    # Prepare XY mesh for pcolormesh (X vs Z at fixed Y)
    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")  # shapes (NX, NZ)
    # We will produce one figure per test case (side-by-side)
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]   # (NX, NZ)
        T_pred_slice = preds[:, y_slice_index, :, k] # (NX, NZ)

        fig = plt.figure(figsize=(12, 5), dpi=150)
        plt.suptitle(f"Case #{case_idx+1} | Y={ys[y_slice_index]:.3f} m")

        # Left: Ground truth
        ax1 = fig.add_subplot(1, 2, 1)
        c1 = ax1.pcolormesh(Xg, Zg, T_true_slice, shading="auto")
        ax1.set_title("Ground Truth")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        fig.colorbar(c1, ax=ax1, label="Temperature")

        # Right: Prediction
        ax2 = fig.add_subplot(1, 2, 2)
        c2 = ax2.pcolormesh(Xg, Zg, T_pred_slice, shading="auto")
        ax2.set_title("Prediction")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        fig.colorbar(c2, ax=ax2, label="Temperature")

        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx+1}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"  Saved {out_path}")

    print("=== Done. Figures and metrics saved in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    # Optional: handle large printouts better
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
