#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved: Tucker Decomposition + Data-Driven Regression (Adaptive)
------------------------------------------------------------------
What's new vs. previous version:
1) Adaptive rank selection per mode by cumulative energy threshold (ENERGY_THRESH).
2) Optional mean-centering along N-mode (across cases), then add back for reconstruction.
3) Standardization of 14D inputs; optional standardization of N-mode coefficients.
4) More flexible regressors:
   - GPR with RBF or Matérn kernel (Kriging-like)
   - MLP (2x128) as non-linear baseline
5) Visualization:
   - Unified color scale between GT & Prediction
   - Flexible plotting method: 'contourf' (levels) or 'imshow' (interpolation)
   - Optional XZ upsampling for smoother slice plots
   - Error map |GT - Pred| per test case
6) Exports metrics summary JSON and per-case figures.

Author: (your name)
Date: 2025-10-15
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
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.colors import Normalize

# Optional interpolation
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -------------------------- User Config --------------------------

# Paths
PARAMS_PATH   = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR  = r"C:\Users\Lenovo\Desktop\insert"                   # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES    = 89
Y_SLICE    = 1.71  # meters

# Adaptive rank selection
USE_ADAPTIVE_RANKS = True
ENERGY_THRESH = 0.999  # cumulative energy threshold per mode; only used if USE_ADAPTIVE_RANKS=True
FALLBACK_RANKS = (20, 20, 20, 25)  # used if adaptive fails

# Centering and scaling
CENTER_ALONG_N = True      # mean-remove across N before HOSVD
SCALE_INPUTS = True        # Standardize 14D parameters
SCALE_OUTPUT_COEFF = True  # Standardize Un for regression targets (will inverse-transform before reconstruction)

# Regression options
REGRESSOR = "GPR"          # "GPR" or "MLP"
GPR_KERNEL = "Matern"      # "RBF" or "Matern" (if REGRESSOR="GPR")
MLP_HIDDEN = (128, 128)
RANDOM_STATE = 42

# Visualization
PLOT_METHOD = "contourf"   # "contourf" | "imshow"
LEVELS = 80                # contourf levels
UPSAMPLE_FX = 1            # upsample factor in X (>=1). If >1 and SCIPY_OK, will interpolate
UPSAMPLE_FZ = 1            # upsample factor in Z (>=1)
SAVE_ERROR_MAP = True      # also save |GT-Pred| slice

# Output directory
FIG_DIR = "./figures_improved"
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------- Utils --------------------------

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
    T_perm = np.moveaxis(T, mode, 0)  # (I_mode, ...)
    out = U @ T_perm.reshape(T_perm.shape[0], -1)  # (J, prod(others))
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
    name = name.upper()
    if name == "GPR":
        if GPR_KERNEL.upper() == "RBF":
            base_kernel = RBF(length_scale=np.ones(14), length_scale_bounds=(1e-2, 1e3))
        else:
            base_kernel = Matern(length_scale=np.ones(14), length_scale_bounds=(1e-2, 1e3), nu=1.5)
        kernel = C(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10, 1e-1))
        base = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RANDOM_STATE, alpha=0.0)
        model = MultiOutputRegressor(base, n_jobs=None)
    elif name == "MLP":
        model = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, activation="relu", solver="adam",
                             alpha=1e-4, batch_size=16, learning_rate_init=1e-3,
                             max_iter=4000, random_state=RANDOM_STATE, verbose=True)
    else:
        raise ValueError("Unknown REGRESSOR")
    return model


def upsample_if_needed(xs, zs, Txz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, Txz
    xi = np.linspace(xs.min(), xs.max(), len(xs)*UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs)*UPSAMPLE_FZ)
    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)
    return xi, zi, Txz_hi


def plot_slice_pair(xs, zs, T_true_slice, T_pred_slice, y_val, out_path, method="contourf", levels=80):
    vmin = min(T_true_slice.min(), T_pred_slice.min())
    vmax = max(T_true_slice.max(), T_pred_slice.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")

    fig = plt.figure(figsize=(16, 5), dpi=150)
    plt.suptitle(f"Y={y_val:.3f} m | Unified colormap")

    ax1 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 1)
    if method == "contourf":
        c1 = ax1.contourf(Xg, Zg, T_true_slice, levels=levels, norm=norm)
    else:
        extent = (xs.min(), xs.max(), zs.min(), zs.max())
        c1 = ax1.imshow(T_true_slice.T, origin="lower", extent=extent, aspect="auto",
                        vmin=vmin, vmax=vmax, interpolation="bicubic")
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
    fig.colorbar(c1, ax=ax1, label="Temperature")

    ax2 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 2)
    if method == "contourf":
        c2 = ax2.contourf(Xg, Zg, T_pred_slice, levels=levels, norm=norm)
    else:
        extent = (xs.min(), xs.max(), zs.min(), zs.max())
        c2 = ax2.imshow(T_pred_slice.T, origin="lower", extent=extent, aspect="auto",
                        vmin=vmin, vmax=vmax, interpolation="bicubic")
    ax2.set_title("Prediction")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)")
    fig.colorbar(c2, ax=ax2, label="Temperature")

    if SAVE_ERROR_MAP:
        err = np.abs(T_true_slice - T_pred_slice)
        ax3 = fig.add_subplot(1, 3, 3)
        if method == "contourf":
            c3 = ax3.contourf(Xg, Zg, err, levels=levels)
        else:
            extent = (xs.min(), xs.max(), zs.min(), zs.max())
            c3 = ax3.imshow(err.T, origin="lower", extent=extent, aspect="auto", interpolation="bicubic")
        ax3.set_title("Abs Error |GT - Pred|")
        ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
        fig.colorbar(c3, ax=ax3, label="Temperature Error")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
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

    # Regression targets
    Y_coeff = Un  # (N_CASES, rn)

    print("=== Train/Test split (80/9) ===")
    idx_all = np.arange(N_CASES)
    idx_train, idx_test = train_test_split(idx_all, test_size=9, random_state=RANDOM_STATE, shuffle=True)

    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff[idx_train], Y_coeff[idx_test]

    # Scale inputs
    if SCALE_INPUTS:
        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train_raw)
        X_test  = x_scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    # Scale outputs
    if SCALE_OUTPUT_COEFF:
        y_scaler = StandardScaler()
        Y_train = y_scaler.fit_transform(Y_train_raw)
    else:
        y_scaler = None
        Y_train = Y_train_raw

    print(f"=== Fit regressor: {REGRESSOR} (kernel={GPR_KERNEL if REGRESSOR.upper()=='GPR' else 'n/a'}) ===")
    model = choose_regressor(REGRESSOR, rn)
    model.fit(X_train, Y_train)

    print("=== Predict coefficients on test set ===")
    Y_pred_scaled = model.predict(X_test)
    if SCALE_OUTPUT_COEFF:
        Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    else:
        Y_pred = Y_pred_scaled

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

    print("=== Metrics ===")
    mae_list, rmse_list = [], []
    mae_slice_list, rmse_slice_list = [], []
    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_hat  = preds[..., k].ravel()
        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        mae_list.append(mae); rmse_list.append(rmse)

        y_true_s = gts[:, y_slice_index, :, k].ravel()
        y_hat_s  = preds[:, y_slice_index, :, k].ravel()
        mae_s = mean_absolute_error(y_true_s, y_hat_s)
        rmse_s = math.sqrt(mean_squared_error(y_true_s, y_hat_s))
        mae_slice_list.append(mae_s); rmse_slice_list.append(rmse_s)

    print(f"[Full 3D]   MAE={np.mean(mae_list):.4f}, RMSE={np.mean(rmse_list):.4f}")
    print(f"[Y={ys[y_slice_index]:.3f} m] MAE={np.mean(mae_slice_list):.4f}, RMSE={np.mean(rmse_slice_list):.4f}")

    # Save summary
    summary = {
        "ranks": {"rx": int(rx), "ry": int(ry), "rz": int(rz), "rn": int(rn)},
        "adaptive": bool(USE_ADAPTIVE_RANKS),
        "energy_threshold": float(ENERGY_THRESH),
        "center_along_N": bool(CENTER_ALONG_N),
        "scale_inputs": bool(SCALE_INPUTS),
        "scale_output_coeff": bool(SCALE_OUTPUT_COEFF),
        "regressor": REGRESSOR,
        "gpr_kernel": GPR_KERNEL if REGRESSOR.upper()=="GPR" else None,
        "random_state": RANDOM_STATE,
        "test_indices": [int(x) for x in idx_test.tolist()],
        "full3d_mae_each": [float(x) for x in mae_list],
        "full3d_rmse_each": [float(x) for x in rmse_list],
        "slice_mae_each": [float(x) for x in mae_slice_list],
        "slice_rmse_each": [float(x) for x in rmse_slice_list],
        "full3d_mae_mean": float(np.mean(mae_list)),
        "full3d_rmse_mean": float(np.mean(rmse_list)),
        "slice_mae_mean": float(np.mean(mae_slice_list)),
        "slice_rmse_mean": float(np.mean(rmse_slice_list)),
    }
    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Plot Y-slice maps (GT vs Pred; +Error) ===")
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]

        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice)
        _,       _,       T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice)

        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx+1}_improved.png")
        plot_slice_pair(xs_plot, zs_plot, T_true_plot, T_pred_plot, ys[y_slice_index],
                        out_path, method=PLOT_METHOD, levels=LEVELS)
        print(f"  Saved {out_path}")

    print("Done. Outputs in:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
