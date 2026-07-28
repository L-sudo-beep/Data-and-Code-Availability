#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unstructured → Tensor (Transfer/Binning) + Tucker + NN
with Neighboring Y-slice Averaging for Visualization
"""
import os, sys, math, json, warnings
from typing import Tuple, Optional
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.colors import Normalize

PARAMS_PATH   = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
SNAPSHOT_DIR  = r"C:\Users\Lenovo\Desktop\condition_data_files"

X_LEN, Y_LEN, Z_LEN = 6.6, 4.5, 9.0
NX, NY, NZ = 75, 190, 229
N_CASES    = 89

ENABLE_PLOTS = True
Y_SLICE_M    = 1.71
DY_LAYERS    = 2
AVG_MODE     = "mean"

FIG_DIR = "./figures_tensor_transfer_avg"
os.makedirs(FIG_DIR, exist_ok=True)

CENTER_ALONG_N = True
ENERGY_X, ENERGY_Y, ENERGY_Z, ENERGY_N = 0.999, 0.995, 0.999, 0.99
FALLBACK_RANKS = (30, 40, 60, 40)

USE_SCALERS = True
MLP_HIDDEN = (192, 192)
MLP_MAX_ITER = 5000
RANDOM_STATE = 42

FILL_STRATEGY = "global_mean"

def safe_params_path(p: str) -> str:
    if os.path.isdir(p):
        cands = [f for f in os.listdir(p) if f.lower().endswith(".csv")]
        if not cands:
            raise FileNotFoundError("No .csv found in Boundary_Conditions directory.")
        if len(cands) > 1:
            warnings.warn(f"Multiple CSVs found in {p}; using the first: {cands[0]}")
        return os.path.join(p, cands[0])
    return p

def read_params_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain at least 14 columns.")
    df14 = df.iloc[:, :14].copy()
    if df14.shape[0] != N_CASES:
        warnings.warn(f"[WARN] Parameter CSV rows={df14.shape[0]}, expected {N_CASES}. Proceeding.")
    return df14.to_numpy(dtype=np.float64)

def grid_axes():
    xs = np.linspace(0.0, X_LEN, NX, endpoint=True)
    ys = np.linspace(0.0, Y_LEN, NY, endpoint=True)
    zs = np.linspace(0.0, Z_LEN, NZ, endpoint=True)
    return xs, ys, zs

def bin_points_to_voxels(x, y, z, val, xs, ys, zs) -> np.ndarray:
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]; dz = zs[1] - zs[0]
    ix = np.floor((x - xs[0]) / dx).astype(int)
    iy = np.floor((y - ys[0]) / dy).astype(int)
    iz = np.floor((z - zs[0]) / dz).astype(int)
    ix = np.clip(ix, 0, NX-1); iy = np.clip(iy, 0, NY-1); iz = np.clip(iz, 0, NZ-1)
    lin = ix * (NY*NZ) + iy * NZ + iz
    flat_sum = np.bincount(lin, weights=val, minlength=NX*NY*NZ).astype(np.float64)
    flat_cnt = np.bincount(lin, minlength=NX*NY*NZ).astype(np.int64)
    with np.errstate(invalid="ignore"):
        flat_avg = flat_sum / flat_cnt
    arr = flat_avg.reshape(NX, NY, NZ)
    arr[flat_cnt.reshape(NX, NY, NZ) == 0] = np.nan
    return arr

def unfold(T, mode): return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

def nmode(T, U, mode):
    Tp = np.moveaxis(T, mode, 0)
    out = U @ Tp.reshape(Tp.shape[0], -1)
    out = out.reshape((U.shape[0],) + Tp.shape[1:])
    return np.moveaxis(out, 0, mode)

def svd_energy(X, thr, min_rank=2):
    n_comp = min(X.shape)
    U, S, VT = randomized_svd(X, n_components=n_comp, random_state=RANDOM_STATE)
    e = S**2; cum = np.cumsum(e) / np.sum(e)
    r = int(np.searchsorted(cum, thr) + 1)
    r = max(min_rank, min(r, U.shape[1]))
    return U[:, :r], r

def fill_empty_voxels(T, strategy="global_mean"):
    T_filled = T.copy()
    if strategy == "zero":
        T_filled[np.isnan(T_filled)] = 0.0
        return T_filled
    if strategy == "per_case_mean":
        for k in range(T.shape[-1]):
            case = T[..., k]
            m = np.nanmean(case)
            case_f = np.where(np.isnan(case), m, case)
            T_filled[..., k] = case_f
        return T_filled
    m = np.nanmean(T_filled)
    T_filled[np.isnan(T_filled)] = float(m)
    return T_filled

def stack_y_slices(arr_xynz, y_idx_center, dy_layers=2, mode="mean"):
    lo = max(0, y_idx_center - dy_layers)
    hi = min(arr_xynz.shape[1] - 1, y_idx_center + dy_layers)
    sl = arr_xynz[:, lo:hi+1, :]
    if mode == "median":
        return np.nanmedian(sl, axis=1), (lo, hi)
    return np.nanmean(sl, axis=1), (lo, hi)

def plot_yslice_pair(xs, zs, GT, PD, yval, out_path, levels=100, idx_range=None):
    vmin = min(GT.min(), PD.min()); vmax = max(GT.max(), PD.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
    fig = plt.figure(figsize=(16,5), dpi=150)
    if idx_range is not None:
        lo, hi = idx_range
        plt.suptitle(f"Y={yval:.3f} m | Averaged over {hi-lo+1} layers (indices {lo}..{hi}) | Unified colormap")
    else:
        plt.suptitle(f"Y={yval:.3f} m | Unified colormap")
    ax1 = fig.add_subplot(1,3,1); c1=ax1.contourf(Xg, Zg, GT, levels=levels, norm=norm)
    ax1.set_title("Ground Truth"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)"); fig.colorbar(c1, ax=ax1)
    ax2 = fig.add_subplot(1,3,2); c2=ax2.contourf(Xg, Zg, PD, levels=levels, norm=norm)
    ax2.set_title("Prediction"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); fig.colorbar(c2, ax=ax2)
    err = np.abs(GT-PD); ax3 = fig.add_subplot(1,3,3); c3=ax3.contourf(Xg, Zg, err, levels=levels)
    ax3.set_title("Absolute Error"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)"); fig.colorbar(c3, ax=ax3)
    plt.tight_layout(); plt.savefig(out_path); plt.close(fig)

def main():
    print("=== Load parameters ===")
    params_path = safe_params_path(PARAMS_PATH)
    X_params = read_params_csv(params_path)

    print("=== Build grid axes ===")
    xs = np.linspace(0.0, X_LEN, NX, endpoint=True)
    ys = np.linspace(0.0, Y_LEN, NY, endpoint=True)
    zs = np.linspace(0.0, Z_LEN, NZ, endpoint=True)
    y_idx = int(np.argmin(np.abs(ys - Y_SLICE_M)))
    print(f"Y target: {Y_SLICE_M} m -> index {y_idx}, y={ys[y_idx]:.3f}")

    print("=== Read & bin all snapshots ===")
    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64); T[:] = np.nan
    for i in range(1, N_CASES+1):
        df = pd.read_csv(os.path.join(SNAPSHOT_DIR, f"{i}.csv"), encoding="utf-8")
        x = df["X (m)"].to_numpy(np.float64)
        y = df["Y (m)"].to_numpy(np.float64)
        z = df["Z (m)"].to_numpy(np.float64)
        t = df["Temperature"].to_numpy(np.float64)
        T[..., i-1] = bin_points_to_voxels(x, y, z, t, xs, ys, zs)
        if i % 10 == 0 or i == N_CASES: print(f"  Binned {i}/{N_CASES}")

    print("=== Fill empty voxels ===")
    T_filled = fill_empty_voxels(T, strategy=FILL_STRATEGY)

    print("=== Center along N ===")
    T_mean = None; T_used = T_filled
    if CENTER_ALONG_N:
        T_mean = T_filled.mean(axis=3, keepdims=True); T_used = T_filled - T_mean

    print("=== HOSVD / Adaptive ranks ===")
    Ux, rx = svd_energy(unfold(T_used, 0), ENERGY_X)
    Uy, ry = svd_energy(unfold(T_used, 1), ENERGY_Y)
    Uz, rz = svd_energy(unfold(T_used, 2), ENERGY_Z)
    Un, rn = svd_energy(unfold(T_used, 3), ENERGY_N)
    print(f"Ranks: rx={rx}, ry={ry}, rz={rz}, rn={rn}")

    G = T_used.copy()
    G = nmode(G, Ux.T, 0); G = nmode(G, Uy.T, 1); G = nmode(G, Uz.T, 2); G = nmode(G, Un.T, 3)

    print("=== Prepare NN training ===")
    Y_coeff = Un
    idx_all = np.arange(N_CASES)
    idx_train, idx_test = train_test_split(idx_all, test_size=9, random_state=RANDOM_STATE, shuffle=True)
    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff[idx_train], Y_coeff[idx_test]

    if USE_SCALERS:
        x_scaler = StandardScaler().fit(X_train_raw)
        y_scaler = StandardScaler().fit(Y_train_raw)
        X_train = x_scaler.transform(X_train_raw); X_test  = x_scaler.transform(X_test_raw)
        Y_train = y_scaler.transform(Y_train_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw; y_scaler = None; Y_train = Y_train_raw

    print("=== Train MLP ===")
    mlp = MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, activation='relu', solver='adam',
                       alpha=1e-4, batch_size=16, learning_rate_init=1e-3,
                       max_iter=MLP_MAX_ITER, random_state=RANDOM_STATE, verbose=True)
    mlp.fit(X_train, Y_train)

    print("=== Predict & reconstruct ===")
    Y_pred_scaled = mlp.predict(X_test); Y_pred = y_scaler.inverse_transform(Y_pred_scaled) if USE_SCALERS else Y_pred_scaled
    preds = np.empty((NX, NY, NZ, len(idx_test)), dtype=np.float64); gts = np.empty_like(preds)
    for j, case_idx in enumerate(idx_test):
        coeff = Y_pred[j]
        M = np.tensordot(G, coeff, axes=([3],[0]))
        T_rec = nmode(nmode(nmode(M, Ux, 0), Uy, 1), Uz, 2)
        if CENTER_ALONG_N: T_rec = T_rec + T_mean[..., 0]
        preds[..., j] = T_rec; gts[..., j] = T_filled[..., case_idx]

    print("=== Metrics ===")
    mae_all, rmse_all = [], []
    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel(); y_hat  = preds[..., k].ravel()
        mae_all.append(mean_absolute_error(y_true, y_hat))
        rmse_all.append(math.sqrt(mean_squared_error(y_true, y_hat)))
    print(f"[3D] MAE={np.mean(mae_all):.3f}, RMSE={np.mean(rmse_all):.3f}")

    if ENABLE_PLOTS:
        for k, case_idx in enumerate(idx_test):
            GT3 = gts[..., k]; PD3 = preds[..., k]
            # Average neighboring Y layers
            def stack_y_slices(arr_xynz, y_idx_center, dy_layers=2, mode="mean"):
                lo = max(0, y_idx_center - dy_layers)
                hi = min(arr_xynz.shape[1] - 1, y_idx_center + dy_layers)
                sl = arr_xynz[:, lo:hi+1, :]
                if mode == "median": return np.nanmedian(sl, axis=1), (lo, hi)
                return np.nanmean(sl, axis=1), (lo, hi)
            GT_avg, idx_range = stack_y_slices(GT3, y_idx, dy_layers=DY_LAYERS, mode=AVG_MODE)
            PD_avg, _         = stack_y_slices(PD3, y_idx, dy_layers=DY_LAYERS, mode=AVG_MODE)
            out = os.path.join(FIG_DIR, f"yslice_case_{case_idx+1}_avg{2*DY_LAYERS+1}_{AVG_MODE}.png")
            # Plot
            vmin = min(GT_avg.min(), PD_avg.min()); vmax = max(GT_avg.max(), PD_avg.max())
            norm = Normalize(vmin=vmin, vmax=vmax)
            Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
            fig = plt.figure(figsize=(16,5), dpi=150)
            lo, hi = idx_range
            plt.suptitle(f"Y={ys[y_idx]:.3f} m | Averaged over {hi-lo+1} layers (indices {lo}..{hi}) | Unified colormap")
            ax1 = fig.add_subplot(1,3,1); c1=ax1.contourf(Xg, Zg, GT_avg, levels=100, norm=norm)
            ax1.set_title("Ground Truth"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)"); fig.colorbar(c1, ax=ax1)
            ax2 = fig.add_subplot(1,3,2); c2=ax2.contourf(Xg, Zg, PD_avg, levels=100, norm=norm)
            ax2.set_title("Prediction"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); fig.colorbar(c2, ax=ax2)
            err = np.abs(GT_avg - PD_avg); ax3 = fig.add_subplot(1,3,3); c3=ax3.contourf(Xg, Zg, err, levels=100)
            ax3.set_title("Absolute Error"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)"); fig.colorbar(c3, ax=ax3)
            plt.tight_layout(); plt.savefig(out); plt.close(fig)
            print("  Saved", out)

    print("Done. Outputs in:", os.path.abspath(FIG_DIR))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e); sys.exit(1)
"""
"""
