#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker (TRAIN-only) with Per-Mode Energy-Based Rank Selection + MLP
-------------------------------------------------------------------
- Tucker 分解仅在【训练集】上完成（避免信息泄漏）
- 每个模式 (X/Y/Z/N) 的秩由【该模展开矩阵的能量阈值】自动选取
- 统一子空间 (Ux,Uy,Uz,G) 对全体样本投影得到系数；仅用训练集系数训练回归器
- 增加了 rRMSE 和 rRMSE_n 指标计算
- 能量加权损失（可选）、标准化（仅 fit 在训练集）
- 统一误差色标（若提供 POD 的 global_error_max.npy）
- 保存逐工况 MAE/RMSE/rRMSE、折线图、切片云图与 metrics_summary.json
- 在指定 Y 平面（如 2.52m）计算：
    1) SSIM_raw：原网格切片 (NX×NZ) 上计算（温度场）
    2) SSIM_interp：griddata(cubic) 插值到轮廓图网格(如150×150)后计算（温度场）
- 新增：梯度 SSIM（更贴近“气流组织边界/羽流轮廓”）
    3) SSIM_raw_grad：在原网格上对 |∇T| 计算 SSIM
    4) SSIM_interp_grad：在插值网格上对 |∇T| 计算 SSIM

Author: (your name)
Date: 2025-10-21
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

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C, Matern
from sklearn.svm import SVR

# PyTorch (MLP)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optional interpolation for upsampling plots
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# griddata 用于“轮廓图插值网格”SSIM
try:
    from scipy.interpolate import griddata
    GRIDDATA_OK = True
except Exception:
    GRIDDATA_OK = False

# ========================== User Config ==========================

# Paths
PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"  # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES = 89
Y_SLICE = 1.53  # meters (用于切片可视化)

# 你之前用于 rRMSE 的平面（保留）
Y_PLANE_FOR_RRMSE = 1.53

# 你要评估气流组织的 SSIM 平面（按需求：2.52m）
Y_PLANE_FOR_SSIM = 2.52

# 插值网格分辨率（与你 POD 轮廓图一致）
SSIM_GRID_RES = 150
SSIM_GRID_METHOD = "cubic"  # 'cubic' / 'linear' / 'nearest'

# Energy thresholds (per-mode)
ENERGY_THRESH = 0.99

# Centering and scaling
CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

# Regressor choice
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
USE_ENERGY_WEIGHTS = True

# PCA bottleneck
USE_PCA_BOTTLENECK = False
PCA_LATENT_Q = 16

# Baselines
GPR_KERNEL = "Matern"
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = "scale"
SVR_CACHE_MB = 500

RANDOM_STATE = 42

# Visualization
PLOT_METHOD = "contourf"
LEVELS = 80
UPSAMPLE_FX = 1
UPSAMPLE_FZ = 1
SAVE_ERROR_MAP = True

# Output directory
FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved"
os.makedirs(FIG_DIR, exist_ok=True)

# Unified error colormap
POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"

# ========================== SSIM Utils ==========================
SSIM_METHOD = "skimage"  # 'skimage' or 'fallback'


def _fill_nans_with_mean(a: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(a)):
        return np.zeros_like(a, dtype=float)
    m = np.nanmean(a)
    return np.where(np.isnan(a), m, a)


def _normalize_to_01(a: np.ndarray, ref_min=None, ref_max=None, eps: float = 1e-12) -> np.ndarray:
    if ref_min is None:
        ref_min = np.nanmin(a)
    if ref_max is None:
        ref_max = np.nanmax(a)
    denom = (ref_max - ref_min)
    if np.abs(denom) < eps:
        return np.zeros_like(a, dtype=float)
    return (a - ref_min) / (denom + eps)


def compute_ssim_2d(true_2d: np.ndarray, pred_2d: np.ndarray) -> float:
    """
    SSIM between two 2D fields.
    - Fill NaNs
    - Normalize to [0,1] based on TRUE field range
    - Clip pred to [0,1] to avoid overshoot penalty (common after cubic interpolation)
    - Prefer skimage.metrics.structural_similarity; fallback to a global SSIM-like formula.
    """
    t = _fill_nans_with_mean(true_2d.astype(float))
    p = _fill_nans_with_mean(pred_2d.astype(float))

    tmin, tmax = np.min(t), np.max(t)
    t01 = _normalize_to_01(t, ref_min=tmin, ref_max=tmax)
    p01 = _normalize_to_01(p, ref_min=tmin, ref_max=tmax)
    p01 = np.clip(p01, 0.0, 1.0)

    if SSIM_METHOD == "skimage":
        try:
            from skimage.metrics import structural_similarity as ssim
            return float(ssim(t01, p01, data_range=1.0))
        except Exception:
            pass

    # fallback (global SSIM-like; robust backup)
    mu_t = np.mean(t01)
    mu_p = np.mean(p01)
    var_t = np.var(t01)
    var_p = np.var(p01)
    cov_tp = np.mean((t01 - mu_t) * (p01 - mu_p))

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    numerator = (2 * mu_t * mu_p + C1) * (2 * cov_tp + C2)
    denominator = (mu_t ** 2 + mu_p ** 2 + C1) * (var_t + var_p + C2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_grad_mag(T_xz: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    Compute gradient magnitude |∇T| on a 2D field defined on (x,z).
    T_xz shape: (NX, NZ) (or (res,res))
    x_axis shape: (NX,) (or (res,))
    z_axis shape: (NZ,) (or (res,))
    """
    T_xz = _fill_nans_with_mean(T_xz.astype(float))
    dTdx, dTdz = np.gradient(T_xz, x_axis, z_axis, edge_order=1)
    return np.sqrt(dTdx**2 + dTdz**2)


def interpolate_slice_to_contour_grid(xs_1d: np.ndarray, zs_1d: np.ndarray, T_xz: np.ndarray,
                                     res: int = 150, method: str = "cubic") -> np.ndarray:
    """
    将规则网格切片 T_xz (shape: NX×NZ) 当作散点 (x,z,T)，使用 griddata 插值到 res×res 网格。
    这与 POD 脚本中的 “griddata + contourf” 口径一致。
    输出 out 的 shape 为 (res, res)，且 indexing="ij"：轴0对应x，轴1对应z
    """
    if not GRIDDATA_OK:
        raise RuntimeError("scipy.interpolate.griddata is not available. Please install scipy.")

    X, Z = np.meshgrid(xs_1d, zs_1d, indexing="ij")  # (NX, NZ)
    pts = np.column_stack([X.ravel(), Z.ravel()])
    vals = T_xz.ravel()

    gx = np.linspace(xs_1d.min(), xs_1d.max(), res)
    gz = np.linspace(zs_1d.min(), zs_1d.max(), res)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    out = griddata(pts, vals, (GX, GZ), method=method)

    # cubic 在边界可能 NaN：后续 SSIM 会填充，这里也先处理一次
    out = _fill_nans_with_mean(out)
    return out


# ========================== Utils (原逻辑) ==========================
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
        raise ValueError(f"Grid size mismatch: {(len(xs), len(ys), len(zs))} vs {(NX, NY, NZ)}")
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


def upsample_if_needed(xs, zs, Txz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, Txz
    xi = np.linspace(xs.min(), xs.max(), len(xs) * UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs) * UPSAMPLE_FZ)
    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)
    return xi, zi, Txz_hi


# ========================== Per-mode energy-based HOSVD (原逻辑) ==========================
def select_rank_by_energy_unfold(X: np.ndarray, thr: float, max_rank: Optional[int] = None):
    I = X.shape[0]
    n_comp = I if max_rank is None else min(I, max_rank)
    U, S, VT = randomized_svd(X, n_components=n_comp, random_state=RANDOM_STATE)
    eng = (S ** 2)
    cum = np.cumsum(eng) / np.sum(eng)
    r = int(np.searchsorted(cum, thr) + 1)
    r = max(1, min(r, n_comp))
    return U[:, :r], r, S


def hosvd_per_mode_energy_train_only(T_centered: np.ndarray, idx_train: np.ndarray, thr: float):
    T_train = T_centered[..., idx_train]
    Ux, rx, Sx = select_rank_by_energy_unfold(mode_n_unfold(T_train, 0), thr)
    Uy, ry, Sy = select_rank_by_energy_unfold(mode_n_unfold(T_train, 1), thr)
    Uz, rz, Sz = select_rank_by_energy_unfold(mode_n_unfold(T_train, 2), thr)
    Un, rn, Sn = select_rank_by_energy_unfold(mode_n_unfold(T_train, 3), thr)
    ranks = (rx, ry, rz, rn)
    G = T_train.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)
    return (G, [Ux, Uy, Uz, Un], ranks, {"Sx": Sx, "Sy": Sy, "Sz": Sz, "Sn": Sn})


def project_case_to_coeff(T_case: np.ndarray, Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, G: np.ndarray) -> np.ndarray:
    B = n_mode_product(n_mode_product(n_mode_product(T_case.copy(), Ux.T, 0), Uy.T, 1), Uz.T, 2)
    b = B.reshape(-1)
    A = G.reshape(-1, G.shape[-1])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


# ========================== Regressors (原逻辑) ==========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers, use_layernorm=True, dropout=0.1):
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


def fit_mlp_torch(X_train: np.ndarray, Y_train: np.ndarray, weight_per_dim: Optional[np.ndarray] = None,
                  lr=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, hidden_layers=HIDDEN_LAYERS,
                  seed=RANDOM_STATE, use_layernorm=USE_LAYERNORM, dropout=DROPOUT,
                  weight_decay=WEIGHT_DECAY) -> nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(X_train.shape[1], Y_train.shape[1], hidden_layers, use_layernorm=use_layernorm, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if weight_per_dim is None:
        weight = torch.ones(Y_train.shape[1], dtype=torch.float32, device=device)
    else:
        weight = torch.tensor(weight_per_dim, dtype=torch.float32, device=device)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            diff = pred - yb
            loss = torch.mean(torch.sum((diff ** 2) * (weight ** 2), dim=1))
            loss.backward()
            opt.step()
        if (ep + 1) % 500 == 0:
            print(f"[MLP] Epoch {ep + 1}/{epochs} | Loss={loss.item():.6f}")
    return model


def predict_mlp_torch(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Yp = model(X_t).cpu().numpy()
    return Yp


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
    elif name == "SVR":
        base = SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA, cache_size=SVR_CACHE_MB)
        model = MultiOutputRegressor(base, n_jobs=None)
    elif name == "MLP_TORCH":
        return None
    else:
        raise ValueError("Unknown REGRESSOR.")
    return model


# ========================== Main ==========================
def main():
    np.random.seed(RANDOM_STATE)
    start_main_time = time.time()

    print("=== Load parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read first snapshot, build grid ===")
    df0 = read_one_snapshot_csv(os.path.join(SNAPSHOT_DIR, "1.csv"))
    xs, ys, zs = build_grid_from_df(df0)

    # --- 平面索引定位 ---
    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))
    print(f"Y slice for visualization ≈ {Y_SLICE} m -> index {y_slice_index}, y={ys[y_slice_index]:.4f}")

    y_plane_rrmse_index = int(np.argmin(np.abs(ys - Y_PLANE_FOR_RRMSE)))
    y_plane_rrmse_value = ys[y_plane_rrmse_index]
    print(f"Y plane for rRMSE calculation ≈ {Y_PLANE_FOR_RRMSE} m -> index {y_plane_rrmse_index}, y={y_plane_rrmse_value:.4f}")

    y_plane_ssim_index = int(np.argmin(np.abs(ys - Y_PLANE_FOR_SSIM)))
    y_plane_ssim_value = ys[y_plane_ssim_index]
    print(f"Y plane for SSIM calculation ≈ {Y_PLANE_FOR_SSIM} m -> index {y_plane_ssim_index}, y={y_plane_ssim_value:.4f}")

    if not GRIDDATA_OK:
        print("[WARN] scipy.griddata not available -> SSIM_interp & SSIM_interp_grad will be skipped (set to NaN).")

    print("=== Build tensor T ===")
    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    T[..., 0] = df_to_grid_values(df0, xs, ys, zs)
    for i in range(2, N_CASES + 1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        dfi = read_one_snapshot_csv(p)
        T[..., i - 1] = df_to_grid_values(dfi, xs, ys, zs)
        if i % 20 == 0 or i == N_CASES:
            print(f"  Loaded {i}/{N_CASES}")

    print("=== Train/Test split (fixed indices) ===")
    TEST_IDX_ONE_BASED = [8, 9, 20, 21, 58, 68, 72, 76, 84]
    idx_test = np.array([i - 1 for i in TEST_IDX_ONE_BASED], dtype=int)
    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(idx_all, idx_test, assume_unique=False)
    print(f"  Train size = {len(idx_train)}, Test size = {len(idx_test)}")

    if CENTER_ALONG_N:
        T_mean_train = T[..., idx_train].mean(axis=3, keepdims=True)
        T_centered = T - T_mean_train
    else:
        T_mean_train = None
        T_centered = T

    print(f"=== Tucker (TRAIN-only) with per-mode energy threshold = {ENERGY_THRESH} ===")
    (G_train, [Ux, Uy, Uz, Un_train], ranks, singvals) = hosvd_per_mode_energy_train_only(
        T_centered, idx_train, ENERGY_THRESH
    )
    rx, ry, rz, rn = ranks
    print(f"Selected ranks: rx={rx}, ry={ry}, rz={rz}, rn={rn}")

    print("=== Build coefficients for ALL cases via projection ===")
    Y_coeff_all = np.zeros((N_CASES, rn), dtype=np.float64)
    for n in range(N_CASES):
        Y_coeff_all[n, :] = project_case_to_coeff(T_centered[..., n], Ux, Uy, Uz, G_train)

    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff_all[idx_train], Y_coeff_all[idx_test]

    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        Gn = np.linalg.norm(G_train.reshape(-1, G_train.shape[-1]), axis=0)
        w = np.clip(Gn / (Gn.max() + 1e-12), 1e-3, None)
    else:
        w = None

    if USE_PCA_BOTTLENECK:
        pca = PCA(n_components=PCA_LATENT_Q, random_state=RANDOM_STATE)
        Z_train = pca.fit_transform(Y_train_raw)
        Z_test = pca.transform(Y_test_raw)

        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        if SCALE_OUTPUT_COEFF:
            z_scaler = StandardScaler().fit(Z_train)
            Z_train_s = z_scaler.transform(Z_train)
        else:
            z_scaler = None
            Z_train_s = Z_train

        print(f"=== Fit regressor: {REGRESSOR} (on PCA scores) ===")
        t0 = time.time()
        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(X_train, Z_train_s)
            Z_pred_s = predict_mlp_torch(model, X_test)
        else:
            base = choose_regressor(REGRESSOR, PCA_LATENT_Q)
            base.fit(X_train, Z_train_s)
            Z_pred_s = base.predict(X_test)
        print(f"Regressor fit time: {time.time() - t0:.2f}s")

        Z_pred = z_scaler.inverse_transform(Z_pred_s) if SCALE_OUTPUT_COEFF else Z_pred_s
        Y_pred = pca.inverse_transform(Z_pred)

    else:
        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        if SCALE_OUTPUT_COEFF:
            y_scaler = StandardScaler().fit(Y_train_raw)
            Y_train = y_scaler.transform(Y_train_raw)
        else:
            y_scaler = None
            Y_train = Y_train_raw

        print(f"=== Fit regressor: {REGRESSOR} (energy-weighted={USE_ENERGY_WEIGHTS}) ===")
        t0 = time.time()
        if REGRESSOR.upper() == "MLP_TORCH":
            model = fit_mlp_torch(X_train, Y_train, weight_per_dim=w)
            Y_pred_s = predict_mlp_torch(model, X_test)
        else:
            base = choose_regressor(REGRESSOR, Y_train.shape[1])
            base.fit(X_train, Y_train)
            Y_pred_s = base.predict(X_test)
        print(f"Regressor fit time: {time.time() - t0:.2f}s")

        Y_pred = y_scaler.inverse_transform(Y_pred_s) if SCALE_OUTPUT_COEFF else Y_pred_s

    print("=== Reconstruct test 3D fields ===")
    preds, gts = [], []
    for j, case_idx in enumerate(idx_test):
        coeff = Y_pred[j]
        M = np.tensordot(G_train, coeff, axes=([3], [0]))  # (rx,ry,rz)
        T1 = n_mode_product(M, Ux, 0)
        T2 = n_mode_product(T1, Uy, 1)
        T3 = n_mode_product(T2, Uz, 2)
        T_pred = T3 + (T_mean_train[..., 0] if CENTER_ALONG_N else 0.0)
        preds.append(T_pred)
        gts.append(T[..., case_idx])

    preds = np.stack(preds, axis=-1)
    gts = np.stack(gts, axis=-1)

    # =================== Metrics ===================
    print("=== Metrics (per-case & mean) ===")
    mae_list, rmse_list, rRMSE_list = [], [], []
    rRMSE_plane_list = []

    # 温度场 SSIM（已有）
    ssim_raw_list = []     # 原网格 (NX×NZ)
    ssim_interp_list = []  # 插值网格 (res×res)

    # 新增：梯度 SSIM（|∇T|）
    ssim_raw_grad_list = []
    ssim_interp_grad_list = []

    for kk in range(preds.shape[-1]):
        # 1) 全局 3D
        y_true = gts[..., kk].ravel()
        y_hat = preds[..., kk].ravel()

        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        norm_true = np.linalg.norm(y_true)
        norm_err = np.linalg.norm(y_true - y_hat)
        rRMSE_case = norm_err / (norm_true + 1e-9)

        mae_list.append(mae)
        rmse_list.append(rmse)
        rRMSE_list.append(rRMSE_case)

        # 2) 平面 rRMSE（保留）
        true_slice_rrmse = gts[:, y_plane_rrmse_index, :, kk]  # (NX, NZ)
        pred_slice_rrmse = preds[:, y_plane_rrmse_index, :, kk]
        y_true_plane = true_slice_rrmse.ravel()
        y_hat_plane = pred_slice_rrmse.ravel()
        norm_true_plane = np.linalg.norm(y_true_plane)
        norm_err_plane = np.linalg.norm(y_true_plane - y_hat_plane)
        rRMSE_case_plane = norm_err_plane / (norm_true_plane + 1e-9)
        rRMSE_plane_list.append(rRMSE_case_plane)

        # 3) SSIM（气流组织）——温度场：原网格
        true_slice_ssim = gts[:, y_plane_ssim_index, :, kk]  # (NX, NZ)
        pred_slice_ssim = preds[:, y_plane_ssim_index, :, kk]
        ssim_raw = compute_ssim_2d(true_slice_ssim, pred_slice_ssim)
        ssim_raw_list.append(ssim_raw)

        # 3b) 新增：梯度 SSIM（|∇T|）——原网格
        grad_true_raw = compute_grad_mag(true_slice_ssim, xs, zs)
        grad_pred_raw = compute_grad_mag(pred_slice_ssim, xs, zs)
        ssim_raw_grad = compute_ssim_2d(grad_true_raw, grad_pred_raw)
        ssim_raw_grad_list.append(ssim_raw_grad)

        # 4) SSIM（气流组织）——温度场：插值到轮廓图网格
        if GRIDDATA_OK:
            gx = np.linspace(xs.min(), xs.max(), SSIM_GRID_RES)
            gz = np.linspace(zs.min(), zs.max(), SSIM_GRID_RES)

            Tt_interp = interpolate_slice_to_contour_grid(xs, zs, true_slice_ssim,
                                                          res=SSIM_GRID_RES, method=SSIM_GRID_METHOD)
            Tp_interp = interpolate_slice_to_contour_grid(xs, zs, pred_slice_ssim,
                                                          res=SSIM_GRID_RES, method=SSIM_GRID_METHOD)
            ssim_interp = compute_ssim_2d(Tt_interp, Tp_interp)

            # 4b) 新增：梯度 SSIM（|∇T|）——插值网格
            grad_true_interp = compute_grad_mag(Tt_interp, gx, gz)
            grad_pred_interp = compute_grad_mag(Tp_interp, gx, gz)
            ssim_interp_grad = compute_ssim_2d(grad_true_interp, grad_pred_interp)
        else:
            ssim_interp = np.nan
            ssim_interp_grad = np.nan

        ssim_interp_list.append(ssim_interp)
        ssim_interp_grad_list.append(ssim_interp_grad)

        print(
            f"  Case {idx_test[kk] + 1:2d}: "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, rRMSE={rRMSE_case:.4f}, "
            f"rRMSE_plane(Y={y_plane_rrmse_value:.2f})={rRMSE_case_plane:.4f}, "
            f"SSIM_raw_T(Y={y_plane_ssim_value:.2f})={ssim_raw:.6f}, "
            f"SSIM_interp_T({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={y_plane_ssim_value:.2f})={ssim_interp:.6f}, "
            f"SSIM_raw_grad(Y={y_plane_ssim_value:.2f})={ssim_raw_grad:.6f}, "
            f"SSIM_interp_grad({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={y_plane_ssim_value:.2f})={ssim_interp_grad:.6f}"
        )

    print(
        f"\n[Full 3D Mean] MAE={np.mean(mae_list):.4f} °C, RMSE={np.mean(rmse_list):.4f} °C, rRMSE={np.mean(rRMSE_list):.4f}"
    )
    print(f"[Plane Y={y_plane_rrmse_value:.2f}m Mean] rRMSE={np.mean(rRMSE_plane_list):.4f}")
    print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_raw_T={np.mean(ssim_raw_list):.6f}")
    print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_raw_grad={np.mean(ssim_raw_grad_list):.6f}")
    if np.any(np.isfinite(ssim_interp_list)):
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_T={np.nanmean(ssim_interp_list):.6f}")
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_grad={np.nanmean(ssim_interp_grad_list):.6f}")
    else:
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_T=NaN (griddata unavailable)")
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_grad=NaN (griddata unavailable)")

    # --- rRMSE_n 部分不改 ---
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
        norm_true_series = np.linalg.norm(true_series)
        rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (norm_true_series + 1e-9)
        avg_temp_at_point = mean_temp_field[i, j, k_dim]
        rRMSE_n_results[int(flat_idx)] = float(rRMSE_n_value)
        print(f"  Grid Point Index {flat_idx:<7} (Avg T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {rRMSE_n_value:.4f}")

    # =================== Save Metrics and Plots ===================
    df = pd.DataFrame({
        "Case": [int(i) + 1 for i in idx_test],
        "MAE": [float(x) for x in mae_list],
        "RMSE": [float(x) for x in rmse_list],
        "rRMSE": [float(x) for x in rRMSE_list],
        f"rRMSE_plane_Y={y_plane_rrmse_value:.2f}m": [float(x) for x in rRMSE_plane_list],

        # 温度SSIM
        f"SSIM_raw_T_plane_Y={y_plane_ssim_value:.2f}m": [float(x) for x in ssim_raw_list],
        f"SSIM_interp_T_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m": [
            float(x) if np.isfinite(x) else np.nan for x in ssim_interp_list
        ],

        # 梯度SSIM
        f"SSIM_raw_grad_plane_Y={y_plane_ssim_value:.2f}m": [float(x) for x in ssim_raw_grad_list],
        f"SSIM_interp_grad_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m": [
            float(x) if np.isfinite(x) else np.nan for x in ssim_interp_grad_list
        ],
    })
    df = df.sort_values("Case")
    df.to_csv(os.path.join(FIG_DIR, "per_case_metrics.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Case"], df["MAE"], '-o', label="MAE (°C) [Full 3D]")
    plt.plot(df["Case"], df["RMSE"], '-s', label="RMSE (°C) [Full 3D]")
    plt.plot(df["Case"], df["rRMSE"], '-^', label="rRMSE [Full 3D]")
    plt.plot(df["Case"], df[f"rRMSE_plane_Y={y_plane_rrmse_value:.2f}m"], '-x', color='purple',
             label=f"rRMSE [Plane Y={y_plane_rrmse_value:.2f}m]")

    plt.plot(df["Case"], df[f"SSIM_raw_T_plane_Y={y_plane_ssim_value:.2f}m"], '-d', color='green',
             label=f"SSIM_raw_T [Plane Y={y_plane_ssim_value:.2f}m]")
    plt.plot(df["Case"], df[f"SSIM_interp_T_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m"], '-p', color='orange',
             label=f"SSIM_interp_T({SSIM_GRID_RES}x{SSIM_GRID_RES}) [Plane Y={y_plane_ssim_value:.2f}m]")

    plt.plot(df["Case"], df[f"SSIM_raw_grad_plane_Y={y_plane_ssim_value:.2f}m"], '-v', color='teal',
             label=f"SSIM_raw_grad [Plane Y={y_plane_ssim_value:.2f}m]")
    plt.plot(df["Case"], df[f"SSIM_interp_grad_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m"], '-h', color='red',
             label=f"SSIM_interp_grad({SSIM_GRID_RES}x{SSIM_GRID_RES}) [Plane Y={y_plane_ssim_value:.2f}m]")

    plt.xlabel("Case")
    plt.ylabel("Metric Value")
    plt.title("Tucker+MLP Per-case Metrics (Full 3D vs. Plane Metrics incl. SSIM & Grad-SSIM)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_metrics_compare.png"), dpi=200)
    plt.close()

    # --- 可视化部分保持原逻辑 ---
    if os.path.exists(POD_ERROR_MAX_PATH):
        try:
            shared_error_max = float(np.load(POD_ERROR_MAX_PATH))
            if not np.isfinite(shared_error_max) or shared_error_max <= 0:
                shared_error_max = None
            else:
                print(f"[Info] Unified error colormap vmax from POD: {shared_error_max:.4f}")
        except Exception:
            shared_error_max = None
    else:
        shared_error_max = None

    print("=== Plot Y-slice maps (GT vs Pred; +Error) ===")
    for kk, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, kk]
        T_pred_slice = preds[:, y_slice_index, :, kk]
        err_slice = np.abs(T_true_slice - T_pred_slice)
        vmax = shared_error_max if (shared_error_max is not None) else float(np.nanmax(err_slice))
        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice.T)
        _, _, T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice.T)
        _, _, err_plot = upsample_if_needed(xs, zs, err_slice.T)
        Xg, Zg = np.meshgrid(xs_plot, zs_plot, indexing='ij')
        fig = plt.figure(figsize=(18, 5))
        plt.suptitle(f"Y-slice Reconstruction | Case {case_idx + 1}", fontsize=16)

        ax1 = fig.add_subplot(1, 3, 1)
        c1 = ax1.contourf(Xg, Zg, T_true_plot.T, levels=LEVELS, cmap='jet')
        ax1.set_title("True Temperature")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_aspect('equal', adjustable='box')
        fig.colorbar(c1, ax=ax1)

        ax2 = fig.add_subplot(1, 3, 2)
        c2 = ax2.contourf(Xg, Zg, T_pred_plot.T, levels=LEVELS, cmap='jet',
                          vmin=c1.get_clim()[0], vmax=c1.get_clim()[1])
        ax2.set_title("Reconstructed Temperature")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax2.set_aspect('equal', adjustable='box')
        fig.colorbar(c2, ax=ax2)

        if SAVE_ERROR_MAP:
            ax3 = fig.add_subplot(1, 3, 3)
            c3 = ax3.contourf(Xg, Zg, err_plot.T, levels=LEVELS, cmap='YlOrRd', vmin=0, vmax=vmax)
            ax3.set_title(f"Absolute Error (vmax={vmax:.2f})")
            ax3.set_xlabel("X (m)")
            ax3.set_ylabel("Z (m)")
            ax3.set_aspect('equal', adjustable='box')
            fig.colorbar(c3, ax=ax3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx + 1}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    # =================== Summary ===================
    def sv_info(S):
        e = (S ** 2)
        cum = np.cumsum(e) / np.sum(e)
        k = min(50, len(S))
        return cum[:k].tolist()

    summary = {
        "model_type": "Tucker_HOSVD",
        "decomposition_set": "TRAIN",
        "ranks": {"rx": int(rx), "ry": int(ry), "rz": int(rz), "rn": int(rn)},
        "energy_threshold_per_mode": float(ENERGY_THRESH),
        "config": {"center_along_N": bool(CENTER_ALONG_N),
                   "scale_inputs": bool(SCALE_INPUTS),
                   "scale_output_coeff": bool(SCALE_OUTPUT_COEFF)},
        "regressor": REGRESSOR,
        "mlp_params": {"hidden_layers": HIDDEN_LAYERS, "use_layernorm": USE_LAYERNORM, "dropout": DROPOUT,
                       "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                       "energy_weighted_loss": USE_ENERGY_WEIGHTS, "pca_bottleneck": USE_PCA_BOTTLENECK,
                       "pca_latent_q": PCA_LATENT_Q if USE_PCA_BOTTLENECK else None},
        "random_state": RANDOM_STATE,
        "test_indices_one_based": [int(x) for x in (idx_test + 1).tolist()],
        "metrics_summary": {
            "full3d_mae_mean": float(np.mean(mae_list)),
            "full3d_rmse_mean": float(np.mean(rmse_list)),
            "full3d_rRMSE_mean": float(np.mean(rRMSE_list)),
            "plane_rRMSE_mean": {"y_value": float(y_plane_rrmse_value), "value": float(np.mean(rRMSE_plane_list))},

            "plane_SSIM_raw_T_mean": {"y_value": float(y_plane_ssim_value), "value": float(np.mean(ssim_raw_list))},
            "plane_SSIM_raw_grad_mean": {"y_value": float(y_plane_ssim_value), "value": float(np.mean(ssim_raw_grad_list))},

            "plane_SSIM_interp_T_mean": {
                "y_value": float(y_plane_ssim_value),
                "grid_res": int(SSIM_GRID_RES),
                "method": SSIM_GRID_METHOD,
                "value": float(np.nanmean(ssim_interp_list)) if np.any(np.isfinite(ssim_interp_list)) else None
            },
            "plane_SSIM_interp_grad_mean": {
                "y_value": float(y_plane_ssim_value),
                "grid_res": int(SSIM_GRID_RES),
                "method": SSIM_GRID_METHOD,
                "value": float(np.nanmean(ssim_interp_grad_list)) if np.any(np.isfinite(ssim_interp_grad_list)) else None
            },

            "full3d_mae_each": [float(x) for x in mae_list],
            "full3d_rmse_each": [float(x) for x in rmse_list],
            "full3d_rRMSE_each": [float(x) for x in rRMSE_list],
            "plane_rRMSE_each": [float(x) for x in rRMSE_plane_list],

            "plane_SSIM_raw_T_each": [float(x) for x in ssim_raw_list],
            "plane_SSIM_raw_grad_each": [float(x) for x in ssim_raw_grad_list],
            "plane_SSIM_interp_T_each": [float(x) if np.isfinite(x) else None for x in ssim_interp_list],
            "plane_SSIM_interp_grad_each": [float(x) if np.isfinite(x) else None for x in ssim_interp_grad_list],

            "rRMSE_n_top10_points": rRMSE_n_results
        },
        "y_slice_visualization": {
            "y_value": float(ys[y_slice_index]),
            "unified_error_vmax_from_pod": float(shared_error_max) if shared_error_max is not None else None
        },
        "cumulative_energy_curves": {
            "mode_x": sv_info(singvals["Sx"]),
            "mode_y": sv_info(singvals["Sy"]),
            "mode_z": sv_info(singvals["Sz"]),
            "mode_n": sv_info(singvals["Sn"])
        },
        "execution_time_seconds": time.time() - start_main_time
    }

    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Total execution time: {time.time() - start_main_time:.2f}s. Outputs in:",
          os.path.abspath(FIG_DIR))


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
