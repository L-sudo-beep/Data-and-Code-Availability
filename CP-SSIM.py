#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CP (PARAFAC) (TRAIN-only) + MLP Regression + Metrics
---------------------------------------------------
- CP 分解仅在【训练集】上完成（避免信息泄漏）
- 训练集求均值后可选中心化（CENTER_ALONG_N）
- 统一投影矩阵 M_cp，对全体样本投影得到系数；仅用训练集系数训练回归器
- 指标：
  * 全场 3D: MAE / RMSE / rRMSE
  * 指定 Y 平面: rRMSE_plane
  * 指定 Y 平面（用于气流组织捕获评估）：
      1) SSIM_raw_T：原网格 (NX×NZ) 温度场 SSIM
      2) SSIM_interp_T：griddata 插值到轮廓图网格(res×res) 后温度场 SSIM
      3) SSIM_raw_grad：原网格 |∇T| 梯度幅值 SSIM
      4) SSIM_interp_grad：插值网格 |∇T| 梯度幅值 SSIM
- 保存 per_case_metrics_cp.csv、error_metrics_compare_cp.png、yslice_case_*_cp.png、metrics_summary_cp.json

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

# ============== CP（PARAFAC）相关依赖 ==============
try:
    import tensorly as tl
    from tensorly.decomposition import parafac
    TL_OK = True
except Exception:
    TL_OK = False

# ========================== User Config ==========================

# Paths
PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"  # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES = 89

# 仍用于“切片可视化”的平面
Y_SLICE = 2.52  # meters (用于切片可视化)

# 平面 rRMSE
Y_PLANE_FOR_RRMSE = 2.52

# 平面 SSIM（气流组织）
Y_PLANE_FOR_SSIM = 2.52

# 插值网格分辨率（与你 POD 轮廓图一致）
SSIM_GRID_RES = 150
SSIM_GRID_METHOD = "cubic"  # 'cubic' / 'linear' / 'nearest'

# SSIM 计算实现
SSIM_METHOD = "skimage"  # 'skimage' or 'fallback'

# Centering and scaling
CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

# CP 分解设置
CP_RANK = 16
CP_N_ITER_MAX = 2000
CP_TOL = 1e-7

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

# 能量加权
USE_ENERGY_WEIGHTS = True

# Baselines
GPR_KERNEL = "Matern"
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = "scale"
SVR_CACHE_MB = 500

RANDOM_STATE = 42

# Visualization
LEVELS = 80
UPSAMPLE_FX = 1
UPSAMPLE_FZ = 1
SAVE_ERROR_MAP = True

# Output directory
FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved_CP"
os.makedirs(FIG_DIR, exist_ok=True)

# 与 POD 统一误差色标
POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"


# ========================== SSIM / Gradient SSIM Utils ==========================
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
    T_xz shape: (NX, NZ) or (res, res)
    x_axis shape: (NX,) or (res,)
    z_axis shape: (NZ,) or (res,)
    """
    T_xz = _fill_nans_with_mean(T_xz.astype(float))
    dTdx, dTdz = np.gradient(T_xz, x_axis, z_axis, edge_order=1)
    return np.sqrt(dTdx**2 + dTdz**2)


def interpolate_slice_to_contour_grid(xs_1d: np.ndarray, zs_1d: np.ndarray, T_xz: np.ndarray,
                                     res: int = 150, method: str = "cubic") -> np.ndarray:
    """
    将规则网格切片 T_xz (shape: NX×NZ) 当作散点 (x,z,T)，使用 griddata 插值到 res×res 网格。
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

    out = _fill_nans_with_mean(out)
    return out


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


def upsample_if_needed(xs, zs, Txz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, Txz
    xi = np.linspace(xs.min(), xs.max(), len(xs) * UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs) * UPSAMPLE_FZ)
    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)
    return xi, zi, Txz_hi


# ========================== MLP & 回归器 ==========================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers, use_layernorm=USE_LAYERNORM, dropout=DROPOUT):
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

    model = MLP(X_train.shape[1], Y_train.shape[1], hidden_layers,
                use_layernorm=use_layernorm, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, Y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if weight_per_dim is None:
        weight = torch.ones(Y_train.shape[1], dtype=torch.float32, device=device)
    else:
        w = weight_per_dim.astype(np.float32)
        w = w / (np.mean(w) + 1e-12)
        weight = torch.tensor(w, dtype=torch.float32, device=device)

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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


# ========================== CP（PARAFAC）核心实现 ==========================
def cp_fit_train_only(T_centered: np.ndarray, idx_train: np.ndarray, R: int,
                      n_iter_max: int = 2000, tol: float = 1e-7, random_state: int = 42):
    if not TL_OK:
        raise RuntimeError("需要安装 tensorly： pip install tensorly")
    tl.set_backend('numpy')
    T_train = T_centered[..., idx_train]
    cp = parafac(
        T_train, rank=R, init='svd', tol=tol, n_iter_max=n_iter_max,
        random_state=random_state, normalize_factors=True
    )
    lambdas = cp.weights
    Ax, Ay, Az, An_train = cp.factors
    return lambdas, Ax, Ay, Az, An_train


def cp_build_projection_matrix(Ax: np.ndarray, Ay: np.ndarray, Az: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    nx, R = Ax.shape
    ny = Ay.shape[0]
    nz = Az.shape[0]
    M = np.empty((nx * ny * nz, R), dtype=np.float64)
    for r in range(R):
        outer3 = np.multiply.outer(np.multiply.outer(Ax[:, r], Ay[:, r]), Az[:, r])
        M[:, r] = float(lambdas[r]) * outer3.reshape(-1)
    return M


def project_case_to_cp_coeff(T_case: np.ndarray, M: np.ndarray) -> np.ndarray:
    b = T_case.reshape(-1)
    d, *_ = np.linalg.lstsq(M, b, rcond=None)
    return d


def cp_reconstruct_from_d(d: np.ndarray, Ax, Ay, Az, lambdas, T_mean_train=None):
    nx, R = Ax.shape
    ny = Ay.shape[0]
    nz = Az.shape[0]
    T_rec = np.zeros((nx, ny, nz), dtype=np.float64)
    for r in range(R):
        T_rec += (lambdas[r] * d[r]) * np.multiply.outer(np.multiply.outer(Ax[:, r], Ay[:, r]), Az[:, r])
    if T_mean_train is not None:
        T_rec = T_rec + T_mean_train[..., 0]
    return T_rec


def cp_component_energy_weights(lambdas, Ax, Ay, Az):
    R = Ax.shape[1]
    w = np.zeros(R, dtype=np.float64)
    for r in range(R):
        w[r] = abs(float(lambdas[r])) * np.linalg.norm(Ax[:, r]) * np.linalg.norm(Ay[:, r]) * np.linalg.norm(Az[:, r])
    w = w / (np.max(w) + 1e-12)
    w = np.clip(w, 1e-3, None)
    return w


# ========================== Main ==========================
def main():
    np.random.seed(RANDOM_STATE)
    start_main_time = time.time()

    if not TL_OK:
        raise RuntimeError("未检测到 tensorly，请先安装： pip install tensorly")

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
        print("[WARN] scipy.griddata not available -> SSIM_interp_T & SSIM_interp_grad will be NaN.")

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
        print("Applied N-mode mean centering using TRAIN set mean.")
    else:
        T_mean_train = None
        T_centered = T

    print(f"=== CP (TRAIN-only) rank = {CP_RANK} ===")
    t0 = time.time()
    lambdas, Ax, Ay, Az, An_train = cp_fit_train_only(
        T_centered, idx_train, CP_RANK, n_iter_max=CP_N_ITER_MAX, tol=CP_TOL, random_state=RANDOM_STATE
    )
    print(f"CP fit time: {time.time() - t0:.2f}s")

    print("=== Build CP coefficients for ALL cases via projection ===")
    M_cp = cp_build_projection_matrix(Ax, Ay, Az, lambdas)
    R = Ax.shape[1]
    Y_coeff_all = np.zeros((N_CASES, R), dtype=np.float64)
    for n in range(N_CASES):
        Y_coeff_all[n, :] = project_case_to_cp_coeff(T_centered[..., n], M_cp)

    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff_all[idx_train], Y_coeff_all[idx_test]

    if USE_ENERGY_WEIGHTS:
        w = cp_component_energy_weights(lambdas, Ax, Ay, Az)
    else:
        w = None

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

    print("=== Reconstruct test 3D fields (CP) ===")
    preds, gts = [], []
    for j, case_idx in enumerate(idx_test):
        preds.append(cp_reconstruct_from_d(Y_pred[j], Ax, Ay, Az, lambdas, T_mean_train))
        gts.append(T[..., case_idx])
    preds = np.stack(preds, axis=-1)
    gts = np.stack(gts, axis=-1)

    # =================== Metrics ===================
    print("=== Metrics (per-case & mean) ===")
    mae_list, rmse_list, rRMSE_list = [], [], []
    rRMSE_plane_list = []

    # 温度SSIM
    ssim_raw_T_list = []
    ssim_interp_T_list = []

    # 梯度SSIM
    ssim_raw_grad_list = []
    ssim_interp_grad_list = []

    for k in range(preds.shape[-1]):
        # 1) 全局误差（3D）
        y_true = gts[..., k].ravel()
        y_hat = preds[..., k].ravel()

        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))
        norm_true = np.linalg.norm(y_true)
        norm_err = np.linalg.norm(y_true - y_hat)
        rRMSE_case = norm_err / (norm_true + 1e-9)

        mae_list.append(mae)
        rmse_list.append(rmse)
        rRMSE_list.append(rRMSE_case)

        # 2) 平面 rRMSE（2D）
        true_slice_rrmse = gts[:, y_plane_rrmse_index, :, k]  # (NX, NZ)
        pred_slice_rrmse = preds[:, y_plane_rrmse_index, :, k]
        y_true_plane = true_slice_rrmse.ravel()
        y_hat_plane = pred_slice_rrmse.ravel()
        norm_true_plane = np.linalg.norm(y_true_plane)
        norm_err_plane = np.linalg.norm(y_true_plane - y_hat_plane)
        rRMSE_case_plane = norm_err_plane / (norm_true_plane + 1e-9)
        rRMSE_plane_list.append(rRMSE_case_plane)

        # 3) SSIM（温度场）——原网格
        true_slice_ssim = gts[:, y_plane_ssim_index, :, k]  # (NX, NZ)
        pred_slice_ssim = preds[:, y_plane_ssim_index, :, k]
        ssim_raw_T = compute_ssim_2d(true_slice_ssim, pred_slice_ssim)
        ssim_raw_T_list.append(ssim_raw_T)

        # 4) 梯度 SSIM（|∇T|）——原网格
        grad_true_raw = compute_grad_mag(true_slice_ssim, xs, zs)
        grad_pred_raw = compute_grad_mag(pred_slice_ssim, xs, zs)
        ssim_raw_grad = compute_ssim_2d(grad_true_raw, grad_pred_raw)
        ssim_raw_grad_list.append(ssim_raw_grad)

        # 5) SSIM（温度场）——插值网格（轮廓图一致）
        if GRIDDATA_OK:
            gx = np.linspace(xs.min(), xs.max(), SSIM_GRID_RES)
            gz = np.linspace(zs.min(), zs.max(), SSIM_GRID_RES)

            Tt_interp = interpolate_slice_to_contour_grid(xs, zs, true_slice_ssim,
                                                          res=SSIM_GRID_RES, method=SSIM_GRID_METHOD)
            Tp_interp = interpolate_slice_to_contour_grid(xs, zs, pred_slice_ssim,
                                                          res=SSIM_GRID_RES, method=SSIM_GRID_METHOD)
            ssim_interp_T = compute_ssim_2d(Tt_interp, Tp_interp)

            # 6) 梯度 SSIM（|∇T|）——插值网格
            grad_true_interp = compute_grad_mag(Tt_interp, gx, gz)
            grad_pred_interp = compute_grad_mag(Tp_interp, gx, gz)
            ssim_interp_grad = compute_ssim_2d(grad_true_interp, grad_pred_interp)
        else:
            ssim_interp_T = np.nan
            ssim_interp_grad = np.nan

        ssim_interp_T_list.append(ssim_interp_T)
        ssim_interp_grad_list.append(ssim_interp_grad)

        print(
            f"  Case {idx_test[k] + 1:2d}: "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, rRMSE={rRMSE_case:.4f}, "
            f"rRMSE_plane(Y={y_plane_rrmse_value:.2f})={rRMSE_case_plane:.4f}, "
            f"SSIM_raw_T(Y={y_plane_ssim_value:.2f})={ssim_raw_T:.6f}, "
            f"SSIM_interp_T({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={y_plane_ssim_value:.2f})={ssim_interp_T:.6f}, "
            f"SSIM_raw_grad(Y={y_plane_ssim_value:.2f})={ssim_raw_grad:.6f}, "
            f"SSIM_interp_grad({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={y_plane_ssim_value:.2f})={ssim_interp_grad:.6f}"
        )

    print(
        f"\n[Full 3D Mean] MAE={np.mean(mae_list):.4f}, RMSE={np.mean(rmse_list):.4f}, rRMSE={np.mean(rRMSE_list):.4f}"
    )
    print(f"[Plane Y={y_plane_rrmse_value:.2f}m Mean] rRMSE={np.mean(rRMSE_plane_list):.4f}")
    print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_raw_T={np.mean(ssim_raw_T_list):.6f}")
    print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_raw_grad={np.mean(ssim_raw_grad_list):.6f}")
    if np.any(np.isfinite(ssim_interp_T_list)):
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_T={np.nanmean(ssim_interp_T_list):.6f}")
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_grad={np.nanmean(ssim_interp_grad_list):.6f}")
    else:
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_T=NaN (griddata unavailable)")
        print(f"[Plane Y={y_plane_ssim_value:.2f}m Mean] SSIM_interp_grad=NaN (griddata unavailable)")

    # --- rRMSE_n（保持你原逻辑） ---
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

        # 温度 SSIM
        f"SSIM_raw_T_plane_Y={y_plane_ssim_value:.2f}m": [float(x) for x in ssim_raw_T_list],
        f"SSIM_interp_T_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m": [
            float(x) if np.isfinite(x) else np.nan for x in ssim_interp_T_list
        ],

        # 梯度 SSIM
        f"SSIM_raw_grad_plane_Y={y_plane_ssim_value:.2f}m": [float(x) for x in ssim_raw_grad_list],
        f"SSIM_interp_grad_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={y_plane_ssim_value:.2f}m": [
            float(x) if np.isfinite(x) else np.nan for x in ssim_interp_grad_list
        ],
    })
    df = df.sort_values("Case")
    df.to_csv(os.path.join(FIG_DIR, "per_case_metrics_cp.csv"), index=False)

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
    plt.title("CP+MLP Per-case Metrics (Full 3D vs. Plane Metrics incl. SSIM & Grad-SSIM)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_metrics_compare_cp.png"), dpi=200)
    plt.close()

    # =================== Visualization (Y-slice) ===================
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
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]
        err_slice = np.abs(T_true_slice - T_pred_slice)
        vmax = shared_error_max if shared_error_max is not None else float(np.nanmax(err_slice))

        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice.T)
        _, _, T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice.T)
        _, _, err_plot = upsample_if_needed(xs, zs, err_slice.T)

        Xg, Zg = np.meshgrid(xs_plot, zs_plot, indexing='ij')
        fig = plt.figure(figsize=(18, 5))
        plt.suptitle(f"Y-slice Reconstruction | Case {case_idx + 1} (CP)", fontsize=16)

        ax1 = fig.add_subplot(1, 3, 1)
        c1 = ax1.contourf(Xg, Zg, T_true_plot.T, levels=LEVELS, cmap='jet')
        ax1.set_title("True")
        fig.colorbar(c1, ax=ax1)

        ax2 = fig.add_subplot(1, 3, 2)
        c2 = ax2.contourf(Xg, Zg, T_pred_plot.T, levels=LEVELS, cmap='jet',
                          vmin=c1.get_clim()[0], vmax=c1.get_clim()[1])
        ax2.set_title("Reconstructed (CP)")
        fig.colorbar(c2, ax=ax2)

        if SAVE_ERROR_MAP:
            ax3 = fig.add_subplot(1, 3, 3)
            c3 = ax3.contourf(Xg, Zg, err_plot.T, levels=LEVELS, cmap='YlOrRd', vmin=0, vmax=vmax)
            ax3.set_title(f"Absolute Error (vmax={vmax:.2f})")
            fig.colorbar(c3, ax=ax3)

        for ax in fig.get_axes():
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(FIG_DIR, f"yslice_case_{case_idx + 1}_cp.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    # =================== Summary ===================
    summary = {
        "model_type": "CP_PARAFAC",
        "decomposition_set": "TRAIN",
        "cp_rank_R": int(CP_RANK),
        "config": {"center_along_N": bool(CENTER_ALONG_N),
                   "scale_inputs": bool(SCALE_INPUTS),
                   "scale_output_coeff": bool(SCALE_OUTPUT_COEFF)},
        "regressor": REGRESSOR,
        "mlp_params": {"hidden_layers": HIDDEN_LAYERS, "use_layernorm": USE_LAYERNORM, "dropout": DROPOUT,
                       "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                       "energy_weighted_loss": USE_ENERGY_WEIGHTS},
        "random_state": RANDOM_STATE,
        "test_indices_one_based": [int(x) for x in (idx_test + 1).tolist()],
        "metrics_summary": {
            "full3d_mae_mean": float(np.mean(mae_list)),
            "full3d_rmse_mean": float(np.mean(rmse_list)),
            "full3d_rRMSE_mean": float(np.mean(rRMSE_list)),
            "plane_rRMSE_mean": {"y_value": float(y_plane_rrmse_value), "value": float(np.mean(rRMSE_plane_list))},

            "plane_SSIM_raw_T_mean": {"y_value": float(y_plane_ssim_value), "value": float(np.mean(ssim_raw_T_list))},
            "plane_SSIM_raw_grad_mean": {"y_value": float(y_plane_ssim_value), "value": float(np.mean(ssim_raw_grad_list))},

            "plane_SSIM_interp_T_mean": {
                "y_value": float(y_plane_ssim_value),
                "grid_res": int(SSIM_GRID_RES),
                "method": SSIM_GRID_METHOD,
                "value": float(np.nanmean(ssim_interp_T_list)) if np.any(np.isfinite(ssim_interp_T_list)) else None
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

            "plane_SSIM_raw_T_each": [float(x) for x in ssim_raw_T_list],
            "plane_SSIM_raw_grad_each": [float(x) for x in ssim_raw_grad_list],
            "plane_SSIM_interp_T_each": [float(x) if np.isfinite(x) else None for x in ssim_interp_T_list],
            "plane_SSIM_interp_grad_each": [float(x) if np.isfinite(x) else None for x in ssim_interp_grad_list],

            "rRMSE_n_top10_points": rRMSE_n_results
        },
        "y_slice_visualization": {
            "y_value": float(ys[y_slice_index]),
            "unified_error_vmax_from_pod": float(shared_error_max) if shared_error_max is not None else None
        },
        "cp_component_weights_preview": [float(x) for x in cp_component_energy_weights(lambdas, Ax, Ay, Az).tolist()],
        "execution_time_seconds": time.time() - start_main_time
    }

    with open(os.path.join(FIG_DIR, "metrics_summary_cp.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone. Outputs in:", os.path.abspath(FIG_DIR))


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
