#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker (TRAIN-only) with Per-Mode Energy-Based Rank Selection + MLP
-------------------------------------------------------------------
- Tucker 分解仅在【训练集】上完成（避免信息泄漏）
- 每个模式 (X/Y/Z/N) 的秩由【该模展开矩阵的能量阈值】自动选取
- 统一子空间 (Ux,Uy,Uz,G) 对全体样本投影得到系数；仅用训练集系数训练回归器
- 增加了 rRMSE 和 rRMSE_n 指标计算
- 增加 DCR 数据压缩率计算：
    DCR = 原始张量元素数量 / 压缩表示元素数量
  可用于比较不同能量阈值保留率 0.90 / 0.95 / 0.99 / 0.999 / ...
- 能量加权损失（可选）、标准化（仅 fit 在训练集）
- 统一误差色标（若提供 POD 的 global_error_max.npy）
- 保存逐工况 MAE/RMSE/rRMSE、DCR、折线图、切片云图与 metrics_summary.json

Author: (your name)
Date: 2025-10-21
"""

import os
import sys
import math
import json
import time
import warnings
from datetime import datetime
from typing import Tuple, Optional

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


# ========================== User Config ==========================

# Paths
PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"  # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES = 89
Y_SLICE = 1.26  # meters (用于切片可视化)

# Energy thresholds (per-mode)
# 对每个模式的展开矩阵做 SVD，累计能量 >= ENERGY_THRESH 即截断
# 当前模型实际采用的能量阈值
ENERGY_THRESH = 0.95  # 可切换 0.90 / 0.95 / 0.99 / 0.999 / 0.9999

# 用于一次性计算不同能量阈值下 DCR 的列表
# 注意：这些阈值只用于计算 DCR 对比表，不会分别训练 MLP。
# 真正训练和重构使用的是上面的 ENERGY_THRESH。
DCR_SWEEP_THRESHOLDS = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]

# DCR 是否把训练集均值场计入压缩存储量
# 如果 CENTER_ALONG_N=True，重构绝对温度场需要保存训练集均值场。
# 论文中有时为了只比较低秩张量表示，会不计入均值场；如需这种口径，可设为 False。
DCR_COUNT_MEAN_FIELD = True

# 每个温度数据默认按 float64 计，用于估计 MB。
# DCR 本身是元素数量之比，因此和字节数单位无关。
DCR_BYTES_PER_VALUE = 8

# Centering and scaling
CENTER_ALONG_N = True  # 按 N 模中心化（用训练集均值）
SCALE_INPUTS = True  # 14 维参数标准化（仅 fit 在训练集）
SCALE_OUTPUT_COEFF = True  # 系数标准化（仅 fit 在训练集）

# Regressor choice
# 'MLP_TORCH'（默认） | 'SVR' | 'GPR'
REGRESSOR = "MLP_TORCH"

# MLP (Torch) hyperparameters
HIDDEN_LAYERS = [128, 256, 128]
USE_LAYERNORM = True
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 2000
BATCH_SIZE = 16

# Energy-weighted loss (沿 rn 对输出维度加权)
USE_ENERGY_WEIGHTS = True

# PCA bottleneck on N-mode coefficients（默认关闭）
USE_PCA_BOTTLENECK = False
PCA_LATENT_Q = 16

# Baselines (if switch REGRESSOR)
GPR_KERNEL = "Matern"  # "RBF" or "Matern"
SVR_KERNEL = "rbf"
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = "scale"
SVR_CACHE_MB = 500

RANDOM_STATE = 42

# Visualization
PLOT_METHOD = "contourf"  # "contourf" | "imshow"
LEVELS = 80
UPSAMPLE_FX = 1
UPSAMPLE_FZ = 1
SAVE_ERROR_MAP = True

# Output directory
# 为了便于不同能量阈值的结果并存，这里增加能量阈值子文件夹。
FIG_BASE_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved"
RUN_TAG = f"energy_{ENERGY_THRESH:.5f}".replace(".", "p")
FIG_DIR = os.path.join(FIG_BASE_DIR, RUN_TAG)
os.makedirs(FIG_DIR, exist_ok=True)

# Unified error colormap with POD
POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"


# ========================== Utils ==========================

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    # 跳过表头行
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


# ========================== DCR Utils ==========================

def rank_from_singular_values(S: np.ndarray, thr: float) -> int:
    """
    根据奇异值累计能量选择秩。

    Parameters
    ----------
    S : np.ndarray
        某一模展开矩阵的奇异值。
    thr : float
        能量阈值，例如 0.90, 0.95, 0.99, 0.999。

    Returns
    -------
    int
        满足累计能量 >= thr 的最小秩。
    """
    if S is None or len(S) == 0:
        return 1

    thr = float(np.clip(thr, 0.0, 1.0))
    energy = S ** 2
    total_energy = np.sum(energy)

    if total_energy <= 0:
        return 1

    cum_energy = np.cumsum(energy) / total_energy
    r = int(np.searchsorted(cum_energy, thr) + 1)
    r = max(1, min(r, len(S)))
    return r


def compute_tucker_dcr_counts(
        nx: int,
        ny: int,
        nz: int,
        n_cases: int,
        ranks: Tuple[int, int, int, int],
        count_mean_field: bool = True,
        bytes_per_value: int = 8
) -> dict:
    """
    计算 Tucker 压缩表示下的 DCR。

    原始张量元素数量：
        nx * ny * nz * n_cases

    压缩后元素数量：
        core_entries + spatial_factor_entries + n_mode_entries + mean_entries

    其中：
        core_entries = rx * ry * rz * rn
        spatial_factor_entries = nx * rx + ny * ry + nz * rz
        n_mode_entries = n_cases * rn
        mean_entries = nx * ny * nz, 如果 count_mean_field=True

    DCR:
        DCR = original_entries / compressed_entries

    compression_percent:
        compression_percent = (1 - compressed_entries / original_entries) * 100%

    注意：
    - 这里统计的是 Tucker 数据表示本身的压缩率，不统计神经网络权重。
    - 如果想将 MLP 参数也作为压缩模型的一部分，可另外将 MLP 参数量加入 compressed_entries。
    """
    rx, ry, rz, rn = ranks

    original_entries = int(nx * ny * nz * n_cases)

    core_entries = int(rx * ry * rz * rn)
    spatial_factor_entries = int(nx * rx + ny * ry + nz * rz)
    n_mode_entries = int(n_cases * rn)
    mean_entries = int(nx * ny * nz) if count_mean_field else 0

    compressed_entries = int(core_entries + spatial_factor_entries + n_mode_entries + mean_entries)

    dcr = float(original_entries / (compressed_entries + 1e-12))
    compression_percent = float((1.0 - compressed_entries / (original_entries + 1e-12)) * 100.0)

    original_mb = float(original_entries * bytes_per_value / (1024 ** 2))
    compressed_mb = float(compressed_entries * bytes_per_value / (1024 ** 2))

    return {
        "original_entries": original_entries,
        "compressed_entries": compressed_entries,
        "core_entries": core_entries,
        "spatial_factor_entries": spatial_factor_entries,
        "n_mode_entries": n_mode_entries,
        "mean_entries": mean_entries,
        "DCR": dcr,
        "compression_percent": compression_percent,
        "original_MB_float64": original_mb,
        "compressed_MB_float64": compressed_mb,
    }


def build_dcr_sweep_table(
        singvals: dict,
        thresholds,
        nx: int,
        ny: int,
        nz: int,
        n_train: int,
        n_all: int,
        count_mean_field: bool = True,
        bytes_per_value: int = 8
) -> pd.DataFrame:
    """
    根据各模奇异值，一次性计算多个能量阈值对应的秩和 DCR。

    输出两个 DCR 口径：
    1. DCR_train_tucker:
       原始训练集张量 vs Tucker 训练集压缩表示。
       压缩表示包含 Ux, Uy, Uz, Un_train, G, mean。

    2. DCR_all_projected_coeff:
       原始全体样本张量 vs 共享空间基 + 核张量 + 每个样本的 rn 维系数。
       压缩表示包含 Ux, Uy, Uz, C_all, G, mean。
       这里 C_all 的存储量为 N_CASES * rn。
    """
    thresholds = sorted(set([float(x) for x in thresholds]))

    rows = []
    for thr in thresholds:
        rx = rank_from_singular_values(singvals["Sx"], thr)
        ry = rank_from_singular_values(singvals["Sy"], thr)
        rz = rank_from_singular_values(singvals["Sz"], thr)
        rn = rank_from_singular_values(singvals["Sn"], thr)
        ranks = (rx, ry, rz, rn)

        train_counts = compute_tucker_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_train,
            ranks=ranks,
            count_mean_field=count_mean_field,
            bytes_per_value=bytes_per_value
        )

        all_counts = compute_tucker_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            ranks=ranks,
            count_mean_field=count_mean_field,
            bytes_per_value=bytes_per_value
        )

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "energy_threshold": float(thr),
            "rx": int(rx),
            "ry": int(ry),
            "rz": int(rz),
            "rn": int(rn),

            "train_original_entries": train_counts["original_entries"],
            "train_compressed_entries": train_counts["compressed_entries"],
            "train_core_entries": train_counts["core_entries"],
            "train_spatial_factor_entries": train_counts["spatial_factor_entries"],
            "train_n_mode_entries": train_counts["n_mode_entries"],
            "train_mean_entries": train_counts["mean_entries"],
            "DCR_train_tucker": train_counts["DCR"],
            "compression_percent_train_tucker": train_counts["compression_percent"],
            "train_original_MB_float64": train_counts["original_MB_float64"],
            "train_compressed_MB_float64": train_counts["compressed_MB_float64"],

            "all_original_entries": all_counts["original_entries"],
            "all_compressed_entries": all_counts["compressed_entries"],
            "all_core_entries": all_counts["core_entries"],
            "all_spatial_factor_entries": all_counts["spatial_factor_entries"],
            "all_coeff_entries": all_counts["n_mode_entries"],
            "all_mean_entries": all_counts["mean_entries"],
            "DCR_all_projected_coeff": all_counts["DCR"],
            "compression_percent_all_projected_coeff": all_counts["compression_percent"],
            "all_original_MB_float64": all_counts["original_MB_float64"],
            "all_compressed_MB_float64": all_counts["compressed_MB_float64"],
        })

    return pd.DataFrame(rows)


def save_dcr_outputs(
        dcr_sweep_df: pd.DataFrame,
        fig_dir: str,
        fig_base_dir: str,
        current_energy_threshold: float
) -> pd.DataFrame:
    """
    保存 DCR 结果：
    - 当前 ENERGY_THRESH 对应的 dcr_current.csv
    - 当前运行得到的 dcr_sweep.csv
    - 跨运行累计/更新的 dcr_vs_energy_threshold.csv
    - DCR 与能量阈值关系图 dcr_vs_energy_threshold.png
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(fig_base_dir, exist_ok=True)

    dcr_sweep_path = os.path.join(fig_dir, "dcr_sweep.csv")
    dcr_sweep_df.to_csv(dcr_sweep_path, index=False, encoding="utf-8-sig")

    current_df = dcr_sweep_df[np.isclose(dcr_sweep_df["energy_threshold"], current_energy_threshold)]
    if current_df.empty:
        current_df = dcr_sweep_df.iloc[[0]]

    dcr_current_path = os.path.join(fig_dir, "dcr_current.csv")
    current_df.to_csv(dcr_current_path, index=False, encoding="utf-8-sig")

    # 将 sweep 结果汇总到总目录，便于多次运行不同阈值后对比。
    comparison_path = os.path.join(fig_base_dir, "dcr_vs_energy_threshold.csv")

    if os.path.exists(comparison_path):
        try:
            old_df = pd.read_csv(comparison_path)
            merged_df = pd.concat([old_df, dcr_sweep_df], ignore_index=True)
        except Exception:
            merged_df = dcr_sweep_df.copy()
    else:
        merged_df = dcr_sweep_df.copy()

    # 同一阈值保留最新一次结果
    merged_df = merged_df.drop_duplicates(subset=["energy_threshold"], keep="last")
    merged_df = merged_df.sort_values("energy_threshold").reset_index(drop=True)
    merged_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    # 绘制 DCR 随能量阈值变化曲线
    plt.figure(figsize=(9, 5))
    plt.plot(
        merged_df["energy_threshold"],
        merged_df["DCR_train_tucker"],
        "-o",
        label="DCR: Train Tucker representation"
    )
    plt.plot(
        merged_df["energy_threshold"],
        merged_df["DCR_all_projected_coeff"],
        "-s",
        label="DCR: All cases with projected coefficients"
    )
    plt.xlabel("Energy Threshold")
    plt.ylabel("Data Compression Ratio, DCR")
    plt.title("DCR vs Energy Threshold")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_base_dir, "dcr_vs_energy_threshold.png"), dpi=200)
    plt.close()

    return merged_df


def print_dcr_report(current_dcr_row: pd.Series):
    """
    在终端打印当前 ENERGY_THRESH 的 DCR 信息。
    """
    print("\n=== DCR Report for Current ENERGY_THRESH ===")
    print(f"  Energy threshold = {current_dcr_row['energy_threshold']:.6f}")
    print(
        f"  Selected ranks: "
        f"rx={int(current_dcr_row['rx'])}, "
        f"ry={int(current_dcr_row['ry'])}, "
        f"rz={int(current_dcr_row['rz'])}, "
        f"rn={int(current_dcr_row['rn'])}"
    )

    print("\n  [Train Tucker representation]")
    print(f"    Original entries   = {int(current_dcr_row['train_original_entries'])}")
    print(f"    Compressed entries = {int(current_dcr_row['train_compressed_entries'])}")
    print(f"    DCR                = {current_dcr_row['DCR_train_tucker']:.6f}")
    print(f"    Saving             = {current_dcr_row['compression_percent_train_tucker']:.2f}%")
    print(f"    Original size      = {current_dcr_row['train_original_MB_float64']:.2f} MB")
    print(f"    Compressed size    = {current_dcr_row['train_compressed_MB_float64']:.2f} MB")

    print("\n  [All cases with projected coefficients]")
    print(f"    Original entries   = {int(current_dcr_row['all_original_entries'])}")
    print(f"    Compressed entries = {int(current_dcr_row['all_compressed_entries'])}")
    print(f"    DCR                = {current_dcr_row['DCR_all_projected_coeff']:.6f}")
    print(f"    Saving             = {current_dcr_row['compression_percent_all_projected_coeff']:.2f}%")
    print(f"    Original size      = {current_dcr_row['all_original_MB_float64']:.2f} MB")
    print(f"    Compressed size    = {current_dcr_row['all_compressed_MB_float64']:.2f} MB")


# ========================== Per-mode energy-based HOSVD (TRAIN-only) ==========================

def select_rank_by_energy_unfold(X: np.ndarray, thr: float, max_rank: Optional[int] = None):
    """
    在某一模展开矩阵 X 上做 randomized SVD，按累计能量阈值 thr 选择秩。
    返回 (U_trunc, r, sing_vals)。

    注意：
    - sing_vals 返回的是本次 randomized_svd 得到的全部奇异值；
    - 由于 n_components 默认取 min(X.shape)，因此可以用于后续 DCR_SWEEP_THRESHOLDS 的秩估计。
    """
    max_possible_rank = min(X.shape[0], X.shape[1])
    n_comp = max_possible_rank if max_rank is None else min(max_possible_rank, max_rank)

    U, S, VT = randomized_svd(X, n_components=n_comp, random_state=RANDOM_STATE)

    r = rank_from_singular_values(S, thr)
    r = max(1, min(r, n_comp))

    return U[:, :r], r, S


def hosvd_per_mode_energy_train_only(T_centered: np.ndarray, idx_train: np.ndarray, thr: float):
    """
    只用训练集子张量 T_centered[..., idx_train] 做 HOSVD；
    每个模按该模展开矩阵的能量阈值 thr 选秩，返回截断后的 (G, [Ux,Uy,Uz,Un], ranks)。
    """
    T_train = T_centered[..., idx_train]

    # X 模
    Ux, rx, Sx = select_rank_by_energy_unfold(mode_n_unfold(T_train, 0), thr)
    # Y 模
    Uy, ry, Sy = select_rank_by_energy_unfold(mode_n_unfold(T_train, 1), thr)
    # Z 模
    Uz, rz, Sz = select_rank_by_energy_unfold(mode_n_unfold(T_train, 2), thr)
    # N 模（工况维，维度为 len(idx_train)）
    Un, rn, Sn = select_rank_by_energy_unfold(mode_n_unfold(T_train, 3), thr)

    ranks = (rx, ry, rz, rn)

    # 核心张量（已是截断因子）
    G = T_train.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)

    return (G, [Ux, Uy, Uz, Un], ranks, {"Sx": Sx, "Sy": Sy, "Sz": Sz, "Sn": Sn})


def project_case_to_coeff(T_case: np.ndarray, Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray,
                          G: np.ndarray) -> np.ndarray:
    """
    将一个 3D 温度场投影到 (Ux,Uy,Uz) 的空间子域后，解最小二乘得到 N 模系数 c。
    G 的形状为 (rx, ry, rz, rn)。
    """
    B = n_mode_product(n_mode_product(n_mode_product(T_case.copy(), Ux.T, 0), Uy.T, 1), Uz.T, 2)  # (rx,ry,rz)
    b = B.reshape(-1)
    A = G.reshape(-1, G.shape[-1])  # (rx*ry*rz, rn)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c  # (rn,)


# ========================== Regressors (Torch/Sklearn) ==========================

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
    if torch.cuda.is_available():
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
    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))
    print(f"Y slice ≈ {Y_SLICE} m -> index {y_slice_index}, y={ys[y_slice_index]:.4f}")

    print("=== Build tensor T ===")
    T = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    T[..., 0] = df_to_grid_values(df0, xs, ys, zs)
    for i in range(2, N_CASES + 1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        dfi = read_one_snapshot_csv(p)
        T[..., i - 1] = df_to_grid_values(dfi, xs, ys, zs)
        if i % 20 == 0 or i == N_CASES:
            print(f"  Loaded {i}/{N_CASES}")

    # ----------------- Fixed train/test indices -----------------
    print("=== Train/Test split (fixed indices) ===")
    TEST_IDX_ONE_BASED = [8, 9, 20, 21, 58, 68, 72, 76, 84]
    idx_test = np.array([i - 1 for i in TEST_IDX_ONE_BASED], dtype=int)
    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(idx_all, idx_test, assume_unique=False)
    print(f"  Test (1-based): {TEST_IDX_ONE_BASED}")
    print(f"  Test (0-based): {idx_test.tolist()}")
    print(f"  Train size = {len(idx_train)}, Test size = {len(idx_test)}")
    # ------------------------------------------------------------

    # === Mean centering defined by TRAIN set only ===
    if CENTER_ALONG_N:
        T_mean_train = T[..., idx_train].mean(axis=3, keepdims=True)  # mean over TRAIN only
        T_centered = T - T_mean_train
        print("Applied N-mode mean centering using TRAIN set mean.")
    else:
        T_mean_train = None
        T_centered = T

    # === HOSVD on TRAIN set with per-mode energy-based ranks ===
    print(f"=== Tucker (TRAIN-only) with per-mode energy threshold = {ENERGY_THRESH} ===")
    (G_train, [Ux, Uy, Uz, Un_train], ranks, singvals) = hosvd_per_mode_energy_train_only(
        T_centered, idx_train, ENERGY_THRESH
    )
    rx, ry, rz, rn = ranks
    print(f"Selected ranks: rx={rx}, ry={ry}, rz={rz}, rn={rn}")
    print(f"Core (TRAIN) shape: {G_train.shape}")

    # =================== DCR Calculation ===================
    print("=== Calculate DCR for current and swept energy thresholds ===")

    dcr_thresholds = sorted(set([float(ENERGY_THRESH)] + [float(x) for x in DCR_SWEEP_THRESHOLDS]))

    count_mean_for_dcr = bool(DCR_COUNT_MEAN_FIELD and CENTER_ALONG_N)

    dcr_sweep_df = build_dcr_sweep_table(
        singvals=singvals,
        thresholds=dcr_thresholds,
        nx=NX,
        ny=NY,
        nz=NZ,
        n_train=len(idx_train),
        n_all=N_CASES,
        count_mean_field=count_mean_for_dcr,
        bytes_per_value=DCR_BYTES_PER_VALUE
    )

    dcr_history_df = save_dcr_outputs(
        dcr_sweep_df=dcr_sweep_df,
        fig_dir=FIG_DIR,
        fig_base_dir=FIG_BASE_DIR,
        current_energy_threshold=ENERGY_THRESH
    )

    current_dcr_df = dcr_sweep_df[np.isclose(dcr_sweep_df["energy_threshold"], ENERGY_THRESH)]
    if current_dcr_df.empty:
        current_dcr_row = dcr_sweep_df.iloc[0]
    else:
        current_dcr_row = current_dcr_df.iloc[0]

    print_dcr_report(current_dcr_row)

    print("\nDCR sweep table:")
    print(
        dcr_sweep_df[
            [
                "energy_threshold", "rx", "ry", "rz", "rn",
                "DCR_train_tucker", "compression_percent_train_tucker",
                "DCR_all_projected_coeff", "compression_percent_all_projected_coeff"
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved DCR current result to: {os.path.join(FIG_DIR, 'dcr_current.csv')}")
    print(f"Saved DCR sweep result to:   {os.path.join(FIG_DIR, 'dcr_sweep.csv')}")
    print(f"Saved DCR comparison to:     {os.path.join(FIG_BASE_DIR, 'dcr_vs_energy_threshold.csv')}")
    print(f"Saved DCR plot to:           {os.path.join(FIG_BASE_DIR, 'dcr_vs_energy_threshold.png')}")

    # === Build coefficients for ALL cases via projection (using TRAIN subspace) ===
    print("=== Build coefficients for ALL cases via projection ===")
    Y_coeff_all = np.zeros((N_CASES, rn), dtype=np.float64)
    for n in range(N_CASES):
        Tc = T_centered[..., n]
        Y_coeff_all[n, :] = project_case_to_coeff(Tc, Ux, Uy, Uz, G_train)

    # Train/Test split for regression
    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff_all[idx_train], Y_coeff_all[idx_test]

    # Energy weights from TRAIN core (length rn)
    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        Gn = np.linalg.norm(G_train.reshape(-1, G_train.shape[-1]), axis=0)
        w = np.clip(Gn / (Gn.max() + 1e-12), 1e-3, None)
    else:
        w = None

    # =================== Regression ===================
    if USE_PCA_BOTTLENECK:
        print(f"=== PCA bottleneck on Y (q={PCA_LATENT_Q}) ===")
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
            model = fit_mlp_torch(
                X_train,
                Z_train_s,
                weight_per_dim=None,
                lr=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                hidden_layers=HIDDEN_LAYERS,
                seed=RANDOM_STATE,
                use_layernorm=USE_LAYERNORM,
                dropout=DROPOUT,
                weight_decay=WEIGHT_DECAY
            )
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
            model = fit_mlp_torch(
                X_train,
                Y_train,
                weight_per_dim=w,
                lr=LEARNING_RATE,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                hidden_layers=HIDDEN_LAYERS,
                seed=RANDOM_STATE,
                use_layernorm=USE_LAYERNORM,
                dropout=DROPOUT,
                weight_decay=WEIGHT_DECAY
            )
            Y_pred_s = predict_mlp_torch(model, X_test)
        else:
            base = choose_regressor(REGRESSOR, Y_train.shape[1])
            base.fit(X_train, Y_train)
            Y_pred_s = base.predict(X_test)
        print(f"Regressor fit time: {time.time() - t0:.2f}s")

        Y_pred = y_scaler.inverse_transform(Y_pred_s) if SCALE_OUTPUT_COEFF else Y_pred_s

    # =================== Reconstruction ===================
    print("=== Reconstruct test 3D fields ===")
    preds, gts = [], []
    for j, case_idx in enumerate(idx_test):
        coeff = Y_pred[j]
        M = np.tensordot(G_train, coeff, axes=([3], [0]))
        T1 = n_mode_product(M, Ux, 0)
        T2 = n_mode_product(T1, Uy, 1)
        T3 = n_mode_product(T2, Uz, 2)
        T_pred = T3 + (T_mean_train[..., 0] if CENTER_ALONG_N else 0.0)
        preds.append(T_pred)
        gts.append(T[..., case_idx])
    preds = np.stack(preds, axis=-1)
    gts = np.stack(gts, axis=-1)

    # =================== Metrics (MAE, RMSE, rRMSE, rRMSE_n) ===================
    print("=== Metrics (per-case & mean) ===")
    mae_list, rmse_list, rRMSE_list = [], [], []

    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_hat = preds[..., k].ravel()

        mae = mean_absolute_error(y_true, y_hat)
        rmse = math.sqrt(mean_squared_error(y_true, y_hat))

        err_vector = y_true - y_hat
        norm_true = np.linalg.norm(y_true)
        norm_err = np.linalg.norm(err_vector)
        rRMSE_case = norm_err / (norm_true + 1e-9)

        mae_list.append(mae)
        rmse_list.append(rmse)
        rRMSE_list.append(rRMSE_case)
        print(f"  Case {idx_test[k] + 1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, rRMSE={rRMSE_case:.4f}")

    print(
        f"\n[Full 3D Mean] "
        f"MAE={np.mean(mae_list):.4f} °C, "
        f"RMSE={np.mean(rmse_list):.4f} °C, "
        f"rRMSE={np.mean(rRMSE_list):.4f}"
    )

    # --- rRMSE_n for top 10 hottest unique points ---
    print("\n=== Calculating rRMSE_n for top 10 hottest unique points ===")
    if T_mean_train is not None:
        mean_temp_field = T_mean_train.squeeze()
    else:
        mean_temp_field = T[..., idx_train].mean(axis=3)

    mean_temp_flat = mean_temp_field.ravel()
    unique_temps_sorted = np.unique(mean_temp_flat)
    # 确保我们有至少10个点，以防万一
    num_points_to_select = min(10, len(unique_temps_sorted))
    top_10_unique_temps = unique_temps_sorted[-num_points_to_select:]

    top_10_indices = []
    for temp_val in reversed(top_10_unique_temps):
        idx = np.where(mean_temp_flat == temp_val)[0][0]
        if idx not in top_10_indices:
            top_10_indices.append(idx)

    print("(Points selected based on highest unique average temperatures in the training set)")
    rRMSE_n_results = {}
    for flat_idx in top_10_indices:
        i, j, k_dim = np.unravel_index(flat_idx, (NX, NY, NZ))
        true_series = gts[i, j, k_dim, :]
        pred_series = preds[i, j, k_dim, :]
        norm_true_series = np.linalg.norm(true_series)
        rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (norm_true_series + 1e-9)

        avg_temp_at_point = mean_temp_field[i, j, k_dim]
        rRMSE_n_results[int(flat_idx)] = float(rRMSE_n_value)
        print(
            f"  Grid Point Index {flat_idx:<7} "
            f"(Avg T ≈ {avg_temp_at_point:.1f}°C): "
            f"rRMSE_n = {rRMSE_n_value:.4f}"
        )

    # =================== Save Metrics and Plots ===================
    df = pd.DataFrame({
        "Case": [int(i) + 1 for i in idx_test],
        "MAE": [float(x) for x in mae_list],
        "RMSE": [float(x) for x in rmse_list],
        "rRMSE": [float(x) for x in rRMSE_list]
    })
    df = df.sort_values("Case")
    df.to_csv(os.path.join(FIG_DIR, "per_case_metrics.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Case"], df["MAE"], '-o', label="MAE (°C)")
    plt.plot(df["Case"], df["RMSE"], '-s', label="RMSE (°C)")
    plt.plot(df["Case"], df["rRMSE"], '-^', label="rRMSE (relative)")
    plt.xlabel("Case")
    plt.ylabel("Error Value")
    plt.title("Tucker+MLP Per-case Error Metrics (TRAIN-only)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_metrics_compare.png"), dpi=200)
    plt.close()

    # =================== Visualization (Y-slice) ===================
    if os.path.exists(POD_ERROR_MAX_PATH):
        try:
            shared_error_max = float(np.load(POD_ERROR_MAX_PATH))
            if not np.isfinite(shared_error_max) or shared_error_max <= 0:
                shared_error_max = None
                print("[Warn] Loaded POD error max is non-positive; fallback to per-case vmax.")
            else:
                print(f"[Info] Unified error colormap vmax from POD: {shared_error_max:.4f}")
        except Exception:
            shared_error_max = None
            print("[Warn] Failed to load POD error max file.")
    else:
        shared_error_max = None
        print("[Info] POD global_error_max.npy not found; Tucker error maps will use per-case vmax.")

    print("=== Plot Y-slice maps (GT vs Pred; +Error) ===")
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]
        err_slice = np.abs(T_true_slice - T_pred_slice)
        vmax = shared_error_max if (shared_error_max is not None) else float(np.nanmax(err_slice))

        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice.T)  # Transpose for plot
        _, _, T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice.T)  # Transpose for plot
        _, _, err_plot = upsample_if_needed(xs, zs, err_slice.T)  # Transpose for plot
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
        c2 = ax2.contourf(
            Xg,
            Zg,
            T_pred_plot.T,
            levels=LEVELS,
            cmap='jet',
            vmin=c1.get_clim()[0],
            vmax=c1.get_clim()[1]
        )
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
        print(f"  Saved {out_path}")

    # =================== Summary ===================
    def sv_info(S):
        e = (S ** 2)
        cum = np.cumsum(e) / np.sum(e)
        k = min(50, len(S))
        return cum[:k].tolist()

    # 当前阈值的 DCR 信息整理为 dict
    current_dcr_info = current_dcr_row.to_dict()

    summary = {
        "model_type": "Tucker_HOSVD",
        "decomposition_set": "TRAIN",
        "ranks": {"rx": int(rx), "ry": int(ry), "rz": int(rz), "rn": int(rn)},
        "energy_threshold_per_mode": float(ENERGY_THRESH),

        "DCR_definition": {
            "formula": "DCR = original_entries / compressed_entries",
            "compressed_entries_for_Tucker": "rx*ry*rz*rn + NX*rx + NY*ry + NZ*rz + N*rn + mean_entries",
            "mean_field_counted": bool(count_mean_for_dcr),
            "note": (
                "DCR here measures Tucker data representation compression. "
                "Neural-network parameters are not included in compressed_entries."
            )
        },

        "DCR_current_energy_threshold": current_dcr_info,

        "DCR_sweep_thresholds": [
            {
                "energy_threshold": float(row["energy_threshold"]),
                "rx": int(row["rx"]),
                "ry": int(row["ry"]),
                "rz": int(row["rz"]),
                "rn": int(row["rn"]),
                "DCR_train_tucker": float(row["DCR_train_tucker"]),
                "compression_percent_train_tucker": float(row["compression_percent_train_tucker"]),
                "DCR_all_projected_coeff": float(row["DCR_all_projected_coeff"]),
                "compression_percent_all_projected_coeff": float(row["compression_percent_all_projected_coeff"]),
            }
            for _, row in dcr_sweep_df.iterrows()
        ],

        "config": {
            "center_along_N": bool(CENTER_ALONG_N),
            "scale_inputs": bool(SCALE_INPUTS),
            "scale_output_coeff": bool(SCALE_OUTPUT_COEFF),
            "dcr_count_mean_field": bool(DCR_COUNT_MEAN_FIELD),
            "dcr_bytes_per_value": int(DCR_BYTES_PER_VALUE),
        },

        "regressor": REGRESSOR,
        "mlp_params": {
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
            "unified_error_vmax_from_pod": float(shared_error_max) if shared_error_max is not None else None,
        },

        "cumulative_energy_curves": {
            "mode_x": sv_info(singvals["Sx"]),
            "mode_y": sv_info(singvals["Sy"]),
            "mode_z": sv_info(singvals["Sz"]),
            "mode_n": sv_info(singvals["Sn"]),
        },

        "output_paths": {
            "run_output_dir": os.path.abspath(FIG_DIR),
            "dcr_current_csv": os.path.abspath(os.path.join(FIG_DIR, "dcr_current.csv")),
            "dcr_sweep_csv": os.path.abspath(os.path.join(FIG_DIR, "dcr_sweep.csv")),
            "dcr_vs_energy_threshold_csv": os.path.abspath(os.path.join(FIG_BASE_DIR, "dcr_vs_energy_threshold.csv")),
            "dcr_vs_energy_threshold_png": os.path.abspath(os.path.join(FIG_BASE_DIR, "dcr_vs_energy_threshold.png")),
        },

        "execution_time_seconds": time.time() - start_main_time
    }

    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Total execution time: {time.time() - start_main_time:.2f}s.")
    print("Outputs in:", os.path.abspath(FIG_DIR))
    print("DCR comparison outputs in:", os.path.abspath(FIG_BASE_DIR))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
