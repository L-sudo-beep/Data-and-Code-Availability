#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CP/PARAFAC (TRAIN-only) + MLP with DCR and computational-time benchmarking
------------------------------------------------------------------------
新增并分别统计以下计算时间：
1. Decomposition time
2. DNN training time
3. DNN inference time per case
4. Field reconstruction time per case
5. Total online prediction time per case

计时口径
--------
Decomposition time:
    训练集平均场计算、训练集中心化、训练集 CP/PARAFAC 分解、
    CP 空间投影矩阵构建，以及训练样本 CP 系数投影。

DNN training time:
    神经网络优化与训练循环。模型、张量和 DataLoader 的初始化不计入。

DNN inference time:
    已完成标准化的单个 14 维输入经过 DNN 前向传播的时间。
    输入标准化、张量创建、输出复制回 CPU 和场重构均不计入。

Field reconstruction time:
    预测输出反标准化、可选 PCA 逆变换、CP 三维场重构及训练集平均场恢复。

Total online prediction time:
    原始 14 维边界条件标准化、张量创建、DNN 前向传播、输出复制回 CPU、
    预测输出反标准化、可选 PCA 逆变换、CP 三维场重构及平均场恢复。

以上核心时间均不包括 CSV 数据读取、DCR 计算、精度指标计算、绘图和文件保存。
"""

import os
import sys
import math
import json
import time
import warnings
import platform
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel as C,
    Matern,
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
Y_SLICE = 1.53

CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

CP_RANK = 6
CP_RANK_SWEEP = [6, 8, 10, 12, 14, 16, 20, 24]
CP_N_ITER_MAX = 2000
CP_TOL = 1e-7

DCR_COUNT_MEAN_FIELD = True
DCR_COUNT_LAMBDAS = True
DCR_BYTES_PER_VALUE = 8
DCR_COUNT_MLP_PARAMS = False

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

POD_ERROR_MAX_PATH = (
    r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"
)

# 为保证 POD、Tucker、CP 的时间对比公平，三个程序应使用相同的设置。
TIMING_INFERENCE_WARMUP_RUNS = 5
TIMING_INFERENCE_REPEATS = 200

TIMING_RECONSTRUCTION_WARMUP_RUNS = 1
TIMING_RECONSTRUCTION_REPEATS = 5

TIMING_TOTAL_ONLINE_WARMUP_RUNS = 1
TIMING_TOTAL_ONLINE_REPEATS = 5


# ========================== Reproducibility ==========================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def sync_device(device: Optional[torch.device] = None):
    """CUDA 计时前后同步；CPU 环境下不执行任何操作。"""
    if torch.cuda.is_available():
        if device is None or device.type == "cuda":
            torch.cuda.synchronize()


def get_hardware_info() -> dict:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "tensorly_version": getattr(tl, "__version__", None) if TL_OK else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


# ========================== Basic Utils ==========================

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")

    if df.shape[0] == N_CASES + 1:
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[0] != N_CASES:
        warnings.warn(
            f"[WARN] Parameter CSV has {df.shape[0]} rows, "
            f"expected {N_CASES}. Proceeding."
        )

    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain at least 14 columns.")

    return df.iloc[:, :14]


def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    required_columns = {"X (m)", "Y (m)", "Z (m)", "Temperature"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Snapshot {path} missing columns {required_columns}"
        )

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


def upsample_if_needed(xs, zs, field_xz):
    if (UPSAMPLE_FX <= 1 and UPSAMPLE_FZ <= 1) or not SCIPY_OK:
        return xs, zs, field_xz

    xi = np.linspace(xs.min(), xs.max(), len(xs) * UPSAMPLE_FX)
    zi = np.linspace(zs.min(), zs.max(), len(zs) * UPSAMPLE_FZ)

    spline = RectBivariateSpline(xs, zs, field_xz)
    high_resolution_field = spline(xi, zi)

    return xi, zi, high_resolution_field


# ========================== DCR for CP/PARAFAC ==========================

def count_mlp_parameters(
        in_dim: int,
        out_dim: int,
        hidden_layers,
        use_layernorm: bool = True,
) -> int:
    total = 0
    previous_dim = in_dim

    for hidden_dim in hidden_layers:
        total += previous_dim * hidden_dim + hidden_dim

        if use_layernorm:
            total += 2 * hidden_dim

        previous_dim = hidden_dim

    total += previous_dim * out_dim + out_dim
    return int(total)


def compute_cp_dcr_counts(
        nx: int,
        ny: int,
        nz: int,
        n_cases: int,
        rank_r: int,
        count_mean_field: bool = True,
        count_lambdas: bool = True,
        bytes_per_value: int = 8,
        extra_entries: int = 0,
) -> dict:
    rank_r = int(rank_r)

    original_entries = int(nx * ny * nz * n_cases)

    spatial_factor_entries = int((nx + ny + nz) * rank_r)
    case_factor_entries = int(n_cases * rank_r)
    lambda_entries = int(rank_r) if count_lambdas else 0
    mean_entries = int(nx * ny * nz) if count_mean_field else 0
    extra_entries = int(extra_entries)

    compressed_entries = int(
        spatial_factor_entries
        + case_factor_entries
        + lambda_entries
        + mean_entries
        + extra_entries
    )

    dcr = float(original_entries / (compressed_entries + 1e-12))
    compression_percent = float(
        (1.0 - compressed_entries / (original_entries + 1e-12)) * 100.0
    )

    original_mb = float(original_entries * bytes_per_value / (1024 ** 2))
    compressed_mb = float(compressed_entries * bytes_per_value / (1024 ** 2))

    return {
        "original_entries": original_entries,
        "compressed_entries": compressed_entries,
        "spatial_factor_entries": spatial_factor_entries,
        "case_factor_entries": case_factor_entries,
        "lambda_entries": lambda_entries,
        "mean_entries": mean_entries,
        "extra_entries": extra_entries,
        "DCR": dcr,
        "compression_percent": compression_percent,
        "original_MB_float64": original_mb,
        "compressed_MB_float64": compressed_mb,
    }


def build_cp_dcr_sweep_table(
        ranks,
        nx: int,
        ny: int,
        nz: int,
        n_train: int,
        n_all: int,
        count_mean_field: bool = True,
        count_lambdas: bool = True,
        count_mlp_params: bool = False,
        bytes_per_value: int = 8,
) -> pd.DataFrame:
    ranks = sorted(set(int(rank) for rank in ranks))
    rows = []

    for rank_r in ranks:
        if USE_PCA_BOTTLENECK:
            mlp_output_dim = min(
                PCA_LATENT_Q,
                rank_r,
                n_train,
            )
        else:
            mlp_output_dim = rank_r

        mlp_parameter_entries = count_mlp_parameters(
            in_dim=14,
            out_dim=mlp_output_dim,
            hidden_layers=HIDDEN_LAYERS,
            use_layernorm=USE_LAYERNORM,
        )

        train_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_train,
            rank_r=rank_r,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=0,
        )

        all_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            rank_r=rank_r,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=0,
        )

        all_with_mlp_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            rank_r=rank_r,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=(
                mlp_parameter_entries
                if count_mlp_params
                else 0
            ),
        )

        rows.append({
            "cp_rank_R": int(rank_r),

            "train_original_entries": train_counts["original_entries"],
            "train_compressed_entries": train_counts["compressed_entries"],
            "train_spatial_factor_entries": (
                train_counts["spatial_factor_entries"]
            ),
            "train_case_factor_entries": train_counts["case_factor_entries"],
            "train_lambda_entries": train_counts["lambda_entries"],
            "train_mean_entries": train_counts["mean_entries"],
            "DCR_train_cp": train_counts["DCR"],
            "compression_percent_train_cp": (
                train_counts["compression_percent"]
            ),
            "train_original_MB_float64": (
                train_counts["original_MB_float64"]
            ),
            "train_compressed_MB_float64": (
                train_counts["compressed_MB_float64"]
            ),

            "all_original_entries": all_counts["original_entries"],
            "all_compressed_entries": all_counts["compressed_entries"],
            "all_spatial_factor_entries": (
                all_counts["spatial_factor_entries"]
            ),
            "all_case_coeff_entries": all_counts["case_factor_entries"],
            "all_lambda_entries": all_counts["lambda_entries"],
            "all_mean_entries": all_counts["mean_entries"],
            "DCR_all_projected_coeff": all_counts["DCR"],
            "compression_percent_all_projected_coeff": (
                all_counts["compression_percent"]
            ),
            "all_original_MB_float64": all_counts["original_MB_float64"],
            "all_compressed_MB_float64": (
                all_counts["compressed_MB_float64"]
            ),

            "mlp_output_dim": int(mlp_output_dim),
            "mlp_param_entries": int(mlp_parameter_entries),
            "DCR_count_mlp_params": bool(count_mlp_params),
            "all_with_mlp_compressed_entries": (
                all_with_mlp_counts["compressed_entries"]
            ),
            "DCR_all_with_mlp_model": all_with_mlp_counts["DCR"],
            "compression_percent_all_with_mlp_model": (
                all_with_mlp_counts["compression_percent"]
            ),
            "all_with_mlp_compressed_MB_float64": (
                all_with_mlp_counts["compressed_MB_float64"]
            ),
        })

    return pd.DataFrame(rows)


def save_cp_dcr_outputs(
        dcr_df: pd.DataFrame,
        fig_dir: str,
        current_rank: int,
) -> pd.Series:
    os.makedirs(fig_dir, exist_ok=True)

    sweep_path = os.path.join(fig_dir, "cp_dcr_rank_sweep.csv")
    dcr_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    current_df = dcr_df[
        dcr_df["cp_rank_R"] == int(current_rank)
    ]

    current_row = (
        dcr_df.iloc[0]
        if current_df.empty
        else current_df.iloc[0]
    )

    current_path = os.path.join(fig_dir, "cp_dcr_current.csv")
    pd.DataFrame([current_row]).to_csv(
        current_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        dcr_df["cp_rank_R"],
        dcr_df["DCR_train_cp"],
        "-o",
        label="DCR: Train CP representation",
    )
    plt.plot(
        dcr_df["cp_rank_R"],
        dcr_df["DCR_all_projected_coeff"],
        "-s",
        label="DCR: All cases with projected coefficients",
    )

    if bool(DCR_COUNT_MLP_PARAMS):
        plt.plot(
            dcr_df["cp_rank_R"],
            dcr_df["DCR_all_with_mlp_model"],
            "-^",
            label="DCR: CP representation with MLP model",
        )

    plt.xlabel("CP Rank R")
    plt.ylabel("Data Compression Ratio, DCR")
    plt.title("CP DCR vs Rank")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, "cp_dcr_vs_rank.png"),
        dpi=200,
    )
    plt.close()

    return current_row


def print_cp_dcr_report(
        current_row: pd.Series,
        dcr_df: pd.DataFrame,
):
    print("\n=== CP DCR Report for Current CP_RANK ===")
    print(
        f"  Current CP rank R = "
        f"{int(current_row['cp_rank_R'])}"
    )

    print("\n  [Train CP representation]")
    print(
        f"    Original entries   = "
        f"{int(current_row['train_original_entries'])}"
    )
    print(
        f"    Compressed entries = "
        f"{int(current_row['train_compressed_entries'])}"
    )
    print(
        f"    DCR                = "
        f"{current_row['DCR_train_cp']:.6f}"
    )
    print(
        f"    Saving             = "
        f"{current_row['compression_percent_train_cp']:.2f}%"
    )

    print("\n  [All cases with projected coefficients]")
    print(
        f"    Original entries   = "
        f"{int(current_row['all_original_entries'])}"
    )
    print(
        f"    Compressed entries = "
        f"{int(current_row['all_compressed_entries'])}"
    )
    print(
        f"    DCR                = "
        f"{current_row['DCR_all_projected_coeff']:.6f}"
    )
    print(
        f"    Saving             = "
        f"{current_row['compression_percent_all_projected_coeff']:.2f}%"
    )

    if bool(DCR_COUNT_MLP_PARAMS):
        print("\n  [CP representation with MLP model]")
        print(
            f"    MLP parameter entries = "
            f"{int(current_row['mlp_param_entries'])}"
        )
        print(
            f"    DCR                   = "
            f"{current_row['DCR_all_with_mlp_model']:.6f}"
        )

    print("\n=== CP DCR Sweep Table ===")

    columns = [
        "cp_rank_R",
        "DCR_train_cp",
        "compression_percent_train_cp",
        "DCR_all_projected_coeff",
        "compression_percent_all_projected_coeff",
        "all_compressed_entries",
    ]

    if bool(DCR_COUNT_MLP_PARAMS):
        columns.extend([
            "mlp_param_entries",
            "DCR_all_with_mlp_model",
            "compression_percent_all_with_mlp_model",
        ])

    print(dcr_df[columns].to_string(index=False))


# ========================== MLP and Regressors ==========================

class MLP(nn.Module):
    """Linear -> optional LayerNorm -> ReLU -> Dropout -> output Linear."""

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
            layers = [nn.Linear(input_dim, output_dim)]

            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))

            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

            return nn.Sequential(*layers)

        layers = []
        previous_dim = in_dim

        for hidden_dim in hidden_layers:
            layers.append(block(previous_dim, hidden_dim))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, out_dim))
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
):
    """
    返回：
        model
        dnn_training_time_seconds

    训练时间只统计 optimizer/training loop。
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    if weight_per_dim is None:
        weight = torch.ones(
            Y_train.shape[1],
            dtype=torch.float32,
            device=device,
        )
    else:
        normalized_weight = weight_per_dim.astype(np.float32)
        normalized_weight = normalized_weight / (
            np.mean(normalized_weight) + 1e-12
        )

        weight = torch.tensor(
            normalized_weight,
            dtype=torch.float32,
            device=device,
        )

    model.train()
    sync_device(device)
    training_loop_start = time.perf_counter()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xb)
            difference = prediction - yb

            loss = torch.mean(
                torch.sum(
                    (difference ** 2) * (weight ** 2),
                    dim=1,
                )
            )

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"[MLP] Epoch {epoch + 1}/{epochs} | "
                f"Loss={loss.item():.6f}"
            )

    sync_device(device)
    training_time_seconds = (
        time.perf_counter() - training_loop_start
    )

    return model, training_time_seconds


def predict_mlp_torch(
        model: nn.Module,
        X: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=device,
        )
        prediction = model(X_tensor)
        sync_device(device)

    return prediction.detach().cpu().numpy()


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

        base_model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=0.0,
        )

        return MultiOutputRegressor(
            base_model,
            n_jobs=None,
        )

    if name == "SVR":
        base_model = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            cache_size=SVR_CACHE_MB,
        )

        return MultiOutputRegressor(
            base_model,
            n_jobs=None,
        )

    if name == "MLP_TORCH":
        return None

    raise ValueError(
        "Unknown REGRESSOR. Supported: MLP_TORCH, SVR, GPR."
    )


# ========================== CP/PARAFAC Core ==========================

def cp_fit_train_tensor(
        T_train_centered: np.ndarray,
        rank_r: int,
        n_iter_max: int = 2000,
        tol: float = 1e-7,
        random_state: int = 42,
):
    """
    只对中心化后的训练集张量进行 CP/PARAFAC 分解。

    T_train_centered:
        shape = (NX, NY, NZ, N_train)
    """
    if not TL_OK:
        raise RuntimeError(
            "TensorLy is required. Install it with: pip install tensorly"
        )

    tl.set_backend("numpy")

    cp_tensor = parafac(
        T_train_centered,
        rank=rank_r,
        init="svd",
        tol=tol,
        n_iter_max=n_iter_max,
        random_state=random_state,
        normalize_factors=True,
    )

    lambdas = cp_tensor.weights
    factors = cp_tensor.factors

    Ax, Ay, Az, An_train = factors

    if lambdas is None:
        lambdas = np.ones(
            Ax.shape[1],
            dtype=np.float64,
        )
    else:
        lambdas = np.asarray(
            lambdas,
            dtype=np.float64,
        )

    return (
        lambdas,
        np.asarray(Ax, dtype=np.float64),
        np.asarray(Ay, dtype=np.float64),
        np.asarray(Az, dtype=np.float64),
        np.asarray(An_train, dtype=np.float64),
    )


def cp_build_projection_matrix(
        Ax: np.ndarray,
        Ay: np.ndarray,
        Az: np.ndarray,
        lambdas: np.ndarray,
) -> np.ndarray:
    """
    构造：
        M[:, r] = lambda_r * vec(a_r ∘ b_r ∘ c_r)

    M shape:
        (NX * NY * NZ, R)

    该矩阵同时用于：
    1. 将训练场投影为 CP 系数；
    2. 根据预测的 CP 系数快速重构完整三维场。
    """
    nx, rank_r = Ax.shape
    ny = Ay.shape[0]
    nz = Az.shape[0]

    projection_matrix = np.empty(
        (nx * ny * nz, rank_r),
        dtype=np.float64,
    )

    for component_index in range(rank_r):
        outer_xy = np.multiply.outer(
            Ax[:, component_index],
            Ay[:, component_index],
        )
        outer_xyz = np.multiply.outer(
            outer_xy,
            Az[:, component_index],
        )

        projection_matrix[:, component_index] = (
            float(lambdas[component_index])
            * outer_xyz.reshape(-1)
        )

    return projection_matrix


def project_case_to_cp_coeff(
        centered_field: np.ndarray,
        projection_matrix: np.ndarray,
) -> np.ndarray:
    flattened_field = centered_field.reshape(-1)

    coefficient, *_ = np.linalg.lstsq(
        projection_matrix,
        flattened_field,
        rcond=None,
    )

    return coefficient


def cp_reconstruct_from_coeff(
        coefficient: np.ndarray,
        projection_matrix: np.ndarray,
        field_shape,
        T_mean_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    快速 CP 重构：
        vec(T_rec) = M @ d
    """
    coefficient = np.asarray(
        coefficient,
        dtype=np.float64,
    ).reshape(-1)

    reconstructed_field = (
        projection_matrix @ coefficient
    ).reshape(field_shape)

    if T_mean_train is not None:
        reconstructed_field = (
            reconstructed_field + T_mean_train[..., 0]
        )

    return reconstructed_field


def cp_component_energy_weights(
        lambdas,
        Ax,
        Ay,
        Az,
):
    rank_r = Ax.shape[1]
    weights = np.zeros(
        rank_r,
        dtype=np.float64,
    )

    for component_index in range(rank_r):
        weights[component_index] = (
            abs(float(lambdas[component_index]))
            * np.linalg.norm(Ax[:, component_index])
            * np.linalg.norm(Ay[:, component_index])
            * np.linalg.norm(Az[:, component_index])
        )

    weights = weights / (
        np.max(weights) + 1e-12
    )
    weights = np.clip(
        weights,
        1e-3,
        None,
    )

    return weights


# ========================== Output Decoding ==========================

def decode_scaled_regressor_output(
        scaled_output: np.ndarray,
        output_scaler=None,
        pca_model: Optional[PCA] = None,
) -> np.ndarray:
    scaled_output = np.asarray(
        scaled_output,
        dtype=np.float64,
    )

    if scaled_output.ndim == 1:
        scaled_output = scaled_output.reshape(1, -1)

    latent_output = (
        output_scaler.inverse_transform(scaled_output)
        if output_scaler is not None
        else scaled_output
    )

    coefficients = (
        pca_model.inverse_transform(latent_output)
        if pca_model is not None
        else latent_output
    )

    return coefficients


def reconstruct_from_scaled_output(
        scaled_output: np.ndarray,
        output_scaler,
        pca_model: Optional[PCA],
        projection_matrix: np.ndarray,
        field_shape,
        T_mean_train: Optional[np.ndarray],
) -> np.ndarray:
    coefficients = decode_scaled_regressor_output(
        scaled_output=scaled_output,
        output_scaler=output_scaler,
        pca_model=pca_model,
    )

    return cp_reconstruct_from_coeff(
        coefficient=coefficients[0],
        projection_matrix=projection_matrix,
        field_shape=field_shape,
        T_mean_train=T_mean_train,
    )


# ========================== Timing Benchmarks ==========================

def benchmark_torch_inference_per_case(
        model: nn.Module,
        X_test_scaled: np.ndarray,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    只统计 DNN 前向传播。

    不统计：
    - 输入标准化
    - NumPy -> Torch 张量创建
    - GPU -> CPU 输出复制
    - 输出反标准化
    - CP 场重构
    """
    device = next(model.parameters()).device
    model.eval()

    per_case_times = []

    with torch.no_grad():
        for row in X_test_scaled:
            input_tensor = torch.as_tensor(
                row.reshape(1, -1),
                dtype=torch.float32,
                device=device,
            )

            for _ in range(warmup_runs):
                _ = model(input_tensor)

            sync_device(device)
            repeated_times = []

            for _ in range(repeats):
                sync_device(device)
                start = time.perf_counter()

                _ = model(input_tensor)

                sync_device(device)
                repeated_times.append(
                    time.perf_counter() - start
                )

            per_case_times.append(
                float(np.mean(repeated_times))
            )

    return np.asarray(
        per_case_times,
        dtype=np.float64,
    )


def benchmark_sklearn_inference_per_case(
        model,
        X_test_scaled: np.ndarray,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    per_case_times = []

    for row in X_test_scaled:
        input_row = row.reshape(1, -1)

        for _ in range(warmup_runs):
            _ = model.predict(input_row)

        repeated_times = []

        for _ in range(repeats):
            start = time.perf_counter()
            _ = model.predict(input_row)
            repeated_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeated_times))
        )

    return np.asarray(
        per_case_times,
        dtype=np.float64,
    )


def benchmark_field_reconstruction_per_case(
        predicted_scaled_outputs: np.ndarray,
        output_scaler,
        pca_model: Optional[PCA],
        projection_matrix: np.ndarray,
        field_shape,
        T_mean_train: Optional[np.ndarray],
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    统计：
    输出反标准化 + 可选 PCA 逆变换 + CP 重构 + 平均场恢复。
    """
    per_case_times = []

    for scaled_output in predicted_scaled_outputs:
        for _ in range(warmup_runs):
            _ = reconstruct_from_scaled_output(
                scaled_output=scaled_output,
                output_scaler=output_scaler,
                pca_model=pca_model,
                projection_matrix=projection_matrix,
                field_shape=field_shape,
                T_mean_train=T_mean_train,
            )

        repeated_times = []

        for _ in range(repeats):
            start = time.perf_counter()

            _ = reconstruct_from_scaled_output(
                scaled_output=scaled_output,
                output_scaler=output_scaler,
                pca_model=pca_model,
                projection_matrix=projection_matrix,
                field_shape=field_shape,
                T_mean_train=T_mean_train,
            )

            repeated_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeated_times))
        )

    return np.asarray(
        per_case_times,
        dtype=np.float64,
    )


def predict_scaled_output_one_case(
        model,
        raw_condition: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
) -> np.ndarray:
    raw_condition = np.asarray(
        raw_condition,
        dtype=np.float64,
    ).reshape(1, -1)

    scaled_condition = (
        input_scaler.transform(raw_condition)
        if scale_inputs
        else raw_condition
    )

    if regressor_name.upper() == "MLP_TORCH":
        device = next(model.parameters()).device

        condition_tensor = torch.as_tensor(
            scaled_condition,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            scaled_output_tensor = model(
                condition_tensor
            )

        sync_device(device)

        return (
            scaled_output_tensor
            .detach()
            .cpu()
            .numpy()
        )

    return model.predict(scaled_condition)


def predict_full_field_online(
        model,
        raw_condition: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
        output_scaler,
        pca_model: Optional[PCA],
        projection_matrix: np.ndarray,
        field_shape,
        T_mean_train: Optional[np.ndarray],
) -> np.ndarray:
    scaled_output = predict_scaled_output_one_case(
        model=model,
        raw_condition=raw_condition,
        input_scaler=input_scaler,
        scale_inputs=scale_inputs,
        regressor_name=regressor_name,
    )

    return reconstruct_from_scaled_output(
        scaled_output=scaled_output[0],
        output_scaler=output_scaler,
        pca_model=pca_model,
        projection_matrix=projection_matrix,
        field_shape=field_shape,
        T_mean_train=T_mean_train,
    )


def benchmark_total_online_prediction_per_case(
        model,
        X_test_raw: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
        output_scaler,
        pca_model: Optional[PCA],
        projection_matrix: np.ndarray,
        field_shape,
        T_mean_train: Optional[np.ndarray],
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    完整在线预测时间：
    输入标准化 + 张量创建 + DNN 推理 + 输出复制回 CPU +
    输出反标准化 + 可选 PCA 逆变换 + CP 重构 + 平均场恢复。
    """
    per_case_times = []

    device = None
    if regressor_name.upper() == "MLP_TORCH":
        device = next(model.parameters()).device

    for raw_condition in X_test_raw:
        for _ in range(warmup_runs):
            _ = predict_full_field_online(
                model=model,
                raw_condition=raw_condition,
                input_scaler=input_scaler,
                scale_inputs=scale_inputs,
                regressor_name=regressor_name,
                output_scaler=output_scaler,
                pca_model=pca_model,
                projection_matrix=projection_matrix,
                field_shape=field_shape,
                T_mean_train=T_mean_train,
            )

        repeated_times = []

        for _ in range(repeats):
            sync_device(device)
            start = time.perf_counter()

            _ = predict_full_field_online(
                model=model,
                raw_condition=raw_condition,
                input_scaler=input_scaler,
                scale_inputs=scale_inputs,
                regressor_name=regressor_name,
                output_scaler=output_scaler,
                pca_model=pca_model,
                projection_matrix=projection_matrix,
                field_shape=field_shape,
                T_mean_train=T_mean_train,
            )

            sync_device(device)
            repeated_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeated_times))
        )

    return np.asarray(
        per_case_times,
        dtype=np.float64,
    )


def summarize_latency_seconds(
        values: np.ndarray,
) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=0)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "mean_ms": float(np.mean(values) * 1000.0),
        "std_ms": float(np.std(values, ddof=0) * 1000.0),
        "min_ms": float(np.min(values) * 1000.0),
        "max_ms": float(np.max(values) * 1000.0),
    }


def save_timing_outputs(
        fig_dir: str,
        test_indices_one_based,
        regressor_name: str,
        device_name: str,
        decomposition_time_seconds: float,
        training_time_seconds: float,
        inference_times_seconds: np.ndarray,
        reconstruction_times_seconds: np.ndarray,
        total_online_times_seconds: np.ndarray,
) -> dict:
    inference_summary = summarize_latency_seconds(
        inference_times_seconds
    )
    reconstruction_summary = summarize_latency_seconds(
        reconstruction_times_seconds
    )
    online_summary = summarize_latency_seconds(
        total_online_times_seconds
    )

    total_offline_time_seconds = float(
        decomposition_time_seconds
        + training_time_seconds
    )

    aggregate_df = pd.DataFrame([{
        "Method": "CP/PARAFAC",
        "Regressor": regressor_name,
        "Device": device_name,
        "Decomposition time (s)": decomposition_time_seconds,
        "DNN training time (s)": training_time_seconds,
        "Total offline time (s)": total_offline_time_seconds,
        "DNN inference time per case - mean (ms)": (
            inference_summary["mean_ms"]
        ),
        "DNN inference time per case - std (ms)": (
            inference_summary["std_ms"]
        ),
        "Field reconstruction time per case - mean (ms)": (
            reconstruction_summary["mean_ms"]
        ),
        "Field reconstruction time per case - std (ms)": (
            reconstruction_summary["std_ms"]
        ),
        "Total online prediction time per case - mean (ms)": (
            online_summary["mean_ms"]
        ),
        "Total online prediction time per case - std (ms)": (
            online_summary["std_ms"]
        ),
    }])

    aggregate_path = os.path.join(
        fig_dir,
        "computational_time_metrics_cp.csv",
    )
    aggregate_df.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    per_case_df = pd.DataFrame({
        "Case": test_indices_one_based,
        "DNN inference time (ms)": (
            inference_times_seconds * 1000.0
        ),
        "Field reconstruction time (ms)": (
            reconstruction_times_seconds * 1000.0
        ),
        "Total online prediction time (ms)": (
            total_online_times_seconds * 1000.0
        ),
    })

    per_case_path = os.path.join(
        fig_dir,
        "computational_time_per_case_cp.csv",
    )
    per_case_df.to_csv(
        per_case_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "decomposition_time_seconds": float(
            decomposition_time_seconds
        ),
        "dnn_training_time_seconds": float(
            training_time_seconds
        ),
        "total_offline_time_seconds": (
            total_offline_time_seconds
        ),
        "dnn_inference_time_per_case": (
            inference_summary
        ),
        "field_reconstruction_time_per_case": (
            reconstruction_summary
        ),
        "total_online_prediction_time_per_case": (
            online_summary
        ),
        "aggregate_csv": os.path.abspath(
            aggregate_path
        ),
        "per_case_csv": os.path.abspath(
            per_case_path
        ),
    }


# ========================== Main ==========================

def main():
    np.random.seed(RANDOM_STATE)
    total_script_start = time.perf_counter()

    if not TL_OK:
        raise RuntimeError(
            "TensorLy was not detected. Install it with: pip install tensorly"
        )

    print("=== Load parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read first snapshot and build grid ===")
    first_snapshot = read_one_snapshot_csv(
        os.path.join(SNAPSHOT_DIR, "1.csv")
    )
    xs, ys, zs = build_grid_from_df(first_snapshot)

    y_slice_index = int(
        np.argmin(np.abs(ys - Y_SLICE))
    )
    print(
        f"Y slice ≈ {Y_SLICE} m -> "
        f"index {y_slice_index}, "
        f"y={ys[y_slice_index]:.4f}"
    )

    print("=== Build tensor T ===")
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

    # ----------------- Fixed train/test indices -----------------
    print("=== Train/Test split (fixed indices) ===")

    TEST_IDX_ONE_BASED = [
        8, 9, 20, 21, 58, 68, 72, 76, 84
    ]

    idx_test = np.asarray(
        [index - 1 for index in TEST_IDX_ONE_BASED],
        dtype=int,
    )

    idx_all = np.arange(N_CASES)
    idx_train = np.setdiff1d(
        idx_all,
        idx_test,
        assume_unique=False,
    )

    print(f"  Test (1-based): {TEST_IDX_ONE_BASED}")
    print(f"  Test (0-based): {idx_test.tolist()}")
    print(
        f"  Train size = {len(idx_train)}, "
        f"Test size = {len(idx_test)}"
    )

    # =================== CP DCR Calculation ===================
    # DCR 不属于核心分解计时，避免文件和表格操作污染 Decomposition time。
    print("=== Calculate CP DCR for different ranks ===")

    count_mean_for_dcr = bool(
        DCR_COUNT_MEAN_FIELD and CENTER_ALONG_N
    )

    dcr_ranks = sorted(
        set(
            [int(CP_RANK)]
            + [int(rank) for rank in CP_RANK_SWEEP]
        )
    )

    cp_dcr_df = build_cp_dcr_sweep_table(
        ranks=dcr_ranks,
        nx=NX,
        ny=NY,
        nz=NZ,
        n_train=len(idx_train),
        n_all=N_CASES,
        count_mean_field=count_mean_for_dcr,
        count_lambdas=DCR_COUNT_LAMBDAS,
        count_mlp_params=DCR_COUNT_MLP_PARAMS,
        bytes_per_value=DCR_BYTES_PER_VALUE,
    )

    current_dcr_row = save_cp_dcr_outputs(
        dcr_df=cp_dcr_df,
        fig_dir=FIG_DIR,
        current_rank=CP_RANK,
    )

    print_cp_dcr_report(
        current_dcr_row,
        cp_dcr_df,
    )

    # ============================================================
    # Decomposition Time
    # ============================================================
    # 包含：
    # 1. 训练集平均场计算
    # 2. 训练集中心化
    # 3. 训练集 CP/PARAFAC 分解
    # 4. CP 空间投影矩阵构建
    # 5. 训练样本 CP 系数投影
    #
    # 不包含：
    # - CSV 数据读取
    # - DCR 表格计算和保存
    # - DNN 训练
    # - 测试工况预测
    # ============================================================
    print(
        f"\n=== CP/PARAFAC TRAIN-only decomposition, "
        f"rank = {CP_RANK} ==="
    )

    decomposition_start = time.perf_counter()

    if CENTER_ALONG_N:
        T_mean_train = T[..., idx_train].mean(
            axis=3,
            keepdims=True,
        )

        T_train_centered = (
            T[..., idx_train] - T_mean_train
        )

        print(
            "Applied N-mode mean centering "
            "using TRAIN-set mean."
        )
    else:
        T_mean_train = None
        T_train_centered = (
            T[..., idx_train].copy()
        )

    (
        lambdas,
        Ax,
        Ay,
        Az,
        An_train,
    ) = cp_fit_train_tensor(
        T_train_centered=T_train_centered,
        rank_r=CP_RANK,
        n_iter_max=CP_N_ITER_MAX,
        tol=CP_TOL,
        random_state=RANDOM_STATE,
    )

    projection_matrix = cp_build_projection_matrix(
        Ax=Ax,
        Ay=Ay,
        Az=Az,
        lambdas=lambdas,
    )

    rank_r = Ax.shape[1]

    # 只投影训练样本作为回归目标。
    # 测试场不参与 CP 分解，也不参与回归器训练。
    Y_train_raw = np.zeros(
        (len(idx_train), rank_r),
        dtype=np.float64,
    )

    for local_train_index in range(len(idx_train)):
        Y_train_raw[local_train_index, :] = (
            project_case_to_cp_coeff(
                centered_field=(
                    T_train_centered[
                        ..., local_train_index
                    ]
                ),
                projection_matrix=projection_matrix,
            )
        )

    decomposition_time_seconds = (
        time.perf_counter() - decomposition_start
    )

    print(
        f"[CP] Shapes: "
        f"Ax={Ax.shape}, Ay={Ay.shape}, "
        f"Az={Az.shape}, An_train={An_train.shape}"
    )
    print(
        f"Decomposition time: "
        f"{decomposition_time_seconds:.6f} s"
    )

    # 训练中心化张量在系数投影完成后不再需要。
    del T_train_centered

    # =================== Regression Data ===================
    X_train_raw = X_params[idx_train]
    X_test_raw = X_params[idx_test]

    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        energy_weights = cp_component_energy_weights(
            lambdas=lambdas,
            Ax=Ax,
            Ay=Ay,
            Az=Az,
        )
    else:
        energy_weights = None

    input_scaler = None
    output_scaler = None
    pca_model = None
    fitted_model = None

    if SCALE_INPUTS:
        input_scaler = StandardScaler().fit(
            X_train_raw
        )
        X_train = input_scaler.transform(
            X_train_raw
        )
        X_test = input_scaler.transform(
            X_test_raw
        )
    else:
        X_train = X_train_raw.copy()
        X_test = X_test_raw.copy()

    # =================== Regression Output Preparation ===================
    if USE_PCA_BOTTLENECK:
        actual_pca_components = min(
            PCA_LATENT_Q,
            Y_train_raw.shape[0],
            Y_train_raw.shape[1],
        )

        print(
            f"=== PCA bottleneck on CP coefficients "
            f"(q={actual_pca_components}) ==="
        )

        pca_model = PCA(
            n_components=actual_pca_components,
            random_state=RANDOM_STATE,
        )

        latent_train = pca_model.fit_transform(
            Y_train_raw
        )

        if SCALE_OUTPUT_COEFF:
            output_scaler = StandardScaler().fit(
                latent_train
            )
            regressor_target_train = (
                output_scaler.transform(
                    latent_train
                )
            )
        else:
            regressor_target_train = latent_train

        regressor_output_dim = actual_pca_components
        regression_weights = None

    else:
        if SCALE_OUTPUT_COEFF:
            output_scaler = StandardScaler().fit(
                Y_train_raw
            )
            regressor_target_train = (
                output_scaler.transform(
                    Y_train_raw
                )
            )
        else:
            regressor_target_train = Y_train_raw

        regressor_output_dim = rank_r
        regression_weights = energy_weights

    print(
        f"=== Fit regressor: {REGRESSOR} | "
        f"output_dim={regressor_output_dim} | "
        f"energy_weighted={USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK} ==="
    )

    # ============================================================
    # DNN Training Time
    # ============================================================
    if REGRESSOR.upper() == "MLP_TORCH":
        (
            fitted_model,
            dnn_training_time_seconds,
        ) = fit_mlp_torch(
            X_train=X_train,
            Y_train=regressor_target_train,
            weight_per_dim=regression_weights,
            lr=LEARNING_RATE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            hidden_layers=HIDDEN_LAYERS,
            seed=RANDOM_STATE,
            use_layernorm=USE_LAYERNORM,
            dropout=DROPOUT,
            weight_decay=WEIGHT_DECAY,
        )

        model_device = next(
            fitted_model.parameters()
        ).device

        if model_device.type == "cuda":
            device_name = (
                f"CUDA: {torch.cuda.get_device_name(model_device)}"
            )
        else:
            device_name = "CPU"

    else:
        fitted_model = choose_regressor(
            REGRESSOR,
            regressor_output_dim,
        )

        training_start = time.perf_counter()

        fitted_model.fit(
            X_train,
            regressor_target_train,
        )

        dnn_training_time_seconds = (
            time.perf_counter() - training_start
        )

        model_device = None
        device_name = "CPU"

    print(
        f"DNN/regressor training time: "
        f"{dnn_training_time_seconds:.6f} s"
    )

    # 一次常规批量预测，用于精度评估和重构计时。
    if REGRESSOR.upper() == "MLP_TORCH":
        predicted_scaled_outputs = predict_mlp_torch(
            fitted_model,
            X_test,
        )
    else:
        predicted_scaled_outputs = (
            fitted_model.predict(X_test)
        )

    predicted_coefficients = (
        decode_scaled_regressor_output(
            scaled_output=predicted_scaled_outputs,
            output_scaler=output_scaler,
            pca_model=pca_model,
        )
    )

    # ============================================================
    # DNN Inference Time
    # ============================================================
    print("=== Benchmark DNN inference time ===")

    if REGRESSOR.upper() == "MLP_TORCH":
        inference_times_seconds = (
            benchmark_torch_inference_per_case(
                model=fitted_model,
                X_test_scaled=X_test,
                warmup_runs=(
                    TIMING_INFERENCE_WARMUP_RUNS
                ),
                repeats=TIMING_INFERENCE_REPEATS,
            )
        )
    else:
        inference_times_seconds = (
            benchmark_sklearn_inference_per_case(
                model=fitted_model,
                X_test_scaled=X_test,
                warmup_runs=(
                    TIMING_INFERENCE_WARMUP_RUNS
                ),
                repeats=TIMING_INFERENCE_REPEATS,
            )
        )

    # ============================================================
    # Field Reconstruction Time
    # ============================================================
    print("=== Benchmark field reconstruction time ===")

    reconstruction_times_seconds = (
        benchmark_field_reconstruction_per_case(
            predicted_scaled_outputs=(
                predicted_scaled_outputs
            ),
            output_scaler=output_scaler,
            pca_model=pca_model,
            projection_matrix=projection_matrix,
            field_shape=(NX, NY, NZ),
            T_mean_train=T_mean_train,
            warmup_runs=(
                TIMING_RECONSTRUCTION_WARMUP_RUNS
            ),
            repeats=(
                TIMING_RECONSTRUCTION_REPEATS
            ),
        )
    )

    # ============================================================
    # Total Online Prediction Time
    # ============================================================
    print("=== Benchmark total online prediction time ===")

    total_online_times_seconds = (
        benchmark_total_online_prediction_per_case(
            model=fitted_model,
            X_test_raw=X_test_raw,
            input_scaler=input_scaler,
            scale_inputs=SCALE_INPUTS,
            regressor_name=REGRESSOR,
            output_scaler=output_scaler,
            pca_model=pca_model,
            projection_matrix=projection_matrix,
            field_shape=(NX, NY, NZ),
            T_mean_train=T_mean_train,
            warmup_runs=(
                TIMING_TOTAL_ONLINE_WARMUP_RUNS
            ),
            repeats=(
                TIMING_TOTAL_ONLINE_REPEATS
            ),
        )
    )

    timing_results = save_timing_outputs(
        fig_dir=FIG_DIR,
        test_indices_one_based=TEST_IDX_ONE_BASED,
        regressor_name=REGRESSOR,
        device_name=device_name,
        decomposition_time_seconds=(
            decomposition_time_seconds
        ),
        training_time_seconds=(
            dnn_training_time_seconds
        ),
        inference_times_seconds=(
            inference_times_seconds
        ),
        reconstruction_times_seconds=(
            reconstruction_times_seconds
        ),
        total_online_times_seconds=(
            total_online_times_seconds
        ),
    )

    print(
        "\n"
        + "=" * 16
        + " COMPUTATIONAL TIME "
        + "=" * 16
    )
    print(
        f"Decomposition time                    : "
        f"{decomposition_time_seconds:.6f} s"
    )
    print(
        f"DNN training time                    : "
        f"{dnn_training_time_seconds:.6f} s"
    )
    print(
        "DNN inference time per case          : "
        f"{timing_results['dnn_inference_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['dnn_inference_time_per_case']['std_ms']:.6f} ms"
    )
    print(
        "Field reconstruction time per case   : "
        f"{timing_results['field_reconstruction_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['field_reconstruction_time_per_case']['std_ms']:.6f} ms"
    )
    print(
        "Total online prediction time per case: "
        f"{timing_results['total_online_prediction_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['total_online_prediction_time_per_case']['std_ms']:.6f} ms"
    )
    print("=" * 53)

    # =================== Reconstruction ===================
    print("=== Reconstruct test 3-D fields (CP) ===")

    predicted_fields = []
    ground_truth_fields = []

    for test_position, case_index in enumerate(idx_test):
        predicted_field = cp_reconstruct_from_coeff(
            coefficient=(
                predicted_coefficients[test_position]
            ),
            projection_matrix=projection_matrix,
            field_shape=(NX, NY, NZ),
            T_mean_train=T_mean_train,
        )

        predicted_fields.append(
            predicted_field
        )
        ground_truth_fields.append(
            T[..., case_index]
        )

    preds = np.stack(
        predicted_fields,
        axis=-1,
    )
    gts = np.stack(
        ground_truth_fields,
        axis=-1,
    )


    # =================== Accuracy Metrics ===================
    print("=== Metrics (per case and mean) ===")

    mae_list = []
    rmse_list = []
    rrmse_list = []

    for test_position in range(preds.shape[-1]):
        y_true = gts[..., test_position].ravel()
        y_pred = preds[..., test_position].ravel()

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )
        rmse = math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        error_vector = y_true - y_pred
        relative_rmse = (
            np.linalg.norm(error_vector)
            / (np.linalg.norm(y_true) + 1e-9)
        )

        mae_list.append(float(mae))
        rmse_list.append(float(rmse))
        rrmse_list.append(float(relative_rmse))

        print(
            f"  Case {idx_test[test_position] + 1:2d}: "
            f"MAE={mae:.4f} °C, "
            f"RMSE={rmse:.4f} °C, "
            f"rRMSE={relative_rmse:.4f}"
        )

    print(
        "\n[Full 3-D Mean] "
        f"MAE={np.mean(mae_list):.4f} °C, "
        f"RMSE={np.mean(rmse_list):.4f} °C, "
        f"rRMSE={np.mean(rrmse_list):.4f}"
    )

    # =================== rRMSE_n ===================
    print(
        "\n=== Calculating rRMSE_n for "
        "top 10 hottest unique points ==="
    )

    if T_mean_train is not None:
        mean_temperature_field = (
            T_mean_train.squeeze()
        )
    else:
        mean_temperature_field = (
            T[..., idx_train].mean(axis=3)
        )

    mean_temperature_flat = (
        mean_temperature_field.ravel()
    )

    unique_temperatures = np.unique(
        mean_temperature_flat
    )

    number_of_points = min(
        10,
        len(unique_temperatures),
    )

    hottest_unique_temperatures = (
        unique_temperatures[-number_of_points:]
    )

    hottest_point_indices = []

    for temperature_value in reversed(
            hottest_unique_temperatures
    ):
        flat_index = np.where(
            mean_temperature_flat
            == temperature_value
        )[0][0]

        if flat_index not in hottest_point_indices:
            hottest_point_indices.append(
                int(flat_index)
            )

    print(
        "(Points selected from the highest unique "
        "training-set average temperatures)"
    )

    rrmse_n_results = {}

    for flat_index in hottest_point_indices:
        ix, iy, iz = np.unravel_index(
            flat_index,
            (NX, NY, NZ),
        )

        true_series = gts[ix, iy, iz, :]
        predicted_series = preds[ix, iy, iz, :]

        relative_rmse_at_point = (
            np.linalg.norm(
                true_series - predicted_series
            )
            / (np.linalg.norm(true_series) + 1e-9)
        )

        average_temperature = (
            mean_temperature_field[ix, iy, iz]
        )

        rrmse_n_results[
            int(flat_index)
        ] = float(relative_rmse_at_point)

        print(
            f"  Grid Point Index {flat_index:<7} "
            f"(Avg T ≈ {average_temperature:.1f}°C): "
            f"rRMSE_n = {relative_rmse_at_point:.4f}"
        )

    # =================== Save Accuracy Metrics ===================
    metrics_df = pd.DataFrame({
        "Case": [
            int(case_index) + 1
            for case_index in idx_test
        ],
        "MAE": mae_list,
        "RMSE": rmse_list,
        "rRMSE": rrmse_list,
    }).sort_values("Case")

    per_case_metrics_path = os.path.join(
        FIG_DIR,
        "per_case_metrics_cp.csv",
    )

    metrics_df.to_csv(
        per_case_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        metrics_df["Case"],
        metrics_df["MAE"],
        "-o",
        label="MAE (°C)",
    )
    plt.plot(
        metrics_df["Case"],
        metrics_df["RMSE"],
        "-s",
        label="RMSE (°C)",
    )
    plt.plot(
        metrics_df["Case"],
        metrics_df["rRMSE"],
        "-^",
        label="rRMSE (relative)",
    )
    plt.xlabel("Case")
    plt.ylabel("Error Value")
    plt.title(
        "CP+MLP Per-case Error Metrics "
        "(TRAIN-only)"
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
            "error_metrics_compare_cp.png",
        ),
        dpi=200,
    )
    plt.close()

    # =================== Visualization ===================
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
                print(
                    "[Warn] Loaded POD error max is "
                    "non-positive; use per-case vmax."
                )
            else:
                print(
                    "[Info] Unified error-map vmax "
                    f"from POD: {shared_error_max:.4f}"
                )

        except Exception:
            shared_error_max = None
            print(
                "[Warn] Failed to load POD "
                "global_error_max.npy."
            )
    else:
        shared_error_max = None
        print(
            "[Info] POD global_error_max.npy was not found; "
            "CP error maps use per-case vmax."
        )

    print(
        "=== Plot Y-slice maps "
        "(ground truth, prediction, error) ==="
    )

    for test_position, case_index in enumerate(idx_test):
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

        xs_plot, zs_plot, true_plot = (
            upsample_if_needed(
                xs,
                zs,
                true_slice.T,
            )
        )
        _, _, predicted_plot = (
            upsample_if_needed(
                xs,
                zs,
                predicted_slice.T,
            )
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
            f"Case {case_index + 1} (CP)",
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

        axis_predicted = figure.add_subplot(
            1,
            3,
            2,
        )
        contour_predicted = (
            axis_predicted.contourf(
                X_grid,
                Z_grid,
                predicted_plot.T,
                levels=LEVELS,
                cmap="jet",
                vmin=contour_true.get_clim()[0],
                vmax=contour_true.get_clim()[1],
            )
        )
        axis_predicted.set_title(
            "Reconstructed Temperature (CP)"
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
            axis_error = figure.add_subplot(
                1,
                3,
                3,
            )
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
                f"Absolute Error "
                f"(vmax={error_vmax:.2f})"
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
            f"yslice_case_{case_index + 1}_cp.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"  Saved {output_path}")

    # =================== JSON Summary ===================
    total_script_time_seconds = (
        time.perf_counter() - total_script_start
    )

    cp_weight_preview = (
        cp_component_energy_weights(
            lambdas,
            Ax,
            Ay,
            Az,
        )
    )

    summary = {
        "model_type": "CP_PARAFAC_MLP",
        "decomposition_set": "TRAIN",
        "cp_rank_R": int(CP_RANK),

        "hardware": get_hardware_info(),
        "regressor_device": device_name,

        "timing_definition": {
            "decomposition_time": (
                "Training mean field, training-set centering, "
                "train-only CP/PARAFAC decomposition, CP projection "
                "matrix construction, and training coefficient projection."
            ),
            "dnn_training_time": (
                "Neural-network optimizer/training loop only. "
                "Model, tensor, and DataLoader initialization are excluded."
            ),
            "dnn_inference_time": (
                "DNN forward propagation only for one already-standardized "
                "14-D input. Tensor creation, output CPU transfer, output "
                "inverse scaling, and field reconstruction are excluded."
            ),
            "field_reconstruction_time": (
                "Prediction inverse standardization, optional inverse PCA, "
                "CP full-field reconstruction, and training-mean restoration."
            ),
            "total_online_prediction_time": (
                "Raw-input standardization, tensor creation, DNN forward "
                "propagation, output CPU transfer, prediction inverse "
                "standardization, optional inverse PCA, CP full-field "
                "reconstruction, and training-mean restoration."
            ),
            "excluded_from_core_timing": (
                "CSV loading, DCR calculation, accuracy metrics, plotting, "
                "and file writing."
            ),
        },

        "timing_settings": {
            "inference_warmup_runs": int(
                TIMING_INFERENCE_WARMUP_RUNS
            ),
            "inference_repeats": int(
                TIMING_INFERENCE_REPEATS
            ),
            "reconstruction_warmup_runs": int(
                TIMING_RECONSTRUCTION_WARMUP_RUNS
            ),
            "reconstruction_repeats": int(
                TIMING_RECONSTRUCTION_REPEATS
            ),
            "total_online_warmup_runs": int(
                TIMING_TOTAL_ONLINE_WARMUP_RUNS
            ),
            "total_online_repeats": int(
                TIMING_TOTAL_ONLINE_REPEATS
            ),
        },

        "computational_time": timing_results,

        "DCR_definition": {
            "formula": (
                "DCR = original_entries / compressed_entries"
            ),
            "compressed_entries_for_CP": (
                "(NX + NY + NZ) * R + N * R "
                "+ lambda_entries + mean_entries"
            ),
            "lambda_counted": bool(
                DCR_COUNT_LAMBDAS
            ),
            "mean_field_counted": bool(
                count_mean_for_dcr
            ),
            "mlp_parameters_counted": bool(
                DCR_COUNT_MLP_PARAMS
            ),
            "note": (
                "DCR mainly measures CP/PARAFAC data-representation "
                "compression. Neural-network parameters are excluded by default."
            ),
        },

        "DCR_current_rank": (
            current_dcr_row.to_dict()
        ),

        "DCR_rank_sweep": [
            {
                "cp_rank_R": int(
                    row["cp_rank_R"]
                ),
                "DCR_train_cp": float(
                    row["DCR_train_cp"]
                ),
                "compression_percent_train_cp": float(
                    row[
                        "compression_percent_train_cp"
                    ]
                ),
                "DCR_all_projected_coeff": float(
                    row[
                        "DCR_all_projected_coeff"
                    ]
                ),
                "compression_percent_all_projected_coeff": float(
                    row[
                        "compression_percent_all_projected_coeff"
                    ]
                ),
                "DCR_all_with_mlp_model": float(
                    row[
                        "DCR_all_with_mlp_model"
                    ]
                ),
                "compression_percent_all_with_mlp_model": float(
                    row[
                        "compression_percent_all_with_mlp_model"
                    ]
                ),
                "mlp_param_entries": int(
                    row["mlp_param_entries"]
                ),
            }
            for _, row in cp_dcr_df.iterrows()
        ],

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
            "dcr_count_mean_field": bool(
                DCR_COUNT_MEAN_FIELD
            ),
            "dcr_count_lambdas": bool(
                DCR_COUNT_LAMBDAS
            ),
            "dcr_count_mlp_params": bool(
                DCR_COUNT_MLP_PARAMS
            ),
            "dcr_bytes_per_value": int(
                DCR_BYTES_PER_VALUE
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
            "energy_weighted_loss": (
                USE_ENERGY_WEIGHTS
            ),
            "pca_bottleneck": (
                USE_PCA_BOTTLENECK
            ),
            "pca_latent_q": (
                int(pca_model.n_components_)
                if pca_model is not None
                else None
            ),
        },

        "random_state": RANDOM_STATE,

        "test_indices_one_based": [
            int(index)
            for index in TEST_IDX_ONE_BASED
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
            "full3d_mae_each": mae_list,
            "full3d_rmse_each": rmse_list,
            "full3d_rRMSE_each": rrmse_list,
            "rRMSE_n_top10_points": (
                rrmse_n_results
            ),
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

        "cp_component_weights_preview": [
            float(value)
            for value in cp_weight_preview.tolist()
        ],

        "output_paths": {
            "fig_dir": os.path.abspath(
                FIG_DIR
            ),
            "cp_dcr_current_csv": os.path.abspath(
                os.path.join(
                    FIG_DIR,
                    "cp_dcr_current.csv",
                )
            ),
            "cp_dcr_rank_sweep_csv": (
                os.path.abspath(
                    os.path.join(
                        FIG_DIR,
                        "cp_dcr_rank_sweep.csv",
                    )
                )
            ),
            "cp_dcr_vs_rank_png": os.path.abspath(
                os.path.join(
                    FIG_DIR,
                    "cp_dcr_vs_rank.png",
                )
            ),
            "per_case_metrics_cp_csv": (
                os.path.abspath(
                    per_case_metrics_path
                )
            ),
            "computational_time_metrics_cp_csv": (
                timing_results["aggregate_csv"]
            ),
            "computational_time_per_case_cp_csv": (
                timing_results["per_case_csv"]
            ),
            "metrics_summary_cp_json": (
                os.path.abspath(
                    os.path.join(
                        FIG_DIR,
                        "metrics_summary_cp.json",
                    )
                )
            ),
        },

        "total_script_execution_time_seconds": float(
            total_script_time_seconds
        ),
    }

    summary_path = os.path.join(
        FIG_DIR,
        "metrics_summary_cp.json",
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

    # =================== Final Console Summary ===================
    print(
        "\n"
        + "=" * 20
        + " FINAL SUMMARY "
        + "=" * 20
    )

    print(
        f"Total script execution time: "
        f"{total_script_time_seconds:.2f} s"
    )
    print(
        f"Outputs in: "
        f"{os.path.abspath(FIG_DIR)}"
    )

    print("\n--- CP Configuration ---")
    print(f"CP rank: {CP_RANK}")
    print(
        f"CP iterations: {CP_N_ITER_MAX}, "
        f"tolerance: {CP_TOL}"
    )

    print("\n--- Core Computational-Time Metrics ---")
    print(
        f"Decomposition time: "
        f"{decomposition_time_seconds:.6f} s"
    )
    print(
        f"DNN training time: "
        f"{dnn_training_time_seconds:.6f} s"
    )
    print(
        "DNN inference time per case: "
        f"{timing_results['dnn_inference_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['dnn_inference_time_per_case']['std_ms']:.6f} ms"
    )
    print(
        "Field reconstruction time per case: "
        f"{timing_results['field_reconstruction_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['field_reconstruction_time_per_case']['std_ms']:.6f} ms"
    )
    print(
        "Total online prediction time per case: "
        f"{timing_results['total_online_prediction_time_per_case']['mean_ms']:.6f} "
        f"± "
        f"{timing_results['total_online_prediction_time_per_case']['std_ms']:.6f} ms"
    )

    print("\n--- DCR at Current Rank ---")
    print(
        f"DCR_train_cp: "
        f"{current_dcr_row['DCR_train_cp']:.6f}"
    )
    print(
        "Compression saving, training representation: "
        f"{current_dcr_row['compression_percent_train_cp']:.2f}%"
    )
    print(
        f"DCR_all_projected_coeff: "
        f"{current_dcr_row['DCR_all_projected_coeff']:.6f}"
    )
    print(
        "Compression saving, all projected coefficients: "
        f"{current_dcr_row['compression_percent_all_projected_coeff']:.2f}%"
    )

    print("\n--- Average Accuracy Metrics ---")
    print(
        f"Mean MAE  : "
        f"{np.mean(mae_list):.4f} °C"
    )
    print(
        f"Mean RMSE : "
        f"{np.mean(rmse_list):.4f} °C"
    )
    print(
        f"Mean rRMSE: "
        f"{np.mean(rrmse_list):.4f}"
    )

    print("\n--- Saved Files ---")
    print("per_case_metrics_cp.csv")
    print("metrics_summary_cp.json")
    print("computational_time_metrics_cp.csv")
    print("computational_time_per_case_cp.csv")
    print("cp_dcr_current.csv")
    print("cp_dcr_rank_sweep.csv")
    print("cp_dcr_vs_rank.png")
    print("error_metrics_compare_cp.png")
    print("=" * 55)


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
