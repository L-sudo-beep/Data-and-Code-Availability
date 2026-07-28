#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker (TRAIN-only) + MLP with DCR and computational-time benchmarking
----------------------------------------------------------------------
新增并分别统计以下计算时间：
1. Decomposition time
2. DNN training time
3. DNN inference time per case
4. Field reconstruction time per case
5. Total online prediction time per case

计时口径：
- Decomposition time:
  训练集平均场计算、训练集中心化、训练集 HOSVD、秩选择、核心张量构建，
  以及训练样本低维系数投影。
- DNN training time:
  神经网络优化与训练循环。
- DNN inference time:
  已完成标准化的单个 14 维输入经过 DNN 前向传播的时间。
- Field reconstruction time:
  预测输出反标准化、可选 PCA 逆变换、Tucker 三维场重构和平均场恢复。
- Total online prediction time:
  原始边界条件标准化、张量转换、DNN 前向传播、预测输出反标准化、
  可选 PCA 逆变换、Tucker 三维场重构和平均场恢复。

上述核心时间均不包括数据读取、误差计算、绘图和文件保存。

Author: (your name)
"""

import os
import sys
import math
import json
import time
import warnings
import platform
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


# ========================== User Config ==========================

PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"

NX, NY, NZ = 75, 51, 103
N_CASES = 89
Y_SLICE = 1.71

ENERGY_THRESH = 0.995
DCR_SWEEP_THRESHOLDS = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
DCR_COUNT_MEAN_FIELD = True
DCR_BYTES_PER_VALUE = 8

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

FIG_BASE_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved"
RUN_TAG = f"energy_{ENERGY_THRESH:.5f}".replace(".", "p")
FIG_DIR = os.path.join(FIG_BASE_DIR, RUN_TAG)
os.makedirs(FIG_DIR, exist_ok=True)

POD_ERROR_MAX_PATH = (
    r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"
)

# Timing settings
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
    """Synchronize CUDA before/after timing; no-op on CPU."""
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
    required = {"X (m)", "Y (m)", "Z (m)", "Temperature"}

    if not required.issubset(df.columns):
        raise ValueError(f"Snapshot {path} missing columns {required}")

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

    spline = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = spline(xi, zi)

    return xi, zi, Txz_hi


# ========================== DCR Utils ==========================

def rank_from_singular_values(S: np.ndarray, threshold: float) -> int:
    if S is None or len(S) == 0:
        return 1

    threshold = float(np.clip(threshold, 0.0, 1.0))
    energy = S ** 2
    total_energy = np.sum(energy)

    if total_energy <= 0:
        return 1

    cumulative_energy = np.cumsum(energy) / total_energy
    rank = int(np.searchsorted(cumulative_energy, threshold) + 1)

    return max(1, min(rank, len(S)))


def compute_tucker_dcr_counts(
        nx: int,
        ny: int,
        nz: int,
        n_cases: int,
        ranks: Tuple[int, int, int, int],
        count_mean_field: bool = True,
        bytes_per_value: int = 8,
) -> dict:
    rx, ry, rz, rn = ranks

    original_entries = int(nx * ny * nz * n_cases)
    core_entries = int(rx * ry * rz * rn)
    spatial_factor_entries = int(nx * rx + ny * ry + nz * rz)
    n_mode_entries = int(n_cases * rn)
    mean_entries = int(nx * ny * nz) if count_mean_field else 0

    compressed_entries = int(
        core_entries
        + spatial_factor_entries
        + n_mode_entries
        + mean_entries
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
        bytes_per_value: int = 8,
) -> pd.DataFrame:
    thresholds = sorted(set(float(x) for x in thresholds))
    rows = []

    for threshold in thresholds:
        rx = rank_from_singular_values(singvals["Sx"], threshold)
        ry = rank_from_singular_values(singvals["Sy"], threshold)
        rz = rank_from_singular_values(singvals["Sz"], threshold)
        rn = rank_from_singular_values(singvals["Sn"], threshold)

        ranks = (rx, ry, rz, rn)

        train_counts = compute_tucker_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_train,
            ranks=ranks,
            count_mean_field=count_mean_field,
            bytes_per_value=bytes_per_value,
        )

        all_counts = compute_tucker_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            ranks=ranks,
            count_mean_field=count_mean_field,
            bytes_per_value=bytes_per_value,
        )

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "energy_threshold": float(threshold),
            "rx": int(rx),
            "ry": int(ry),
            "rz": int(rz),
            "rn": int(rn),

            "train_original_entries": train_counts["original_entries"],
            "train_compressed_entries": train_counts["compressed_entries"],
            "train_core_entries": train_counts["core_entries"],
            "train_spatial_factor_entries": (
                train_counts["spatial_factor_entries"]
            ),
            "train_n_mode_entries": train_counts["n_mode_entries"],
            "train_mean_entries": train_counts["mean_entries"],
            "DCR_train_tucker": train_counts["DCR"],
            "compression_percent_train_tucker": (
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
            "all_core_entries": all_counts["core_entries"],
            "all_spatial_factor_entries": (
                all_counts["spatial_factor_entries"]
            ),
            "all_coeff_entries": all_counts["n_mode_entries"],
            "all_mean_entries": all_counts["mean_entries"],
            "DCR_all_projected_coeff": all_counts["DCR"],
            "compression_percent_all_projected_coeff": (
                all_counts["compression_percent"]
            ),
            "all_original_MB_float64": (
                all_counts["original_MB_float64"]
            ),
            "all_compressed_MB_float64": (
                all_counts["compressed_MB_float64"]
            ),
        })

    return pd.DataFrame(rows)


def save_dcr_outputs(
        dcr_sweep_df: pd.DataFrame,
        fig_dir: str,
        fig_base_dir: str,
        current_energy_threshold: float,
) -> pd.DataFrame:
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(fig_base_dir, exist_ok=True)

    dcr_sweep_df.to_csv(
        os.path.join(fig_dir, "dcr_sweep.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    current_df = dcr_sweep_df[
        np.isclose(
            dcr_sweep_df["energy_threshold"],
            current_energy_threshold,
        )
    ]

    if current_df.empty:
        current_df = dcr_sweep_df.iloc[[0]]

    current_df.to_csv(
        os.path.join(fig_dir, "dcr_current.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    comparison_path = os.path.join(
        fig_base_dir,
        "dcr_vs_energy_threshold.csv",
    )

    if os.path.exists(comparison_path):
        try:
            old_df = pd.read_csv(comparison_path)
            merged_df = pd.concat(
                [old_df, dcr_sweep_df],
                ignore_index=True,
            )
        except Exception:
            merged_df = dcr_sweep_df.copy()
    else:
        merged_df = dcr_sweep_df.copy()

    merged_df = merged_df.drop_duplicates(
        subset=["energy_threshold"],
        keep="last",
    )
    merged_df = merged_df.sort_values(
        "energy_threshold"
    ).reset_index(drop=True)

    merged_df.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        merged_df["energy_threshold"],
        merged_df["DCR_train_tucker"],
        "-o",
        label="DCR: Train Tucker representation",
    )
    plt.plot(
        merged_df["energy_threshold"],
        merged_df["DCR_all_projected_coeff"],
        "-s",
        label="DCR: All cases with projected coefficients",
    )
    plt.xlabel("Energy Threshold")
    plt.ylabel("Data Compression Ratio, DCR")
    plt.title("Tucker DCR vs Energy Threshold")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_base_dir, "dcr_vs_energy_threshold.png"),
        dpi=200,
    )
    plt.close()

    return merged_df


def print_dcr_report(current_dcr_row: pd.Series):
    print("\n=== DCR Report for Current ENERGY_THRESH ===")
    print(
        f"  Energy threshold = "
        f"{current_dcr_row['energy_threshold']:.6f}"
    )
    print(
        "  Selected ranks: "
        f"rx={int(current_dcr_row['rx'])}, "
        f"ry={int(current_dcr_row['ry'])}, "
        f"rz={int(current_dcr_row['rz'])}, "
        f"rn={int(current_dcr_row['rn'])}"
    )

    print("\n  [Train Tucker representation]")
    print(
        f"    Original entries   = "
        f"{int(current_dcr_row['train_original_entries'])}"
    )
    print(
        f"    Compressed entries = "
        f"{int(current_dcr_row['train_compressed_entries'])}"
    )
    print(
        f"    DCR                = "
        f"{current_dcr_row['DCR_train_tucker']:.6f}"
    )
    print(
        f"    Saving             = "
        f"{current_dcr_row['compression_percent_train_tucker']:.2f}%"
    )

    print("\n  [All cases with projected coefficients]")
    print(
        f"    Original entries   = "
        f"{int(current_dcr_row['all_original_entries'])}"
    )
    print(
        f"    Compressed entries = "
        f"{int(current_dcr_row['all_compressed_entries'])}"
    )
    print(
        f"    DCR                = "
        f"{current_dcr_row['DCR_all_projected_coeff']:.6f}"
    )
    print(
        f"    Saving             = "
        f"{current_dcr_row['compression_percent_all_projected_coeff']:.2f}%"
    )


# ========================== Tucker/HOSVD ==========================

def select_rank_by_energy_unfold(
        X: np.ndarray,
        threshold: float,
        max_rank: Optional[int] = None,
):
    max_possible_rank = min(X.shape[0], X.shape[1])
    n_components = (
        max_possible_rank
        if max_rank is None
        else min(max_possible_rank, max_rank)
    )

    U, S, VT = randomized_svd(
        X,
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    rank = rank_from_singular_values(S, threshold)
    rank = max(1, min(rank, n_components))

    return U[:, :rank], rank, S


def hosvd_per_mode_energy_train_tensor(
        T_train_centered: np.ndarray,
        threshold: float,
):
    """
    Perform HOSVD using the centered training tensor only.

    T_train_centered shape:
        (NX, NY, NZ, N_train)
    """
    Ux, rx, Sx = select_rank_by_energy_unfold(
        mode_n_unfold(T_train_centered, 0),
        threshold,
    )
    Uy, ry, Sy = select_rank_by_energy_unfold(
        mode_n_unfold(T_train_centered, 1),
        threshold,
    )
    Uz, rz, Sz = select_rank_by_energy_unfold(
        mode_n_unfold(T_train_centered, 2),
        threshold,
    )
    Un, rn, Sn = select_rank_by_energy_unfold(
        mode_n_unfold(T_train_centered, 3),
        threshold,
    )

    ranks = (rx, ry, rz, rn)

    G = T_train_centered.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)

    singular_values = {
        "Sx": Sx,
        "Sy": Sy,
        "Sz": Sz,
        "Sn": Sn,
    }

    return G, [Ux, Uy, Uz, Un], ranks, singular_values


def project_case_to_coeff(
        T_case: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        G: np.ndarray,
) -> np.ndarray:
    """
    Project one centered 3-D temperature field onto the train-only
    Tucker representation and solve for its N-mode coefficient vector.
    """
    B = n_mode_product(T_case.copy(), Ux.T, 0)
    B = n_mode_product(B, Uy.T, 1)
    B = n_mode_product(B, Uz.T, 2)

    b = B.reshape(-1)
    A = G.reshape(-1, G.shape[-1])

    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coeff


def reconstruct_tucker_field_from_coeff(
        coeff: np.ndarray,
        G_train: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        T_mean_train: Optional[np.ndarray],
        center_along_n: bool,
) -> np.ndarray:
    """
    Reconstruct one full 3-D temperature field from one Tucker
    coefficient vector.
    """
    coeff = np.asarray(coeff, dtype=np.float64).reshape(-1)

    reduced_field = np.tensordot(
        G_train,
        coeff,
        axes=([3], [0]),
    )

    field_x = n_mode_product(reduced_field, Ux, 0)
    field_xy = n_mode_product(field_x, Uy, 1)
    centered_field = n_mode_product(field_xy, Uz, 2)

    if center_along_n:
        return centered_field + T_mean_train[..., 0]

    return centered_field


# ========================== Regressors ==========================

class MLP(nn.Module):
    """Linear -> LayerNorm -> ReLU -> Dropout blocks + output Linear."""

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers,
            use_layernorm=USE_LAYERNORM,
            dropout=DROPOUT,
    ):
        super().__init__()

        def block(d_in, d_out):
            layers = [nn.Linear(d_in, d_out)]

            if use_layernorm:
                layers.append(nn.LayerNorm(d_out))

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
) -> nn.Module:
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
        weight = torch.tensor(
            weight_per_dim,
            dtype=torch.float32,
            device=device,
        )

    model.train()

    # Only the optimizer/training loop is included in DNN training time.
    # Model construction, tensor creation, and DataLoader construction are excluded.
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
            optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"[MLP] Epoch {epoch + 1}/{epochs} | "
                f"Loss={loss.item():.6f}"
            )

    sync_device(device)
    training_loop_time_seconds = (
        time.perf_counter() - training_loop_start
    )

    return model, training_loop_time_seconds


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

        base = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=RANDOM_STATE,
            alpha=0.0,
        )

        return MultiOutputRegressor(base, n_jobs=None)

    if name == "SVR":
        base = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA,
            cache_size=SVR_CACHE_MB,
        )

        return MultiOutputRegressor(base, n_jobs=None)

    if name == "MLP_TORCH":
        return None

    raise ValueError(
        "Unknown REGRESSOR. Supported: MLP_TORCH, SVR, GPR."
    )


# ========================== Output Decoding ==========================

def decode_scaled_regressor_output(
        scaled_output: np.ndarray,
        output_scaler=None,
        pca_model: Optional[PCA] = None,
) -> np.ndarray:
    """
    Convert scaled regressor output to the original Tucker coefficient space.

    Without PCA:
        inverse StandardScaler -> Tucker coefficients

    With PCA:
        inverse StandardScaler -> PCA scores -> inverse PCA ->
        Tucker coefficients
    """
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
        G_train: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        T_mean_train: Optional[np.ndarray],
        center_along_n: bool,
) -> np.ndarray:
    coefficients = decode_scaled_regressor_output(
        scaled_output,
        output_scaler=output_scaler,
        pca_model=pca_model,
    )

    return reconstruct_tucker_field_from_coeff(
        coeff=coefficients[0],
        G_train=G_train,
        Ux=Ux,
        Uy=Uy,
        Uz=Uz,
        T_mean_train=T_mean_train,
        center_along_n=center_along_n,
    )


# ========================== Timing Benchmarks ==========================

def benchmark_torch_inference_per_case(
        model: nn.Module,
        X_test_scaled: np.ndarray,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    DNN forward-propagation time only.

    Input scaling and tensor creation are excluded.
    """
    device = next(model.parameters()).device
    model.eval()
    per_case_times = []

    with torch.no_grad():
        for row in X_test_scaled:
            x_tensor = torch.as_tensor(
                row.reshape(1, -1),
                dtype=torch.float32,
                device=device,
            )

            for _ in range(warmup_runs):
                _ = model(x_tensor)

            sync_device(device)

            repeat_times = []

            for _ in range(repeats):
                sync_device(device)
                start = time.perf_counter()

                _ = model(x_tensor)

                sync_device(device)
                repeat_times.append(
                    time.perf_counter() - start
                )

            per_case_times.append(
                float(np.mean(repeat_times))
            )

    return np.asarray(per_case_times, dtype=np.float64)


def benchmark_sklearn_inference_per_case(
        model,
        X_test_scaled: np.ndarray,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    Fallback inference benchmark for SVR/GPR.
    """
    per_case_times = []

    for row in X_test_scaled:
        x_row = row.reshape(1, -1)

        for _ in range(warmup_runs):
            _ = model.predict(x_row)

        repeat_times = []

        for _ in range(repeats):
            start = time.perf_counter()
            _ = model.predict(x_row)
            repeat_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeat_times))
        )

    return np.asarray(per_case_times, dtype=np.float64)


def benchmark_field_reconstruction_per_case(
        predicted_scaled_outputs: np.ndarray,
        output_scaler,
        pca_model: Optional[PCA],
        G_train: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        T_mean_train: Optional[np.ndarray],
        center_along_n: bool,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    Time output inverse scaling + optional inverse PCA +
    Tucker reconstruction + mean restoration.
    """
    per_case_times = []

    for scaled_output in predicted_scaled_outputs:
        for _ in range(warmup_runs):
            _ = reconstruct_from_scaled_output(
                scaled_output=scaled_output,
                output_scaler=output_scaler,
                pca_model=pca_model,
                G_train=G_train,
                Ux=Ux,
                Uy=Uy,
                Uz=Uz,
                T_mean_train=T_mean_train,
                center_along_n=center_along_n,
            )

        repeat_times = []

        for _ in range(repeats):
            start = time.perf_counter()

            _ = reconstruct_from_scaled_output(
                scaled_output=scaled_output,
                output_scaler=output_scaler,
                pca_model=pca_model,
                G_train=G_train,
                Ux=Ux,
                Uy=Uy,
                Uz=Uz,
                T_mean_train=T_mean_train,
                center_along_n=center_along_n,
            )

            repeat_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeat_times))
        )

    return np.asarray(per_case_times, dtype=np.float64)


def predict_scaled_output_one_case(
        model,
        raw_condition: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
) -> np.ndarray:
    """
    Normalize one raw 14-D condition and run one regressor prediction.
    """
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
            scaled_output = model(condition_tensor)

        sync_device(device)
        return scaled_output.detach().cpu().numpy()

    return model.predict(scaled_condition)


def predict_full_field_online(
        model,
        raw_condition: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
        output_scaler,
        pca_model: Optional[PCA],
        G_train: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        T_mean_train: Optional[np.ndarray],
        center_along_n: bool,
) -> np.ndarray:
    """
    Complete online prediction for one unseen boundary-condition vector.
    """
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
        G_train=G_train,
        Ux=Ux,
        Uy=Uy,
        Uz=Uz,
        T_mean_train=T_mean_train,
        center_along_n=center_along_n,
    )


def benchmark_total_online_prediction_per_case(
        model,
        X_test_raw: np.ndarray,
        input_scaler,
        scale_inputs: bool,
        regressor_name: str,
        output_scaler,
        pca_model: Optional[PCA],
        G_train: np.ndarray,
        Ux: np.ndarray,
        Uy: np.ndarray,
        Uz: np.ndarray,
        T_mean_train: Optional[np.ndarray],
        center_along_n: bool,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
    """
    Complete online time:
    input scaling + tensor conversion + inference + output inverse scaling
    + optional inverse PCA + Tucker reconstruction + mean restoration.
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
                G_train=G_train,
                Ux=Ux,
                Uy=Uy,
                Uz=Uz,
                T_mean_train=T_mean_train,
                center_along_n=center_along_n,
            )

        repeat_times = []

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
                G_train=G_train,
                Ux=Ux,
                Uy=Uy,
                Uz=Uz,
                T_mean_train=T_mean_train,
                center_along_n=center_along_n,
            )

            sync_device(device)
            repeat_times.append(
                time.perf_counter() - start
            )

        per_case_times.append(
            float(np.mean(repeat_times))
        )

    return np.asarray(per_case_times, dtype=np.float64)


def summarize_latency_seconds(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)

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

    total_offline_time = float(
        decomposition_time_seconds + training_time_seconds
    )

    aggregate_df = pd.DataFrame([{
        "Method": "Tucker",
        "Regressor": regressor_name,
        "Device": device_name,
        "Decomposition time (s)": decomposition_time_seconds,
        "DNN training time (s)": training_time_seconds,
        "Total offline time (s)": total_offline_time,
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
        "computational_time_metrics.csv",
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
        "computational_time_per_case.csv",
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
        "total_offline_time_seconds": total_offline_time,
        "dnn_inference_time_per_case": inference_summary,
        "field_reconstruction_time_per_case": (
            reconstruction_summary
        ),
        "total_online_prediction_time_per_case": online_summary,
        "aggregate_csv": os.path.abspath(aggregate_path),
        "per_case_csv": os.path.abspath(per_case_path),
    }


# ========================== Main ==========================

def main():
    total_script_start = time.perf_counter()
    np.random.seed(RANDOM_STATE)

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

    # ============================================================
    # Decomposition Time
    # ============================================================
    # Included:
    # - training mean field
    # - training-set centering
    # - train-only HOSVD
    # - rank selection
    # - core tensor construction
    # - coefficient projection for all TRAINING cases
    #
    # Excluded:
    # - raw CSV loading
    # - DCR calculation and file writing
    # - test prediction
    # ============================================================
    print(
        "=== Tucker decomposition and "
        "training-coefficient projection ==="
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
        T_train_centered = T[..., idx_train].copy()

    (
        G_train,
        [Ux, Uy, Uz, Un_train],
        ranks,
        singvals,
    ) = hosvd_per_mode_energy_train_tensor(
        T_train_centered,
        ENERGY_THRESH,
    )

    rx, ry, rz, rn = ranks

    # Only training fields are projected to obtain DNN targets.
    # Test fields are never used to construct the reduced model.
    Y_train_raw = np.zeros(
        (len(idx_train), rn),
        dtype=np.float64,
    )

    for local_index in range(len(idx_train)):
        Y_train_raw[local_index, :] = project_case_to_coeff(
            T_case=T_train_centered[..., local_index],
            Ux=Ux,
            Uy=Uy,
            Uz=Uz,
            G=G_train,
        )

    decomposition_time_seconds = (
        time.perf_counter() - decomposition_start
    )

    print(
        f"Selected ranks: "
        f"rx={rx}, ry={ry}, rz={rz}, rn={rn}"
    )
    print(f"Core tensor shape: {G_train.shape}")
    print(
        f"Decomposition time: "
        f"{decomposition_time_seconds:.6f} s"
    )

    # =================== DCR Calculation ===================
    print(
        "=== Calculate DCR for current and "
        "swept energy thresholds ==="
    )

    dcr_thresholds = sorted(
        set(
            [float(ENERGY_THRESH)]
            + [
                float(value)
                for value in DCR_SWEEP_THRESHOLDS
            ]
        )
    )

    count_mean_for_dcr = bool(
        DCR_COUNT_MEAN_FIELD and CENTER_ALONG_N
    )

    dcr_sweep_df = build_dcr_sweep_table(
        singvals=singvals,
        thresholds=dcr_thresholds,
        nx=NX,
        ny=NY,
        nz=NZ,
        n_train=len(idx_train),
        n_all=N_CASES,
        count_mean_field=count_mean_for_dcr,
        bytes_per_value=DCR_BYTES_PER_VALUE,
    )

    dcr_history_df = save_dcr_outputs(
        dcr_sweep_df=dcr_sweep_df,
        fig_dir=FIG_DIR,
        fig_base_dir=FIG_BASE_DIR,
        current_energy_threshold=ENERGY_THRESH,
    )

    current_dcr_df = dcr_sweep_df[
        np.isclose(
            dcr_sweep_df["energy_threshold"],
            ENERGY_THRESH,
        )
    ]

    if current_dcr_df.empty:
        current_dcr_row = dcr_sweep_df.iloc[0]
    else:
        current_dcr_row = current_dcr_df.iloc[0]

    print_dcr_report(current_dcr_row)

    print("\nDCR sweep table:")
    print(
        dcr_sweep_df[
            [
                "energy_threshold",
                "rx",
                "ry",
                "rz",
                "rn",
                "DCR_train_tucker",
                "compression_percent_train_tucker",
                "DCR_all_projected_coeff",
                "compression_percent_all_projected_coeff",
            ]
        ].to_string(index=False)
    )

    # =================== Regression Data ===================
    X_train_raw = X_params[idx_train]
    X_test_raw = X_params[idx_test]

    if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
        core_energy = np.linalg.norm(
            G_train.reshape(-1, G_train.shape[-1]),
            axis=0,
        )
        energy_weights = np.clip(
            core_energy / (core_energy.max() + 1e-12),
            1e-3,
            None,
        )
    else:
        energy_weights = None

    # Variables shared by timing/reconstruction code
    input_scaler = None
    output_scaler = None
    pca_model = None
    fitted_model = None
    predicted_scaled_outputs = None

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

    # =================== Regression ===================
    if USE_PCA_BOTTLENECK:
        print(
            f"=== PCA bottleneck on Tucker "
            f"coefficients (q={PCA_LATENT_Q}) ==="
        )

        pca_model = PCA(
            n_components=PCA_LATENT_Q,
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
                output_scaler.transform(latent_train)
            )
        else:
            regressor_target_train = latent_train

        regressor_output_dim = PCA_LATENT_Q
        regression_weights = None

    else:
        if SCALE_OUTPUT_COEFF:
            output_scaler = StandardScaler().fit(
                Y_train_raw
            )
            regressor_target_train = (
                output_scaler.transform(Y_train_raw)
            )
        else:
            regressor_target_train = Y_train_raw

        regressor_output_dim = rn
        regression_weights = energy_weights

    print(
        f"=== Fit regressor: {REGRESSOR} | "
        f"output_dim={regressor_output_dim} ==="
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
        device_name = str(model_device)

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

    # One regular prediction for all test cases.
    # This result is used for accuracy evaluation and reconstruction timing.
    if REGRESSOR.upper() == "MLP_TORCH":
        predicted_scaled_outputs = predict_mlp_torch(
            fitted_model,
            X_test,
        )
    else:
        predicted_scaled_outputs = fitted_model.predict(
            X_test
        )

    predicted_coefficients = decode_scaled_regressor_output(
        scaled_output=predicted_scaled_outputs,
        output_scaler=output_scaler,
        pca_model=pca_model,
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
                warmup_runs=TIMING_INFERENCE_WARMUP_RUNS,
                repeats=TIMING_INFERENCE_REPEATS,
            )
        )
    else:
        inference_times_seconds = (
            benchmark_sklearn_inference_per_case(
                model=fitted_model,
                X_test_scaled=X_test,
                warmup_runs=TIMING_INFERENCE_WARMUP_RUNS,
                repeats=TIMING_INFERENCE_REPEATS,
            )
        )

    # ============================================================
    # Field Reconstruction Time
    # ============================================================
    print("=== Benchmark field reconstruction time ===")

    reconstruction_times_seconds = (
        benchmark_field_reconstruction_per_case(
            predicted_scaled_outputs=predicted_scaled_outputs,
            output_scaler=output_scaler,
            pca_model=pca_model,
            G_train=G_train,
            Ux=Ux,
            Uy=Uy,
            Uz=Uz,
            T_mean_train=T_mean_train,
            center_along_n=CENTER_ALONG_N,
            warmup_runs=(
                TIMING_RECONSTRUCTION_WARMUP_RUNS
            ),
            repeats=TIMING_RECONSTRUCTION_REPEATS,
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
            G_train=G_train,
            Ux=Ux,
            Uy=Uy,
            Uz=Uz,
            T_mean_train=T_mean_train,
            center_along_n=CENTER_ALONG_N,
            warmup_runs=(
                TIMING_TOTAL_ONLINE_WARMUP_RUNS
            ),
            repeats=TIMING_TOTAL_ONLINE_REPEATS,
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
    print("=== Reconstruct test 3-D fields ===")

    predicted_fields = []
    ground_truth_fields = []

    for test_position, case_index in enumerate(idx_test):
        predicted_field = reconstruct_tucker_field_from_coeff(
            coeff=predicted_coefficients[test_position],
            G_train=G_train,
            Ux=Ux,
            Uy=Uy,
            Uz=Uz,
            T_mean_train=T_mean_train,
            center_along_n=CENTER_ALONG_N,
        )

        predicted_fields.append(predicted_field)
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
        mean_temperature_field = T_mean_train.squeeze()
    else:
        mean_temperature_field = T[
            ..., idx_train
        ].mean(axis=3)

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

    top_unique_temperatures = unique_temperatures[
        -number_of_points:
    ]

    top_point_indices = []

    for temperature_value in reversed(
            top_unique_temperatures
    ):
        flat_index = np.where(
            mean_temperature_flat == temperature_value
        )[0][0]

        if flat_index not in top_point_indices:
            top_point_indices.append(
                int(flat_index)
            )

    print(
        "(Points selected based on highest unique "
        "average temperatures in the training set)"
    )

    rrmse_n_results = {}

    for flat_index in top_point_indices:
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

        rrmse_n_results[int(flat_index)] = float(
            relative_rmse_at_point
        )

        print(
            f"  Grid Point Index {flat_index:<7} "
            f"(Avg T ≈ {average_temperature:.1f}°C): "
            f"rRMSE_n = {relative_rmse_at_point:.4f}"
        )

    # =================== Save Accuracy Metrics ===================
    metrics_df = pd.DataFrame({
        "Case": [int(index) + 1 for index in idx_test],
        "MAE": mae_list,
        "RMSE": rmse_list,
        "rRMSE": rrmse_list,
    }).sort_values("Case")

    metrics_df.to_csv(
        os.path.join(
            FIG_DIR,
            "per_case_metrics.csv",
        ),
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
        "Tucker+MLP Per-case Error Metrics "
        "(TRAIN-only)"
    )
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIG_DIR,
            "error_metrics_compare.png",
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
            "[Info] POD global_error_max.npy not found; "
            "Tucker error maps use per-case vmax."
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
            f"Case {case_index + 1}",
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

        axis_pred = figure.add_subplot(1, 3, 2)
        contour_pred = axis_pred.contourf(
            X_grid,
            Z_grid,
            predicted_plot.T,
            levels=LEVELS,
            cmap="jet",
            vmin=contour_true.get_clim()[0],
            vmax=contour_true.get_clim()[1],
        )
        axis_pred.set_title(
            "Reconstructed Temperature"
        )
        axis_pred.set_xlabel("X (m)")
        axis_pred.set_ylabel("Z (m)")
        axis_pred.set_aspect(
            "equal",
            adjustable="box",
        )
        figure.colorbar(
            contour_pred,
            ax=axis_pred,
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
            f"yslice_case_{case_index + 1}.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(f"  Saved {output_path}")

    # =================== JSON Summary ===================
    def singular_value_energy_info(S):
        energy = S ** 2
        cumulative = np.cumsum(energy) / np.sum(energy)
        count = min(50, len(S))
        return cumulative[:count].tolist()

    current_dcr_info = current_dcr_row.to_dict()
    total_script_time_seconds = (
        time.perf_counter() - total_script_start
    )

    summary = {
        "model_type": "Tucker_HOSVD_MLP",
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

        "hardware": get_hardware_info(),
        "regressor_device": device_name,

        "timing_definition": {
            "decomposition_time": (
                "Training mean field, training-set centering, "
                "train-only HOSVD, rank selection, core tensor "
                "construction, and training coefficient projection."
            ),
            "dnn_training_time": (
                "Neural-network optimizer/training loop."
            ),
            "dnn_inference_time": (
                "DNN forward propagation only; the 14-D input "
                "is already standardized."
            ),
            "field_reconstruction_time": (
                "Prediction inverse standardization, optional "
                "inverse PCA, Tucker field reconstruction, and "
                "training-mean-field restoration."
            ),
            "total_online_prediction_time": (
                "Raw-input standardization, tensor conversion, "
                "DNN forward propagation, prediction inverse "
                "standardization, optional inverse PCA, Tucker "
                "field reconstruction, and mean-field restoration."
            ),
            "excluded_from_core_timing": (
                "CSV loading, DCR calculation, error metrics, "
                "plotting, and file writing."
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
            "compressed_entries_for_Tucker": (
                "rx*ry*rz*rn + NX*rx + NY*ry + "
                "NZ*rz + N*rn + mean_entries"
            ),
            "mean_field_counted": bool(
                count_mean_for_dcr
            ),
            "note": (
                "Neural-network parameters are not "
                "included in compressed_entries."
            ),
        },

        "DCR_current_energy_threshold": (
            current_dcr_info
        ),

        "DCR_sweep_thresholds": [
            {
                "energy_threshold": float(
                    row["energy_threshold"]
                ),
                "rx": int(row["rx"]),
                "ry": int(row["ry"]),
                "rz": int(row["rz"]),
                "rn": int(row["rn"]),
                "DCR_train_tucker": float(
                    row["DCR_train_tucker"]
                ),
                "compression_percent_train_tucker": float(
                    row[
                        "compression_percent_train_tucker"
                    ]
                ),
                "DCR_all_projected_coeff": float(
                    row["DCR_all_projected_coeff"]
                ),
                "compression_percent_all_projected_coeff": float(
                    row[
                        "compression_percent_all_projected_coeff"
                    ]
                ),
            }
            for _, row in dcr_sweep_df.iterrows()
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
                PCA_LATENT_Q
                if USE_PCA_BOTTLENECK
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

        "cumulative_energy_curves": {
            "mode_x": singular_value_energy_info(
                singvals["Sx"]
            ),
            "mode_y": singular_value_energy_info(
                singvals["Sy"]
            ),
            "mode_z": singular_value_energy_info(
                singvals["Sz"]
            ),
            "mode_n": singular_value_energy_info(
                singvals["Sn"]
            ),
        },

        "output_paths": {
            "run_output_dir": os.path.abspath(
                FIG_DIR
            ),
            "dcr_current_csv": os.path.abspath(
                os.path.join(
                    FIG_DIR,
                    "dcr_current.csv",
                )
            ),
            "dcr_sweep_csv": os.path.abspath(
                os.path.join(
                    FIG_DIR,
                    "dcr_sweep.csv",
                )
            ),
            "dcr_vs_energy_threshold_csv": (
                os.path.abspath(
                    os.path.join(
                        FIG_BASE_DIR,
                        "dcr_vs_energy_threshold.csv",
                    )
                )
            ),
            "dcr_vs_energy_threshold_png": (
                os.path.abspath(
                    os.path.join(
                        FIG_BASE_DIR,
                        "dcr_vs_energy_threshold.png",
                    )
                )
            ),
            "computational_time_metrics_csv": (
                timing_results["aggregate_csv"]
            ),
            "computational_time_per_case_csv": (
                timing_results["per_case_csv"]
            ),
        },

        "total_script_execution_time_seconds": float(
            total_script_time_seconds
        ),
    }

    with open(
        os.path.join(
            FIG_DIR,
            "metrics_summary.json",
        ),
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

    print("\n--- Tucker Configuration ---")
    print(f"Energy threshold: {ENERGY_THRESH}")
    print(
        f"Ranks: ({rx}, {ry}, {rz}, {rn})"
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

    print("\n--- DCR at Current Setting ---")
    print(
        f"DCR_train_tucker: "
        f"{current_dcr_row['DCR_train_tucker']:.6f}"
    )
    print(
        "Compression saving, training representation: "
        f"{current_dcr_row['compression_percent_train_tucker']:.2f}%"
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
    print("per_case_metrics.csv")
    print("metrics_summary.json")
    print("computational_time_metrics.csv")
    print("computational_time_per_case.csv")
    print("dcr_current.csv")
    print("dcr_sweep.csv")
    print("dcr_vs_energy_threshold.csv")
    print("dcr_vs_energy_threshold.png")
    print("error_metrics_compare.png")
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
