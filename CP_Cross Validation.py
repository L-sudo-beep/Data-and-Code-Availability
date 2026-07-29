#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import json
import time
import gc
import warnings
import platform
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
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


PARAMS_PATH = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
SNAPSHOT_DIR = r"C:\Users\Lenovo\Desktop\insert"

NX, NY, NZ = 75, 51, 103
N_CASES = 89
Y_SLICE = 1.26

CENTER_ALONG_N = True
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True

CP_RANK_LIST = [6, 8, 10, 12, 14, 16, 18, 20, 22]

CP_N_ITER_MAX = 2000
CP_TOL = 1e-7

FINAL_CP_N_STARTS = 1
FINAL_CP_INIT = "svd"


RUN_INTERNAL_CV = True
CV_N_SPLITS = 5
CV_SHUFFLE = True
CV_RANDOM_STATE = 2026
CV_SELECTION_RULE = "one_se"
CV_MLP_EPOCHS = 2000

CV_CP_N_STARTS = 1
CV_CP_INIT = "svd"

FORCE_CP_RANK = None
RUN_FINAL_TEST_RANK_SWEEP = False

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
SAVE_FINAL_TEST_SLICE_MAPS = True

FIG_ROOT_DIR = (
    r"C:\Users\Lenovo\Desktop\TensorPOD89"
    r"\figures_improved_CP_rank_cv"
)
os.makedirs(FIG_ROOT_DIR, exist_ok=True)

CV_OUTPUT_DIR = os.path.join(FIG_ROOT_DIR, "cross_validation")
FINAL_TEST_OUTPUT_DIR = os.path.join(FIG_ROOT_DIR, "final_test")

os.makedirs(CV_OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_TEST_OUTPUT_DIR, exist_ok=True)

POD_ERROR_MAX_PATH = (
    r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1"
    r"\global_error_max.npy"
)


TIMING_INFERENCE_WARMUP_RUNS = 5
TIMING_INFERENCE_REPEATS = 200

TIMING_RECONSTRUCTION_WARMUP_RUNS = 1
TIMING_RECONSTRUCTION_REPEATS = 5

TIMING_TOTAL_ONLINE_WARMUP_RUNS = 1
TIMING_TOTAL_ONLINE_REPEATS = 5


np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def reset_random_seed(seed: int = RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_device(device: Optional[torch.device] = None):
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
        "tensorly_version": (
            getattr(tl, "__version__", None) if TL_OK else None
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None

    return value


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

    required_columns = {
        "X (m)",
        "Y (m)",
        "Z (m)",
        "Temperature",
    }

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


def load_all_data():
    print("=== Load parameters ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(dtype=np.float64)

    print("=== Read first snapshot and build grid ===")
    first_snapshot = read_one_snapshot_csv(
        os.path.join(SNAPSHOT_DIR, "1.csv")
    )

    xs, ys, zs = build_grid_from_df(first_snapshot)

    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))

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

        snapshot_df = read_one_snapshot_csv(snapshot_path)

        T[..., case_number - 1] = df_to_grid_values(
            snapshot_df,
            xs,
            ys,
            zs,
        )

        if case_number % 20 == 0 or case_number == N_CASES:
            print(f"  Loaded {case_number}/{N_CASES}")

    return X_params, T, xs, ys, zs, y_slice_index


def get_fixed_train_test_indices():
    print("=== Fixed development/test split ===")

    test_idx_one_based = [
        8, 9, 20, 21, 58, 68, 72, 76, 84
    ]

    idx_test = np.asarray(
        [index - 1 for index in test_idx_one_based],
        dtype=int,
    )

    idx_all = np.arange(N_CASES)

    idx_train = np.setdiff1d(
        idx_all,
        idx_test,
        assume_unique=False,
    )

    print(f"  Independent test cases (1-based): {test_idx_one_based}")
    print(f"  Independent test cases (0-based): {idx_test.tolist()}")
    print(
        f"  Development/train size = {len(idx_train)}, "
        f"independent test size = {len(idx_test)}"
    )

    return test_idx_one_based, idx_train, idx_test


def load_shared_error_max():
    if os.path.exists(POD_ERROR_MAX_PATH):
        try:
            shared_error_max = float(np.load(POD_ERROR_MAX_PATH))

            if (
                not np.isfinite(shared_error_max)
                or shared_error_max <= 0
            ):
                shared_error_max = None
                print(
                    "[Warn] Loaded POD error max is non-positive; "
                    "use per-case vmax."
                )
            else:
                print(
                    "[Info] Unified error-map vmax from POD: "
                    f"{shared_error_max:.4f}"
                )

        except Exception:
            shared_error_max = None
            print("[Warn] Failed to load POD global_error_max.npy.")
    else:
        shared_error_max = None
        print(
            "[Info] POD global_error_max.npy was not found; "
            "CP error maps use per-case vmax."
        )

    return shared_error_max


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    return float(
        np.linalg.norm(y_true - y_pred)
        / (np.linalg.norm(y_true) + 1e-12)
    )


def calculate_field_metrics(
        true_field: np.ndarray,
        predicted_field: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(true_field, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(predicted_field, dtype=np.float64).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = relative_rmse(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "rRMSE": float(rrmse),
    }


def calculate_rrmse_n_hot_points(
        T_mean_train: Optional[np.ndarray],
        T: np.ndarray,
        idx_train: np.ndarray,
        preds: np.ndarray,
        gts: np.ndarray,
        number_of_points: int = 10,
) -> pd.DataFrame:

    if T_mean_train is not None:
        mean_temperature_field = T_mean_train.squeeze()
    else:
        mean_temperature_field = T[..., idx_train].mean(axis=3)

    mean_temperature_flat = mean_temperature_field.ravel()
    unique_temperatures = np.unique(mean_temperature_flat)

    n_points = min(
        int(number_of_points),
        len(unique_temperatures),
    )

    hottest_unique_temperatures = unique_temperatures[-n_points:]
    hottest_point_indices = []

    for temperature_value in reversed(hottest_unique_temperatures):
        flat_index = int(
            np.where(mean_temperature_flat == temperature_value)[0][0]
        )

        if flat_index not in hottest_point_indices:
            hottest_point_indices.append(flat_index)

    rows = []

    for order, flat_index in enumerate(hottest_point_indices, start=1):
        ix, iy, iz = np.unravel_index(
            flat_index,
            (NX, NY, NZ),
        )

        true_series = gts[ix, iy, iz, :]
        predicted_series = preds[ix, iy, iz, :]

        point_rrmse = relative_rmse(
            y_true=true_series,
            y_pred=predicted_series,
        )

        rows.append({
            "hot_point_order": int(order),
            "flat_index": int(flat_index),
            "ix": int(ix),
            "iy": int(iy),
            "iz": int(iz),
            "training_mean_temperature": float(
                mean_temperature_field[ix, iy, iz]
            ),
            "rRMSE_n": float(point_rrmse),
        })

    return pd.DataFrame(rows)


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
            "train_spatial_factor_entries": train_counts["spatial_factor_entries"],
            "train_case_factor_entries": train_counts["case_factor_entries"],
            "train_lambda_entries": train_counts["lambda_entries"],
            "train_mean_entries": train_counts["mean_entries"],
            "DCR_train_cp": train_counts["DCR"],
            "compression_percent_train_cp": train_counts["compression_percent"],
            "train_original_MB_float64": train_counts["original_MB_float64"],
            "train_compressed_MB_float64": train_counts["compressed_MB_float64"],

            "all_original_entries": all_counts["original_entries"],
            "all_compressed_entries": all_counts["compressed_entries"],
            "all_spatial_factor_entries": all_counts["spatial_factor_entries"],
            "all_case_coeff_entries": all_counts["case_factor_entries"],
            "all_lambda_entries": all_counts["lambda_entries"],
            "all_mean_entries": all_counts["mean_entries"],
            "DCR_all_projected_coeff": all_counts["DCR"],
            "compression_percent_all_projected_coeff": (
                all_counts["compression_percent"]
            ),
            "all_original_MB_float64": all_counts["original_MB_float64"],
            "all_compressed_MB_float64": all_counts["compressed_MB_float64"],

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


def save_global_cp_dcr_outputs(
        dcr_df: pd.DataFrame,
        fig_root_dir: str,
):
    os.makedirs(fig_root_dir, exist_ok=True)

    sweep_path = os.path.join(
        fig_root_dir,
        "cp_dcr_rank_sweep.csv",
    )

    dcr_df.to_csv(
        sweep_path,
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
        os.path.join(fig_root_dir, "cp_dcr_vs_rank.png"),
        dpi=200,
    )

    plt.close()

    return sweep_path


def save_current_cp_dcr_output(
        current_row: pd.Series,
        fig_dir: str,
):
    current_path = os.path.join(
        fig_dir,
        "cp_dcr_current.csv",
    )

    pd.DataFrame([current_row]).to_csv(
        current_path,
        index=False,
        encoding="utf-8-sig",
    )

    return current_path


def print_cp_dcr_report(current_row: pd.Series):
    print("\n=== CP DCR Report for Current CP Rank ===")
    print(f"  Current CP rank R = {int(current_row['cp_rank_R'])}")

    print("\n  [Train CP representation]")
    print(
        f"    Original entries   = "
        f"{int(current_row['train_original_entries'])}"
    )
    print(
        f"    Compressed entries = "
        f"{int(current_row['train_compressed_entries'])}"
    )
    print(f"    DCR                = {current_row['DCR_train_cp']:.6f}")
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
    reset_random_seed(seed)

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


def cp_fit_train_tensor(
        T_train_centered: np.ndarray,
        rank_r: int,
        n_iter_max: int = 2000,
        tol: float = 1e-7,
        random_state: int = 42,
        init_method: str = "svd",
):
    if not TL_OK:
        raise RuntimeError(
            "TensorLy is required. Install it with: pip install tensorly"
        )

    tl.set_backend("numpy")

    cp_tensor = parafac(
        T_train_centered,
        rank=rank_r,
        init=init_method,
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


def project_tensor_cases_to_cp_coeff(
        centered_tensor: np.ndarray,
        projection_matrix: np.ndarray,
) -> Tuple[np.ndarray, float]:
    n_cases = centered_tensor.shape[-1]

    field_matrix = centered_tensor.reshape(
        -1,
        n_cases,
    )

    coefficient_matrix, residuals, _, _ = np.linalg.lstsq(
        projection_matrix,
        field_matrix,
        rcond=None,
    )

    if residuals.size == n_cases:
        residual_sum_squares = float(np.sum(residuals))
    else:
        residual_sum_squares = 0.0

        for case_index in range(n_cases):
            difference = (
                field_matrix[:, case_index]
                - projection_matrix @ coefficient_matrix[:, case_index]
            )
            residual_sum_squares += float(difference @ difference)

    denominator = float(np.sum(field_matrix ** 2))

    projection_rrmse = math.sqrt(
        residual_sum_squares / (denominator + 1e-12)
    )

    return coefficient_matrix.T, float(projection_rrmse)


def fit_cp_basis_and_project_train(
        T_train_centered: np.ndarray,
        rank_r: int,
        n_starts: int,
        init_method: str,
        n_iter_max: int,
        tol: float,
        base_seed: int,
) -> Dict[str, Any]:
    n_starts = max(1, int(n_starts))
    start_records = []
    best_result = None

    for start_id in range(n_starts):
        seed = int(base_seed + start_id)

        start_time = time.perf_counter()

        (
            lambdas,
            Ax,
            Ay,
            Az,
            An_train,
        ) = cp_fit_train_tensor(
            T_train_centered=T_train_centered,
            rank_r=rank_r,
            n_iter_max=n_iter_max,
            tol=tol,
            random_state=seed,
            init_method=init_method,
        )

        projection_matrix = cp_build_projection_matrix(
            Ax=Ax,
            Ay=Ay,
            Az=Az,
            lambdas=lambdas,
        )

        Y_train_raw, projection_rrmse = (
            project_tensor_cases_to_cp_coeff(
                centered_tensor=T_train_centered,
                projection_matrix=projection_matrix,
            )
        )

        gram_matrix = projection_matrix.T @ projection_matrix
        gram_condition_number = float(
            math.sqrt(max(np.linalg.cond(gram_matrix), 0.0))
        )

        elapsed = time.perf_counter() - start_time

        start_record = {
            "start_id": int(start_id),
            "seed": int(seed),
            "init_method": init_method,
            "train_projection_rRMSE": float(projection_rrmse),
            "projection_condition_number": gram_condition_number,
            "elapsed_seconds": float(elapsed),
        }

        start_records.append(start_record)

        candidate = {
            "lambdas": lambdas,
            "Ax": Ax,
            "Ay": Ay,
            "Az": Az,
            "An_train": An_train,
            "projection_matrix": projection_matrix,
            "Y_train_raw": Y_train_raw,
            "train_projection_rRMSE": float(projection_rrmse),
            "projection_condition_number": gram_condition_number,
            "selected_start_id": int(start_id),
            "selected_seed": int(seed),
        }

        if (
            best_result is None
            or projection_rrmse
            < best_result["train_projection_rRMSE"]
        ):
            best_result = candidate

    best_result["start_records"] = start_records
    return best_result


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


def fit_regression_pipeline(
        X_train_raw: np.ndarray,
        Y_train_raw: np.ndarray,
        energy_weights: Optional[np.ndarray],
        epochs: int,
        seed: int,
) -> Dict[str, Any]:
    input_scaler = None
    output_scaler = None
    pca_model = None

    if SCALE_INPUTS:
        input_scaler = StandardScaler().fit(X_train_raw)
        X_train = input_scaler.transform(X_train_raw)
    else:
        X_train = X_train_raw.copy()

    if USE_PCA_BOTTLENECK:
        actual_pca_components = min(
            PCA_LATENT_Q,
            Y_train_raw.shape[0],
            Y_train_raw.shape[1],
        )

        pca_model = PCA(
            n_components=actual_pca_components,
            random_state=seed,
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

        regressor_output_dim = Y_train_raw.shape[1]
        regression_weights = energy_weights

    if REGRESSOR.upper() == "MLP_TORCH":
        model, training_time_seconds = fit_mlp_torch(
            X_train=X_train,
            Y_train=regressor_target_train,
            weight_per_dim=regression_weights,
            lr=LEARNING_RATE,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            hidden_layers=HIDDEN_LAYERS,
            seed=seed,
            use_layernorm=USE_LAYERNORM,
            dropout=DROPOUT,
            weight_decay=WEIGHT_DECAY,
        )

        model_device = next(model.parameters()).device

        if model_device.type == "cuda":
            device_name = (
                f"CUDA: {torch.cuda.get_device_name(model_device)}"
            )
        else:
            device_name = "CPU"

    else:
        model = choose_regressor(
            REGRESSOR,
            regressor_output_dim,
        )

        training_start = time.perf_counter()

        model.fit(
            X_train,
            regressor_target_train,
        )

        training_time_seconds = (
            time.perf_counter() - training_start
        )

        device_name = "CPU"

    return {
        "model": model,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "pca_model": pca_model,
        "training_time_seconds": float(training_time_seconds),
        "device_name": device_name,
        "regressor_output_dim": int(regressor_output_dim),
    }


def transform_regression_inputs(
        X_raw: np.ndarray,
        input_scaler,
) -> np.ndarray:
    if SCALE_INPUTS:
        return input_scaler.transform(X_raw)

    return np.asarray(X_raw, dtype=np.float64).copy()


def predict_scaled_regressor_outputs(
        model,
        X_scaled: np.ndarray,
) -> np.ndarray:
    if REGRESSOR.upper() == "MLP_TORCH":
        return predict_mlp_torch(model, X_scaled)

    return model.predict(X_scaled)


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


def select_cp_rank_from_cv_summary(
        summary_df: pd.DataFrame,
        selection_rule: str,
) -> Dict[str, Any]:
    summary_sorted = summary_df.sort_values("cp_rank_R").copy()

    best_row_index = summary_sorted["cv_mean_rRMSE"].idxmin()
    best_row = summary_sorted.loc[best_row_index]

    min_error_rank = int(best_row["cp_rank_R"])
    min_error = float(best_row["cv_mean_rRMSE"])
    min_error_se = float(best_row["cv_SE_rRMSE"])

    one_se_threshold = min_error + min_error_se

    selection_rule = selection_rule.lower().strip()

    if selection_rule == "min_error":
        selected_rank = min_error_rank

    elif selection_rule == "one_se":
        eligible = summary_sorted[
            summary_sorted["cv_mean_rRMSE"] <= one_se_threshold
        ]

        if eligible.empty:
            selected_rank = min_error_rank
        else:
            selected_rank = int(eligible["cp_rank_R"].min())

    else:
        raise ValueError(
            "CV_SELECTION_RULE must be 'one_se' or 'min_error'."
        )

    return {
        "selection_rule": selection_rule,
        "min_error_rank": min_error_rank,
        "min_error_cv_mean_rRMSE": min_error,
        "min_error_cv_SE_rRMSE": min_error_se,
        "one_se_threshold": float(one_se_threshold),
        "selected_rank": int(selected_rank),
    }


def save_cv_outputs(
        fold_df: pd.DataFrame,
        case_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        selection: Dict[str, Any],
        output_dir: str,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    fold_path = os.path.join(
        output_dir,
        "cp_rank_cv_fold_metrics.csv",
    )

    case_path = os.path.join(
        output_dir,
        "cp_rank_cv_case_metrics.csv",
    )

    summary_path = os.path.join(
        output_dir,
        "cp_rank_cv_summary.csv",
    )

    selection_path = os.path.join(
        output_dir,
        "cp_rank_cv_selection.json",
    )

    fold_df.to_csv(
        fold_path,
        index=False,
        encoding="utf-8-sig",
    )

    case_df.to_csv(
        case_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    with open(selection_path, "w", encoding="utf-8") as file:
        json.dump(
            to_jsonable(selection),
            file,
            indent=2,
            ensure_ascii=False,
        )

    # Mean ± SE curve
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        summary_df["cp_rank_R"],
        summary_df["cv_mean_rRMSE"],
        yerr=summary_df["cv_SE_rRMSE"],
        fmt="-o",
        capsize=5,
        label="K-fold mean rRMSE ± SE",
    )

    plt.axhline(
        selection["one_se_threshold"],
        linestyle="--",
        label="One-SE threshold",
    )

    plt.axvline(
        selection["min_error_rank"],
        linestyle=":",
        label=f"Minimum-error rank = {selection['min_error_rank']}",
    )

    plt.axvline(
        selection["selected_rank"],
        linestyle="-.",
        label=f"Selected rank = {selection['selected_rank']}",
    )

    plt.xlabel("CP Rank R")
    plt.ylabel("Cross-validated rRMSE")
    plt.title("Internal Cross-Validation for CP Rank Selection")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()

    curve_path = os.path.join(
        output_dir,
        "cp_rank_cv_curve.png",
    )

    plt.savefig(curve_path, dpi=220)
    plt.close()

    # Pooled out-of-fold case-error distribution
    plt.figure(figsize=(11, 6))

    rank_order = sorted(case_df["cp_rank_R"].unique())
    box_data = [
        case_df.loc[
            case_df["cp_rank_R"] == rank,
            "rRMSE",
        ].to_numpy()
        for rank in rank_order
    ]

    plt.boxplot(
        box_data,
        labels=[str(rank) for rank in rank_order],
        showmeans=True,
    )

    plt.axvline(
        rank_order.index(selection["selected_rank"]) + 1,
        linestyle="--",
        label=f"Selected rank = {selection['selected_rank']}",
    )

    plt.xlabel("CP Rank R")
    plt.ylabel("Out-of-fold case rRMSE")
    plt.title("Out-of-Fold rRMSE Distribution vs CP Rank")
    plt.grid(True, axis="y", linestyle="--")
    plt.legend()
    plt.tight_layout()

    boxplot_path = os.path.join(
        output_dir,
        "cp_rank_cv_boxplot.png",
    )

    plt.savefig(boxplot_path, dpi=220)
    plt.close()

    return {
        "fold_metrics_csv": os.path.abspath(fold_path),
        "case_metrics_csv": os.path.abspath(case_path),
        "summary_csv": os.path.abspath(summary_path),
        "selection_json": os.path.abspath(selection_path),
        "curve_png": os.path.abspath(curve_path),
        "boxplot_png": os.path.abspath(boxplot_path),
    }


def run_internal_cp_rank_cross_validation(
        X_params: np.ndarray,
        T: np.ndarray,
        idx_development: np.ndarray,
        rank_list: List[int],
        output_dir: str,
) -> Dict[str, Any]:
    if CV_N_SPLITS < 2:
        raise ValueError("CV_N_SPLITS must be at least 2.")

    if CV_N_SPLITS > len(idx_development):
        raise ValueError(
            "CV_N_SPLITS cannot exceed the number of development cases."
        )

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("INTERNAL CROSS-VALIDATION FOR CP RANK SELECTION")
    print("=" * 80)

    print(
        f"Development cases: {len(idx_development)} | "
        f"CV folds: {CV_N_SPLITS} | "
        f"Candidate ranks: {rank_list}"
    )

    local_case_ids = np.arange(len(idx_development))

    kfold = KFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=(
            CV_RANDOM_STATE if CV_SHUFFLE else None
        ),
    )

    splits = list(kfold.split(local_case_ids))

    fold_records = []
    case_records = []

    cv_total_start = time.perf_counter()

    for rank_position, cp_rank in enumerate(rank_list, start=1):
        print("\n" + "#" * 72)
        print(
            f"CV rank {cp_rank} "
            f"({rank_position}/{len(rank_list)})"
        )
        print("#" * 72)

        for fold_index, (fold_train_local, fold_val_local) in enumerate(
                splits,
                start=1,
        ):
            fold_start = time.perf_counter()

            fold_train_global = idx_development[fold_train_local]
            fold_val_global = idx_development[fold_val_local]

            print(
                f"\n[R={cp_rank}] Fold {fold_index}/{CV_N_SPLITS} | "
                f"train={len(fold_train_global)}, "
                f"validation={len(fold_val_global)}"
            )

            if CENTER_ALONG_N:
                T_mean_fold = T[..., fold_train_global].mean(
                    axis=3,
                    keepdims=True,
                )

                T_fold_train_centered = (
                    T[..., fold_train_global] - T_mean_fold
                )
            else:
                T_mean_fold = None
                T_fold_train_centered = (
                    T[..., fold_train_global].copy()
                )

            cp_seed = (
                CV_RANDOM_STATE
                + cp_rank * 1000
                + fold_index * 100
            )

            decomposition_start = time.perf_counter()

            cp_result = fit_cp_basis_and_project_train(
                T_train_centered=T_fold_train_centered,
                rank_r=cp_rank,
                n_starts=CV_CP_N_STARTS,
                init_method=CV_CP_INIT,
                n_iter_max=CP_N_ITER_MAX,
                tol=CP_TOL,
                base_seed=cp_seed,
            )

            decomposition_time = (
                time.perf_counter() - decomposition_start
            )

            lambdas = cp_result["lambdas"]
            Ax = cp_result["Ax"]
            Ay = cp_result["Ay"]
            Az = cp_result["Az"]
            projection_matrix = cp_result["projection_matrix"]
            Y_fold_train_raw = cp_result["Y_train_raw"]

            if USE_ENERGY_WEIGHTS and not USE_PCA_BOTTLENECK:
                energy_weights = cp_component_energy_weights(
                    lambdas=lambdas,
                    Ax=Ax,
                    Ay=Ay,
                    Az=Az,
                )
            else:
                energy_weights = None

            regression_seed = (
                RANDOM_STATE
                + cp_rank * 100
                + fold_index
            )

            regression_pipeline = fit_regression_pipeline(
                X_train_raw=X_params[fold_train_global],
                Y_train_raw=Y_fold_train_raw,
                energy_weights=energy_weights,
                epochs=CV_MLP_EPOCHS,
                seed=regression_seed,
            )

            model = regression_pipeline["model"]
            input_scaler = regression_pipeline["input_scaler"]
            output_scaler = regression_pipeline["output_scaler"]
            pca_model = regression_pipeline["pca_model"]

            X_val_scaled = transform_regression_inputs(
                X_raw=X_params[fold_val_global],
                input_scaler=input_scaler,
            )

            predicted_scaled_outputs = (
                predict_scaled_regressor_outputs(
                    model=model,
                    X_scaled=X_val_scaled,
                )
            )

            predicted_coefficients = decode_scaled_regressor_output(
                scaled_output=predicted_scaled_outputs,
                output_scaler=output_scaler,
                pca_model=pca_model,
            )

            fold_rrmse_values = []
            fold_mae_values = []
            fold_rmse_values = []

            for local_val_position, global_case_index in enumerate(
                    fold_val_global
            ):
                predicted_field = cp_reconstruct_from_coeff(
                    coefficient=(
                        predicted_coefficients[local_val_position]
                    ),
                    projection_matrix=projection_matrix,
                    field_shape=(NX, NY, NZ),
                    T_mean_train=T_mean_fold,
                )

                true_field = T[..., global_case_index]

                metrics = calculate_field_metrics(
                    true_field=true_field,
                    predicted_field=predicted_field,
                )

                fold_mae_values.append(metrics["MAE"])
                fold_rmse_values.append(metrics["RMSE"])
                fold_rrmse_values.append(metrics["rRMSE"])

                case_records.append({
                    "cp_rank_R": int(cp_rank),
                    "fold": int(fold_index),
                    "case_index_zero_based": int(global_case_index),
                    "case_index_one_based": int(global_case_index + 1),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "rRMSE": metrics["rRMSE"],
                })

            fold_elapsed = time.perf_counter() - fold_start

            fold_record = {
                "cp_rank_R": int(cp_rank),
                "fold": int(fold_index),
                "n_train_cases": int(len(fold_train_global)),
                "n_validation_cases": int(len(fold_val_global)),
                "train_indices_one_based": json.dumps(
                    [int(i + 1) for i in fold_train_global]
                ),
                "validation_indices_one_based": json.dumps(
                    [int(i + 1) for i in fold_val_global]
                ),
                "mean_MAE": float(np.mean(fold_mae_values)),
                "std_MAE": float(np.std(fold_mae_values, ddof=1)),
                "mean_RMSE": float(np.mean(fold_rmse_values)),
                "std_RMSE": float(np.std(fold_rmse_values, ddof=1)),
                "mean_rRMSE": float(np.mean(fold_rrmse_values)),
                "std_rRMSE": float(np.std(fold_rrmse_values, ddof=1)),
                "train_projection_rRMSE": float(
                    cp_result["train_projection_rRMSE"]
                ),
                "projection_condition_number": float(
                    cp_result["projection_condition_number"]
                ),
                "selected_cp_start_id": int(
                    cp_result["selected_start_id"]
                ),
                "selected_cp_seed": int(
                    cp_result["selected_seed"]
                ),
                "cp_start_records": json.dumps(
                    to_jsonable(cp_result["start_records"]),
                    ensure_ascii=False,
                ),
                "decomposition_and_projection_time_seconds": float(
                    decomposition_time
                ),
                "regressor_training_time_seconds": float(
                    regression_pipeline["training_time_seconds"]
                ),
                "fold_total_time_seconds": float(fold_elapsed),
            }

            fold_records.append(fold_record)

            print(
                f"[R={cp_rank}, fold={fold_index}] "
                f"mean rRMSE={fold_record['mean_rRMSE']:.6f}, "
                f"train projection rRMSE="
                f"{fold_record['train_projection_rRMSE']:.6f}"
            )

            pd.DataFrame(fold_records).to_csv(
                os.path.join(
                    output_dir,
                    "cp_rank_cv_fold_metrics_progress.csv",
                ),
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(case_records).to_csv(
                os.path.join(
                    output_dir,
                    "cp_rank_cv_case_metrics_progress.csv",
                ),
                index=False,
                encoding="utf-8-sig",
            )

            del T_fold_train_centered
            del cp_result
            del lambdas
            del Ax
            del Ay
            del Az
            del projection_matrix
            del Y_fold_train_raw
            del predicted_scaled_outputs
            del predicted_coefficients
            del model
            del regression_pipeline

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_records)
    case_df = pd.DataFrame(case_records)

    summary_rows = []
    sorted_ranks = sorted(set(int(r) for r in rank_list))
    previous_mean = None

    for cp_rank in sorted_ranks:
        rank_fold = fold_df[
            fold_df["cp_rank_R"] == cp_rank
        ]

        rank_case = case_df[
            case_df["cp_rank_R"] == cp_rank
        ]

        fold_means = rank_fold["mean_rRMSE"].to_numpy(
            dtype=np.float64
        )

        cv_mean = float(np.mean(fold_means))
        cv_std = float(np.std(fold_means, ddof=1))
        cv_se = float(cv_std / math.sqrt(len(fold_means)))

        pooled_mean = float(rank_case["rRMSE"].mean())
        pooled_std = float(rank_case["rRMSE"].std(ddof=1))
        pooled_median = float(rank_case["rRMSE"].median())
        pooled_max = float(rank_case["rRMSE"].max())

        if previous_mean is None:
            improvement = np.nan
        else:
            improvement = float(
                (previous_mean - cv_mean)
                / (previous_mean + 1e-12)
                * 100.0
            )

        summary_rows.append({
            "cp_rank_R": int(cp_rank),
            "cv_mean_rRMSE": cv_mean,
            "cv_std_rRMSE": cv_std,
            "cv_SE_rRMSE": cv_se,
            "pooled_oof_mean_rRMSE": pooled_mean,
            "pooled_oof_std_rRMSE": pooled_std,
            "pooled_oof_median_rRMSE": pooled_median,
            "pooled_oof_max_rRMSE": pooled_max,
            "relative_improvement_from_previous_rank_percent": improvement,
            "mean_train_projection_rRMSE": float(
                rank_fold["train_projection_rRMSE"].mean()
            ),
            "mean_projection_condition_number": float(
                rank_fold["projection_condition_number"].mean()
            ),
            "mean_decomposition_time_seconds": float(
                rank_fold[
                    "decomposition_and_projection_time_seconds"
                ].mean()
            ),
            "mean_regressor_training_time_seconds": float(
                rank_fold[
                    "regressor_training_time_seconds"
                ].mean()
            ),
            "mean_fold_total_time_seconds": float(
                rank_fold["fold_total_time_seconds"].mean()
            ),
        })

        previous_mean = cv_mean

    summary_df = pd.DataFrame(summary_rows)

    selection = select_cp_rank_from_cv_summary(
        summary_df=summary_df,
        selection_rule=CV_SELECTION_RULE,
    )

    summary_df["one_se_threshold"] = selection["one_se_threshold"]
    summary_df["is_min_error_rank"] = (
        summary_df["cp_rank_R"] == selection["min_error_rank"]
    )
    summary_df["is_selected_rank"] = (
        summary_df["cp_rank_R"] == selection["selected_rank"]
    )
    summary_df["within_one_se"] = (
        summary_df["cv_mean_rRMSE"]
        <= selection["one_se_threshold"]
    )

    cv_total_time = time.perf_counter() - cv_total_start

    selection.update({
        "candidate_ranks": sorted_ranks,
        "n_development_cases": int(len(idx_development)),
        "n_splits": int(CV_N_SPLITS),
        "shuffle": bool(CV_SHUFFLE),
        "cv_random_state": int(CV_RANDOM_STATE),
        "cv_mlp_epochs": int(CV_MLP_EPOCHS),
        "cp_n_starts": int(CV_CP_N_STARTS),
        "cp_init_method": CV_CP_INIT,
        "test_set_used_for_rank_selection": False,
        "cv_total_time_seconds": float(cv_total_time),
    })

    paths = save_cv_outputs(
        fold_df=fold_df,
        case_df=case_df,
        summary_df=summary_df,
        selection=selection,
        output_dir=output_dir,
    )

    selection["output_paths"] = paths

    print("\n" + "=" * 80)
    print("CP RANK SELECTION RESULT")
    print("=" * 80)

    print(
        summary_df[
            [
                "cp_rank_R",
                "cv_mean_rRMSE",
                "cv_SE_rRMSE",
                "pooled_oof_mean_rRMSE",
                "relative_improvement_from_previous_rank_percent",
                "within_one_se",
                "is_selected_rank",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nMinimum-error rank : {selection['min_error_rank']}"
    )
    print(
        f"One-SE threshold   : {selection['one_se_threshold']:.6f}"
    )
    print(
        f"Selected CP rank   : {selection['selected_rank']}"
    )
    print(
        f"Selection rule     : {selection['selection_rule']}"
    )
    print(
        "Independent 9-case test set was not used in rank selection."
    )
    print("=" * 80)

    return {
        "selected_rank": int(selection["selected_rank"]),
        "selection": selection,
        "fold_df": fold_df,
        "case_df": case_df,
        "summary_df": summary_df,
        "output_paths": paths,
    }


def benchmark_torch_inference_per_case(
        model: nn.Module,
        X_test_scaled: np.ndarray,
        warmup_runs: int,
        repeats: int,
) -> np.ndarray:
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
            scaled_output_tensor = model(condition_tensor)

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


def summarize_latency_seconds(values: np.ndarray) -> dict:
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
        cp_rank: int,
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
        decomposition_time_seconds + training_time_seconds
    )

    aggregate_df = pd.DataFrame([{
        "Method": "CP/PARAFAC",
        "CP Rank": int(cp_rank),
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
        "total_offline_time_seconds": float(
            total_offline_time_seconds
        ),
        "dnn_inference_time_per_case": inference_summary,
        "field_reconstruction_time_per_case": reconstruction_summary,
        "total_online_prediction_time_per_case": online_summary,
        "aggregate_csv": os.path.abspath(aggregate_path),
        "per_case_csv": os.path.abspath(per_case_path),
    }


def save_error_metrics_plot(
        metrics_df: pd.DataFrame,
        fig_dir: str,
        cp_rank: int,
):
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
        label="rRMSE",
    )

    plt.xlabel("Case")
    plt.ylabel("Error Value")
    plt.title(
        f"CP+MLP Independent-Test Error Metrics (R={cp_rank})"
    )
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fig_dir,
            "error_metrics_compare_cp.png",
        ),
        dpi=200,
    )

    plt.close()


def save_y_slice_maps(
        xs,
        zs,
        y_slice_index: int,
        idx_test: np.ndarray,
        preds: np.ndarray,
        gts: np.ndarray,
        fig_dir: str,
        shared_error_max,
        cp_rank: int,
):
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
            f"Case {case_index + 1} | CP Rank {cp_rank}",
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
        axis_true.set_aspect("equal", adjustable="box")
        figure.colorbar(contour_true, ax=axis_true)

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
            f"Reconstructed Temperature (CP, R={cp_rank})"
        )
        axis_predicted.set_xlabel("X (m)")
        axis_predicted.set_ylabel("Z (m)")
        axis_predicted.set_aspect("equal", adjustable="box")
        figure.colorbar(contour_predicted, ax=axis_predicted)

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
            axis_error.set_aspect("equal", adjustable="box")
            figure.colorbar(contour_error, ax=axis_error)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(
            fig_dir,
            f"yslice_case_{case_index + 1}_cp_rank_{cp_rank}.png",
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(figure)

        print(f"  Saved {output_path}")


def save_final_test_summary_outputs(
        summary_df: pd.DataFrame,
        output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(
        output_dir,
        "cp_final_test_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["mean_rRMSE"],
        "-o",
        label="Independent-test mean rRMSE",
    )

    plt.xlabel("CP Rank R")
    plt.ylabel("Mean rRMSE")
    plt.title("Independent-Test Accuracy vs CP Rank")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "cp_final_test_rank_vs_accuracy.png",
        ),
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["decomposition_time_seconds"],
        "-o",
        label="Decomposition and projection time (s)",
    )

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["dnn_training_time_seconds"],
        "-s",
        label="Regressor training time (s)",
    )

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["total_offline_time_seconds"],
        "-^",
        label="Total offline time (s)",
    )

    plt.xlabel("CP Rank R")
    plt.ylabel("Time (s)")
    plt.title("Independent-Test Models: Offline Time vs CP Rank")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "cp_final_test_rank_vs_offline_time.png",
        ),
        dpi=200,
    )

    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["dnn_inference_mean_ms"],
        "-o",
        label="DNN inference per case (ms)",
    )

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["field_reconstruction_mean_ms"],
        "-s",
        label="Field reconstruction per case (ms)",
    )

    plt.plot(
        summary_df["cp_rank_R"],
        summary_df["total_online_prediction_mean_ms"],
        "-^",
        label="Total online prediction per case (ms)",
    )

    plt.xlabel("CP Rank R")
    plt.ylabel("Time (ms)")
    plt.title("Independent-Test Models: Online Time vs CP Rank")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "cp_final_test_rank_vs_online_time.png",
        ),
        dpi=200,
    )

    plt.close()

    return summary_path


def run_one_rank_final_test(
        cp_rank: int,
        X_params: np.ndarray,
        T: np.ndarray,
        xs,
        ys,
        zs,
        y_slice_index: int,
        idx_train: np.ndarray,
        idx_test: np.ndarray,
        test_idx_one_based,
        cp_dcr_df: pd.DataFrame,
        shared_error_max,
        hardware_info: dict,
        output_root_dir: str,
) -> dict:
    reset_random_seed(RANDOM_STATE)

    rank_start = time.perf_counter()

    fig_dir = os.path.join(
        output_root_dir,
        f"rank_{int(cp_rank):02d}",
    )

    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "#" * 72)
    print(f"FINAL INDEPENDENT-TEST EXPERIMENT | CP RANK = {cp_rank}")
    print("#" * 72)

    current_df = cp_dcr_df[
        cp_dcr_df["cp_rank_R"] == int(cp_rank)
    ]

    if current_df.empty:
        raise ValueError(
            f"Rank {cp_rank} was not found in CP DCR table."
        )

    current_dcr_row = current_df.iloc[0]

    save_current_cp_dcr_output(
        current_row=current_dcr_row,
        fig_dir=fig_dir,
    )

    print_cp_dcr_report(current_dcr_row)
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
            "Applied N-mode mean centering using all 80 development cases."
        )
    else:
        T_mean_train = None
        T_train_centered = T[..., idx_train].copy()

    cp_result = fit_cp_basis_and_project_train(
        T_train_centered=T_train_centered,
        rank_r=cp_rank,
        n_starts=FINAL_CP_N_STARTS,
        init_method=FINAL_CP_INIT,
        n_iter_max=CP_N_ITER_MAX,
        tol=CP_TOL,
        base_seed=RANDOM_STATE + cp_rank * 1000,
    )

    lambdas = cp_result["lambdas"]
    Ax = cp_result["Ax"]
    Ay = cp_result["Ay"]
    Az = cp_result["Az"]
    An_train = cp_result["An_train"]
    projection_matrix = cp_result["projection_matrix"]
    Y_train_raw = cp_result["Y_train_raw"]

    decomposition_time_seconds = (
        time.perf_counter() - decomposition_start
    )

    print(
        f"[CP] Shapes: "
        f"Ax={Ax.shape}, Ay={Ay.shape}, "
        f"Az={Az.shape}, An_train={An_train.shape}"
    )

    print(
        f"Selected CP start = {cp_result['selected_start_id']} | "
        f"training projection rRMSE = "
        f"{cp_result['train_projection_rRMSE']:.6f}"
    )

    print(
        f"Decomposition and projection time: "
        f"{decomposition_time_seconds:.6f} s"
    )

    del T_train_centered
    gc.collect()

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

    regression_pipeline = fit_regression_pipeline(
        X_train_raw=X_train_raw,
        Y_train_raw=Y_train_raw,
        energy_weights=energy_weights,
        epochs=EPOCHS,
        seed=RANDOM_STATE,
    )

    fitted_model = regression_pipeline["model"]
    input_scaler = regression_pipeline["input_scaler"]
    output_scaler = regression_pipeline["output_scaler"]
    pca_model = regression_pipeline["pca_model"]
    dnn_training_time_seconds = (
        regression_pipeline["training_time_seconds"]
    )
    device_name = regression_pipeline["device_name"]

    X_test = transform_regression_inputs(
        X_raw=X_test_raw,
        input_scaler=input_scaler,
    )

    predicted_scaled_outputs = predict_scaled_regressor_outputs(
        model=fitted_model,
        X_scaled=X_test,
    )

    predicted_coefficients = decode_scaled_regressor_output(
        scaled_output=predicted_scaled_outputs,
        output_scaler=output_scaler,
        pca_model=pca_model,
    )

    if REGRESSOR.upper() == "MLP_TORCH":
        inference_times_seconds = benchmark_torch_inference_per_case(
            model=fitted_model,
            X_test_scaled=X_test,
            warmup_runs=TIMING_INFERENCE_WARMUP_RUNS,
            repeats=TIMING_INFERENCE_REPEATS,
        )
    else:
        inference_times_seconds = benchmark_sklearn_inference_per_case(
            model=fitted_model,
            X_test_scaled=X_test,
            warmup_runs=TIMING_INFERENCE_WARMUP_RUNS,
            repeats=TIMING_INFERENCE_REPEATS,
        )

    reconstruction_times_seconds = (
        benchmark_field_reconstruction_per_case(
            predicted_scaled_outputs=predicted_scaled_outputs,
            output_scaler=output_scaler,
            pca_model=pca_model,
            projection_matrix=projection_matrix,
            field_shape=(NX, NY, NZ),
            T_mean_train=T_mean_train,
            warmup_runs=TIMING_RECONSTRUCTION_WARMUP_RUNS,
            repeats=TIMING_RECONSTRUCTION_REPEATS,
        )
    )

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
            warmup_runs=TIMING_TOTAL_ONLINE_WARMUP_RUNS,
            repeats=TIMING_TOTAL_ONLINE_REPEATS,
        )
    )

    timing_results = save_timing_outputs(
        fig_dir=fig_dir,
        test_indices_one_based=test_idx_one_based,
        cp_rank=cp_rank,
        regressor_name=REGRESSOR,
        device_name=device_name,
        decomposition_time_seconds=decomposition_time_seconds,
        training_time_seconds=dnn_training_time_seconds,
        inference_times_seconds=inference_times_seconds,
        reconstruction_times_seconds=reconstruction_times_seconds,
        total_online_times_seconds=total_online_times_seconds,
    )

    predicted_fields = []
    ground_truth_fields = []
    metric_rows = []

    for test_position, case_index in enumerate(idx_test):
        predicted_field = cp_reconstruct_from_coeff(
            coefficient=predicted_coefficients[test_position],
            projection_matrix=projection_matrix,
            field_shape=(NX, NY, NZ),
            T_mean_train=T_mean_train,
        )

        true_field = T[..., case_index]

        predicted_fields.append(predicted_field)
        ground_truth_fields.append(true_field)

        metrics = calculate_field_metrics(
            true_field=true_field,
            predicted_field=predicted_field,
        )

        metric_rows.append({
            "Case": int(case_index + 1),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "rRMSE": metrics["rRMSE"],
        })

        print(
            f"  Test case {case_index + 1:2d}: "
            f"MAE={metrics['MAE']:.4f} °C, "
            f"RMSE={metrics['RMSE']:.4f} °C, "
            f"rRMSE={metrics['rRMSE']:.6f}"
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("Case")

    mean_mae = float(metrics_df["MAE"].mean())
    mean_rmse = float(metrics_df["RMSE"].mean())
    mean_rrmse = float(metrics_df["rRMSE"].mean())
    std_rrmse = float(metrics_df["rRMSE"].std(ddof=1))
    se_rrmse = float(std_rrmse / math.sqrt(len(metrics_df)))
    median_rrmse = float(metrics_df["rRMSE"].median())
    max_rrmse = float(metrics_df["rRMSE"].max())

    print(
        "\n[Independent-test mean] "
        f"MAE={mean_mae:.4f} °C, "
        f"RMSE={mean_rmse:.4f} °C, "
        f"rRMSE={mean_rrmse:.6f} ± {se_rrmse:.6f} (SE)"
    )

    per_case_metrics_path = os.path.join(
        fig_dir,
        "per_case_metrics_cp.csv",
    )

    metrics_df.to_csv(
        per_case_metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_error_metrics_plot(
        metrics_df=metrics_df,
        fig_dir=fig_dir,
        cp_rank=cp_rank,
    )

    preds = np.stack(predicted_fields, axis=-1)
    gts = np.stack(ground_truth_fields, axis=-1)

    print(
        "\n=== rRMSE_n at top 10 hottest unique training-mean points ==="
    )

    rrmse_n_df = calculate_rrmse_n_hot_points(
        T_mean_train=T_mean_train,
        T=T,
        idx_train=idx_train,
        preds=preds,
        gts=gts,
        number_of_points=10,
    )

    rrmse_n_path = os.path.join(
        fig_dir,
        "rRMSE_n_top10_hot_points_cp.csv",
    )

    rrmse_n_df.to_csv(
        rrmse_n_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(rrmse_n_df.to_string(index=False))

    if SAVE_FINAL_TEST_SLICE_MAPS:
        save_y_slice_maps(
            xs=xs,
            zs=zs,
            y_slice_index=y_slice_index,
            idx_test=idx_test,
            preds=preds,
            gts=gts,
            fig_dir=fig_dir,
            shared_error_max=shared_error_max,
            cp_rank=cp_rank,
        )

    rank_total_time_seconds = (
        time.perf_counter() - rank_start
    )

    summary = {
        "model_type": "CP_PARAFAC_MLP",
        "rank_selection_source": "80-case internal cross-validation",
        "independent_test_set_used_for_rank_selection": False,
        "cp_rank_R": int(cp_rank),
        "hardware": hardware_info,
        "regressor_device": device_name,
        "test_indices_one_based": [
            int(index) for index in test_idx_one_based
        ],
        "cp_fit": {
            "n_starts": int(FINAL_CP_N_STARTS),
            "init_method": FINAL_CP_INIT,
            "selected_start_id": int(
                cp_result["selected_start_id"]
            ),
            "selected_seed": int(
                cp_result["selected_seed"]
            ),
            "train_projection_rRMSE": float(
                cp_result["train_projection_rRMSE"]
            ),
            "projection_condition_number": float(
                cp_result["projection_condition_number"]
            ),
            "start_records": cp_result["start_records"],
        },
        "computational_time": timing_results,
        "DCR_current_rank": current_dcr_row.to_dict(),
        "metrics_summary": {
            "full3d_mae_mean": mean_mae,
            "full3d_rmse_mean": mean_rmse,
            "full3d_rRMSE_mean": mean_rrmse,
            "full3d_rRMSE_std": std_rrmse,
            "full3d_rRMSE_SE": se_rrmse,
            "full3d_rRMSE_median": median_rrmse,
            "full3d_rRMSE_max": max_rrmse,
            "full3d_mae_each": metrics_df["MAE"].tolist(),
            "full3d_rmse_each": metrics_df["RMSE"].tolist(),
            "full3d_rRMSE_each": metrics_df["rRMSE"].tolist(),
            "rRMSE_n_top10_hot_points": (
                rrmse_n_df.to_dict(orient="records")
            ),
        },
        "output_paths": {
            "fig_dir": os.path.abspath(fig_dir),
            "per_case_metrics_cp_csv": os.path.abspath(
                per_case_metrics_path
            ),
            "rRMSE_n_top10_hot_points_csv": os.path.abspath(
                rrmse_n_path
            ),
            "computational_time_metrics_cp_csv": (
                timing_results["aggregate_csv"]
            ),
            "computational_time_per_case_cp_csv": (
                timing_results["per_case_csv"]
            ),
        },
        "rank_total_execution_time_seconds": float(
            rank_total_time_seconds
        ),
    }

    summary_path = os.path.join(
        fig_dir,
        "metrics_summary_cp.json",
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            to_jsonable(summary),
            file,
            indent=2,
            ensure_ascii=False,
        )

    rank_summary_row = {
        "cp_rank_R": int(cp_rank),
        "mean_MAE": mean_mae,
        "mean_RMSE": mean_rmse,
        "mean_rRMSE": mean_rrmse,
        "std_rRMSE": std_rrmse,
        "SE_rRMSE": se_rrmse,
        "median_rRMSE": median_rrmse,
        "max_rRMSE": max_rrmse,
        "train_projection_rRMSE": float(
            cp_result["train_projection_rRMSE"]
        ),
        "projection_condition_number": float(
            cp_result["projection_condition_number"]
        ),
        "decomposition_time_seconds": float(
            decomposition_time_seconds
        ),
        "dnn_training_time_seconds": float(
            dnn_training_time_seconds
        ),
        "total_offline_time_seconds": float(
            timing_results["total_offline_time_seconds"]
        ),
        "dnn_inference_mean_ms": float(
            timing_results[
                "dnn_inference_time_per_case"
            ]["mean_ms"]
        ),
        "field_reconstruction_mean_ms": float(
            timing_results[
                "field_reconstruction_time_per_case"
            ]["mean_ms"]
        ),
        "total_online_prediction_mean_ms": float(
            timing_results[
                "total_online_prediction_time_per_case"
            ]["mean_ms"]
        ),
        "DCR_train_cp": float(
            current_dcr_row["DCR_train_cp"]
        ),
        "DCR_all_projected_coeff": float(
            current_dcr_row["DCR_all_projected_coeff"]
        ),
        "rank_output_dir": os.path.abspath(fig_dir),
        "rank_total_execution_time_seconds": float(
            rank_total_time_seconds
        ),
    }

    del preds
    del gts
    del predicted_fields
    del ground_truth_fields
    del predicted_scaled_outputs
    del predicted_coefficients
    del projection_matrix
    del Ax
    del Ay
    del Az
    del An_train
    del lambdas
    del Y_train_raw
    del fitted_model
    del regression_pipeline
    del cp_result

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rank_summary_row


def main():
    total_script_start = time.perf_counter()

    reset_random_seed(RANDOM_STATE)

    if not TL_OK:
        raise RuntimeError(
            "TensorLy was not detected. Install it with: pip install tensorly"
        )

    os.makedirs(FIG_ROOT_DIR, exist_ok=True)
    os.makedirs(CV_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FINAL_TEST_OUTPUT_DIR, exist_ok=True)

    X_params, T, xs, ys, zs, y_slice_index = load_all_data()

    (
        test_idx_one_based,
        idx_train,
        idx_test,
    ) = get_fixed_train_test_indices()

    hardware_info = get_hardware_info()
    print("=== Calculate CP DCR for all candidate ranks ===")

    count_mean_for_dcr = bool(
        DCR_COUNT_MEAN_FIELD and CENTER_ALONG_N
    )

    cp_dcr_df = build_cp_dcr_sweep_table(
        ranks=CP_RANK_LIST,
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

    save_global_cp_dcr_outputs(
        dcr_df=cp_dcr_df,
        fig_root_dir=FIG_ROOT_DIR,
    )
    cv_result = None

    if RUN_INTERNAL_CV:
        cv_result = run_internal_cp_rank_cross_validation(
            X_params=X_params,
            T=T,
            idx_development=idx_train,
            rank_list=CP_RANK_LIST,
            output_dir=CV_OUTPUT_DIR,
        )

        selected_rank = int(cv_result["selected_rank"])

    else:
        if FORCE_CP_RANK is None:
            raise ValueError(
                "RUN_INTERNAL_CV=False requires FORCE_CP_RANK to be set."
            )

        selected_rank = int(FORCE_CP_RANK)

    if FORCE_CP_RANK is not None:
        selected_rank = int(FORCE_CP_RANK)
        print(
            f"[Info] FORCE_CP_RANK overrides CV selection: "
            f"R={selected_rank}"
        )

    if selected_rank not in CP_RANK_LIST:
        raise ValueError(
            f"Selected rank {selected_rank} is not in CP_RANK_LIST."
        )

    shared_error_max = load_shared_error_max()

    if RUN_FINAL_TEST_RANK_SWEEP:
        final_ranks = list(CP_RANK_LIST)

        print(
            "\n[Warning] Final test rank sweep is enabled. "
            "The selected rank remains the internal-CV rank; "
            "do not re-select rank from test errors."
        )
    else:
        final_ranks = [selected_rank]

    final_summary_rows = []

    for cp_rank in final_ranks:
        final_summary_row = run_one_rank_final_test(
            cp_rank=cp_rank,
            X_params=X_params,
            T=T,
            xs=xs,
            ys=ys,
            zs=zs,
            y_slice_index=y_slice_index,
            idx_train=idx_train,
            idx_test=idx_test,
            test_idx_one_based=test_idx_one_based,
            cp_dcr_df=cp_dcr_df,
            shared_error_max=shared_error_max,
            hardware_info=hardware_info,
            output_root_dir=FINAL_TEST_OUTPUT_DIR,
        )

        final_summary_row["selected_by_internal_cv"] = (
            int(cp_rank) == int(selected_rank)
        )

        final_summary_rows.append(final_summary_row)

    final_summary_df = pd.DataFrame(
        final_summary_rows
    ).sort_values("cp_rank_R")

    final_summary_path = save_final_test_summary_outputs(
        summary_df=final_summary_df,
        output_dir=FINAL_TEST_OUTPUT_DIR,
    )

    total_script_time_seconds = (
        time.perf_counter() - total_script_start
    )

    global_summary = {
        "model_type": "CP_PARAFAC_MLP_WITH_INTERNAL_RANK_CV",
        "candidate_cp_ranks": [
            int(rank) for rank in CP_RANK_LIST
        ],
        "selected_cp_rank": int(selected_rank),
        "rank_selection_rule": (
            CV_SELECTION_RULE if RUN_INTERNAL_CV
            else "forced"
        ),
        "rank_selection_data": (
            "80 development cases only"
            if RUN_INTERNAL_CV
            else "forced rank"
        ),
        "independent_test_indices_one_based": [
            int(index) for index in test_idx_one_based
        ],
        "independent_test_used_for_rank_selection": False,
        "run_final_test_rank_sweep": bool(
            RUN_FINAL_TEST_RANK_SWEEP
        ),
        "hardware": hardware_info,
        "regressor": REGRESSOR,
        "cv_result": (
            cv_result["selection"]
            if cv_result is not None
            else None
        ),
        "final_test_summary_csv": os.path.abspath(
            final_summary_path
        ),
        "cp_dcr_rank_sweep_csv": os.path.abspath(
            os.path.join(
                FIG_ROOT_DIR,
                "cp_dcr_rank_sweep.csv",
            )
        ),
        "total_script_execution_time_seconds": float(
            total_script_time_seconds
        ),
        "final_test_records": (
            final_summary_df.to_dict(orient="records")
        ),
    }

    global_summary_path = os.path.join(
        FINAL_TEST_OUTPUT_DIR,
        "cp_final_global_summary.json",
    )

    with open(
        global_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_jsonable(global_summary),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)

    print(f"Selected CP rank: {selected_rank}")
    print(
        "Rank selection used only the 80 development cases; "
        "the 9-case independent test set was untouched."
    )

    print("\n--- Final Independent-Test Summary ---")

    print_columns = [
        "cp_rank_R",
        "selected_by_internal_cv",
        "mean_MAE",
        "mean_RMSE",
        "mean_rRMSE",
        "std_rRMSE",
        "SE_rRMSE",
        "max_rRMSE",
        "DCR_train_cp",
        "total_online_prediction_mean_ms",
    ]

    print(
        final_summary_df[print_columns].to_string(index=False)
    )

    print(
        f"\nTotal script execution time: "
        f"{total_script_time_seconds:.2f} s"
    )

    print(f"All outputs: {os.path.abspath(FIG_ROOT_DIR)}")

    print("\nKey rank-selection files:")
    print(
        os.path.join(
            CV_OUTPUT_DIR,
            "cp_rank_cv_summary.csv",
        )
    )
    print(
        os.path.join(
            CV_OUTPUT_DIR,
            "cp_rank_cv_selection.json",
        )
    )
    print(
        os.path.join(
            CV_OUTPUT_DIR,
            "cp_rank_cv_curve.png",
        )
    )

    print("=" * 80)


if __name__ == "__main__":
    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    pd.set_option(
        "display.width",
        180,
    )

    pd.set_option(
        "display.max_columns",
        100,
    )

    try:
        main()

    except Exception as error:
        print(f"ERROR: {error}")

        import traceback
        traceback.print_exc()

        sys.exit(1)
