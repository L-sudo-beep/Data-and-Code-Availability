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
Y_SLICE = 1.53  # meters (用于切片可视化)

# Centering and scaling
CENTER_ALONG_N = True  # 按 N 模中心化（用训练集均值）
SCALE_INPUTS = True  # 14 维参数标准化（仅 fit 在训练集）
SCALE_OUTPUT_COEFF = True  # 系数标准化（仅 fit 在训练集）

# CP 分解设置（取代 Tucker）
CP_RANK = 16  # 当前真正用于 CP 分解、训练和重构的秩
CP_RANK_SWEEP = [6, 8, 10, 12, 14, 16, 20, 24]  # 用于计算不同 R 下 DCR 的秩集合

CP_N_ITER_MAX = 2000
CP_TOL = 1e-7

# DCR 设置
DCR_COUNT_MEAN_FIELD = True
# 如果 CENTER_ALONG_N=True，重构绝对温度场需要保存训练集均值场。
# 因此默认将均值场计入压缩存储量。
# 如果论文中只想比较 CP 低秩结构本身，可将其设为 False。

DCR_COUNT_LAMBDAS = True
# CP 的 weights/lambdas 是否计入压缩存储量，默认计入。

DCR_BYTES_PER_VALUE = 8
# 默认按 float64 估计 MB。DCR 本身是元素数量之比，与字节数无关。

DCR_COUNT_MLP_PARAMS = False
# 默认不把神经网络参数计入 DCR。
# 因为 DCR 通常衡量的是 CP 数据压缩表示本身。
# 如果你希望统计“CP基 + MLP模型”作为压缩模型，也可设为 True。

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

# 能量加权（对输出维度加权，CP 组件权重来自 lambdas 与因子范数）
USE_ENERGY_WEIGHTS = True

# PCA bottleneck on CP coefficients（默认关闭）
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
FIG_DIR = r"C:\Users\Lenovo\Desktop\TensorPOD89\figures_improved_CP"
os.makedirs(FIG_DIR, exist_ok=True)

# 与 POD 统一误差色标
POD_ERROR_MAX_PATH = r"C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1\global_error_max.npy"


# ========================== Utils ==========================

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    # 跳过表头行（数据集中第一行为表头再重复一行的情况）
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


# ========================== DCR for CP/PARAFAC ==========================

def count_mlp_parameters(
        in_dim: int,
        out_dim: int,
        hidden_layers,
        use_layernorm: bool = True
) -> int:
    """
    估算当前 MLP 的可训练参数量。

    仅当 DCR_COUNT_MLP_PARAMS=True 时用于额外口径。
    默认 DCR 不统计神经网络参数。
    """
    total = 0
    prev = in_dim

    for h in hidden_layers:
        # Linear: weight + bias
        total += prev * h + h

        # LayerNorm: gamma + beta
        if use_layernorm:
            total += 2 * h

        prev = h

    # Output layer
    total += prev * out_dim + out_dim

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
        extra_entries: int = 0
) -> dict:
    """
    计算 CP/PARAFAC 压缩表示下的数据压缩率 DCR。

    原始数据存储量：
        original_entries = nx * ny * nz * n_cases

    CP 压缩表示存储量：
        spatial_factor_entries = nx*R + ny*R + nz*R
        case_factor_entries    = n_cases*R
        lambda_entries         = R
        mean_entries           = nx*ny*nz，如果 count_mean_field=True

    compressed_entries =
        spatial_factor_entries
        + case_factor_entries
        + lambda_entries
        + mean_entries
        + extra_entries

    DCR:
        DCR = original_entries / compressed_entries

    注意：
    - 默认不统计 MLP 参数。
    - 如果 extra_entries 传入 MLP 参数量，则可以得到“CP基 + MLP模型”的压缩率口径。
    """
    R = int(rank_r)

    original_entries = int(nx * ny * nz * n_cases)

    spatial_factor_entries = int((nx + ny + nz) * R)
    case_factor_entries = int(n_cases * R)
    lambda_entries = int(R) if count_lambdas else 0
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
    compression_percent = float((1.0 - compressed_entries / (original_entries + 1e-12)) * 100.0)

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
        bytes_per_value: int = 8
) -> pd.DataFrame:
    """
    计算不同 CP 秩 R 下的数据压缩率。

    输出两个主要口径：

    1. DCR_train_cp:
       原始训练集张量 vs 训练集 CP 表示。
       压缩表示包括：
       Ax, Ay, Az, An_train, lambdas, mean field。

    2. DCR_all_projected_coeff:
       原始全体样本张量 vs 共享空间因子 + 全体样本 CP 系数。
       压缩表示包括：
       Ax, Ay, Az, C_all, lambdas, mean field。

    可选口径：

    3. DCR_all_with_mlp_model:
       原始全体样本张量 vs CP 空间因子 + MLP 模型。
       压缩表示包括：
       Ax, Ay, Az, lambdas, mean field, MLP parameters。
       默认不启用，因为 DCR 通常比较的是低秩数据表达。
    """
    ranks = sorted(set([int(r) for r in ranks]))

    rows = []
    for R in ranks:
        mlp_param_entries = count_mlp_parameters(
            in_dim=14,
            out_dim=R,
            hidden_layers=HIDDEN_LAYERS,
            use_layernorm=USE_LAYERNORM
        )

        train_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_train,
            rank_r=R,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=0
        )

        all_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            rank_r=R,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=0
        )

        all_with_mlp_counts = compute_cp_dcr_counts(
            nx=nx,
            ny=ny,
            nz=nz,
            n_cases=n_all,
            rank_r=R,
            count_mean_field=count_mean_field,
            count_lambdas=count_lambdas,
            bytes_per_value=bytes_per_value,
            extra_entries=mlp_param_entries if count_mlp_params else 0
        )

        rows.append({
            "cp_rank_R": int(R),

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
            "compression_percent_all_projected_coeff": all_counts["compression_percent"],
            "all_original_MB_float64": all_counts["original_MB_float64"],
            "all_compressed_MB_float64": all_counts["compressed_MB_float64"],

            "mlp_param_entries": int(mlp_param_entries),
            "DCR_count_mlp_params": bool(count_mlp_params),
            "all_with_mlp_compressed_entries": all_with_mlp_counts["compressed_entries"],
            "DCR_all_with_mlp_model": all_with_mlp_counts["DCR"],
            "compression_percent_all_with_mlp_model": all_with_mlp_counts["compression_percent"],
            "all_with_mlp_compressed_MB_float64": all_with_mlp_counts["compressed_MB_float64"],
        })

    return pd.DataFrame(rows)


def save_cp_dcr_outputs(
        dcr_df: pd.DataFrame,
        fig_dir: str,
        current_rank: int
) -> pd.Series:
    """
    保存 CP DCR 结果：
    - cp_dcr_rank_sweep.csv
    - cp_dcr_current.csv
    - cp_dcr_vs_rank.png
    """
    os.makedirs(fig_dir, exist_ok=True)

    sweep_path = os.path.join(fig_dir, "cp_dcr_rank_sweep.csv")
    dcr_df.to_csv(sweep_path, index=False, encoding="utf-8-sig")

    current_df = dcr_df[dcr_df["cp_rank_R"] == int(current_rank)]
    if current_df.empty:
        current_row = dcr_df.iloc[0]
    else:
        current_row = current_df.iloc[0]

    current_path = os.path.join(fig_dir, "cp_dcr_current.csv")
    pd.DataFrame([current_row]).to_csv(current_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    plt.plot(
        dcr_df["cp_rank_R"],
        dcr_df["DCR_train_cp"],
        "-o",
        label="DCR: Train CP representation"
    )
    plt.plot(
        dcr_df["cp_rank_R"],
        dcr_df["DCR_all_projected_coeff"],
        "-s",
        label="DCR: All cases with projected coefficients"
    )

    if bool(DCR_COUNT_MLP_PARAMS):
        plt.plot(
            dcr_df["cp_rank_R"],
            dcr_df["DCR_all_with_mlp_model"],
            "-^",
            label="DCR: All cases with MLP model"
        )

    plt.xlabel("CP Rank R")
    plt.ylabel("Data Compression Ratio, DCR")
    plt.title("CP DCR vs Rank")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cp_dcr_vs_rank.png"), dpi=200)
    plt.close()

    return current_row


def print_cp_dcr_report(current_row: pd.Series, dcr_df: pd.DataFrame):
    """
    打印当前 CP_RANK 的 DCR，以及所有 R 的 DCR 表。
    """
    print("\n=== CP DCR Report for Current CP_RANK ===")
    print(f"  Current CP rank R = {int(current_row['cp_rank_R'])}")

    print("\n  [Train CP representation]")
    print(f"    Original entries   = {int(current_row['train_original_entries'])}")
    print(f"    Compressed entries = {int(current_row['train_compressed_entries'])}")
    print(f"    DCR                = {current_row['DCR_train_cp']:.6f}")
    print(f"    Saving             = {current_row['compression_percent_train_cp']:.2f}%")
    print(f"    Original size      = {current_row['train_original_MB_float64']:.2f} MB")
    print(f"    Compressed size    = {current_row['train_compressed_MB_float64']:.2f} MB")

    print("\n  [All cases with projected coefficients]")
    print(f"    Original entries   = {int(current_row['all_original_entries'])}")
    print(f"    Compressed entries = {int(current_row['all_compressed_entries'])}")
    print(f"    DCR                = {current_row['DCR_all_projected_coeff']:.6f}")
    print(f"    Saving             = {current_row['compression_percent_all_projected_coeff']:.2f}%")
    print(f"    Original size      = {current_row['all_original_MB_float64']:.2f} MB")
    print(f"    Compressed size    = {current_row['all_compressed_MB_float64']:.2f} MB")

    if bool(DCR_COUNT_MLP_PARAMS):
        print("\n  [All cases with CP basis + MLP model]")
        print(f"    MLP parameter entries = {int(current_row['mlp_param_entries'])}")
        print(f"    Compressed entries    = {int(current_row['all_with_mlp_compressed_entries'])}")
        print(f"    DCR                   = {current_row['DCR_all_with_mlp_model']:.6f}")
        print(f"    Saving                = {current_row['compression_percent_all_with_mlp_model']:.2f}%")

    print("\n=== CP DCR Sweep Table ===")
    show_cols = [
        "cp_rank_R",
        "DCR_train_cp",
        "compression_percent_train_cp",
        "DCR_all_projected_coeff",
        "compression_percent_all_projected_coeff",
        "all_compressed_entries"
    ]

    if bool(DCR_COUNT_MLP_PARAMS):
        show_cols += [
            "mlp_param_entries",
            "DCR_all_with_mlp_model",
            "compression_percent_all_with_mlp_model"
        ]

    print(dcr_df[show_cols].to_string(index=False))


# ========================== MLP & 回归器 ==========================

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

    if weight_per_dim is None:
        weight = torch.ones(Y_train.shape[1], dtype=torch.float32, device=device)
    else:
        # 归一化到均值=1，避免整体尺度变化
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


# ========================== CP（PARAFAC）核心实现 ==========================

def cp_fit_train_only(T_centered: np.ndarray, idx_train: np.ndarray, R: int,
                      n_iter_max: int = 2000, tol: float = 1e-7, random_state: int = 42):
    """
    只在训练集子张量做 CP 分解：T_train ∈ R^{NX x NY x NZ x N_train}
    返回权重 lambdas 和因子矩阵 Ax, Ay, Az, An_train
    """
    if not TL_OK:
        raise RuntimeError("需要安装 tensorly： pip install tensorly")
    tl.set_backend('numpy')

    T_train = T_centered[..., idx_train]  # (NX, NY, NZ, N_train)

    cp = parafac(
        T_train,
        rank=R,
        init='svd',
        tol=tol,
        n_iter_max=n_iter_max,
        random_state=random_state,
        normalize_factors=True
    )

    lambdas = cp.weights
    Ax, Ay, Az, An_train = cp.factors  # shapes: (NX,R), (NY,R), (NZ,R), (N_train,R)
    return lambdas, Ax, Ay, Az, An_train


def cp_build_projection_matrix(Ax: np.ndarray, Ay: np.ndarray, Az: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    构造 M ∈ R^{(NX*NY*NZ) x R}，列 r 为 lambda_r * vec(a_r ∘ b_r ∘ c_r)
    用于对每个工况的 3D 场做线性最小二乘，解 d^{(n)}。
    """
    nx, R = Ax.shape
    ny = Ay.shape[0]
    nz = Az.shape[0]

    M = np.empty((nx * ny * nz, R), dtype=np.float64)

    for r in range(R):
        outer3 = np.multiply.outer(np.multiply.outer(Ax[:, r], Ay[:, r]), Az[:, r])  # (NX, NY, NZ)
        M[:, r] = float(lambdas[r]) * outer3.reshape(-1)

    return M


def project_case_to_cp_coeff(T_case: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    给定单个工况的 3D 场（已中心化），用预构造的 M 解 d^{(n)}：
        min ||M d - vec(T_case)||_2
    """
    b = T_case.reshape(-1)
    d, *_ = np.linalg.lstsq(M, b, rcond=None)
    return d


def cp_reconstruct_from_d(d: np.ndarray, Ax, Ay, Az, lambdas, T_mean_train=None):
    """
    用预测的 CP 系数 d 重构 3D 温度场
    """
    nx, R = Ax.shape
    ny = Ay.shape[0]
    nz = Az.shape[0]

    T_rec = np.zeros((nx, ny, nz), dtype=np.float64)

    for r in range(R):
        T_rec += (lambdas[r] * d[r]) * np.multiply.outer(
            np.multiply.outer(Ax[:, r], Ay[:, r]),
            Az[:, r]
        )

    if T_mean_train is not None:
        T_rec = T_rec + T_mean_train[..., 0]

    return T_rec


def cp_component_energy_weights(lambdas, Ax, Ay, Az):
    """
    估计每个 CP 组件的“能量”权重：
        |lambda_r| * ||a_r|| * ||b_r|| * ||c_r||

    再归一化并设置下限，返回长度 R 的权重。
    """
    R = Ax.shape[1]
    w = np.zeros(R, dtype=np.float64)

    for r in range(R):
        w[r] = (
            abs(float(lambdas[r]))
            * np.linalg.norm(Ax[:, r])
            * np.linalg.norm(Ay[:, r])
            * np.linalg.norm(Az[:, r])
        )

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
        T_mean_train = T[..., idx_train].mean(axis=3, keepdims=True)
        T_centered = T - T_mean_train
        print("Applied N-mode mean centering using TRAIN set mean.")
    else:
        T_mean_train = None
        T_centered = T

    # =================== CP DCR Calculation ===================
    print("=== Calculate CP DCR for different ranks ===")

    count_mean_for_dcr = bool(DCR_COUNT_MEAN_FIELD and CENTER_ALONG_N)

    dcr_ranks = sorted(set([int(CP_RANK)] + [int(r) for r in CP_RANK_SWEEP]))

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
        bytes_per_value=DCR_BYTES_PER_VALUE
    )

    current_dcr_row = save_cp_dcr_outputs(
        dcr_df=cp_dcr_df,
        fig_dir=FIG_DIR,
        current_rank=CP_RANK
    )

    print_cp_dcr_report(current_dcr_row, cp_dcr_df)

    print(f"\nSaved CP DCR current result to: {os.path.join(FIG_DIR, 'cp_dcr_current.csv')}")
    print(f"Saved CP DCR sweep result to:   {os.path.join(FIG_DIR, 'cp_dcr_rank_sweep.csv')}")
    print(f"Saved CP DCR plot to:           {os.path.join(FIG_DIR, 'cp_dcr_vs_rank.png')}")

    # === CP on TRAIN set ===
    print(f"\n=== CP (TRAIN-only) rank = {CP_RANK} ===")
    t0 = time.time()

    lambdas, Ax, Ay, Az, An_train = cp_fit_train_only(
        T_centered,
        idx_train,
        CP_RANK,
        n_iter_max=CP_N_ITER_MAX,
        tol=CP_TOL,
        random_state=RANDOM_STATE
    )

    print(f"[CP] Done. Shapes: Ax={Ax.shape}, Ay={Ay.shape}, Az={Az.shape}, An_train={An_train.shape}")
    print(f"CP fit time: {time.time() - t0:.2f}s")

    # === Build coefficients for ALL cases via projection (using CP spatial factors) ===
    print("=== Build CP coefficients for ALL cases via projection ===")
    M_cp = cp_build_projection_matrix(Ax, Ay, Az, lambdas)
    R = Ax.shape[1]

    Y_coeff_all = np.zeros((N_CASES, R), dtype=np.float64)

    for n in range(N_CASES):
        Tc = T_centered[..., n]
        Y_coeff_all[n, :] = project_case_to_cp_coeff(Tc, M_cp)

    # Train/Test split for regression
    X_train_raw, X_test_raw = X_params[idx_train], X_params[idx_test]
    Y_train_raw, Y_test_raw = Y_coeff_all[idx_train], Y_coeff_all[idx_test]

    # Energy weights from CP components
    if USE_ENERGY_WEIGHTS:
        w = cp_component_energy_weights(lambdas, Ax, Ay, Az)
    else:
        w = None

    # =================== Regression ===================
    if USE_PCA_BOTTLENECK:
        raise NotImplementedError("PCA Bottleneck with CP is not the focus of this update.")
    else:
        # Scale X
        if SCALE_INPUTS:
            x_scaler = StandardScaler().fit(X_train_raw)
            X_train = x_scaler.transform(X_train_raw)
            X_test = x_scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # Scale Y coefficients
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
    print("=== Reconstruct test 3D fields (CP) ===")
    preds, gts = [], []

    for j, case_idx in enumerate(idx_test):
        d_pred = Y_pred[j]
        T_pred = cp_reconstruct_from_d(d_pred, Ax, Ay, Az, lambdas, T_mean_train)
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

        print(
            f"  Case {idx_test[k] + 1:2d}: "
            f"MAE={mae:.4f} °C, "
            f"RMSE={rmse:.4f} °C, "
            f"rRMSE={rRMSE_case:.4f}"
        )

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
    df.to_csv(os.path.join(FIG_DIR, "per_case_metrics_cp.csv"), index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(df["Case"], df["MAE"], '-o', label="MAE (°C)")
    plt.plot(df["Case"], df["RMSE"], '-s', label="RMSE (°C)")
    plt.plot(df["Case"], df["rRMSE"], '-^', label="rRMSE (relative)")
    plt.xlabel("Case")
    plt.ylabel("Error Value")
    plt.title("CP+MLP Per-case Error Metrics (TRAIN-only)")
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
        print("[Info] POD global_error_max.npy not found; CP error maps will use per-case vmax.")

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
        ax2.set_title("Reconstructed Temperature (CP)")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Z (m)")
        ax2.set_aspect('equal', adjustable='box')
        fig.colorbar(c2, ax=ax2)

        if SAVE_ERROR_MAP:
            ax3 = fig.add_subplot(1, 3, 3)
            c3 = ax3.contourf(
                Xg,
                Zg,
                err_plot.T,
                levels=LEVELS,
                cmap='YlOrRd',
                vmin=0,
                vmax=vmax
            )
            ax3.set_title(f"Absolute Error (vmax={vmax:.2f})")
            ax3.set_xlabel("X (m)")
            ax3.set_ylabel("Z (m)")
            ax3.set_aspect('equal', adjustable='box')
            fig.colorbar(c3, ax=ax3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx + 1}_cp.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved {out_path}")

    # =================== Summary ===================
    summary = {
        "model_type": "CP_PARAFAC",
        "decomposition_set": "TRAIN",
        "cp_rank_R": int(CP_RANK),

        "DCR_definition": {
            "formula": "DCR = original_entries / compressed_entries",
            "compressed_entries_for_CP": "(NX + NY + NZ) * R + N * R + lambda_entries + mean_entries",
            "lambda_counted": bool(DCR_COUNT_LAMBDAS),
            "mean_field_counted": bool(count_mean_for_dcr),
            "mlp_parameters_counted": bool(DCR_COUNT_MLP_PARAMS),
            "note": (
                "DCR here mainly measures CP/PARAFAC data representation compression. "
                "By default, neural-network parameters are not included."
            )
        },

        "DCR_current_rank": current_dcr_row.to_dict(),

        "DCR_rank_sweep": [
            {
                "cp_rank_R": int(row["cp_rank_R"]),
                "DCR_train_cp": float(row["DCR_train_cp"]),
                "compression_percent_train_cp": float(row["compression_percent_train_cp"]),
                "DCR_all_projected_coeff": float(row["DCR_all_projected_coeff"]),
                "compression_percent_all_projected_coeff": float(row["compression_percent_all_projected_coeff"]),
                "DCR_all_with_mlp_model": float(row["DCR_all_with_mlp_model"]),
                "compression_percent_all_with_mlp_model": float(row["compression_percent_all_with_mlp_model"]),
                "mlp_param_entries": int(row["mlp_param_entries"]),
            }
            for _, row in cp_dcr_df.iterrows()
        ],

        "config": {
            "center_along_N": bool(CENTER_ALONG_N),
            "scale_inputs": bool(SCALE_INPUTS),
            "scale_output_coeff": bool(SCALE_OUTPUT_COEFF),
            "dcr_count_mean_field": bool(DCR_COUNT_MEAN_FIELD),
            "dcr_count_lambdas": bool(DCR_COUNT_LAMBDAS),
            "dcr_count_mlp_params": bool(DCR_COUNT_MLP_PARAMS),
            "dcr_bytes_per_value": int(DCR_BYTES_PER_VALUE)
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
            "unified_error_vmax_from_pod": float(shared_error_max) if shared_error_max is not None else None
        },

        "cp_component_weights_preview": [
            float(x) for x in cp_component_energy_weights(lambdas, Ax, Ay, Az).tolist()
        ],

        "output_paths": {
            "fig_dir": os.path.abspath(FIG_DIR),
            "cp_dcr_current_csv": os.path.abspath(os.path.join(FIG_DIR, "cp_dcr_current.csv")),
            "cp_dcr_rank_sweep_csv": os.path.abspath(os.path.join(FIG_DIR, "cp_dcr_rank_sweep.csv")),
            "cp_dcr_vs_rank_png": os.path.abspath(os.path.join(FIG_DIR, "cp_dcr_vs_rank.png")),
            "per_case_metrics_cp_csv": os.path.abspath(os.path.join(FIG_DIR, "per_case_metrics_cp.csv")),
            "metrics_summary_cp_json": os.path.abspath(os.path.join(FIG_DIR, "metrics_summary_cp.json"))
        },

        "execution_time_seconds": time.time() - start_main_time
    }

    with open(os.path.join(FIG_DIR, "metrics_summary_cp.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Total execution time: {time.time() - start_main_time:.2f}s.")
    print("Outputs in:", os.path.abspath(FIG_DIR))


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
