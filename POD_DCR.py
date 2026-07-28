import pandas as pd
import numpy as np
from scipy.linalg import svd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import random
from scipy.interpolate import griddata
import time
from tqdm import tqdm
import json
from datetime import datetime

# ======================================
# Configurable Parameters
# ======================================
SEED = 42

# 当前 POD+MLP 实际训练使用的能量阈值
ENERGY_THRESHOLD = 0.99

# 用于一次性扫描不同能量阈值下 DCR 的列表
# 注意：这些阈值只用于计算不同 K 下的 DCR，不会分别训练多个 MLP。
DCR_SWEEP_THRESHOLDS = [
    0.90, 0.95, 0.97, 0.99, 0.995, 0.999
]

# DCR 统计时是否计入均值场
# 对 POD 重构绝对温度场来说，均值场通常需要保存，因此建议 True。
DCR_COUNT_MEAN_FIELD = True

# 是否将 MLP 参数量也计入压缩存储量
# 如果只是比较 POD / Tucker / CP 的数据压缩率，建议 False。
# 如果想比较完整的“降阶模型存储量”，可设为 True。
DCR_COUNT_MLP_WEIGHTS = False

# 估算 MB 时每个数值按 float64 计算
DCR_BYTES_PER_VALUE = 8

HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16

DATA_DIR = r'C:\Users\Lenovo\Desktop\insert'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'

PLANE_TYPE = 'y'
PLANE_VALUE = 2.52
NUM_ADJACENT_PLANES = 0

SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-1.26'

# ======================================
# Set Seeds for Reproducibility
# ======================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ======================================
# DCR Utility Functions
# ======================================

def rank_from_energy_singular_values(S, energy_threshold):
    """
    根据 POD 奇异值累计能量选择模态数 K。

    Parameters
    ----------
    S : np.ndarray
        POD 奇异值。
    energy_threshold : float
        能量保留阈值，例如 0.90, 0.95, 0.99。

    Returns
    -------
    int
        满足累计能量 >= energy_threshold 的最小 K。
    """
    energy_threshold = float(np.clip(energy_threshold, 0.0, 1.0))

    energy = S ** 2
    total_energy = np.sum(energy)

    if total_energy <= 0:
        return 1

    cumulative_energy = np.cumsum(energy) / total_energy
    K = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)
    K = max(1, min(K, len(S)))

    return K


def count_mlp_parameters(input_dim, output_dim, hidden_layers):
    """
    计算当前 MLP 架构的参数数量。

    这里统计：
    - Linear 层权重和偏置；
    - BatchNorm1d 的 gamma 和 beta。

    Dropout 和 ReLU 不含可训练参数。
    """
    total_params = 0

    prev_dim = input_dim

    for h in hidden_layers:
        # Linear: weight + bias
        total_params += prev_dim * h + h

        # BatchNorm1d: gamma + beta
        total_params += 2 * h

        prev_dim = h

    # Output Linear: weight + bias
    total_params += prev_dim * output_dim + output_dim

    return int(total_params)


def compute_pod_dcr_counts(
        n_points,
        n_cases,
        K,
        input_dim=None,
        hidden_layers=None,
        count_mean_field=True,
        count_mlp_weights=False,
        bytes_per_value=8
):
    """
    计算 POD 数据压缩率 DCR。

    原始数据量：
        n_points * n_cases

    POD 压缩表示数据量：
        mean_entries + mode_entries + coeff_entries

    其中：
        mean_entries = n_points
        mode_entries = n_points * K
        coeff_entries = n_cases * K

    如果 count_mlp_weights=True，则额外加入 MLP 参数量。
    但通常与 POD/Tucker/CP 分解压缩率横向比较时，不建议计入神经网络参数。
    """
    original_entries = int(n_points * n_cases)

    mean_entries = int(n_points) if count_mean_field else 0
    mode_entries = int(n_points * K)
    coeff_entries = int(n_cases * K)

    mlp_entries = 0
    if count_mlp_weights:
        if input_dim is None or hidden_layers is None:
            raise ValueError("When count_mlp_weights=True, input_dim and hidden_layers must be provided.")
        mlp_entries = count_mlp_parameters(
            input_dim=input_dim,
            output_dim=K,
            hidden_layers=hidden_layers
        )

    compressed_entries = int(
        mean_entries
        + mode_entries
        + coeff_entries
        + mlp_entries
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
        "mean_entries": mean_entries,
        "mode_entries": mode_entries,
        "coeff_entries": coeff_entries,
        "mlp_entries": mlp_entries,
        "DCR": dcr,
        "compression_percent": compression_percent,
        "original_MB_float64": original_mb,
        "compressed_MB_float64": compressed_mb
    }


def build_pod_dcr_sweep_table(
        S,
        thresholds,
        n_points,
        n_train,
        n_all,
        input_dim,
        hidden_layers,
        count_mean_field=True,
        count_mlp_weights=False,
        bytes_per_value=8
):
    """
    根据 POD 奇异值，一次性计算多个能量阈值下的 K 和 DCR。

    输出两个主要口径：

    1. DCR_train_pod:
       训练集 POD 压缩率。
       原始数据量为 n_points * n_train。
       压缩数据量为 mean + modes + train_coefficients。

    2. DCR_all_projected_coeff:
       全部样本采用同一 POD 基投影系数后的压缩率。
       原始数据量为 n_points * n_all。
       压缩数据量为 mean + modes + all_coefficients。
    """
    thresholds = sorted(set([float(x) for x in thresholds]))

    rows = []

    for thr in thresholds:
        K = rank_from_energy_singular_values(S, thr)

        train_counts = compute_pod_dcr_counts(
            n_points=n_points,
            n_cases=n_train,
            K=K,
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            count_mean_field=count_mean_field,
            count_mlp_weights=count_mlp_weights,
            bytes_per_value=bytes_per_value
        )

        all_counts = compute_pod_dcr_counts(
            n_points=n_points,
            n_cases=n_all,
            K=K,
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            count_mean_field=count_mean_field,
            count_mlp_weights=count_mlp_weights,
            bytes_per_value=bytes_per_value
        )

        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "energy_threshold": float(thr),
            "K": int(K),

            "train_original_entries": train_counts["original_entries"],
            "train_compressed_entries": train_counts["compressed_entries"],
            "train_mean_entries": train_counts["mean_entries"],
            "train_mode_entries": train_counts["mode_entries"],
            "train_coeff_entries": train_counts["coeff_entries"],
            "train_mlp_entries": train_counts["mlp_entries"],
            "DCR_train_pod": train_counts["DCR"],
            "compression_percent_train_pod": train_counts["compression_percent"],
            "train_original_MB_float64": train_counts["original_MB_float64"],
            "train_compressed_MB_float64": train_counts["compressed_MB_float64"],

            "all_original_entries": all_counts["original_entries"],
            "all_compressed_entries": all_counts["compressed_entries"],
            "all_mean_entries": all_counts["mean_entries"],
            "all_mode_entries": all_counts["mode_entries"],
            "all_coeff_entries": all_counts["coeff_entries"],
            "all_mlp_entries": all_counts["mlp_entries"],
            "DCR_all_projected_coeff": all_counts["DCR"],
            "compression_percent_all_projected_coeff": all_counts["compression_percent"],
            "all_original_MB_float64": all_counts["original_MB_float64"],
            "all_compressed_MB_float64": all_counts["compressed_MB_float64"],
        })

    return pd.DataFrame(rows)


def save_pod_dcr_outputs(
        dcr_sweep_df,
        save_dir,
        current_energy_threshold
):
    """
    保存 DCR 结果：
    - dcr_sweep.csv：当前运行的多个阈值 DCR 结果；
    - dcr_current.csv：当前 ENERGY_THRESHOLD 对应结果；
    - dcr_vs_energy_threshold.csv：用于画图和论文整理；
    - dcr_vs_energy_threshold.png：DCR 随能量阈值变化曲线。
    """
    os.makedirs(save_dir, exist_ok=True)

    dcr_sweep_path = os.path.join(save_dir, "dcr_sweep.csv")
    dcr_sweep_df.to_csv(dcr_sweep_path, index=False, encoding="utf-8-sig")

    current_df = dcr_sweep_df[
        np.isclose(dcr_sweep_df["energy_threshold"], current_energy_threshold)
    ]

    if current_df.empty:
        current_df = dcr_sweep_df.iloc[[0]]

    dcr_current_path = os.path.join(save_dir, "dcr_current.csv")
    current_df.to_csv(dcr_current_path, index=False, encoding="utf-8-sig")

    comparison_path = os.path.join(save_dir, "dcr_vs_energy_threshold.csv")
    dcr_sweep_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    plt.plot(
        dcr_sweep_df["energy_threshold"],
        dcr_sweep_df["DCR_train_pod"],
        "-o",
        label="DCR: Train POD representation"
    )
    plt.plot(
        dcr_sweep_df["energy_threshold"],
        dcr_sweep_df["DCR_all_projected_coeff"],
        "-s",
        label="DCR: All cases with projected coefficients"
    )
    plt.xlabel("Energy Threshold")
    plt.ylabel("Data Compression Ratio, DCR")
    plt.title("POD DCR vs Energy Threshold")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dcr_vs_energy_threshold.png"), dpi=200)
    plt.close()

    return current_df.iloc[0]


def print_dcr_report(current_dcr_row):
    """
    打印当前 ENERGY_THRESHOLD 对应的 DCR 信息。
    """
    print("\n" + "=" * 20 + " POD DCR REPORT " + "=" * 20)
    print(f"Energy threshold = {current_dcr_row['energy_threshold']:.6f}")
    print(f"Selected POD modes K = {int(current_dcr_row['K'])}")

    print("\n--- Train POD representation ---")
    print(f"Original entries   : {int(current_dcr_row['train_original_entries'])}")
    print(f"Compressed entries : {int(current_dcr_row['train_compressed_entries'])}")
    print(f"Mean entries       : {int(current_dcr_row['train_mean_entries'])}")
    print(f"Mode entries       : {int(current_dcr_row['train_mode_entries'])}")
    print(f"Coeff entries      : {int(current_dcr_row['train_coeff_entries'])}")
    print(f"MLP entries        : {int(current_dcr_row['train_mlp_entries'])}")
    print(f"DCR                : {current_dcr_row['DCR_train_pod']:.6f}")
    print(f"Compression saving : {current_dcr_row['compression_percent_train_pod']:.2f}%")
    print(f"Original size      : {current_dcr_row['train_original_MB_float64']:.2f} MB")
    print(f"Compressed size    : {current_dcr_row['train_compressed_MB_float64']:.2f} MB")

    print("\n--- All cases with projected coefficients ---")
    print(f"Original entries   : {int(current_dcr_row['all_original_entries'])}")
    print(f"Compressed entries : {int(current_dcr_row['all_compressed_entries'])}")
    print(f"Mean entries       : {int(current_dcr_row['all_mean_entries'])}")
    print(f"Mode entries       : {int(current_dcr_row['all_mode_entries'])}")
    print(f"Coeff entries      : {int(current_dcr_row['all_coeff_entries'])}")
    print(f"MLP entries        : {int(current_dcr_row['all_mlp_entries'])}")
    print(f"DCR                : {current_dcr_row['DCR_all_projected_coeff']:.6f}")
    print(f"Compression saving : {current_dcr_row['compression_percent_all_projected_coeff']:.2f}%")
    print(f"Original size      : {current_dcr_row['all_original_MB_float64']:.2f} MB")
    print(f"Compressed size    : {current_dcr_row['all_compressed_MB_float64']:.2f} MB")
    print("=" * 58)


# ======================================
# Load Data
# ======================================
start_time = time.time()

os.makedirs(SAVE_DIR, exist_ok=True)

bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]

# Split indices: fixed test cases
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]

# Fit scaler only on training set, avoiding data leakage
scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

# Load coordinates and snapshots
temp_files = [
    os.path.join(DATA_DIR, f"{i}.csv")
    for i in range(1, num_samples + 1)
]

first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = coords.shape[0]

print("Loading all snapshots...")
all_snapshots = np.zeros((N_points, num_samples))

for i in tqdm(range(num_samples), desc="Loading Snapshots"):
    df = pd.read_csv(temp_files[i])
    all_snapshots[:, i] = df['Temperature'].values

snapshots_train = all_snapshots[:, train_idx]
snapshots_val = all_snapshots[:, val_idx]

# ======================================
# POD Analysis
# ======================================
print("\nPerforming POD analysis...")

mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
fluctuations = snapshots_train - mean_temp

U, S, Vt = svd(fluctuations, full_matrices=False)

energy = np.cumsum(S ** 2) / np.sum(S ** 2)
K = rank_from_energy_singular_values(S, ENERGY_THRESHOLD)

print(f"Selected K={K} modes for {ENERGY_THRESHOLD * 100:.1f}% energy")

# ======================================
# DCR Calculation
# ======================================
print("\nCalculating POD DCR for different energy thresholds...")

dcr_thresholds = sorted(
    set([float(ENERGY_THRESHOLD)] + [float(x) for x in DCR_SWEEP_THRESHOLDS])
)

dcr_sweep_df = build_pod_dcr_sweep_table(
    S=S,
    thresholds=dcr_thresholds,
    n_points=N_points,
    n_train=len(train_idx),
    n_all=num_samples,
    input_dim=conditions.shape[1],
    hidden_layers=HIDDEN_LAYERS,
    count_mean_field=DCR_COUNT_MEAN_FIELD,
    count_mlp_weights=DCR_COUNT_MLP_WEIGHTS,
    bytes_per_value=DCR_BYTES_PER_VALUE
)

current_dcr_row = save_pod_dcr_outputs(
    dcr_sweep_df=dcr_sweep_df,
    save_dir=SAVE_DIR,
    current_energy_threshold=ENERGY_THRESHOLD
)

print_dcr_report(current_dcr_row)

print("\nPOD DCR sweep table:")
print(
    dcr_sweep_df[
        [
            "energy_threshold",
            "K",
            "DCR_train_pod",
            "compression_percent_train_pod",
            "DCR_all_projected_coeff",
            "compression_percent_all_projected_coeff"
        ]
    ].to_string(index=False)
)

print(f"\nSaved DCR sweep table to: {os.path.join(SAVE_DIR, 'dcr_sweep.csv')}")
print(f"Saved current DCR table to: {os.path.join(SAVE_DIR, 'dcr_current.csv')}")
print(f"Saved DCR plot to: {os.path.join(SAVE_DIR, 'dcr_vs_energy_threshold.png')}")

# ======================================
# Build POD modes and coefficients for current ENERGY_THRESHOLD
# ======================================
modes = U[:, :K]

# 训练集 POD 系数
coeffs_train = (modes.T @ fluctuations).T

scaler_coeffs = MinMaxScaler().fit(coeffs_train)
coeffs_train_norm = scaler_coeffs.transform(coeffs_train)


# ======================================
# Define MLP
# ======================================
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        layers = []
        prev_size = input_size

        for h in hidden_layers:
            layers += [
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.3)
            ]
            prev_size = h

        layers.append(nn.Linear(prev_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(
    input_size=conditions.shape[1],
    output_size=K,
    hidden_layers=HIDDEN_LAYERS
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Prepare DataLoader
X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ======================================
# Train MLP
# ======================================
print("\nTraining MLP...")

for epoch in tqdm(range(EPOCHS), desc="Training MLP"):
    for bx, by in loader:
        optimizer.zero_grad()

        pred = model(bx)
        loss = criterion(pred, by)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss={loss.item():.6f}")


# ======================================
# Visualization Function
# ======================================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {
        'x': 0,
        'y': 1,
        'z': 2
    }

    axis_idx = axis_map[plane_type]
    unique_vals = np.sort(np.unique(coords[:, axis_idx]))

    closest_idx = np.argmin(np.abs(unique_vals - target_value))

    low = max(0, closest_idx - num_adjacent)
    high = min(len(unique_vals), closest_idx + num_adjacent + 1)

    sel_vals = unique_vals[low:high]
    mask = np.isin(coords[:, axis_idx], sel_vals)

    return mask


def visualize_results(true_temp, pred_temp, coords, case_index, vmax_shared=None):
    mask = select_adjacent_planes(
        coords,
        PLANE_TYPE,
        PLANE_VALUE,
        NUM_ADJACENT_PLANES
    )

    if not np.any(mask):
        return None

    points_to_interp = coords[mask]

    x = points_to_interp[:, 0]
    z = points_to_interp[:, 2]

    t_true = true_temp[mask]
    t_pred = pred_temp[mask]

    grid_x = np.linspace(x.min(), x.max(), 150)
    grid_z = np.linspace(z.min(), z.max(), 150)

    GX, GZ = np.meshgrid(grid_x, grid_z)

    Tt = griddata((x, z), t_true, (GX, GZ), method='cubic')
    Tp = griddata((x, z), t_pred, (GX, GZ), method='cubic')

    err = np.abs(Tt - Tp)

    vmax = vmax_shared if vmax_shared else np.nanmax(err)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Temperature Field Reconstruction - Case {case_index}',
        fontsize=16
    )

    plots_data = [
        (axs[0], Tt, "True Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[1], Tp, "Reconstructed Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[2], err, "Absolute Error", "YlOrRd", 0, vmax)
    ]

    for ax, data, title, cmap, vmin, vmax_plot in plots_data:
        if np.all(np.isnan(data)):
            continue

        cs = ax.contourf(
            GX,
            GZ,
            data,
            levels=50,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot
        )

        fig.colorbar(cs, ax=ax)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(SAVE_DIR, exist_ok=True)

    plt.savefig(
        os.path.join(SAVE_DIR, f"case_{case_index}_reconstruction.png"),
        dpi=200
    )

    plt.close(fig)

    return np.nanmax(err)


# ======================================
# Validation and Error Calculation
# ======================================
print("\nValidating on test set and calculating errors...")

model.eval()

all_true_val_temps = np.zeros((N_points, len(val_idx)))
all_pred_val_temps = np.zeros((N_points, len(val_idx)))

mae_list = []
rmse_list = []
rRMSE_list = []

global_error_max = 0

for i, idx in enumerate(tqdm(val_idx, desc="Validating Cases")):
    cond = torch.tensor(
        conditions_val_norm[i:i + 1],
        dtype=torch.float32
    )

    with torch.no_grad():
        pred_coeff_norm = model(cond).numpy()
        pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

    rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
    true_temp = snapshots_val[:, i]

    all_true_val_temps[:, i] = true_temp
    all_pred_val_temps[:, i] = rec_temp

    err_vector = true_temp - rec_temp

    mae = np.mean(np.abs(err_vector))
    mae_list.append(mae)

    rmse = np.sqrt(np.mean(err_vector ** 2))
    rmse_list.append(rmse)

    norm_true = np.linalg.norm(true_temp)
    rRMSE_case = np.linalg.norm(err_vector) / (norm_true + 1e-9)
    rRMSE_list.append(rRMSE_case)

    print(
        f"Case {idx + 1:2d}: "
        f"MAE={mae:.4f} °C, "
        f"RMSE={rmse:.4f} °C, "
        f"rRMSE={rRMSE_case:.4f}"
    )

    vmax_now = visualize_results(true_temp, rec_temp, coords, idx + 1)

    if vmax_now is not None:
        global_error_max = max(global_error_max, vmax_now)

np.save(os.path.join(SAVE_DIR, "global_error_max.npy"), global_error_max)

# ======================================================================
# Calculate rRMSE_n for 10 pre-selected high-temperature points
# ======================================================================
print("\nCalculating rRMSE_n for 10 pre-selected high-temperature points...")

# User-specified indices for rRMSE_n calculation
# These were identified as high-temperature points in a previous analysis.
pre_selected_indices = [
    113159, 118413, 118310, 113158, 118516,
    113056, 118207, 113262, 113055, 112953
]

rRMSE_n_results = {}

print("(Using a fixed list of grid point indices provided by the user)")

for point_idx in pre_selected_indices:
    if point_idx >= N_points:
        print(
            f"  Grid Point Index {point_idx} is out of bounds "
            f"(max is {N_points - 1}). Skipping."
        )
        continue

    true_series = all_true_val_temps[point_idx, :]
    pred_series = all_pred_val_temps[point_idx, :]

    norm_true_series = np.linalg.norm(true_series)

    rRMSE_n_value = (
        np.linalg.norm(true_series - pred_series)
        / (norm_true_series + 1e-9)
    )

    rRMSE_n_results[int(point_idx)] = float(rRMSE_n_value)

    avg_temp_at_point = mean_temp[point_idx, 0]

    print(
        f"  Grid Point Index {point_idx:<6} "
        f"(Avg T ≈ {avg_temp_at_point:.1f}°C): "
        f"rRMSE_n = {rRMSE_n_value:.4f}"
    )

# ======================================
# Results summary & plots
# ======================================
df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE (°C)": mae_list,
    "RMSE (°C)": rmse_list,
    "rRMSE": rRMSE_list
})

df_metrics.to_csv(
    os.path.join(SAVE_DIR, "per_case_metrics.csv"),
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))

plt.plot(
    df_metrics["Case"],
    df_metrics["MAE (°C)"],
    '-o',
    label="MAE (°C)"
)

plt.plot(
    df_metrics["Case"],
    df_metrics["RMSE (°C)"],
    '-s',
    label="RMSE (°C)"
)

plt.plot(
    df_metrics["Case"],
    df_metrics["rRMSE"],
    '-^',
    label="rRMSE (relative)"
)

plt.xlabel("Case Number")
plt.ylabel("Error Value")
plt.title("POD+MLP Per-Case Error Metrics")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "error_metrics_compare.png"),
    dpi=200
)

plt.close()

# ======================================
# Save metrics summary as JSON
# ======================================
summary = {
    "model_type": "POD_MLP",
    "energy_threshold": float(ENERGY_THRESHOLD),
    "selected_K": int(K),
    "dcr_count_mean_field": bool(DCR_COUNT_MEAN_FIELD),
    "dcr_count_mlp_weights": bool(DCR_COUNT_MLP_WEIGHTS),
    "dcr_current": {
        "energy_threshold": float(current_dcr_row["energy_threshold"]),
        "K": int(current_dcr_row["K"]),
        "DCR_train_pod": float(current_dcr_row["DCR_train_pod"]),
        "compression_percent_train_pod": float(current_dcr_row["compression_percent_train_pod"]),
        "DCR_all_projected_coeff": float(current_dcr_row["DCR_all_projected_coeff"]),
        "compression_percent_all_projected_coeff": float(
            current_dcr_row["compression_percent_all_projected_coeff"]
        ),
        "train_original_entries": int(current_dcr_row["train_original_entries"]),
        "train_compressed_entries": int(current_dcr_row["train_compressed_entries"]),
        "all_original_entries": int(current_dcr_row["all_original_entries"]),
        "all_compressed_entries": int(current_dcr_row["all_compressed_entries"])
    },
    "test_indices_one_based": [int(i + 1) for i in val_idx],
    "metrics_summary": {
        "mean_MAE": float(np.mean(mae_list)),
        "mean_RMSE": float(np.mean(rmse_list)),
        "mean_rRMSE": float(np.mean(rRMSE_list)),
        "MAE_each": [float(x) for x in mae_list],
        "RMSE_each": [float(x) for x in rmse_list],
        "rRMSE_each": [float(x) for x in rRMSE_list],
        "rRMSE_n_pre_selected_points": {
            str(k): float(v)
            for k, v in rRMSE_n_results.items()
        }
    },
    "mlp_params": {
        "hidden_layers": HIDDEN_LAYERS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    },
    "execution_time_seconds": float(time.time() - start_time)
}

with open(
    os.path.join(SAVE_DIR, "metrics_summary.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ======================================
# Print Summary
# ======================================
print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)

print(f"Execution time: {time.time() - start_time:.2f} s")
print(f"Results and plots saved to: {SAVE_DIR}")

print("\n--- POD Configuration ---")
print(f"Energy threshold: {ENERGY_THRESHOLD}")
print(f"Selected POD modes K: {K}")

print("\n--- DCR of Current POD Setting ---")
print(f"DCR_train_pod: {current_dcr_row['DCR_train_pod']:.6f}")
print(
    f"Compression saving train POD: "
    f"{current_dcr_row['compression_percent_train_pod']:.2f}%"
)

print(f"DCR_all_projected_coeff: {current_dcr_row['DCR_all_projected_coeff']:.6f}")
print(
    f"Compression saving all projected coeff: "
    f"{current_dcr_row['compression_percent_all_projected_coeff']:.2f}%"
)

print("\n--- Average Metrics Across All Test Cases ---")
print(f"Mean MAE  : {np.mean(mae_list):.4f} °C")
print(f"Mean RMSE : {np.mean(rmse_list):.4f} °C")
print(f"Mean rRMSE: {np.mean(rRMSE_list):.4f}")

print("\n--- rRMSE_n for Pre-selected High-Temperature Points ---")

if not rRMSE_n_results:
    print("  No rRMSE_n values were calculated. Please check the point indices.")
else:
    for idx, val in rRMSE_n_results.items():
        avg_temp_at_point = mean_temp[idx, 0]
        print(
            f"  Grid Point Index {idx:<6} "
            f"(Avg T ≈ {avg_temp_at_point:.1f}°C): "
            f"rRMSE_n = {val:.4f}"
        )

print("\n--- Saved Files ---")
print(f"per_case_metrics.csv")
print(f"metrics_summary.json")
print(f"dcr_current.csv")
print(f"dcr_sweep.csv")
print(f"dcr_vs_energy_threshold.csv")
print(f"dcr_vs_energy_threshold.png")
print(f"error_metrics_compare.png")
print("=" * 49)
