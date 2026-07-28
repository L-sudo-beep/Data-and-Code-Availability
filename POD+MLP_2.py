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

# ======================================
# Configurable Parameters
# ======================================
SEED = 42
ENERGY_THRESHOLD = 0.99
HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\Lenovo\Desktop\insert'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-1.26'

# --- 核心修改区域：现在可以自由选择 'x', 'y', 'z' ---
PLANE_TYPE = 'x'  # <--- 在这里修改: 'x', 'y', 或 'z'
PLANE_VALUE = 4.54  # <--- 在这里修改: 对应平面的坐标值
# --- 示例:
# PLANE_TYPE = 'x'
# PLANE_VALUE = 0.5
# PLANE_TYPE = 'z'
# PLANE_VALUE = 1.0

NUM_ADJACENT_PLANES = 0

# --- 用于计算特定平面rRMSE的参数 (这个功能保持不变) ---
PLANE_Y_FOR_RRMSE = 4.54

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
# Load Data
# ======================================
start_time = time.time()
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]

# Split indices (fixed test cases)
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]

# Fit scaler only on training set (no leakage)
scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

# Load coordinates and snapshots
temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
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
K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"Selected K={K} modes for {ENERGY_THRESHOLD * 100:.1f}% energy")

modes = U[:, :K]
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
            layers += [nn.Linear(prev_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


model = MLP(input_size=conditions.shape[1], output_size=K, hidden_layers=HIDDEN_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Prepare DataLoader
X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# ======================================
# Train MLP
# ======================================
print("\nTraining MLP...")
for epoch in tqdm(range(EPOCHS), desc="Training MLP"):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss={loss.item():.6f}")


# ======================================
# Visualization Function
# ======================================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if plane_type not in axis_map:
        raise ValueError("Invalid plane_type. Must be 'x', 'y', or 'z'.")
    axis_idx = axis_map[plane_type]

    unique_vals = np.sort(np.unique(coords[:, axis_idx]))
    if len(unique_vals) == 0:
        return np.zeros(coords.shape[0], dtype=bool)  # Return empty mask

    closest_idx = np.argmin(np.abs(unique_vals - target_value))
    low = max(0, closest_idx - num_adjacent)
    high = min(len(unique_vals), closest_idx + num_adjacent + 1)
    sel_vals = unique_vals[low:high]
    mask = np.isin(coords[:, axis_idx], sel_vals)
    return mask


def visualize_results(true_temp, pred_temp, coords, case_index, vmax_shared=None):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        print(
            f"Warning: No points found for plane {PLANE_TYPE}={PLANE_VALUE} in case {case_index}. Skipping visualization.")
        return None

    points_to_interp = coords[mask]
    t_true, t_pred = true_temp[mask], pred_temp[mask]

    # <--- 修改开始：根据PLANE_TYPE动态选择绘图轴 ---
    # 根据选择的平面类型，确定用于绘图的另外两个坐标轴
    if PLANE_TYPE == 'y':
        axis1_idx, axis2_idx = 0, 2  # Y平面，用X和Z坐标绘图
        axis1_label, axis2_label = "X (m)", "Z (m)"
    elif PLANE_TYPE == 'x':
        axis1_idx, axis2_idx = 1, 2  # X平面，用Y和Z坐标绘图
        axis1_label, axis2_label = "Y (m)", "Z (m)"
    elif PLANE_TYPE == 'z':
        axis1_idx, axis2_idx = 0, 1  # Z平面，用X和Y坐标绘图
        axis1_label, axis2_label = "X (m)", "Y (m)"
    else:
        raise ValueError("Invalid PLANE_TYPE in visualize_results. Must be 'x', 'y', or 'z'.")

    # 使用通用变量提取坐标
    axis1_coords = points_to_interp[:, axis1_idx]
    axis2_coords = points_to_interp[:, axis2_idx]

    # 创建网格
    grid_1, grid_2 = np.linspace(axis1_coords.min(), axis1_coords.max(), 150), np.linspace(axis2_coords.min(),
                                                                                           axis2_coords.max(), 150)
    G1, G2 = np.meshgrid(grid_1, grid_2)

    # 插值
    Tt = griddata((axis1_coords, axis2_coords), t_true, (G1, G2), method='cubic')
    Tp = griddata((axis1_coords, axis2_coords), t_pred, (G1, G2), method='cubic')
    # <--- 修改结束 ---

    err = np.abs(Tt - Tp)
    vmax = vmax_shared if vmax_shared else np.nanmax(err)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Temperature Field Reconstruction ({PLANE_TYPE.upper()}={PLANE_VALUE}m plane) - Case {case_index}',
                 fontsize=16)

    plots_data = [
        (axs[0], Tt, "True Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[1], Tp, "Reconstructed Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[2], err, "Absolute Error", "YlOrRd", 0, vmax)
    ]

    for ax, data, title, cmap, vmin, vmax_plot in plots_data:
        if np.all(np.isnan(data)): continue
        # <--- 修改：使用通用变量绘图和设置标签 ---
        cs = ax.contourf(G1, G2, data, levels=50, cmap=cmap, vmin=vmin, vmax=vmax_plot)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel(axis1_label)
        ax.set_ylabel(axis2_label)
        # <--- 修改结束 ---
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f"case_{case_index}_reconstruction_{PLANE_TYPE}{PLANE_VALUE}.png"), dpi=200)
    plt.close(fig)
    return np.nanmax(err)


# ======================================
# Validation and Error Calculation
# ======================================
print("\nValidating on test set and calculating errors...")
model.eval()

all_true_val_temps = np.zeros((N_points, len(val_idx)))
all_pred_val_temps = np.zeros((N_points, len(val_idx)))

mae_list, rmse_list, rRMSE_list = [], [], []
rRMSE_plane_list = []
global_error_max = 0

# (这部分计算 rRMSE_plane 的逻辑保持不变，仍然针对特定的Y平面)
unique_y_coords = np.unique(coords[:, 1])
closest_y_val = unique_y_coords[np.argmin(np.abs(unique_y_coords - PLANE_Y_FOR_RRMSE))]
plane_mask = (coords[:, 1] == closest_y_val)
num_plane_points = np.sum(plane_mask)

print(f"Target plane for rRMSE calculation: Y = {PLANE_Y_FOR_RRMSE:.4f} m")
if num_plane_points > 0:
    print(f"Found {num_plane_points} points on the closest available plane for rRMSE: Y = {closest_y_val:.4f} m")
else:
    print(f"Warning: No points found for the specified plane Y={PLANE_Y_FOR_RRMSE}. Cannot calculate plane rRMSE.")

for i, idx in enumerate(tqdm(val_idx, desc="Validating Cases")):
    cond = torch.tensor(conditions_val_norm[i:i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred_coeff_norm = model(cond).numpy()
        pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

    rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
    true_temp = snapshots_val[:, i]

    all_true_val_temps[:, i] = true_temp
    all_pred_val_temps[:, i] = rec_temp

    err_vector = true_temp - rec_temp

    # 1. 全局误差计算
    mae = np.mean(np.abs(err_vector))
    mae_list.append(mae)

    rmse = np.sqrt(np.mean(err_vector ** 2))
    rmse_list.append(rmse)

    norm_true = np.linalg.norm(true_temp)
    rRMSE_case = np.linalg.norm(err_vector) / (norm_true + 1e-9)
    rRMSE_list.append(rRMSE_case)

    # 2. 特定平面 (Y=...) 的rRMSE计算
    rRMSE_case_plane = np.nan
    if num_plane_points > 0:
        true_temp_plane = true_temp[plane_mask]
        rec_temp_plane = rec_temp[plane_mask]
        err_vector_plane = true_temp_plane - rec_temp_plane
        norm_true_plane = np.linalg.norm(true_temp_plane)
        if norm_true_plane > 1e-9:
            rRMSE_case_plane = np.linalg.norm(err_vector_plane) / norm_true_plane
    rRMSE_plane_list.append(rRMSE_case_plane)

    print(
        f"Case {idx + 1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, rRMSE={rRMSE_case:.4f}, rRMSE_plane(Y={closest_y_val:.2f})={rRMSE_case_plane:.4f}")

    vmax_now = visualize_results(true_temp, rec_temp, coords, idx + 1)
    if vmax_now is not None:
        global_error_max = max(global_error_max, vmax_now)

np.save(os.path.join(SAVE_DIR, "global_error_max.npy"), global_error_max)

# ======================================================================
# Calculate rRMSE_n for 10 pre-selected high-temperature points.
# ======================================================================
print("\nCalculating rRMSE_n for 10 pre-selected high-temperature points...")

pre_selected_indices = [
    113159, 118413, 118310, 113158, 118516,
    113056, 118207, 113262, 113055, 112953
]

rRMSE_n_results = {}
print("(Using a fixed list of grid point indices provided by the user)")
for point_idx in pre_selected_indices:
    if point_idx >= N_points:
        print(f"  Grid Point Index {point_idx} is out of bounds (max is {N_points - 1}). Skipping.")
        continue
    true_series = all_true_val_temps[point_idx, :]
    pred_series = all_pred_val_temps[point_idx, :]
    norm_true_series = np.linalg.norm(true_series)
    rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (norm_true_series + 1e-9)
    rRMSE_n_results[point_idx] = rRMSE_n_value
    avg_temp_at_point = mean_temp[point_idx, 0]
    print(f"  Grid Point Index {point_idx:<6} (Avg T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {rRMSE_n_value:.4f}")

# ======================================
# Results summary & plots
# ======================================
df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE (°C)": mae_list,
    "RMSE (°C)": rmse_list,
    "rRMSE": rRMSE_list,
    f"rRMSE_plane_Y={closest_y_val:.2f}m": rRMSE_plane_list
})
df_metrics.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False)

plt.figure(figsize=(12, 7))
plt.plot(df_metrics["Case"], df_metrics["MAE (°C)"], '-o', label="MAE (°C) [Full Field]")
plt.plot(df_metrics["Case"], df_metrics["RMSE (°C)"], '-s', label="RMSE (°C) [Full Field]")
plt.plot(df_metrics["Case"], df_metrics["rRMSE"], '-^', label="rRMSE [Full Field]")
plt.plot(df_metrics["Case"], df_metrics[f"rRMSE_plane_Y={closest_y_val:.2f}m"], '-x', color='purple',
         label=f"rRMSE [Plane Y={closest_y_val:.2f}m]")
plt.xlabel("Case Number")
plt.ylabel("Error Value")
plt.title("POD+MLP Per-Case Error Metrics (Full Field vs. Specific Plane)")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "error_metrics_compare.png"), dpi=200)
plt.close()

print("\n" + "=" * 20 + " SUMMARY " + "=" * 20)
print(f"Execution time: {time.time() - start_time:.2f} s")
print(f"Results and plots saved to: {SAVE_DIR}")
print("\n--- Average Metrics Across All Test Cases ---")
print(f"Mean MAE            : {np.mean(mae_list):.4f} °C")
print(f"Mean RMSE           : {np.mean(rmse_list):.4f} °C")
print(f"Mean rRMSE (Full)   : {np.mean(rRMSE_list):.4f}")
print(f"Mean rRMSE (Plane Y={closest_y_val:.2f}m): {np.nanmean(rRMSE_plane_list):.4f}")

print("\n--- rRMSE_n for Pre-selected High-Temperature Points ---")
if not rRMSE_n_results:
    print("  No rRMSE_n values were calculated (check indices).")
else:
    for idx, val in rRMSE_n_results.items():
        avg_temp_at_point = mean_temp[idx, 0]
        print(f"  Grid Point Index {idx:<6} (Avg T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {val:.4f}")
print("=" * 49)

