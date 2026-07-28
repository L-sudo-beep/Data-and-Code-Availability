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
ENERGY_THRESHOLD = 0.999
HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
PLANE_TYPE = 'y'
PLANE_VALUE = 1.2334
NUM_ADJACENT_PLANES = 2
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-1'

# ======================================
# Set Seeds for Reproducibility
# ======================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================
# Load Data
# ======================================
start_time = time.time()

# Load boundary conditions
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values  # shape: (num_samples, 14)
num_samples = conditions.shape[0]

# 【FIXED-1】先划分索引，再用训练集拟合条件缩放器，避免信息泄露
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]

all_indices = list(range(num_samples))
train_idx = [i for i in all_indices if i not in val_idx]

print(f"Train samples: {len(train_idx)}, Indices: {train_idx}")
print(f"Validation (Test) samples: {len(val_idx)}, Indices: {val_idx}")

# 【FIXED-2】只用训练集拟合 MinMaxScaler，再分别 transform 训练/验证
scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

# Load coordinates and all snapshots
temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = coords.shape[0]

# 加载所有快照到一个大矩阵中
all_snapshots = np.zeros((N_points, num_samples))
print("Loading all snapshots...")
for i in tqdm(range(num_samples)):
    df = pd.read_csv(temp_files[i])
    all_snapshots[:, i] = df['Temperature'].values

# 根据索引划分训练集和验证集快照（无泄露）
snapshots_train = all_snapshots[:, train_idx]  # shape: (N_points, N_train)
snapshots_val = all_snapshots[:, val_idx]      # shape: (N_points, N_val)

# ======================================
# POD Analysis on Training Data (no leakage)
# ======================================
print("\nPerforming POD analysis on training data...")
mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
fluctuations = snapshots_train - mean_temp
U, S, Vt = svd(fluctuations, full_matrices=False)

energy = np.cumsum(S ** 2) / np.sum(S ** 2)
K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"Selected K: {K} modes to retain {ENERGY_THRESHOLD * 100:.2f}% energy")

modes = U[:, :K]  # spatial modes Φ
# 训练集 POD 系数（未归一化，目标值）
coeffs_train = (modes.T @ fluctuations).T  # shape: (N_train, K)

# 【无泄露】只在训练系数上拟合 scaler
scaler_coeffs = MinMaxScaler().fit(coeffs_train)
coeffs_train_norm = scaler_coeffs.transform(coeffs_train)

# ======================================
# Neural Network Definition (MLP)
# ======================================
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Instantiate model
model = MLP(input_size=conditions.shape[1], output_size=K, hidden_layers=HIDDEN_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Prepare DataLoader
X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================================
# Train Neural Network
# ======================================
print("\nTraining Neural Network...")
model.train()
for epoch in range(EPOCHS):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ======================================
# Visualization helpers
# ======================================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent, case_index):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]
    unique_values = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_values - target_value))
    start_idx = max(0, closest_idx - num_adjacent)
    end_idx = min(len(unique_values), closest_idx + num_adjacent + 1)
    selected_planes = unique_values[start_idx:end_idx]
    mask = np.isin(coords[:, axis_idx], selected_planes)
    return mask, selected_planes

def visualize_results(true_temp, reconstructed_temp, coords, case_index, save_dir):
    # Plane selection logic
    mask, selected_planes = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES, case_index)
    if not mask.any():
        print(f"Case {case_index}: No points found on selected planes. Skipping visualization.")
        return

    axis_map = {'x': (1, 2, 'Y (m)', 'Z (m)'), 'y': (0, 2, 'X (m)', 'Z (m)'), 'z': (0, 1, 'X (m)', 'Y (m)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axis_map[PLANE_TYPE]
    slice_coords_x, slice_coords_y = coords[mask, ax1_idx], coords[mask, ax2_idx]
    true_plane, recon_plane = true_temp[mask], reconstructed_temp[mask]

    grid_x = np.linspace(slice_coords_x.min(), slice_coords_x.max(), 150)
    grid_y = np.linspace(slice_coords_y.min(), slice_coords_y.max(), 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    true_interp = griddata((slice_coords_x, slice_coords_y), true_plane, (grid_X, grid_Y), method='cubic',
                           fill_value=np.nan)
    recon_interp = griddata((slice_coords_x, slice_coords_y), recon_plane, (grid_X, grid_Y), method='cubic',
                            fill_value=np.nan)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    temp_min = min(np.nanmin(true_interp), np.nanmin(recon_interp))
    temp_max = max(np.nanmax(true_interp), np.nanmax(recon_interp))
    levels = np.linspace(temp_min, temp_max, 50)

    error_field = np.abs(true_interp - recon_interp)

    cs1 = axs[0].contourf(grid_X, grid_Y, true_interp, levels=levels, cmap='jet')
    axs[0].set_title(f'True Temperature (Case {case_index})')
    fig.colorbar(cs1, ax=axs[0])

    cs2 = axs[1].contourf(grid_X, grid_Y, recon_interp, levels=levels, cmap='jet')
    axs[1].set_title('Reconstructed Temperature')
    fig.colorbar(cs2, ax=axs[1])

    cs3 = axs[2].contourf(grid_X, grid_Y, error_field, levels=50, cmap='YlOrRd')
    axs[2].set_title('Absolute Error')
    fig.colorbar(cs3, ax=axs[2])

    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'case_{case_index}_reconstruction.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization for Case {case_index} to: {save_path}")

# ======================================
# Validation, Reconstruction, and Visualization
# ======================================
print("\nPerforming validation on the specified test set...")
model.eval()
all_errors = []

# 逐个验证样本
for i, original_idx in enumerate(val_idx):
    case_index_human = original_idx + 1
    print(f"\n--- Processing Validation Case {i + 1}/{len(val_idx)} (Original Index: {case_index_human}) ---")

    # 【FIXED-3】验证条件使用“只在训练集上拟合”的 scaler 变换后的切片
    cond_val_vec = conditions_val_norm[i:i+1]  # 对应 val_idx 中的第 i 个
    condition_val_norm_tensor = torch.tensor(cond_val_vec, dtype=torch.float32)

    true_temp = snapshots_val[:, i]  # 与 cond_val_vec 同一验证样本

    # 预测归一化 POD 系数
    with torch.no_grad():
        pred_coeffs_norm = model(condition_val_norm_tensor).numpy()

    # 逆归一化回到真实尺度
    pred_coeffs = scaler_coeffs.inverse_transform(pred_coeffs_norm).flatten()

    # 重构温度场： T_hat = mean + Φ a_pred
    reconstructed = mean_temp.flatten() + modes @ pred_coeffs

    # 记录误差
    all_errors.append(true_temp - reconstructed)

    # 可视化
    visualize_results(true_temp, reconstructed, coords, case_index_human, SAVE_DIR)

# ======================================
# Final Performance Evaluation over the entire test set
# ======================================
all_errors = np.array(all_errors).flatten()
mae = np.mean(np.abs(all_errors))
rmse = np.sqrt(np.mean(all_errors ** 2))

print("\n" + "=" * 50)
print("  Final Performance on the Entire Test Set")
print("=" * 50)
print(f"  Test Set Indices (Human Readable): {val_indices_human}")
print(f"  Mean Absolute Error (MAE): {mae:.4f} °C")
print(f"  Root Mean Square Error (RMSE): {rmse:.4f} °C")
print("=" * 50)

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
