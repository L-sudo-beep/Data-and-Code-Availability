import pandas as pd
import numpy as np
from scipy.linalg import svd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import random

# ======================================
# Configurable Parameters
# ======================================
SEED = 42  # Fixed seed for reproducibility across all random operations
ENERGY_THRESHOLD = 0.99  # Cumulative energy threshold to determine K (e.g., 0.9999 for 99.99%)
HIDDEN_LAYERS = [128, 64]  # List of hidden layer sizes for the MLP; modifiable for different architectures
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
EPOCHS = 2000  # Number of training epochs
BATCH_SIZE = 16  # Batch size for training
DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'  # Directory containing temperature CSV files (1.csv to 89.csv)
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'  # Path to boundary conditions CSV
TRAIN_SPLIT = 0.9  # Fraction of data for training (90%)
PLANE_TYPE = 'y'  # Plane type for visualization: 'x', 'y', or 'z'
PLANE_VALUE = 1.2334  # Target value for the plane slice (in meters)
NUM_ADJACENT_PLANES = 2  # Number of adjacent planes to include on each side (新增参数)
VAL_INDEX = 3  # Index in validation set to visualize (0 to len(val)-1)
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89'  # Directory to save validation images

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
# Load boundary conditions: (89 rows, 14 columns)
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t')
bc_df = bc_df.astype(float)  # Force conversion to float to ensure numerical types
conditions = bc_df.values  # numpy array (89, 14)
num_samples = conditions.shape[0]
assert num_samples == 89, "Boundary conditions should have 89 rows"

# Normalize conditions
scaler_conditions = MinMaxScaler()
conditions = scaler_conditions.fit_transform(conditions)

# List temperature files: 1.csv to 89.csv
temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]

# Split indices into train and val (random split with fixed seed)
indices = list(range(num_samples))
train_idx, val_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)
print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# Load coordinates from the first file (assumed same for all)
first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values  # (N_points, 3)
N_points = coords.shape[0]

# Print unique X, Y, Z values for debugging plane selection
print("Unique X values:", np.sort(np.unique(coords[:, 0])))
print("Unique Y values:", np.sort(np.unique(coords[:, 1])))
print("Unique Z values:", np.sort(np.unique(coords[:, 2])))

# Load training snapshots: temperature fields as columns
snapshots_train = np.zeros((N_points, len(train_idx)))
for i, idx in enumerate(train_idx):
    df = pd.read_csv(temp_files[idx])
    snapshots_train[:, i] = df['Temperature'].values

# ======================================
# POD Analysis on Training Data
# ======================================
# Compute mean temperature field
mean_temp = np.mean(snapshots_train, axis=1)[:, np.newaxis]

# Subtract mean to get fluctuations
fluctuations = snapshots_train - mean_temp

# Perform SVD
U, S, Vt = svd(fluctuations, full_matrices=False)

# Compute cumulative energy
energy = np.cumsum(S ** 2) / np.sum(S ** 2)

# Determine K based on energy threshold
K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"Selected K: {K} modes to retain {ENERGY_THRESHOLD * 100:.2f}% energy")

# POD modes (basis functions)
modes = U[:, :K]  # (N_points, K)

# Compute POD coefficients for training data
coeffs_train = modes.T @ fluctuations  # (K, len(train_idx))

# Transpose for NN: (len(train_idx), K)
coeffs_train = coeffs_train.T

# Normalize coefficients
scaler_coeffs = MinMaxScaler()
coeffs_train = scaler_coeffs.fit_transform(coeffs_train)

# Training conditions
conditions_train = conditions[train_idx, :]


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
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Instantiate model
input_size = 14  # Condition vector dimension
output_size = K  # Number of POD coefficients
model = MLP(input_size, output_size, HIDDEN_LAYERS)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Prepare DataLoader
X_train = torch.tensor(conditions_train, dtype=torch.float32)
y_train = torch.tensor(coeffs_train, dtype=torch.float32)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================================
# Train Neural Network
# ======================================
model.train()
for epoch in range(EPOCHS):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

# ======================================
# Validation and Reconstruction
# ======================================
# Load validation snapshots for reference
snapshots_val = np.zeros((N_points, len(val_idx)))
for i, idx in enumerate(val_idx):
    df = pd.read_csv(temp_files[idx])
    snapshots_val[:, i] = df['Temperature'].values

conditions_val = conditions[val_idx, :]

# Select validation sample for visualization
val_i = VAL_INDEX % len(val_idx)
condition_val = torch.tensor(conditions_val[val_i:val_i + 1], dtype=torch.float32)
true_temp = snapshots_val[:, val_i]

# Predict coefficients (normalized)
model.eval()
with torch.no_grad():
    pred_coeffs_norm = model(condition_val).numpy()

# Denormalize predicted coefficients
pred_coeffs = scaler_coeffs.inverse_transform(pred_coeffs_norm).flatten()  # (K,)

# Reconstruct temperature field
reconstructed = mean_temp.flatten() + modes @ pred_coeffs

# Compute MSE error
mse = np.mean((true_temp - reconstructed) ** 2)
print(f"Validation MSE for sample {val_i} (original index {val_idx[val_i]}): {mse:.6f}")


# ======================================
# 改进的可视化：选择多个相邻平面
# ======================================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    """
    选择目标平面及其相邻的多个平面上的点

    Args:
        coords: 坐标数组 (N, 3)
        plane_type: 'x', 'y', or 'z'
        target_value: 目标平面的坐标值
        num_adjacent: 每一侧要包含的相邻平面数量

    Returns:
        mask: 布尔数组，标识选中的点
        selected_planes: 实际选中的平面坐标值列表
    """
    # 确定坐标轴索引
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]

    # 获取该轴上的所有唯一值并排序
    unique_values = np.sort(np.unique(coords[:, axis_idx]))

    # 找到最接近目标值的索引
    closest_idx = np.argmin(np.abs(unique_values - target_value))

    # 确定要选择的平面范围
    start_idx = max(0, closest_idx - num_adjacent)
    end_idx = min(len(unique_values), closest_idx + num_adjacent + 1)

    selected_planes = unique_values[start_idx:end_idx]
    print(f"Selected {len(selected_planes)} planes for {plane_type}-axis:")
    print(f"Target value: {target_value}")
    print(f"Selected plane values: {selected_planes}")

    # 创建掩码
    mask = np.zeros(coords.shape[0], dtype=bool)
    for plane_val in selected_planes:
        plane_mask = np.abs(coords[:, axis_idx] - plane_val) < 1e-10  # 很小的容差
        mask |= plane_mask

    return mask, selected_planes


# 使用改进的平面选择
mask, selected_planes = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)

# 获取其他两个轴的坐标用于可视化
if PLANE_TYPE == 'x':
    slice_coords_x = coords[mask, 1]  # Y
    slice_coords_y = coords[mask, 2]  # Z
    xlabel = 'Y (m)'
    ylabel = 'Z (m)'
elif PLANE_TYPE == 'y':
    slice_coords_x = coords[mask, 0]  # X
    slice_coords_y = coords[mask, 2]  # Z
    xlabel = 'X (m)'
    ylabel = 'Z (m)'
else:  # 'z'
    slice_coords_x = coords[mask, 0]  # X
    slice_coords_y = coords[mask, 1]  # Y
    xlabel = 'X (m)'
    ylabel = 'Y (m)'

true_plane = true_temp[mask]
recon_plane = reconstructed[mask]

num_points_on_planes = mask.sum()
print(f"Number of points on selected planes: {num_points_on_planes}")

if num_points_on_planes == 0:
    print(f"No points found on the selected planes. Check your data.")
else:
    # 尝试创建更密集的网格进行插值可视化
    from scipy.interpolate import griddata

    # 定义插值网格的分辨率
    grid_resolution = 100

    # 创建插值网格
    x_min, x_max = slice_coords_x.min(), slice_coords_x.max()
    y_min, y_max = slice_coords_y.min(), slice_coords_y.max()

    grid_x = np.linspace(x_min, x_max, grid_resolution)
    grid_y = np.linspace(y_min, y_max, grid_resolution)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # 使用scipy的griddata进行插值
    try:
        true_interp = griddata(
            (slice_coords_x, slice_coords_y), true_plane,
            (grid_X, grid_Y), method='linear', fill_value=np.nan
        )
        recon_interp = griddata(
            (slice_coords_x, slice_coords_y), recon_plane,
            (grid_X, grid_Y), method='linear', fill_value=np.nan
        )

        # 统一颜色范围 - 找到全局最小值和最大值
        temp_min = min(true_plane.min(), recon_plane.min())
        temp_max = max(true_plane.max(), recon_plane.max())
        print(f"Unified color range: [{temp_min:.3f}, {temp_max:.3f}]")

        # 为等高线图定义统一的levels
        levels = np.linspace(temp_min, temp_max, 50)

        # 计算误差场用于第四个子图
        error_field = true_interp - recon_interp
        error_min = np.nanmin(error_field)
        error_max = np.nanmax(error_field)
        error_abs_max = max(abs(error_min), abs(error_max))
        error_levels = np.linspace(-error_abs_max, error_abs_max, 50)

        # 创建插值后的等高线图 (增加误差图)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

        # 原始散点图 - 使用统一的vmin和vmax
        sc1 = axs[0].scatter(slice_coords_x, slice_coords_y, c=true_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f'True Temperature (Scatter)\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.colorbar(sc1, ax=axs[0], shrink=0.8)

        # 插值后的真实温度场 - 使用统一的levels和vmin/vmax
        cs1 = axs[1].contourf(grid_X, grid_Y, true_interp, levels=levels,
                              cmap='jet', vmin=temp_min, vmax=temp_max)
        axs[1].set_title('True Temperature (Interpolated)')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        fig.colorbar(cs1, ax=axs[1], shrink=0.8)

        # 插值后的重构温度场 - 使用统一的levels和vmin/vmax
        cs2 = axs[2].contourf(grid_X, grid_Y, recon_interp, levels=levels,
                              cmap='jet', vmin=temp_min, vmax=temp_max)
        axs[2].set_title('Reconstructed Temperature (Interpolated)')
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)
        fig.colorbar(cs2, ax=axs[2], shrink=0.8)

        # 误差分布图
        cs3 = axs[3].contourf(grid_X, grid_Y, error_field, levels=error_levels,
                              cmap='RdBu_r', vmin=-error_abs_max, vmax=error_abs_max)
        axs[3].set_title(f'Error Field (True - Reconstructed)\nRange: [{error_min:.4f}, {error_max:.4f}]')
        axs[3].set_xlabel(xlabel)
        axs[3].set_ylabel(ylabel)
        fig.colorbar(cs3, ax=axs[3], shrink=0.8)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR,
                                 f'validation_interpolated_{PLANE_TYPE}_{PLANE_VALUE}_val{val_i}_adj{NUM_ADJACENT_PLANES}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpolated visualization to: {save_path}")
        plt.show()

    except Exception as e:
        print(f"Interpolation failed: {e}")
        print("Falling back to scatter plot...")

        # 备用：散点图 (2x2布局，包含误差图)
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        # 统一颜色范围
        temp_min = min(true_plane.min(), recon_plane.min())
        temp_max = max(true_plane.max(), recon_plane.max())
        print(f"Unified color range for scatter plot: [{temp_min:.3f}, {temp_max:.3f}]")

        # 计算误差
        error_plane = true_plane - recon_plane
        error_abs_max = max(abs(error_plane.min()), abs(error_plane.max()))

        sc1 = axs[0].scatter(slice_coords_x, slice_coords_y, c=true_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f'True Temperature\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.colorbar(sc1, ax=axs[0], shrink=0.8)

        sc2 = axs[1].scatter(slice_coords_x, slice_coords_y, c=recon_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[1].set_title(f'Reconstructed Temperature\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        fig.colorbar(sc2, ax=axs[1], shrink=0.8)

        # 误差散点图
        sc3 = axs[2].scatter(slice_coords_x, slice_coords_y, c=error_plane,
                             cmap='RdBu_r', s=10, vmin=-error_abs_max, vmax=error_abs_max)
        axs[2].set_title(f'Error Field (True - Reconstructed)')
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)
        fig.colorbar(sc3, ax=axs[2], shrink=0.8)

        # 误差直方图
        axs[3].hist(error_plane, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[3].set_title(f'Error Distribution\nMean: {error_plane.mean():.4f}, Std: {error_plane.std():.4f}')
        axs[3].set_xlabel('Temperature Error')
        axs[3].set_ylabel('Frequency')
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR,
                                 f'validation_scatter_{PLANE_TYPE}_{PLANE_VALUE}_val{val_i}_adj{NUM_ADJACENT_PLANES}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter visualization to: {save_path}")
        plt.show()

# 计算并显示平面上的误差统计
plane_mse = np.mean((true_plane - recon_plane) ** 2)
plane_mae = np.mean(np.abs(true_plane - recon_plane))
plane_max_error = np.max(np.abs(true_plane - recon_plane))

print(f"\n=== Error Statistics on Selected Planes ===")
print(f"MSE: {plane_mse:.6f}")
print(f"MAE: {plane_mae:.6f}")
print(f"Max Absolute Error: {plane_max_error:.6f}")
print(f"Temperature range - True: [{true_plane.min():.3f}, {true_plane.max():.3f}]")
print(f"Temperature range - Reconstructed: [{recon_plane.min():.3f}, {recon_plane.max():.3f}]")

# Optional: Save model, scalers, modes, mean for future use
# torch.save(model.state_dict(), 'pod_nn_model.pth')
# import joblib
# joblib.dump(scaler_conditions, 'scaler_conditions.pkl')
# joblib.dump(scaler_coeffs, 'scaler_coeffs.pkl')
# np.save('modes.npy', modes)
# np.save('mean_temp.npy', mean_temp)

print("\nTask complete. Modify parameters as needed for further runs.")