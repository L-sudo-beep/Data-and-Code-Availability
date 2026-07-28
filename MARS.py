import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import random
from scipy.interpolate import griddata
import time
from tqdm import tqdm

# —— 新增：MARS（pyearth）——
try:
    from pyearth import Earth
except Exception as e:
    raise ImportError(
        "需要安装 pyearth 才能使用 MARS 回归器。\n"
        "请先运行：pip install sklearn-contrib-py-earth\n"
        f"原始错误：{e}"
    )

# ======================================
# Configurable Parameters
# ======================================
SEED = 42
ENERGY_THRESHOLD = 0.99

# —— MARS 超参数（可按需调参/交叉验证）——
MARS_MAX_DEGREE = 2        # 基函数交互阶数（常用：1或2）
MARS_PENALTY = 1.0         # 复杂度惩罚（越大越稀疏）
MARS_ENABLE_PRUNING = True # 启用剪枝
MARS_MINSPAN_ALPHA = 0.0   # 可保持 0
MARS_ENDSPAN_ALPHA = 0.0   # 可保持 0

DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
PLANE_TYPE = 'y'
PLANE_VALUE = 1.2334
NUM_ADJACENT_PLANES = 2
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set2'

# ======================================
# Set Seeds for Reproducibility
# ======================================
random.seed(SEED)
np.random.seed(SEED)

# ======================================
# Load Data
# ======================================
start_time = time.time()
# Load boundary conditions
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]

# Normalize conditions
scaler_conditions = MinMaxScaler()
conditions_norm = scaler_conditions.fit_transform(conditions)

# 【固定测试集】（一基→零基）
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]

# 训练集索引 = 全集 \ 测试集
all_indices = list(range(num_samples))
train_idx = [i for i in all_indices if i not in val_idx]

print(f"Train samples: {len(train_idx)}, Indices: {train_idx}")
print(f"Validation (Test) samples: {len(val_idx)}, Indices: {val_idx}")

# Load coordinates and all snapshots
temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = coords.shape[0]

# 加载所有快照到矩阵 (N_points, N_cases)
all_snapshots = np.zeros((N_points, num_samples))
print("Loading all snapshots...")
for i in tqdm(range(num_samples)):
    df = pd.read_csv(temp_files[i])
    all_snapshots[:, i] = df['Temperature'].values

# 按索引划分训练/验证
snapshots_train = all_snapshots[:, train_idx]
snapshots_val   = all_snapshots[:, val_idx]

# ======================================
# POD Analysis on Training Data
# ======================================
print("\nPerforming POD analysis on training data...")
mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
fluctuations = snapshots_train - mean_temp
U, S, Vt = svd(fluctuations, full_matrices=False)

energy = np.cumsum(S ** 2) / np.sum(S ** 2)
K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"Selected K: {K} modes to retain {ENERGY_THRESHOLD * 100:.2f}% energy")

modes = U[:, :K]                       # (N_points, K)
coeffs_train = (modes.T @ fluctuations).T  # (N_train, K)

# 仅在训练集上拟合系数归一化，再用于测试集的反归一化
scaler_coeffs = MinMaxScaler()
coeffs_train_norm = scaler_coeffs.fit_transform(coeffs_train)

# 训练输入（已归一化）
conditions_train_norm = conditions_norm[train_idx, :]

# ======================================
# MARS Regressor（替换 SVR）
# ======================================
print("\nTraining MARS regressors (one Earth per POD coefficient)...")
t0 = time.time()
mars_models = []
for k in range(K):
    model_k = Earth(
        max_degree=MARS_MAX_DEGREE,
        penalty=MARS_PENALTY,
        enable_pruning=MARS_ENABLE_PRUNING,
        minspan_alpha=MARS_MINSPAN_ALPHA,
        endspan_alpha=MARS_ENDSPAN_ALPHA,
        verbose=0,
    )
    model_k.fit(conditions_train_norm, coeffs_train_norm[:, k])
    mars_models.append(model_k)
print(f"MARS training done in {time.time() - t0:.2f} s")

# ======================================
# 可视化/辅助函数
# ======================================
def visualize_results(true_temp, reconstructed_temp, coords, case_index, save_dir):
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

    true_interp = griddata((slice_coords_x, slice_coords_y), true_plane, (grid_X, grid_Y),
                           method='cubic', fill_value=np.nan)
    recon_interp = griddata((slice_coords_x, slice_coords_y), recon_plane, (grid_X, grid_Y),
                            method='cubic', fill_value=np.nan)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    temp_min = np.nanmin([true_interp, recon_interp])
    temp_max = np.nanmax([true_interp, recon_interp])
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

# ======================================
# Validation, Reconstruction, and Visualization
# ======================================
print("\nPerforming validation on the specified test set...")
all_errors = []

for i, original_idx in enumerate(val_idx):
    case_index_human = original_idx + 1
    print(f"\n--- Processing Validation Case {i + 1}/{len(val_idx)} (Original Index: {case_index_human}) ---")

    # 当前验证工况的条件（注意保持 2D 形状）
    condition_val_norm = conditions_norm[original_idx:original_idx + 1, :]
    true_temp = snapshots_val[:, i]

    # 预测归一化后的 POD 系数（逐维 MARS 预测并堆叠）
    pred_coeffs_norm_cols = []
    for k in range(K):
        pred_k = mars_models[k].predict(condition_val_norm)  # shape (1,)
        pred_coeffs_norm_cols.append(pred_k.reshape(-1, 1))
    pred_coeffs_norm = np.concatenate(pred_coeffs_norm_cols, axis=1)  # shape (1, K)

    # 逆归一化到系数原域
    pred_coeffs = scaler_coeffs.inverse_transform(pred_coeffs_norm).flatten()

    # 重构温度场：mean + Phi * a
    reconstructed = mean_temp.flatten() + modes @ pred_coeffs
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
