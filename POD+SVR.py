import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
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

# —— SVR 超参数（可按需调整/做网格搜索）——
SVR_KERNEL = 'rbf'   # 'rbf' | 'linear' | 'poly' | 'sigmoid'
SVR_C = 10.0
SVR_EPSILON = 1e-3
SVR_GAMMA = 'scale'  # 'scale' | 'auto' 或者给浮点数
SVR_CACHE_MB = 500   # 加大缓存提速（视内存情况而定）

DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
PLANE_TYPE = 'y'
PLANE_VALUE = 1.2334
NUM_ADJACENT_PLANES = 2
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set-SVR'

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

# 【固定测试集】（人类索引）
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]

# 训练集索引 = 全集 \ 测试集
all_indices = list(range(num_samples))
train_idx = [i for i in all_indices if i not in val_idx]

print(f"Train samples: {len(train_idx)}, Indices: {train_idx}")
print(f"Validation (Test) samples: {len(val_idx)}, Indices: {val_idx}")

# —— 严格防止泄露：仅用训练集拟合输入归一化器 ——
scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm   = scaler_conditions.transform(conditions[val_idx])

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

modes = U[:, :K]                         # (N_points, K)
coeffs_train = (modes.T @ fluctuations).T  # (N_train, K)

# 仅在训练集上拟合系数归一化，再用于测试集的反归一化（避免泄露）
scaler_coeffs = MinMaxScaler().fit(coeffs_train)
coeffs_train_norm = scaler_coeffs.transform(coeffs_train)

# ======================================
# SVR Regressor
# ======================================
# 使用 MultiOutputRegressor 包装多个 SVR，一次学 K 个系数
svr_base = SVR(
    kernel=SVR_KERNEL,
    C=SVR_C,
    epsilon=SVR_EPSILON,
    gamma=SVR_GAMMA,
    cache_size=SVR_CACHE_MB
)
model = MultiOutputRegressor(svr_base, n_jobs=None)

print("\nTraining SVR (MultiOutput)...")
t0 = time.time()
model.fit(conditions_train_norm, coeffs_train_norm)
print(f"SVR training done in {time.time() - t0:.2f} s")

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
# Validation, Reconstruction, and Per-Case Metrics
# ======================================
print("\nPerforming validation on the specified test set...")
all_errors = []               # for overall MAE/RMSE
per_case_mae = []
per_case_rmse = []
per_case_idx_human = []

for i, original_idx in enumerate(val_idx):
    case_index_human = original_idx + 1
    print(f"\n--- Processing Validation Case {i + 1}/{len(val_idx)} (Original Index: {case_index_human}) ---")

    # 验证工况参数（已用仅在训练集上拟合的 scaler 变换）
    cond_val_vec = conditions_val_norm[i:i+1, :]     # shape (1, 14)
    true_temp = snapshots_val[:, i]                  # shape (N_points,)

    # 预测归一化后的 POD 系数
    pred_coeffs_norm = model.predict(cond_val_vec)   # shape (1, K)

    # 逆归一化到系数原域
    pred_coeffs = scaler_coeffs.inverse_transform(pred_coeffs_norm).flatten()

    # 重构温度场：mean + Phi * a
    reconstructed = mean_temp.flatten() + modes @ pred_coeffs

    # ---- 逐工况误差（全场）----
    err = true_temp - reconstructed
    mae_i = np.mean(np.abs(err))
    rmse_i = np.sqrt(np.mean(err**2))
    per_case_mae.append(mae_i)
    per_case_rmse.append(rmse_i)
    per_case_idx_human.append(case_index_human)

    # 收集整体误差（用于整体 MAE/RMSE）
    all_errors.append(err)

    # 可视化
    visualize_results(true_temp, reconstructed, coords, case_index_human, SAVE_DIR)

# ======================================
# Final Performance Evaluation over the entire test set
# ======================================
all_errors = np.concatenate(all_errors)  # 变为一维向量
mae = np.mean(np.abs(all_errors))
rmse = np.sqrt(np.mean(all_errors ** 2))

print("\n" + "=" * 50)
print("  Final Performance on the Entire Test Set")
print("=" * 50)
print(f"  Test Set Indices (Human Readable): {val_indices_human}")
print(f"  Mean Absolute Error (MAE): {mae:.4f} °C")
print(f"  Root Mean Square Error (RMSE): {rmse:.4f} °C")
print("=" * 50)

# —— 打印逐工况 MAE/RMSE ——
print("\nPer-Case Metrics (Full 3D):")
for idx_h, m, r in zip(per_case_idx_human, per_case_mae, per_case_rmse):
    print(f"  Case {idx_h:>2d}  |  MAE={m:.4f} °C  |  RMSE={r:.4f} °C")

# —— 保存逐工况指标到 CSV ——
metrics_df = pd.DataFrame({
    "case_index_human": per_case_idx_human,
    "MAE": per_case_mae,
    "RMSE": per_case_rmse
})
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
csv_path = os.path.join(SAVE_DIR, "pod_svr_metrics_by_case.csv")
metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\nSaved per-case metrics to: {csv_path}")

end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")
