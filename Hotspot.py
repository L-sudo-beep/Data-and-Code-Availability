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

SEED = 42
ENERGY_THRESHOLD = 0.99
HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\Lenovo\Desktop\insert'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
PLANE_TYPE = 'y'
PLANE_VALUE = 1.26
NUM_ADJACENT_PLANES = 0
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-1.26'


RELATIVE_ERROR_THRESHOLD = 0.10


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


start_time = time.time()
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]


val_indices_human = [1, 2, 3, 4, 5, 6, 7, 8, 9]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]


scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])


temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = coords.shape[0]

print("正在加载所有快照...")
all_snapshots = np.zeros((N_points, num_samples))
for i in tqdm(range(num_samples), desc="加载快照"):
    df = pd.read_csv(temp_files[i])
    all_snapshots[:, i] = df['Temperature'].values

snapshots_train = all_snapshots[:, train_idx]
snapshots_val = all_snapshots[:, val_idx]


print("\n正在进行POD分析...")
mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
fluctuations = snapshots_train - mean_temp
U, S, Vt = svd(fluctuations, full_matrices=False)
energy = np.cumsum(S ** 2) / np.sum(S ** 2)
K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"为达到 {ENERGY_THRESHOLD * 100:.1f}% 的能量，选择 K={K} 个模态")

modes = U[:, :K]
coeffs_train = (modes.T @ fluctuations).T
scaler_coeffs = MinMaxScaler().fit(coeffs_train)
coeffs_train_norm = scaler_coeffs.transform(coeffs_train)



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


X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)


print("\n正在训练MLP...")
for epoch in tqdm(range(EPOCHS), desc="训练 MLP"):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"轮次 {epoch + 1}/{EPOCHS} | 损失={loss.item():.6f}")



def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]
    unique_vals = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_vals - target_value))
    low = max(0, closest_idx - num_adjacent)
    high = min(len(unique_vals), closest_idx + num_adjacent + 1)
    sel_vals = unique_vals[low:high]
    mask = np.isin(coords[:, axis_idx], sel_vals)
    return mask


def visualize_results(true_temp, pred_temp, coords, case_index, vmax_shared=None):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask): return None

    points_to_interp = coords[mask]
    x, z = points_to_interp[:, 0], points_to_interp[:, 2]
    t_true, t_pred = true_temp[mask], pred_temp[mask]

    grid_x, grid_z = np.linspace(x.min(), x.max(), 150), np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z)

    Tt = griddata((x, z), t_true, (GX, GZ), method='cubic')
    Tp = griddata((x, z), t_pred, (GX, GZ), method='cubic')
    err = np.abs(Tt - Tp)

    vmax = vmax_shared if vmax_shared else np.nanmax(err)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'温度场重构 - 案例 {case_index}', fontsize=16)

    plots_data = [
        (axs[0], Tt, "真实温度场", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[1], Tp, "重构温度场", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[2], err, "绝对误差", "YlOrRd", 0, vmax)
    ]

    for ax, data, title, cmap, vmin, vmax_plot in plots_data:
        if np.all(np.isnan(data)): continue
        cs = ax.contourf(GX, GZ, data, levels=50, cmap=cmap, vmin=vmin, vmax=vmax_plot)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f"case_{case_index}_reconstruction.png"), dpi=200)
    plt.close(fig)
    return np.nanmax(err)



print("\n在测试集上进行验证并计算误差...")
model.eval()

all_true_val_temps = np.zeros((N_points, len(val_idx)))
all_pred_val_temps = np.zeros((N_points, len(val_idx)))

mae_list, rmse_list, rRMSE_list = [], [], []
percentage_below_threshold_list = []
global_error_max = 0

for i, idx in enumerate(tqdm(val_idx, desc="验证案例")):
    cond = torch.tensor(conditions_val_norm[i:i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred_coeff_norm = model(cond).numpy()
        pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

    rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
    true_temp = snapshots_val[:, i]

    all_true_val_temps[:, i] = true_temp
    all_pred_val_temps[:, i] = rec_temp

    err_vector = true_temp - rec_temp

    # --- 计算当前案例的各项指标 ---
    mae = np.mean(np.abs(err_vector))
    mae_list.append(mae)

    rmse = np.sqrt(np.mean(err_vector ** 2))
    rmse_list.append(rmse)

    norm_true = np.linalg.norm(true_temp)
    rRMSE_case = np.linalg.norm(err_vector) / (norm_true + 1e-9)
    rRMSE_list.append(rRMSE_case)

    relative_error = np.abs(err_vector) / (np.abs(true_temp) + 1e-9)
    num_points_below = np.sum(relative_error < RELATIVE_ERROR_THRESHOLD)
    total_points = len(true_temp)
    percentage = (num_points_below / total_points) * 100
    percentage_below_threshold_list.append(percentage)


    print(f"案例 {idx + 1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, rRMSE={rRMSE_case:.4f}, "
          f"点占比(相对误差<{int(RELATIVE_ERROR_THRESHOLD * 100)}%)={percentage:.2f}%")

    vmax_now = visualize_results(true_temp, rec_temp, coords, idx + 1)
    if vmax_now is not None:
        global_error_max = max(global_error_max, vmax_now)

np.save(os.path.join(SAVE_DIR, "global_error_max.npy"), global_error_max)


pre_selected_indices = [
    113159, 118413, 118310, 113158, 118516,
    113056, 118207, 113262, 113055, 112953
]


print("\n正在为指定的10个热点汇总温度数据...")


hot_spot_data_list = []


for point_idx in pre_selected_indices:

    if point_idx >= N_points:
        print(f"警告：网格点索引 {point_idx} 超出范围，已跳过。")
        continue


    point_data = {
        'Point_Index': point_idx,
        'X (m)': coords[point_idx, 0],
        'Y (m)': coords[point_idx, 1],
        'Z (m)': coords[point_idx, 2]
    }


    true_series = all_true_val_temps[point_idx, :]
    pred_series = all_pred_val_temps[point_idx, :]
    error_series = np.abs(true_series - pred_series)


    for i, case_idx in enumerate(val_idx):
        case_num = case_idx + 1
        point_data[f'Case_{case_num}_True_T'] = true_series[i]
        point_data[f'Case_{case_num}_Pred_T'] = pred_series[i]
        point_data[f'Case_{case_num}_AbsError_T'] = error_series[i]
        hot_spot_data_list.append(point_data)


if hot_spot_data_list:
    df_hot_spots = pd.DataFrame(hot_spot_data_list)
    summary_file_path = os.path.join(SAVE_DIR, "hot_spots_temperature_summary.csv")
    df_hot_spots.to_csv(summary_file_path, index=False, float_format='%.4f')
    print(f"10个热点的详细温度及误差数据已保存至: {summary_file_path}")
else:
    print("没有有效的热点索引可供汇总。")



print("\n正在为10个预选的高温点计算rRMSE_n...")

rRMSE_n_results = {}
print("(使用用户提供的固定网格点索引列表)")
for point_idx in pre_selected_indices:
    if point_idx >= N_points:
           continue

    true_series = all_true_val_temps[point_idx, :]
    pred_series = all_pred_val_temps[point_idx, :]

    norm_true_series = np.linalg.norm(true_series)
    # 计算 rRMSE_n，公式为 || T_true - T_pred || / || T_true ||
    rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (norm_true_series + 1e-9)
    rRMSE_n_results[point_idx] = rRMSE_n_value

    avg_temp_at_point = mean_temp[point_idx, 0]
    print(f"  网格点索引 {point_idx:<6} (平均 T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {rRMSE_n_value:.4f}")

df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE (°C)": mae_list,
    "RMSE (°C)": rmse_list,
    "rRMSE": rRMSE_list,
    f"Points_Percentage_RelErr_<{int(RELATIVE_ERROR_THRESHOLD * 100)}%": percentage_below_threshold_list
})
df_metrics.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False, float_format='%.6f')

plt.figure(figsize=(10, 6))
plt.plot(df_metrics["Case"], df_metrics["MAE (°C)"], '-o', label="MAE (°C)")
plt.plot(df_metrics["Case"], df_metrics["RMSE (°C)"], '-s', label="RMSE (°C)")
plt.plot(df_metrics["Case"], df_metrics["rRMSE"], '-^', label="rRMSE (相对)")
plt.xlabel("案例编号")
plt.ylabel("误差值")
plt.title("POD+MLP 各案例误差指标")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "error_metrics_compare.png"), dpi=200)
plt.close()

print("\n" + "=" * 20 + " 总结 " + "=" * 20)
print(f"执行时间: {time.time() - start_time:.2f} 秒")
print(f"结果和图片已保存至: {SAVE_DIR}")
print("\n--- 所有测试案例的平均指标 (用于宏观评估) ---")
print(f"平均 MAE  : {np.mean(mae_list):.4f} °C")
print(f"平均 RMSE : {np.mean(rmse_list):.4f} °C")
print(f"平均 rRMSE: {np.mean(rRMSE_list):.4f}")
print(f"平均点占比 (相对误差 < {int(RELATIVE_ERROR_THRESHOLD * 100)}%): {np.mean(percentage_below_threshold_list):.2f}%")

print("\n--- 预选高温点的rRMSE_n ---")
if not rRMSE_n_results:
    print("  未计算rRMSE_n值 (请检查索引)。")
else:
    for idx, val in rRMSE_n_results.items():
        avg_temp_at_point = mean_temp[idx, 0]
        print(f"  网格点索引 {idx:<6} (平均 T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {val:.4f}")
print("=" * 49)

