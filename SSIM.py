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
PLANE_TYPE = 'y'
PLANE_VALUE = 4.32
NUM_ADJACENT_PLANES = 0
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-1.26'

# --- rRMSE 平面（保留你原来的逻辑） ---
PLANE_Y_FOR_RRMSE = 4.32

# --- 新增：SSIM/梯度SSIM 平面（按你的需求：Y=1.52m） ---
PLANE_Y_FOR_SSIM = 2.52

# --- 新增：插值网格分辨率（与轮廓图一致） ---
SSIM_GRID_RES = 150
SSIM_GRID_METHOD = 'cubic'   # 'cubic'/'linear'/'nearest'

# --- SSIM 计算实现（优先 skimage，否则 fallback） ---
SSIM_METHOD = 'skimage'      # 'skimage' or 'fallback'

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
# SSIM / Gradient-SSIM Utils
# ======================================
def _fill_nans_with_mean(a: np.ndarray) -> np.ndarray:
    if np.all(np.isnan(a)):
        return np.zeros_like(a, dtype=float)
    m = np.nanmean(a)
    return np.where(np.isnan(a), m, a)

def _normalize_to_01(a: np.ndarray, ref_min=None, ref_max=None, eps: float = 1e-12) -> np.ndarray:
    if ref_min is None:
        ref_min = np.nanmin(a)
    if ref_max is None:
        ref_max = np.nanmax(a)
    denom = (ref_max - ref_min)
    if np.abs(denom) < eps:
        return np.zeros_like(a, dtype=float)
    return (a - ref_min) / (denom + eps)

def compute_ssim_2d(true_2d: np.ndarray, pred_2d: np.ndarray) -> float:
    """
    与 Tucker 同口径：
    - NaN 用均值填充
    - 以 True 的 min/max 归一化到 [0,1]
    - pred 归一化后 clip 到 [0,1]（避免 cubic 插值过冲带来额外惩罚）
    """
    t = _fill_nans_with_mean(true_2d.astype(float))
    p = _fill_nans_with_mean(pred_2d.astype(float))

    tmin, tmax = np.min(t), np.max(t)
    t01 = _normalize_to_01(t, ref_min=tmin, ref_max=tmax)
    p01 = _normalize_to_01(p, ref_min=tmin, ref_max=tmax)
    p01 = np.clip(p01, 0.0, 1.0)

    if SSIM_METHOD.lower() == 'skimage':
        try:
            from skimage.metrics import structural_similarity as ssim
            return float(ssim(t01, p01, data_range=1.0))
        except Exception:
            pass

    # fallback：非滑窗版 global SSIM-like（不是标准SSIM，但可用于排序/对比）
    mu_t = np.mean(t01)
    mu_p = np.mean(p01)
    var_t = np.var(t01)
    var_p = np.var(p01)
    cov_tp = np.mean((t01 - mu_t) * (p01 - mu_p))

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    numerator = (2 * mu_t * mu_p + C1) * (2 * cov_tp + C2)
    denominator = (mu_t ** 2 + mu_p ** 2 + C1) * (var_t + var_p + C2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)

def compute_grad_mag(T_xz: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    计算 |∇T|，用真实坐标间距做梯度（保持物理尺度一致）
    T_xz shape: (res,res) 或 (Nx,Nz)
    """
    T_xz = _fill_nans_with_mean(T_xz.astype(float))
    dTdx, dTdz = np.gradient(T_xz, x_axis, z_axis, edge_order=1)
    return np.sqrt(dTdx ** 2 + dTdz ** 2)

def interpolate_scatter_to_grid(x: np.ndarray, z: np.ndarray, v: np.ndarray,
                                res: int = 150, method: str = 'cubic'):
    """
    与你 contourf 画图同口径：griddata((x,z), v) -> (GX,GZ)
    返回 (Vg, grid_x, grid_z, GX, GZ)
    """
    grid_x = np.linspace(x.min(), x.max(), res)
    grid_z = np.linspace(z.min(), z.max(), res)
    GX, GZ = np.meshgrid(grid_x, grid_z)
    Vg = griddata((x, z), v, (GX, GZ), method=method)
    Vg = _fill_nans_with_mean(Vg)
    return Vg, grid_x, grid_z, GX, GZ

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

    def forward(self, x):
        return self.net(x)

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
# Visualization Function (保持你的逻辑)
# ======================================
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
    if not np.any(mask):
        return None

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
    fig.suptitle(f'Temperature Field Reconstruction - Case {case_index}', fontsize=16)

    plots_data = [
        (axs[0], Tt, "True Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[1], Tp, "Reconstructed Temperature", "jet", np.nanmin(Tt), np.nanmax(Tt)),
        (axs[2], err, "Absolute Error", "YlOrRd", 0, vmax)
    ]

    for ax, data, title, cmap, vmin, vmax_plot in plots_data:
        if np.all(np.isnan(data)):
            continue
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

# --- 预计算 rRMSE 平面掩码 ---
unique_y_coords = np.unique(coords[:, 1])
closest_y_val_rrmse = unique_y_coords[np.argmin(np.abs(unique_y_coords - PLANE_Y_FOR_RRMSE))]
plane_mask_rrmse = (coords[:, 1] == closest_y_val_rrmse)
num_plane_points_rrmse = np.sum(plane_mask_rrmse)

print(f"Target plane for rRMSE calculation: Y = {PLANE_Y_FOR_RRMSE:.4f} m")
if num_plane_points_rrmse > 0:
    print(f"Found {num_plane_points_rrmse} points on the closest available plane: Y = {closest_y_val_rrmse:.4f} m")
else:
    print(f"Warning: No points found for the specified plane Y={PLANE_Y_FOR_RRMSE}. Cannot calculate plane rRMSE.")

# --- 新增：预计算 SSIM/梯度SSIM 平面掩码（Y=1.52m） ---
closest_y_val_ssim = unique_y_coords[np.argmin(np.abs(unique_y_coords - PLANE_Y_FOR_SSIM))]
plane_mask_ssim = (coords[:, 1] == closest_y_val_ssim)
num_plane_points_ssim = np.sum(plane_mask_ssim)
print(f"\nTarget plane for SSIM calculation: Y = {PLANE_Y_FOR_SSIM:.4f} m")
if num_plane_points_ssim > 0:
    print(f"Found {num_plane_points_ssim} points on the closest available plane: Y = {closest_y_val_ssim:.4f} m")
else:
    print(f"Warning: No points found for the specified plane Y={PLANE_Y_FOR_SSIM}. SSIM will be NaN.")

# --- 新增：SSIM 结果容器（温度/梯度，原网格/插值网格） ---
ssim_raw_T_list = []
ssim_interp_T_list = []
ssim_raw_grad_list = []
ssim_interp_grad_list = []

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

    # 1) 全局误差
    mae = np.mean(np.abs(err_vector))
    rmse = np.sqrt(np.mean(err_vector ** 2))
    norm_true = np.linalg.norm(true_temp)
    rRMSE_case = np.linalg.norm(err_vector) / (norm_true + 1e-9)

    mae_list.append(mae)
    rmse_list.append(rmse)
    rRMSE_list.append(rRMSE_case)

    # 2) 平面 rRMSE
    rRMSE_case_plane = np.nan
    if num_plane_points_rrmse > 0:
        true_temp_plane = true_temp[plane_mask_rrmse]
        rec_temp_plane = rec_temp[plane_mask_rrmse]
        err_vector_plane = true_temp_plane - rec_temp_plane
        norm_true_plane = np.linalg.norm(true_temp_plane)
        if norm_true_plane > 1e-9:
            rRMSE_case_plane = np.linalg.norm(err_vector_plane) / norm_true_plane
    rRMSE_plane_list.append(rRMSE_case_plane)

    # 3) SSIM & 梯度SSIM（Y=1.52m，同 Tucker 口径：原网格 vs 插值网格）
    ssim_raw_T = np.nan
    ssim_interp_T = np.nan
    ssim_raw_grad = np.nan
    ssim_interp_grad = np.nan

    if num_plane_points_ssim > 0:
        # 3.1 先在该平面上做插值（这一步与 contourf 完全一致）
        pts = coords[plane_mask_ssim]
        x, z = pts[:, 0], pts[:, 2]
        t_true_plane = true_temp[plane_mask_ssim]
        t_pred_plane = rec_temp[plane_mask_ssim]

        # 插值到轮廓图网格（SSIM_interp_* 就在这个网格上算）
        Tt_grid, gx, gz, GX, GZ = interpolate_scatter_to_grid(
            x, z, t_true_plane, res=SSIM_GRID_RES, method=SSIM_GRID_METHOD
        )
        Tp_grid, _, _, _, _ = interpolate_scatter_to_grid(
            x, z, t_pred_plane, res=SSIM_GRID_RES, method=SSIM_GRID_METHOD
        )

        # 3.2 SSIM（插值网格，与你“画图网格”一致）
        ssim_interp_T = compute_ssim_2d(Tt_grid, Tp_grid)

        # 3.3 梯度SSIM（插值网格）
        grad_true_interp = compute_grad_mag(Tt_grid, gx, gz)
        grad_pred_interp = compute_grad_mag(Tp_grid, gx, gz)
        ssim_interp_grad = compute_ssim_2d(grad_true_interp, grad_pred_interp)

        # 3.4 “原网格 SSIM”：
        # POD 这里没有 (NX×NZ) 的规则切片矩阵可直接取（因为数据是散点列表），
        # 为了与 Tucker 的“raw vs interp”形式对应，这里用同一 plane 上的“近邻重采样”构造一个原始网格：
        # 方法：用最近邻在 (unique_x, unique_z) 上重建 Nx×Nz，再算 SSIM_raw_*。
        # 注意：这一步只是为了给一个“原网格口径”的对照项，真正最推荐对齐 Tucker/POD 的是 interp 版。
        unique_x = np.sort(np.unique(x))
        unique_z = np.sort(np.unique(z))
        X_raw, Z_raw = np.meshgrid(unique_x, unique_z, indexing='ij')

        Tt_raw = griddata((x, z), t_true_plane, (X_raw, Z_raw), method='nearest')
        Tp_raw = griddata((x, z), t_pred_plane, (X_raw, Z_raw), method='nearest')
        Tt_raw = _fill_nans_with_mean(Tt_raw)
        Tp_raw = _fill_nans_with_mean(Tp_raw)

        ssim_raw_T = compute_ssim_2d(Tt_raw, Tp_raw)

        grad_true_raw = compute_grad_mag(Tt_raw, unique_x, unique_z)
        grad_pred_raw = compute_grad_mag(Tp_raw, unique_x, unique_z)
        ssim_raw_grad = compute_ssim_2d(grad_true_raw, grad_pred_raw)

    ssim_raw_T_list.append(ssim_raw_T)
    ssim_interp_T_list.append(ssim_interp_T)
    ssim_raw_grad_list.append(ssim_raw_grad)
    ssim_interp_grad_list.append(ssim_interp_grad)

    print(
        f"Case {idx + 1:2d}: "
        f"MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, rRMSE={rRMSE_case:.4f}, "
        f"rRMSE_plane(Y={closest_y_val_rrmse:.2f})={rRMSE_case_plane:.4f}, "
        f"SSIM_raw_T(Y={closest_y_val_ssim:.2f})={ssim_raw_T:.6f}, "
        f"SSIM_interp_T({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={closest_y_val_ssim:.2f})={ssim_interp_T:.6f}, "
        f"SSIM_raw_grad(Y={closest_y_val_ssim:.2f})={ssim_raw_grad:.6f}, "
        f"SSIM_interp_grad({SSIM_GRID_RES}x{SSIM_GRID_RES}, Y={closest_y_val_ssim:.2f})={ssim_interp_grad:.6f}"
    )

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
    f"rRMSE_plane_Y={closest_y_val_rrmse:.2f}m": rRMSE_plane_list,

    # 新增：温度 SSIM（原网格 vs 插值网格）
    f"SSIM_raw_T_plane_Y={closest_y_val_ssim:.2f}m": ssim_raw_T_list,
    f"SSIM_interp_T_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={closest_y_val_ssim:.2f}m": ssim_interp_T_list,

    # 新增：梯度 SSIM（原网格 vs 插值网格）
    f"SSIM_raw_grad_plane_Y={closest_y_val_ssim:.2f}m": ssim_raw_grad_list,
    f"SSIM_interp_grad_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={closest_y_val_ssim:.2f}m": ssim_interp_grad_list,
})
df_metrics.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False)

plt.figure(figsize=(12, 7))
plt.plot(df_metrics["Case"], df_metrics["MAE (°C)"], '-o', label="MAE (°C) [Full Field]")
plt.plot(df_metrics["Case"], df_metrics["RMSE (°C)"], '-s', label="RMSE (°C) [Full Field]")
plt.plot(df_metrics["Case"], df_metrics["rRMSE"], '-^', label="rRMSE [Full Field]")
plt.plot(df_metrics["Case"], df_metrics[f"rRMSE_plane_Y={closest_y_val_rrmse:.2f}m"], '-x', color='purple',
         label=f"rRMSE [Plane Y={closest_y_val_rrmse:.2f}m]")

plt.plot(df_metrics["Case"], df_metrics[f"SSIM_raw_T_plane_Y={closest_y_val_ssim:.2f}m"], '-d', color='green',
         label=f"SSIM_raw_T [Plane Y={closest_y_val_ssim:.2f}m]")
plt.plot(df_metrics["Case"], df_metrics[f"SSIM_interp_T_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={closest_y_val_ssim:.2f}m"],
         '-p', color='orange', label=f"SSIM_interp_T({SSIM_GRID_RES}x{SSIM_GRID_RES}) [Plane Y={closest_y_val_ssim:.2f}m]")

plt.plot(df_metrics["Case"], df_metrics[f"SSIM_raw_grad_plane_Y={closest_y_val_ssim:.2f}m"], '-v', color='teal',
         label=f"SSIM_raw_grad [Plane Y={closest_y_val_ssim:.2f}m]")
plt.plot(df_metrics["Case"], df_metrics[f"SSIM_interp_grad_{SSIM_GRID_RES}x{SSIM_GRID_RES}_Y={closest_y_val_ssim:.2f}m"],
         '-h', color='red', label=f"SSIM_interp_grad({SSIM_GRID_RES}x{SSIM_GRID_RES}) [Plane Y={closest_y_val_ssim:.2f}m]")

plt.xlabel("Case Number")
plt.ylabel("Metric Value")
plt.title("POD+MLP Per-Case Metrics (Full Field vs. Plane Metrics incl. SSIM & Grad-SSIM)")
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
print(f"Mean rRMSE (Plane Y={closest_y_val_rrmse:.2f}m): {np.nanmean(rRMSE_plane_list):.4f}")

valid_ssim_interp_T = [v for v in ssim_interp_T_list if (v is not None and not np.isnan(v))]
valid_ssim_interp_g = [v for v in ssim_interp_grad_list if (v is not None and not np.isnan(v))]
print(f"\n--- Plane SSIM/Grad-SSIM Summary (Y≈{closest_y_val_ssim:.2f}m) ---")
print(f"Mean SSIM_raw_T        : {np.nanmean(ssim_raw_T_list):.6f}")
print(f"Mean SSIM_interp_T     : {np.mean(valid_ssim_interp_T):.6f}" if len(valid_ssim_interp_T) else "Mean SSIM_interp_T     : NaN")
print(f"Mean SSIM_raw_grad     : {np.nanmean(ssim_raw_grad_list):.6f}")
print(f"Mean SSIM_interp_grad  : {np.mean(valid_ssim_interp_g):.6f}" if len(valid_ssim_interp_g) else "Mean SSIM_interp_grad  : NaN")

print("\n--- rRMSE_n for Pre-selected High-Temperature Points ---")
if not rRMSE_n_results:
    print("  No rRMSE_n values were calculated (check indices).")
else:
    for idx, val in rRMSE_n_results.items():
        avg_temp_at_point = mean_temp[idx, 0]
        print(f"  Grid Point Index {idx:<6} (Avg T ≈ {avg_temp_at_point:.1f}°C): rRMSE_n = {val:.4f}")
print("=" * 49)
