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
BASE_SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD_Single_Run_Results'  # 主保存目录

# --- 可视化与误差计算参数 ---
PLANE_TYPE = 'y'
PLANE_VALUE = 2.52
NUM_ADJACENT_PLANES = 0

# --- SSIM 计算参数 ---
SSIM_GRID_RES = 150  # 与可视化保持一致
SSIM_METHOD = 'skimage'  # 'skimage' or 'fallback'
INTERP_METHOD_FOR_SSIM = 'cubic'  # 与轮廓图一致（如需更稳健可改为 'linear'）

# ======================================
# Set Seeds for Reproducibility
# ======================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ======================================
# 1. Load Common Data (一次性加载)
# ======================================
print("=" * 20 + " Step 1: Loading Common Data " + "=" * 20)
overall_start_time = time.time()
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]

# 固定的训练集和验证集划分
val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]

# 边界条件归一化
scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

# 加载所有快照和坐标
temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
first_df = pd.read_csv(temp_files[0])
original_coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = original_coords.shape[0]

print("Loading all snapshots into memory...")
all_snapshots_flat = np.zeros((N_points, num_samples))
for i in tqdm(range(num_samples), desc="Loading Snapshots"):
    df = pd.read_csv(temp_files[i])
    all_snapshots_flat[:, i] = df['Temperature'].values

# ======================================
# 2. Reshape Data into 3D Tensor
# ======================================
print("\n" + "=" * 20 + " Step 2: Reshaping flat data to 3D tensor " + "=" * 20)
x_coords_unique = np.unique(original_coords[:, 0])
y_coords_unique = np.unique(original_coords[:, 1])
z_coords_unique = np.unique(original_coords[:, 2])
Nx, Ny, Nz = len(x_coords_unique), len(y_coords_unique), len(z_coords_unique)
print(f"Grid dimensions inferred: Nx={Nx}, Ny={Ny}, Nz={Nz}")
if Nx * Ny * Nz != N_points:
    raise ValueError("Grid dimensions do not match total points. Is the grid regular?")

sort_indices = np.lexsort((original_coords[:, 0], original_coords[:, 1], original_coords[:, 2]))
sorted_coords = original_coords[sort_indices]
all_snapshots_sorted = all_snapshots_flat[sort_indices, :]
all_snapshots_tensor = all_snapshots_sorted.reshape(Nz, Ny, Nx, num_samples)
all_snapshots_tensor = np.transpose(all_snapshots_tensor, (2, 1, 0, 3))
print(f"Successfully created master snapshot tensor with shape: {all_snapshots_tensor.shape}")

# ======================================
# Helper Functions (MLP, Visualization, SSIM)
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

def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if plane_type not in axis_map:
        raise ValueError("plane_type must be 'x', 'y', or 'z'.")
    axis_idx = axis_map[plane_type]
    unique_vals = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_vals - target_value))
    low = max(0, closest_idx - num_adjacent)
    high = min(len(unique_vals), closest_idx + num_adjacent + 1)
    selected_plane_values = unique_vals[low:high]
    mask = np.isin(coords[:, axis_idx], selected_plane_values)
    return mask

def _normalize_to_01(a, ref_min=None, ref_max=None, eps=1e-12):
    """Normalize array to [0,1] using provided ref_min/ref_max or its own min/max (nan-aware)."""
    if ref_min is None:
        ref_min = np.nanmin(a)
    if ref_max is None:
        ref_max = np.nanmax(a)
    denom = (ref_max - ref_min)
    if np.abs(denom) < eps:
        return np.zeros_like(a, dtype=float)
    return (a - ref_min) / (denom + eps)

def _fill_nans_with_mean(a):
    """Replace NaNs with mean of valid entries (or 0 if all NaN)."""
    if np.all(np.isnan(a)):
        return np.zeros_like(a, dtype=float)
    m = np.nanmean(a)
    return np.where(np.isnan(a), m, a)

def compute_ssim_2d(true_2d, pred_2d):
    """
    Compute SSIM between two 2D fields.
    - Normalizes using true field min/max to align with 'structure' comparison.
    - Handles NaNs by filling with mean.
    - Clips pred01 to [0,1] to avoid overshoot penalty (common after cubic interpolation).
    Prefers skimage if available, otherwise uses a fallback global-SSIM approximation.
    """
    t = _fill_nans_with_mean(true_2d.astype(float))
    p = _fill_nans_with_mean(pred_2d.astype(float))

    tmin, tmax = np.min(t), np.max(t)
    t01 = _normalize_to_01(t, ref_min=tmin, ref_max=tmax)
    p01 = _normalize_to_01(p, ref_min=tmin, ref_max=tmax)
    p01 = np.clip(p01, 0.0, 1.0)

    if SSIM_METHOD == 'skimage':
        try:
            from skimage.metrics import structural_similarity as ssim
            val = ssim(t01, p01, data_range=1.0)
            return float(val)
        except Exception:
            pass

    # Fallback: global SSIM-like (not windowed)
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

def compute_grad_mag(T2d, x_axis, z_axis):
    """
    Compute gradient magnitude |∇T| on a 2D field defined on (x,z).
    T2d shape: (Nx, Nz)
    x_axis shape: (Nx,)
    z_axis shape: (Nz,)
    """
    T2d = _fill_nans_with_mean(T2d.astype(float))
    # np.gradient: first axis corresponds to x, second axis corresponds to z
    dTdx, dTdz = np.gradient(T2d, x_axis, z_axis, edge_order=1)
    return np.sqrt(dTdx**2 + dTdz**2)

def extract_plane_raw_grid(true_temp, pred_temp, coords):
    """
    Extract plane (y=PLANE_VALUE) as RAW grid (Nx_raw × Nz_raw) without interpolation.
    Returns: (T_true_raw, T_pred_raw, x_vals, z_vals) or (None,...)
    """
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        return None, None, None, None

    pts = coords[mask]
    x_vals = np.sort(np.unique(pts[:, 0]))
    z_vals = np.sort(np.unique(pts[:, 2]))
    Nx_raw, Nz_raw = len(x_vals), len(z_vals)

    Tt = np.full((Nx_raw, Nz_raw), np.nan, dtype=float)
    Tp = np.full((Nx_raw, Nz_raw), np.nan, dtype=float)

    x_map = {v: i for i, v in enumerate(x_vals)}
    z_map = {v: i for i, v in enumerate(z_vals)}

    tt = true_temp[mask]
    tp = pred_temp[mask]
    for (x, _, z), v_t, v_p in zip(pts, tt, tp):
        Tt[x_map[x], z_map[z]] = v_t
        Tp[x_map[x], z_map[z]] = v_p

    return Tt, Tp, x_vals, z_vals

def compute_plane_fields_for_interp(true_temp, pred_temp, coords):
    """
    Interpolate plane points to a regular X-Z grid (SSIM_GRID_RES × SSIM_GRID_RES),
    consistent with contour visualization.
    Returns: (T_true_grid, T_pred_grid, grid_x, grid_z) or (None,...)
    """
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        return None, None, None, None

    points_to_interp = coords[mask]
    x, z = points_to_interp[:, 0], points_to_interp[:, 2]
    t_true, t_pred = true_temp[mask], pred_temp[mask]

    grid_x = np.linspace(x.min(), x.max(), SSIM_GRID_RES)
    grid_z = np.linspace(z.min(), z.max(), SSIM_GRID_RES)
    GX, GZ = np.meshgrid(grid_x, grid_z, indexing='xy')

    Tt = griddata((x, z), t_true, (GX, GZ), method=INTERP_METHOD_FOR_SSIM)
    Tp = griddata((x, z), t_pred, (GX, GZ), method=INTERP_METHOD_FOR_SSIM)

    return Tt, Tp, grid_x, grid_z

def visualize_results(true_temp, pred_temp, coords, case_index, save_dir):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        print(f"Warning: No points found for visualization on plane {PLANE_TYPE}={PLANE_VALUE} for case {case_index}. Skipping plot.")
        return

    points_to_interp = coords[mask]
    x, z = points_to_interp[:, 0], points_to_interp[:, 2]
    t_true, t_pred = true_temp[mask], pred_temp[mask]

    grid_x, grid_z = np.linspace(x.min(), x.max(), 150), np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z, indexing='xy')

    Tt = griddata((x, z), t_true, (GX, GZ), method='cubic')
    Tp = griddata((x, z), t_pred, (GX, GZ), method='cubic')
    err = np.abs(Tt - Tp)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Temperature Field Reconstruction - Case {case_index}', fontsize=16)

    v_min, v_max = np.nanmin(Tt), np.nanmax(Tt)

    plots_data = [
        (axs[0], Tt, "True Temperature", "jet", v_min, v_max),
        (axs[1], Tp, "Reconstructed Temperature", "jet", v_min, v_max),
        (axs[2], err, "Absolute Error", "YlOrRd", 0, np.nanmax(err))
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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"case_{case_index}_reconstruction.png"), dpi=200)
    plt.close(fig)

# ======================================
# 3. Main Analysis and Execution
# ======================================
# --- 用户输入选择展开方式 ---
chosen_order = ""
while chosen_order not in ['xyz', 'yxz', 'zxy']:
    chosen_order = input("请选择数据展开方式 ('xyz', 'yxz', or 'zxy'): ").lower().strip()
    if chosen_order not in ['xyz', 'yxz', 'zxy']:
        print("输入无效，请重新输入。")

SAVE_DIR = os.path.join(BASE_SAVE_DIR, f'POD_run_{chosen_order}')
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"\n已选择展开方式: '{chosen_order}'. 结果将保存在: {SAVE_DIR}")

# --- Step A: 根据选择的展开方式准备数据 ---
print(f"Preparing data for order '{chosen_order}'...")
if chosen_order == 'xyz':
    permuted_tensor = all_snapshots_tensor
    coords = sorted_coords
elif chosen_order == 'yxz':
    permuted_tensor = np.transpose(all_snapshots_tensor, (1, 0, 2, 3))
    coords = sorted_coords[np.lexsort((sorted_coords[:, 2], sorted_coords[:, 0], sorted_coords[:, 1]))]
elif chosen_order == 'zxy':
    permuted_tensor = np.transpose(all_snapshots_tensor, (2, 0, 1, 3))
    coords = sorted_coords[np.lexsort((sorted_coords[:, 1], sorted_coords[:, 0], sorted_coords[:, 2]))]

all_snapshots = permuted_tensor.reshape(N_points, num_samples)
snapshots_train = all_snapshots[:, train_idx]
snapshots_val = all_snapshots[:, val_idx]
print(f"Snapshot matrix shape for this run: {snapshots_train.shape}")

# --- Step B: POD Analysis ---
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

# --- Step C: MLP Training ---
set_seed(SEED)
model = MLP(input_size=conditions.shape[1], output_size=K, hidden_layers=HIDDEN_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

print("\nTraining MLP...")
for epoch in tqdm(range(EPOCHS), desc=f"Training MLP ({chosen_order})"):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

# --- Step D: Validation & Error Calculation ---
print("\nValidating on test set...")
model.eval()

mae_list, rmse_list, rRMSE_list = [], [], []
ssim_raw_T_list = []
ssim_interp_T_list = []
ssim_raw_grad_list = []
ssim_interp_grad_list = []

for i, original_case_idx in enumerate(tqdm(val_idx, desc="Validating and Visualizing Cases")):
    cond = torch.tensor(conditions_val_norm[i:i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred_coeff_norm = model(cond).numpy()
    pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

    rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
    true_temp = snapshots_val[:, i]

    # --- 3D error metrics ---
    err_vector = true_temp - rec_temp
    mae = np.mean(np.abs(err_vector))
    rmse = np.sqrt(np.mean(err_vector ** 2))
    rRMSE = np.linalg.norm(err_vector) / (np.linalg.norm(true_temp) + 1e-9)

    mae_list.append(mae)
    rmse_list.append(rmse)
    rRMSE_list.append(rRMSE)

    human_case_number = original_case_idx + 1

    # --- visualization ---
    visualize_results(true_temp, rec_temp, coords, human_case_number, SAVE_DIR)

    # =======================
    # SSIM (raw vs interp) + Gradient SSIM
    # =======================
    # 1) RAW grid slice (no interpolation)
    Tt_raw, Tp_raw, x_raw, z_raw = extract_plane_raw_grid(true_temp, rec_temp, coords)
    if Tt_raw is None:
        ssim_raw_T = np.nan
        ssim_raw_grad = np.nan
    else:
        ssim_raw_T = compute_ssim_2d(Tt_raw, Tp_raw)

        grad_t_raw = compute_grad_mag(Tt_raw, x_raw, z_raw)
        grad_p_raw = compute_grad_mag(Tp_raw, x_raw, z_raw)
        ssim_raw_grad = compute_ssim_2d(grad_t_raw, grad_p_raw)

    ssim_raw_T_list.append(ssim_raw_T)
    ssim_raw_grad_list.append(ssim_raw_grad)

    # 2) Interpolated slice (contour grid, SSIM_GRID_RES x SSIM_GRID_RES)
    Tt_i, Tp_i, gx, gz = compute_plane_fields_for_interp(true_temp, rec_temp, coords)
    if Tt_i is None:
        ssim_interp_T = np.nan
        ssim_interp_grad = np.nan
    else:
        ssim_interp_T = compute_ssim_2d(Tt_i, Tp_i)

        # gradient on interpolated grid
        # NOTE: Tt_i/Tp_i shape is (SSIM_GRID_RES, SSIM_GRID_RES) but created with indexing='xy':
        # grid_x aligns with columns, grid_z aligns with rows in that mesh. We treat it as a field over (x,z):
        # We'll interpret axis-0 as z-direction and axis-1 as x-direction if using indexing='xy'.
        # To keep consistent, we transpose to (Nx, Nz) form: use mesh as (x,z) with axis0->x, axis1->z.
        # Here: Tt_i is on (grid_z, grid_x) in typical plotting, so we convert:
        Tt_i_xz = np.array(Tt_i).T  # now shape approx (len(grid_x), len(grid_z)) -> (x,z)
        Tp_i_xz = np.array(Tp_i).T
        grad_t_i = compute_grad_mag(Tt_i_xz, gx, gz)
        grad_p_i = compute_grad_mag(Tp_i_xz, gx, gz)
        ssim_interp_grad = compute_ssim_2d(grad_t_i, grad_p_i)

    ssim_interp_T_list.append(ssim_interp_T)
    ssim_interp_grad_list.append(ssim_interp_grad)

    print(
        f"  Case {human_case_number:<2d}: "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, rRMSE={rRMSE:.6f} | "
        f"SSIM_raw_T={ssim_raw_T:.6f}, SSIM_interp_T={ssim_interp_T:.6f} | "
        f"SSIM_raw_grad={ssim_raw_grad:.6f}, SSIM_interp_grad={ssim_interp_grad:.6f}"
    )

# ======================================
# 4. Final Summary
# ======================================
print("\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
print(f"Unfolding Order       : {chosen_order.upper()}")
print(f"Number of Modes (K)   : {K}")
print("-" * 40)
print("Average Metrics on Test Set:")
print(f"  Mean MAE            : {np.mean(mae_list):.4f}")
print(f"  Mean RMSE           : {np.mean(rmse_list):.4f}")
print(f"  Mean rRMSE          : {np.mean(rRMSE_list):.6f}")

def nanmean_safe(lst):
    arr = np.array(lst, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan

print(f"  Mean SSIM_raw_T (y={PLANE_VALUE})       : {nanmean_safe(ssim_raw_T_list):.6f}")
print(f"  Mean SSIM_interp_T (y={PLANE_VALUE})    : {nanmean_safe(ssim_interp_T_list):.6f}")
print(f"  Mean SSIM_raw_grad (y={PLANE_VALUE})    : {nanmean_safe(ssim_raw_grad_list):.6f}")
print(f"  Mean SSIM_interp_grad (y={PLANE_VALUE}) : {nanmean_safe(ssim_interp_grad_list):.6f}")

print("-" * 40)
print(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds.")
print(f"All results and visualizations saved in: {SAVE_DIR}")

# 保存每个案例的详细指标到CSV（新增 SSIM 与 梯度SSIM）
df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE": mae_list,
    "RMSE": rmse_list,
    "rRMSE": rRMSE_list,
    f"SSIM_raw_T_y{PLANE_VALUE}": ssim_raw_T_list,
    f"SSIM_interp_T_y{PLANE_VALUE}": ssim_interp_T_list,
    f"SSIM_raw_grad_y{PLANE_VALUE}": ssim_raw_grad_list,
    f"SSIM_interp_grad_y{PLANE_VALUE}": ssim_interp_grad_list,
})
df_metrics.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False)

# 额外保存一个总体汇总
df_summary = pd.DataFrame({
    "Unfolding_Order": [chosen_order.upper()],
    "K_modes": [K],
    "Mean_MAE": [float(np.mean(mae_list))],
    "Mean_RMSE": [float(np.mean(rmse_list))],
    "Mean_rRMSE": [float(np.mean(rRMSE_list))],
    f"Mean_SSIM_raw_T_y{PLANE_VALUE}": [nanmean_safe(ssim_raw_T_list)],
    f"Mean_SSIM_interp_T_y{PLANE_VALUE}": [nanmean_safe(ssim_interp_T_list)],
    f"Mean_SSIM_raw_grad_y{PLANE_VALUE}": [nanmean_safe(ssim_raw_grad_list)],
    f"Mean_SSIM_interp_grad_y{PLANE_VALUE}": [nanmean_safe(ssim_interp_grad_list)],
})
df_summary.to_csv(os.path.join(SAVE_DIR, "summary_metrics.csv"), index=False)

print("\nSaved metrics:")
print(" - per_case_metrics.csv")
print(" - summary_metrics.csv")
