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
BASE_SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD_Single_Run_Results'  # 主保存目录


PLANE_TYPE = 'y'
PLANE_VALUE = 2.52
NUM_ADJACENT_PLANES = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
print("=" * 20 + " Step 1: Loading Common Data " + "=" * 20)
overall_start_time = time.time()
bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]


val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]

scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
first_df = pd.read_csv(temp_files[0])
original_coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = original_coords.shape[0]

print("Loading all snapshots into memory...")
all_snapshots_flat = np.zeros((N_points, num_samples))
for i in tqdm(range(num_samples), desc="Loading Snapshots"):
    df = pd.read_csv(temp_files[i])
    all_snapshots_flat[:, i] = df['Temperature'].values

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


def visualize_results(true_temp, pred_temp, coords, case_index, save_dir):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        print(
            f"Warning: No points found for visualization on plane {PLANE_TYPE}={PLANE_VALUE} for case {case_index}. Skipping plot.")
        return

    points_to_interp = coords[mask]
    x, z = points_to_interp[:, 0], points_to_interp[:, 2]
    t_true, t_pred = true_temp[mask], pred_temp[mask]

    grid_x, grid_z = np.linspace(x.min(), x.max(), 150), np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z)

    Tt = griddata((x, z), t_true, (GX, GZ), method='cubic')
    Tp = griddata((x, z), t_pred, (GX, GZ), method='cubic')
    err = np.abs(Tt - Tp)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Temperature Field Reconstruction - Case {case_index}', fontsize=16)

    v_min, v_max = np.nanmin(Tt), np.nanmax(Tt)

    plots_data = [(axs[0], Tt, "True Temperature", "jet", v_min, v_max),
                  (axs[1], Tp, "Reconstructed Temperature", "jet", v_min, v_max),
                  (axs[2], err, "Absolute Error", "YlOrRd", 0, np.nanmax(err))]

    for ax, data, title, cmap, vmin, vmax_plot in plots_data:
        if np.all(np.isnan(data)): continue
        cs = ax.contourf(GX, GZ, data, levels=50, cmap=cmap, vmin=vmin, vmax=vmax_plot)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel("X (m)");
        ax.set_ylabel("Z (m)")
        ax.set_title(title);
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"case_{case_index}_reconstruction.png"), dpi=200)
    plt.close(fig)

chosen_order = ""
while chosen_order not in ['xyz', 'yxz', 'zxy']:
    chosen_order = input("请选择数据展开方式 ('xyz', 'yxz', or 'zxy'): ").lower().strip()
    if chosen_order not in ['xyz', 'yxz', 'zxy']:
        print("输入无效，请重新输入。")

SAVE_DIR = os.path.join(BASE_SAVE_DIR, f'POD_run_{chosen_order}')
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"\n已选择展开方式: '{chosen_order}'. 结果将保存在: {SAVE_DIR}")


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

print("\nValidating on test set...")
model.eval()
mae_list, rmse_list, rRMSE_list = [], [], []

for i, original_case_idx in enumerate(tqdm(val_idx, desc="Validating and Visualizing Cases")):
    cond = torch.tensor(conditions_val_norm[i:i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred_coeff_norm = model(cond).numpy()
    pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

    rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
    true_temp = snapshots_val[:, i]

    err_vector = true_temp - rec_temp
    mae = np.mean(np.abs(err_vector))
    rmse = np.sqrt(np.mean(err_vector ** 2))
    rRMSE = np.linalg.norm(err_vector) / (np.linalg.norm(true_temp) + 1e-9)

    mae_list.append(mae)
    rmse_list.append(rmse)
    rRMSE_list.append(rRMSE)

    human_case_number = original_case_idx + 1
    visualize_results(true_temp, rec_temp, coords, human_case_number, SAVE_DIR)

    print(f"  Case {human_case_number:<2d}: MAE={mae:.4f}, RMSE={rmse:.4f}, rRMSE={rRMSE:.6f}")

print("\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
print(f"Unfolding Order       : {chosen_order.upper()}")
print(f"Number of Modes (K)   : {K}")
print("-" * 40)
print("Average Metrics on Test Set:")
print(f"  Mean MAE            : {np.mean(mae_list):.4f}")
print(f"  Mean RMSE           : {np.mean(rmse_list):.4f}")
print(f"  Mean rRMSE          : {np.mean(rRMSE_list):.6f}")
print("-" * 40)
print(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds.")
print(f"All results and visualizations saved in: {SAVE_DIR}")


df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE": mae_list,
    "RMSE": rmse_list,
    "rRMSE": rRMSE_list
})
df_metrics.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False)
