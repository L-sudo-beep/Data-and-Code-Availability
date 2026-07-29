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
BASE_SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD_Single_Run_Results'

# --- 可视化与误差计算参数 ---
PLANE_TYPE = 'y'
PLANE_VALUE = 1.26
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

x_coords_unique = np.sort(np.unique(original_coords[:, 0]))
y_coords_unique = np.sort(np.unique(original_coords[:, 1]))
z_coords_unique = np.sort(np.unique(original_coords[:, 2]))

Nx, Ny, Nz = len(x_coords_unique), len(y_coords_unique), len(z_coords_unique)

print(f"Grid dimensions inferred: Nx={Nx}, Ny={Ny}, Nz={Nz}")

if Nx * Ny * Nz != N_points:
    raise ValueError("Grid dimensions do not match total points. Is the grid regular?")

# 按 z, y, x 排序，使 reshape(Nz, Ny, Nx) 后再转置为 (Nx, Ny, Nz)
sort_indices = np.lexsort((original_coords[:, 0], original_coords[:, 1], original_coords[:, 2]))
sorted_coords = original_coords[sort_indices]
all_snapshots_sorted = all_snapshots_flat[sort_indices, :]

all_snapshots_tensor = all_snapshots_sorted.reshape(Nz, Ny, Nx, num_samples)
all_snapshots_tensor = np.transpose(all_snapshots_tensor, (2, 1, 0, 3))

print(f"Successfully created master snapshot tensor with shape: {all_snapshots_tensor.shape}")
print("Tensor axis order: (x, y, z, sample)")



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
    mask = select_adjacent_planes(
        coords,
        PLANE_TYPE,
        PLANE_VALUE,
        NUM_ADJACENT_PLANES
    )

    if not np.any(mask):
        print(
            f"Warning: No points found for visualization on plane "
            f"{PLANE_TYPE}={PLANE_VALUE} for case {case_index}. Skipping plot."
        )
        return

    points_to_interp = coords[mask]
    x, z = points_to_interp[:, 0], points_to_interp[:, 2]

    t_true = true_temp[mask]
    t_pred = pred_temp[mask]

    grid_x = np.linspace(x.min(), x.max(), 150)
    grid_z = np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z)

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
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, f"case_{case_index}_reconstruction.png"),
        dpi=200
    )
    plt.close(fig)


def build_ordered_grid_indices(Nx, Ny, Nz, order):

    size_map = {'x': Nx, 'y': Ny, 'z': Nz}

    if sorted(order) != ['x', 'y', 'z']:
        raise ValueError("order must be a permutation of 'x', 'y', and 'z'.")

    grids = np.meshgrid(
        *[np.arange(size_map[axis]) for axis in order],
        indexing='ij'
    )

    idx_by_axis = {}

    for axis, grid in zip(order, grids):
        idx_by_axis[axis] = grid.reshape(-1, order='C')

    index_sequence = np.column_stack([
        idx_by_axis['x'],
        idx_by_axis['y'],
        idx_by_axis['z']
    ])

    return index_sequence


def build_ordered_coords(x_vals, y_vals, z_vals, order):

    coord_map = {
        'x': x_vals,
        'y': y_vals,
        'z': z_vals
    }

    if sorted(order) != ['x', 'y', 'z']:
        raise ValueError("order must be a permutation of 'x', 'y', and 'z'.")

    grids = np.meshgrid(
        *[coord_map[axis] for axis in order],
        indexing='ij'
    )

    coord_by_axis = {}

    for axis, grid in zip(order, grids):
        coord_by_axis[axis] = grid.reshape(-1, order='C')

    coords_ordered = np.column_stack([
        coord_by_axis['x'],
        coord_by_axis['y'],
        coord_by_axis['z']
    ])

    return coords_ordered


def compute_vector_adjacency_stats(index_sequence, Nx, Ny, Nz, order_name):

    diffs = np.abs(np.diff(index_sequence, axis=0))

    x_neighbor_mask = (
        (diffs[:, 0] == 1) &
        (diffs[:, 1] == 0) &
        (diffs[:, 2] == 0)
    )

    y_neighbor_mask = (
        (diffs[:, 0] == 0) &
        (diffs[:, 1] == 1) &
        (diffs[:, 2] == 0)
    )

    z_neighbor_mask = (
        (diffs[:, 0] == 0) &
        (diffs[:, 1] == 0) &
        (diffs[:, 2] == 1)
    )

    count_x = int(np.sum(x_neighbor_mask))
    count_y = int(np.sum(y_neighbor_mask))
    count_z = int(np.sum(z_neighbor_mask))

    total_1d_pairs = index_sequence.shape[0] - 1
    true_neighbor_pairs = count_x + count_y + count_z
    non_neighbor_pairs = total_1d_pairs - true_neighbor_pairs

    total_x_edges = (Nx - 1) * Ny * Nz
    total_y_edges = Nx * (Ny - 1) * Nz
    total_z_edges = Nx * Ny * (Nz - 1)
    total_3d_edges = total_x_edges + total_y_edges + total_z_edges

    stats = {
        "Order": order_name,

        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,

        "Total_1D_adjacent_pairs": total_1d_pairs,

        "X_neighbor_count_in_1D": count_x,
        "Y_neighbor_count_in_1D": count_y,
        "Z_neighbor_count_in_1D": count_z,
        "True_3D_neighbor_count_in_1D": true_neighbor_pairs,
        "Non_3D_neighbor_count_in_1D": non_neighbor_pairs,

        "X_ratio_in_all_1D_pairs": count_x / total_1d_pairs,
        "Y_ratio_in_all_1D_pairs": count_y / total_1d_pairs,
        "Z_ratio_in_all_1D_pairs": count_z / total_1d_pairs,
        "True_3D_neighbor_ratio_in_all_1D_pairs": true_neighbor_pairs / total_1d_pairs,
        "Non_3D_neighbor_ratio_in_all_1D_pairs": non_neighbor_pairs / total_1d_pairs,

        "X_share_among_true_1D_neighbors": count_x / true_neighbor_pairs if true_neighbor_pairs > 0 else 0.0,
        "Y_share_among_true_1D_neighbors": count_y / true_neighbor_pairs if true_neighbor_pairs > 0 else 0.0,
        "Z_share_among_true_1D_neighbors": count_z / true_neighbor_pairs if true_neighbor_pairs > 0 else 0.0,

        "Captured_X_edges_ratio": count_x / total_x_edges if total_x_edges > 0 else 0.0,
        "Captured_Y_edges_ratio": count_y / total_y_edges if total_y_edges > 0 else 0.0,
        "Captured_Z_edges_ratio": count_z / total_z_edges if total_z_edges > 0 else 0.0,
        "Captured_total_3D_edges_ratio": true_neighbor_pairs / total_3d_edges if total_3d_edges > 0 else 0.0,
    }

    return stats


def print_adjacency_stats(stats):

    print("\n" + "=" * 20 + " Vectorization Adjacency Analysis " + "=" * 20)
    print(f"Unfolding order: {stats['Order'].upper()}")
    print(f"Grid dimensions: Nx={stats['Nx']}, Ny={stats['Ny']}, Nz={stats['Nz']}")
    print(f"Total 1D adjacent pairs: {stats['Total_1D_adjacent_pairs']}")
    print("-" * 70)

    print("Counts of true 3D neighbors appearing as adjacent elements in 1D vector:")
    print(f"  X-direction neighbors : {stats['X_neighbor_count_in_1D']}")
    print(f"  Y-direction neighbors : {stats['Y_neighbor_count_in_1D']}")
    print(f"  Z-direction neighbors : {stats['Z_neighbor_count_in_1D']}")
    print(f"  True 3D-neighbor pairs: {stats['True_3D_neighbor_count_in_1D']}")
    print(f"  Non-3D-neighbor pairs : {stats['Non_3D_neighbor_count_in_1D']}")
    print("-" * 70)

    print("Ratios among all adjacent pairs in the 1D vector:")
    print(f"  X-direction ratio      : {stats['X_ratio_in_all_1D_pairs']:.6f}")
    print(f"  Y-direction ratio      : {stats['Y_ratio_in_all_1D_pairs']:.6f}")
    print(f"  Z-direction ratio      : {stats['Z_ratio_in_all_1D_pairs']:.6f}")
    print(f"  True 3D-neighbor ratio : {stats['True_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
    print(f"  Non-3D-neighbor ratio  : {stats['Non_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
    print("-" * 70)

    print("Directional share among true 3D neighbors appearing in the 1D vector:")
    print(f"  X share among true 1D neighbors: {stats['X_share_among_true_1D_neighbors']:.6f}")
    print(f"  Y share among true 1D neighbors: {stats['Y_share_among_true_1D_neighbors']:.6f}")
    print(f"  Z share among true 1D neighbors: {stats['Z_share_among_true_1D_neighbors']:.6f}")
    print("-" * 70)

    print("Captured original 3D grid adjacency edges:")
    print(f"  Captured X edges ratio : {stats['Captured_X_edges_ratio']:.6f}")
    print(f"  Captured Y edges ratio : {stats['Captured_Y_edges_ratio']:.6f}")
    print(f"  Captured Z edges ratio : {stats['Captured_Z_edges_ratio']:.6f}")
    print(f"  Captured total 3D edges: {stats['Captured_total_3D_edges_ratio']:.6f}")
    print("=" * 90)




allowed_orders = ['xyz', 'yzx', 'xzy']

chosen_order = ""

while chosen_order not in allowed_orders:
    chosen_order = input("请选择数据展开方式 ('xyz', 'yzx', or 'xzy'): ").lower().strip()

    if chosen_order not in allowed_orders:
        print("输入无效，请重新输入。")

SAVE_DIR = os.path.join(BASE_SAVE_DIR, f'POD_run_{chosen_order}')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n已选择展开方式: '{chosen_order}'. 结果将保存在: {SAVE_DIR}")



all_order_stats = []

for order in allowed_orders:
    idx_seq_tmp = build_ordered_grid_indices(Nx, Ny, Nz, order)

    stats_tmp = compute_vector_adjacency_stats(
        index_sequence=idx_seq_tmp,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        order_name=order
    )

    all_order_stats.append(stats_tmp)

df_all_adjacency = pd.DataFrame(all_order_stats)

adjacency_compare_path = os.path.join(
    BASE_SAVE_DIR,
    "vectorization_adjacency_stats_all_orders.csv"
)

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
df_all_adjacency.to_csv(adjacency_compare_path, index=False)

print("\nSaved adjacency comparison of all vectorization orders to:")
print(adjacency_compare_path)



print(f"\nPreparing data for order '{chosen_order}'...")


axis_to_tensor_axis = {
    'x': 0,
    'y': 1,
    'z': 2
}


spatial_permutation = tuple(axis_to_tensor_axis[axis] for axis in chosen_order)

permuted_tensor = np.transpose(
    all_snapshots_tensor,
    spatial_permutation + (3,)
)


coords = build_ordered_coords(
    x_coords_unique,
    y_coords_unique,
    z_coords_unique,
    chosen_order
)


index_sequence = build_ordered_grid_indices(
    Nx,
    Ny,
    Nz,
    chosen_order
)


adjacency_stats = compute_vector_adjacency_stats(
    index_sequence=index_sequence,
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    order_name=chosen_order
)

print_adjacency_stats(adjacency_stats)


adjacency_stats_path = os.path.join(
    SAVE_DIR,
    "vectorization_adjacency_stats.csv"
)

pd.DataFrame([adjacency_stats]).to_csv(
    adjacency_stats_path,
    index=False
)

print(f"Saved current vectorization adjacency stats to:")
print(adjacency_stats_path)

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

model = MLP(
    input_size=conditions.shape[1],
    output_size=K,
    hidden_layers=HIDDEN_LAYERS
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.MSELoss()

X_train = torch.tensor(
    conditions_train_norm,
    dtype=torch.float32
)

y_train = torch.tensor(
    coeffs_train_norm,
    dtype=torch.float32
)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

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

mae_list = []
rmse_list = []
rRMSE_list = []

for i, original_case_idx in enumerate(tqdm(val_idx, desc="Validating and Visualizing Cases")):
    cond = torch.tensor(
        conditions_val_norm[i:i + 1],
        dtype=torch.float32
    )

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

    visualize_results(
        true_temp=true_temp,
        pred_temp=rec_temp,
        coords=coords,
        case_index=human_case_number,
        save_dir=SAVE_DIR
    )

    print(
        f"  Case {human_case_number:<2d}: "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, rRMSE={rRMSE:.6f}"
    )

mean_mae = np.mean(mae_list)
mean_rmse = np.mean(rmse_list)
mean_rRMSE = np.mean(rRMSE_list)

print("\n" + "=" * 30 + " FINAL SUMMARY " + "=" * 30)
print(f"Unfolding Order       : {chosen_order.upper()}")
print(f"Number of Modes (K)   : {K}")
print("-" * 40)
print("Average Metrics on Test Set:")
print(f"  Mean MAE            : {mean_mae:.4f}")
print(f"  Mean RMSE           : {mean_rmse:.4f}")
print(f"  Mean rRMSE          : {mean_rRMSE:.6f}")
print("-" * 40)
print("Vectorization Adjacency Metrics:")
print(f"  X ratio in 1D adjacent pairs      : {adjacency_stats['X_ratio_in_all_1D_pairs']:.6f}")
print(f"  Y ratio in 1D adjacent pairs      : {adjacency_stats['Y_ratio_in_all_1D_pairs']:.6f}")
print(f"  Z ratio in 1D adjacent pairs      : {adjacency_stats['Z_ratio_in_all_1D_pairs']:.6f}")
print(f"  True 3D-neighbor ratio in 1D pairs: {adjacency_stats['True_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
print(f"  Non-3D-neighbor ratio in 1D pairs : {adjacency_stats['Non_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
print(f"  Captured total 3D edges ratio     : {adjacency_stats['Captured_total_3D_edges_ratio']:.6f}")
print("-" * 40)
print(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds.")
print(f"All results and visualizations saved in: {SAVE_DIR}")


df_metrics = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE": mae_list,
    "RMSE": rmse_list,
    "rRMSE": rRMSE_list
})

per_case_metrics_path = os.path.join(
    SAVE_DIR,
    "per_case_metrics.csv"
)

df_metrics.to_csv(
    per_case_metrics_path,
    index=False
)

print(f"Saved per-case metrics to:")
print(per_case_metrics_path)



summary_dict = {
    "Unfolding_Order": chosen_order,
    "Number_of_Modes_K": K,
    "Mean_MAE": mean_mae,
    "Mean_RMSE": mean_rmse,
    "Mean_rRMSE": mean_rRMSE,
}

summary_dict.update(adjacency_stats)

summary_path = os.path.join(
    SAVE_DIR,
    "summary_metrics_with_adjacency.csv"
)

pd.DataFrame([summary_dict]).to_csv(
    summary_path,
    index=False
)

print(f"Saved summary metrics with adjacency stats to:")
print(summary_path)
