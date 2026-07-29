import pandas as pd
import numpy as np
from scipy.linalg import svd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
import random
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

# 总保存路径
BASE_SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD_Unfolding_Comparison'

# 可视化截面参数
PLANE_TYPE = 'y'
PLANE_VALUE = 1.26
NUM_ADJACENT_PLANES = 0


RELATIVE_ERROR_THRESHOLD = 0.10
ALLOWED_ORDERS = ['yzx','zyx','zxy', 'xzy', 'yxz', 'xyz']
pre_selected_indices = [
    113159, 118413, 118310, 113158, 118516,
    113056, 118207, 113262, 113055, 112953
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

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
        return None

    points_to_interp = coords[mask]

    if PLANE_TYPE == 'y':
        x_plot = points_to_interp[:, 0]
        y_plot = points_to_interp[:, 2]
        x_label = "X (m)"
        y_label = "Z (m)"
    elif PLANE_TYPE == 'x':
        x_plot = points_to_interp[:, 1]
        y_plot = points_to_interp[:, 2]
        x_label = "Y (m)"
        y_label = "Z (m)"
    else:
        x_plot = points_to_interp[:, 0]
        y_plot = points_to_interp[:, 1]
        x_label = "X (m)"
        y_label = "Y (m)"

    t_true = true_temp[mask]
    t_pred = pred_temp[mask]

    grid_x = np.linspace(x_plot.min(), x_plot.max(), 150)
    grid_y = np.linspace(y_plot.min(), y_plot.max(), 150)
    GX, GY = np.meshgrid(grid_x, grid_y)

    Tt = griddata((x_plot, y_plot), t_true, (GX, GY), method='cubic')
    Tp = griddata((x_plot, y_plot), t_pred, (GX, GY), method='cubic')
    err = np.abs(Tt - Tp)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Temperature Field Reconstruction - Case {case_index}', fontsize=16)

    v_min = np.nanmin(Tt)
    v_max = np.nanmax(Tt)

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
            GY,
            data,
            levels=50,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_plot
        )

        fig.colorbar(cs, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)

    fig_path = os.path.join(save_dir, f"case_{case_index}_reconstruction.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    return np.nanmax(err)



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

    dominant_direction = order_name[-1].upper()

    stats = {
        "Order": order_name,
        "Dominant_Direction": dominant_direction,

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

        "Captured_X_edges_ratio": count_x / total_x_edges if total_x_edges > 0 else 0.0,
        "Captured_Y_edges_ratio": count_y / total_y_edges if total_y_edges > 0 else 0.0,
        "Captured_Z_edges_ratio": count_z / total_z_edges if total_z_edges > 0 else 0.0,
        "Captured_total_3D_edges_ratio": true_neighbor_pairs / total_3d_edges if total_3d_edges > 0 else 0.0,
    }

    if dominant_direction == "X":
        stats["Dominant_directional_adjacency_ratio"] = stats["X_ratio_in_all_1D_pairs"]
    elif dominant_direction == "Y":
        stats["Dominant_directional_adjacency_ratio"] = stats["Y_ratio_in_all_1D_pairs"]
    elif dominant_direction == "Z":
        stats["Dominant_directional_adjacency_ratio"] = stats["Z_ratio_in_all_1D_pairs"]
    else:
        stats["Dominant_directional_adjacency_ratio"] = np.nan

    return stats


def print_adjacency_stats(stats):
    print("\n" + "=" * 20 + " Vectorization Adjacency Analysis " + "=" * 20)
    print(f"Unfolding order: {stats['Order'].upper()}")
    print(f"Dominant direction: {stats['Dominant_Direction']}")
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
    print(f"  Dominant direction ratio: {stats['Dominant_directional_adjacency_ratio']:.6f}")
    print(f"  True 3D-neighbor ratio : {stats['True_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
    print(f"  Non-3D-neighbor ratio  : {stats['Non_3D_neighbor_ratio_in_all_1D_pairs']:.6f}")
    print("-" * 70)

    print("Captured original 3D grid adjacency edges:")
    print(f"  Captured X edges ratio : {stats['Captured_X_edges_ratio']:.6f}")
    print(f"  Captured Y edges ratio : {stats['Captured_Y_edges_ratio']:.6f}")
    print(f"  Captured Z edges ratio : {stats['Captured_Z_edges_ratio']:.6f}")
    print(f"  Captured total 3D edges: {stats['Captured_total_3D_edges_ratio']:.6f}")
    print("=" * 90)


def map_hotspot_indices_to_current_order(pre_selected_indices, original_coords, current_coords):

    valid_original_indices = []

    for idx in pre_selected_indices:
        if idx < original_coords.shape[0]:
            valid_original_indices.append(idx)
        else:
            print(f"Warning: hotspot index {idx} is out of range and will be skipped.")

    if not valid_original_indices:
        return [], []

    hotspot_coords = original_coords[valid_original_indices, :]
    tree = cKDTree(current_coords)
    distances, mapped_indices = tree.query(hotspot_coords, k=1)

    return valid_original_indices, mapped_indices



print("=" * 20 + " Step 1: Loading Data " + "=" * 20)

overall_start_time = time.time()

bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]


val_indices_human = [1, 2, 3, 4, 5, 6, 7, 8, 9]
val_idx = [i - 1 for i in val_indices_human]
train_idx = [i for i in range(num_samples) if i not in val_idx]

print(f"Number of samples: {num_samples}")
print(f"Training samples: {len(train_idx)}")
print(f"Test samples: {len(val_idx)}")


scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
conditions_val_norm = scaler_conditions.transform(conditions[val_idx])


temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]

first_df = pd.read_csv(temp_files[0])
original_coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = original_coords.shape[0]

print("Loading all snapshots...")
all_snapshots_flat = np.zeros((N_points, num_samples))

for i in tqdm(range(num_samples), desc="Loading snapshots"):
    df = pd.read_csv(temp_files[i])
    all_snapshots_flat[:, i] = df['Temperature'].values


print("\n" + "=" * 20 + " Step 2: Reshaping to 3D Tensor " + "=" * 20)

x_coords_unique = np.sort(np.unique(original_coords[:, 0]))
y_coords_unique = np.sort(np.unique(original_coords[:, 1]))
z_coords_unique = np.sort(np.unique(original_coords[:, 2]))

Nx, Ny, Nz = len(x_coords_unique), len(y_coords_unique), len(z_coords_unique)

print(f"Grid dimensions inferred: Nx={Nx}, Ny={Ny}, Nz={Nz}")

if Nx * Ny * Nz != N_points:
    raise ValueError(
        "Grid dimensions do not match total points. "
        "Please check whether the CFD data have been interpolated to a regular grid."
    )

# 按 z, y, x 排序，使 reshape(Nz, Ny, Nx) 后再转置为 (Nx, Ny, Nz)
sort_indices = np.lexsort((
    original_coords[:, 0],
    original_coords[:, 1],
    original_coords[:, 2]
))

sorted_coords = original_coords[sort_indices]
all_snapshots_sorted = all_snapshots_flat[sort_indices, :]

all_snapshots_tensor = all_snapshots_sorted.reshape(Nz, Ny, Nx, num_samples)
all_snapshots_tensor = np.transpose(all_snapshots_tensor, (2, 1, 0, 3))

print(f"Snapshot tensor shape: {all_snapshots_tensor.shape}")
print("Tensor axis order: (x, y, z, sample)")

all_order_adjacency_stats = []

for order in ALLOWED_ORDERS:
    idx_seq_tmp = build_ordered_grid_indices(Nx, Ny, Nz, order)
    stats_tmp = compute_vector_adjacency_stats(
        index_sequence=idx_seq_tmp,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        order_name=order
    )
    all_order_adjacency_stats.append(stats_tmp)

df_all_adjacency = pd.DataFrame(all_order_adjacency_stats)

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

adjacency_compare_path = os.path.join(
    BASE_SAVE_DIR,
    "vectorization_adjacency_stats_all_orders.csv"
)

df_all_adjacency.to_csv(
    adjacency_compare_path,
    index=False,
    float_format="%.6f"
)

print(f"\nSaved adjacency comparison to: {adjacency_compare_path}")


def run_pod_for_order(chosen_order):
    print("\n" + "=" * 90)
    print(f"Running POD + MLP reconstruction for unfolding order: {chosen_order.upper()}")
    print("=" * 90)

    set_seed(SEED)

    order_save_dir = os.path.join(BASE_SAVE_DIR, f"POD_{chosen_order}")
    os.makedirs(order_save_dir, exist_ok=True)

    # all_snapshots_tensor 基础轴顺序为 (x, y, z, sample)
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

    coords_current = build_ordered_coords(
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

    pd.DataFrame([adjacency_stats]).to_csv(
        os.path.join(order_save_dir, "vectorization_adjacency_stats.csv"),
        index=False,
        float_format="%.6f"
    )


    snapshots_ordered = permuted_tensor.reshape(N_points, num_samples)

    snapshots_train = snapshots_ordered[:, train_idx]
    snapshots_val = snapshots_ordered[:, val_idx]

    print(f"Snapshot matrix shape: {snapshots_train.shape}")

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

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss = {loss.item():.6f}")

    print("\nValidating on test set...")

    model.eval()

    all_true_val_temps = np.zeros((N_points, len(val_idx)))
    all_pred_val_temps = np.zeros((N_points, len(val_idx)))

    mae_list = []
    rmse_list = []
    rRMSE_list = []
    percentage_below_threshold_list = []

    global_error_max = 0

    for i, idx in enumerate(tqdm(val_idx, desc=f"Validating cases ({chosen_order})")):
        cond = torch.tensor(
            conditions_val_norm[i:i + 1],
            dtype=torch.float32
        )

        with torch.no_grad():
            pred_coeff_norm = model(cond).numpy()

        pred_coeff = scaler_coeffs.inverse_transform(pred_coeff_norm).flatten()

        rec_temp = mean_temp.flatten() + (modes @ pred_coeff)
        true_temp = snapshots_val[:, i]

        all_true_val_temps[:, i] = true_temp
        all_pred_val_temps[:, i] = rec_temp

        err_vector = true_temp - rec_temp

        mae = np.mean(np.abs(err_vector))
        rmse = np.sqrt(np.mean(err_vector ** 2))
        rRMSE_case = np.linalg.norm(err_vector) / (np.linalg.norm(true_temp) + 1e-9)

        mae_list.append(mae)
        rmse_list.append(rmse)
        rRMSE_list.append(rRMSE_case)

        relative_error = np.abs(err_vector) / (np.abs(true_temp) + 1e-9)
        num_points_below = np.sum(relative_error < RELATIVE_ERROR_THRESHOLD)
        percentage = (num_points_below / len(true_temp)) * 100
        percentage_below_threshold_list.append(percentage)

        print(
            f"Case {idx + 1:2d}: "
            f"MAE={mae:.4f} °C, "
            f"RMSE={rmse:.4f} °C, "
            f"rRMSE={rRMSE_case:.4f}, "
            f"Points ratio(RelErr < {int(RELATIVE_ERROR_THRESHOLD * 100)}%)={percentage:.2f}%"
        )

        vmax_now = visualize_results(
            true_temp=true_temp,
            pred_temp=rec_temp,
            coords=coords_current,
            case_index=idx + 1,
            save_dir=order_save_dir
        )

        if vmax_now is not None:
            global_error_max = max(global_error_max, vmax_now)

    np.save(
        os.path.join(order_save_dir, "global_error_max.npy"),
        global_error_max
    )

    print("\nSummarizing 10 hotspot temperatures and errors...")

    valid_original_indices, current_hotspot_indices = map_hotspot_indices_to_current_order(
        pre_selected_indices=pre_selected_indices,
        original_coords=original_coords,
        current_coords=coords_current
    )

    hot_spot_data_list = []
    rRMSE_n_results = {}

    if len(valid_original_indices) > 0:
        for original_point_idx, current_point_idx in zip(valid_original_indices, current_hotspot_indices):
            point_data = {
                "Original_Point_Index": int(original_point_idx),
                "Current_Order_Point_Index": int(current_point_idx),
                "X (m)": coords_current[current_point_idx, 0],
                "Y (m)": coords_current[current_point_idx, 1],
                "Z (m)": coords_current[current_point_idx, 2]
            }

            true_series = all_true_val_temps[current_point_idx, :]
            pred_series = all_pred_val_temps[current_point_idx, :]
            error_series = np.abs(true_series - pred_series)

            for j, case_idx in enumerate(val_idx):
                case_num = case_idx + 1
                point_data[f"Case_{case_num}_True_T"] = true_series[j]
                point_data[f"Case_{case_num}_Pred_T"] = pred_series[j]
                point_data[f"Case_{case_num}_AbsError_T"] = error_series[j]

            rRMSE_n_value = np.linalg.norm(true_series - pred_series) / (np.linalg.norm(true_series) + 1e-9)
            point_data["rRMSE_n"] = rRMSE_n_value

            rRMSE_n_results[int(current_point_idx)] = rRMSE_n_value
            hot_spot_data_list.append(point_data)

        df_hot_spots = pd.DataFrame(hot_spot_data_list)
        df_hot_spots.to_csv(
            os.path.join(order_save_dir, "hot_spots_temperature_summary.csv"),
            index=False,
            float_format="%.6f"
        )

        print(f"Hotspot summary saved to: {os.path.join(order_save_dir, 'hot_spots_temperature_summary.csv')}")
    else:
        print("No valid hotspot indices were found.")

    df_metrics = pd.DataFrame({
        "Case": [i + 1 for i in val_idx],
        "MAE (°C)": mae_list,
        "RMSE (°C)": rmse_list,
        "rRMSE": rRMSE_list,
        f"Points_Percentage_RelErr_<{int(RELATIVE_ERROR_THRESHOLD * 100)}%": percentage_below_threshold_list
    })

    df_metrics.to_csv(
        os.path.join(order_save_dir, "per_case_metrics.csv"),
        index=False,
        float_format="%.6f"
    )

    plt.figure(figsize=(10, 6))
    plt.plot(df_metrics["Case"], df_metrics["MAE (°C)"], '-o', label="MAE (°C)")
    plt.plot(df_metrics["Case"], df_metrics["RMSE (°C)"], '-s', label="RMSE (°C)")
    plt.plot(df_metrics["Case"], df_metrics["rRMSE"], '-^', label="rRMSE")
    plt.xlabel("Case")
    plt.ylabel("Error")
    plt.title(f"POD + MLP Error Metrics ({chosen_order.upper()})")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(
        os.path.join(order_save_dir, "error_metrics_compare.png"),
        dpi=200
    )
    plt.close()

    mean_mae = np.mean(mae_list)
    mean_rmse = np.mean(rmse_list)
    mean_rRMSE = np.mean(rRMSE_list)
    mean_percentage = np.mean(percentage_below_threshold_list)

    mean_hotspot_rRMSE_n = (
        np.mean(list(rRMSE_n_results.values()))
        if len(rRMSE_n_results) > 0
        else np.nan
    )

    summary_dict = {
        "Unfolding_Order": chosen_order,
        "Dominant_Direction": adjacency_stats["Dominant_Direction"],
        "Number_of_Modes_K": K,

        "Mean_MAE": mean_mae,
        "Mean_RMSE": mean_rmse,
        "Mean_rRMSE": mean_rRMSE,
        f"Mean_Points_Percentage_RelErr_<{int(RELATIVE_ERROR_THRESHOLD * 100)}%": mean_percentage,

        "Mean_hotspot_rRMSE_n": mean_hotspot_rRMSE_n,
        "Global_Error_Max": global_error_max,
    }

    summary_dict.update(adjacency_stats)

    summary_path = os.path.join(
        order_save_dir,
        "summary_metrics_with_adjacency.csv"
    )

    pd.DataFrame([summary_dict]).to_csv(
        summary_path,
        index=False,
        float_format="%.6f"
    )

    print("\n" + "=" * 30 + f" SUMMARY {chosen_order.upper()} " + "=" * 30)
    print(f"Dominant direction: {adjacency_stats['Dominant_Direction']}")
    print(f"K modes: {K}")
    print(f"Mean MAE: {mean_mae:.4f} °C")
    print(f"Mean RMSE: {mean_rmse:.4f} °C")
    print(f"Mean rRMSE: {mean_rRMSE:.6f}")
    print(f"Mean points ratio(RelErr < {int(RELATIVE_ERROR_THRESHOLD * 100)}%): {mean_percentage:.2f}%")
    print(f"Mean hotspot rRMSE_n: {mean_hotspot_rRMSE_n:.6f}")
    print(f"Dominant directional adjacency ratio: {adjacency_stats['Dominant_directional_adjacency_ratio'] * 100:.2f}%")
    print(f"Captured total 3D edges ratio: {adjacency_stats['Captured_total_3D_edges_ratio'] * 100:.2f}%")
    print(f"Results saved in: {order_save_dir}")
    print("=" * 90)

    return summary_dict

all_summary_list = []

for order in ALLOWED_ORDERS:
    summary = run_pod_for_order(order)
    all_summary_list.append(summary)

df_all_summary = pd.DataFrame(all_summary_list)

all_summary_path = os.path.join(
    BASE_SAVE_DIR,
    "all_unfolding_orders_summary.csv"
)

df_all_summary.to_csv(
    all_summary_path,
    index=False,
    float_format="%.6f"
)

print("\n" + "=" * 30 + " ALL UNFOLDING ORDERS SUMMARY " + "=" * 30)

summary_columns_to_print = [
    "Unfolding_Order",
    "Dominant_Direction",
    "Number_of_Modes_K",
    "Mean_MAE",
    "Mean_RMSE",
    "Mean_rRMSE",
    f"Mean_Points_Percentage_RelErr_<{int(RELATIVE_ERROR_THRESHOLD * 100)}%",
    "Mean_hotspot_rRMSE_n",
    "Dominant_directional_adjacency_ratio",
    "Captured_total_3D_edges_ratio"
]

print(df_all_summary[summary_columns_to_print])

print(f"\nAll unfolding order results saved to: {all_summary_path}")
print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")
