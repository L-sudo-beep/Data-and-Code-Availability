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
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
BASE_SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD_Single_Run_Results'

# --- 温度场切片可视化与误差计算参数 ---
PLANE_TYPE = 'y'
PLANE_VALUE = 2.52
NUM_ADJACENT_PLANES = 0


# ======================================
# 三维邻接可视化参数
# ======================================
PLOT_3D_ADJACENCY = True

# 是否对 xyz / yzx / xzy 三种展开方式都绘制三维邻接图
PLOT_ALL_ORDERS_3D_ADJACENCY = True

# 为避免网格点太多导致图片过密或绘图过慢，设置最大绘制点数
MAX_POINTS_TO_PLOT_3D = 30000

# 每个方向最多绘制多少条真实邻接边
MAX_TRUE_EDGES_PER_DIRECTION = 8000

# 如果想看一维展开导致的非真实三维邻接跳跃，可以改成 1000 或 3000
MAX_NON_NEIGHBOR_JUMPS = 0

ADJACENCY_POINT_SIZE = 5
ADJACENCY_LINE_WIDTH = 0.6


# ======================================
# 三维邻接方向箭头参数
# ======================================
SHOW_DOMINANT_DIRECTION_ARROW = True

# 箭头长度，占对应坐标轴尺寸的比例
ADJACENCY_ARROW_LENGTH_RATIO = 0.35

# 箭头线宽
ADJACENCY_ARROW_LINE_WIDTH = 5.5

# 箭头标签偏移，占点云最大尺寸的比例
ADJACENCY_ARROW_LABEL_OFFSET_RATIO = 0.03


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
# 1. Load Common Data
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


# ======================================
# Helper Functions: MLP
# ======================================
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


# ======================================
# Helper Functions: Temperature Visualization
# ======================================
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


# ======================================
# Helper Functions: Vectorization Adjacency Analysis
# ======================================
def build_ordered_grid_indices(Nx, Ny, Nz, order):
    """
    根据向量化展开顺序，生成一维向量中每个位置对应的原始三维网格索引 (ix, iy, iz)。

    order:
        'xyz'：张量轴顺序为 (x, y, z)，C-order flatten 时 z 方向变化最快；
        'yzx'：张量轴顺序为 (y, z, x)，C-order flatten 时 x 方向变化最快；
        'xzy'：张量轴顺序为 (x, z, y)，C-order flatten 时 y 方向变化最快。
    """
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
    """
    根据向量化展开顺序，生成与 all_snapshots 行顺序严格一致的坐标数组。

    返回的 coords_ordered 始终为 [X, Y, Z] 三列。
    """
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
    """
    统计一维向量中相邻元素映射回三维网格后，分别属于 x/y/z 方向真实邻接的比例。
    """
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

    # 原始三维规则网格中的全部真实邻接边数量，采用 6-neighbor 邻接定义
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
    """
    打印当前展开方式的空间邻接统计结果。
    """
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


# ======================================
# Helper Functions: 3D 邻接关系可视化
# 重要：
# 物理坐标中 X 和 Z 是水平方向，Y 是垂直方向。
# 因此绘图时使用 Matplotlib 坐标：
#   plot_x = physical X
#   plot_y = physical Z
#   plot_z = physical Y
# ======================================
def classify_1d_adjacent_pairs(index_sequence):
    """
    判断一维向量中相邻元素对，在三维网格中是否为真实邻接点。
    """
    diffs = np.abs(np.diff(index_sequence, axis=0))

    x_mask = (
        (diffs[:, 0] == 1) &
        (diffs[:, 1] == 0) &
        (diffs[:, 2] == 0)
    )

    y_mask = (
        (diffs[:, 0] == 0) &
        (diffs[:, 1] == 1) &
        (diffs[:, 2] == 0)
    )

    z_mask = (
        (diffs[:, 0] == 0) &
        (diffs[:, 1] == 0) &
        (diffs[:, 2] == 1)
    )

    true_mask = x_mask | y_mask | z_mask
    non_neighbor_mask = ~true_mask

    pair_indices_by_type = {
        'x': np.where(x_mask)[0],
        'y': np.where(y_mask)[0],
        'z': np.where(z_mask)[0],
        'non_neighbor': np.where(non_neighbor_mask)[0]
    }

    return pair_indices_by_type


def convert_physical_coords_to_plot_coords(coords_xyz):
    """
    将物理坐标 [X, Y, Z] 转换成用于绘图的坐标 [X, Z, Y]。
    """
    coords_plot = np.column_stack([
        coords_xyz[:, 0],  # plot_x = physical X
        coords_xyz[:, 2],  # plot_y = physical Z
        coords_xyz[:, 1]   # plot_z = physical Y, vertical
    ])

    return coords_plot


def convert_physical_segments_to_plot_segments(segments_xyz):
    """
    将线段坐标从物理 [X, Y, Z] 转成绘图 [X, Z, Y]。

    segments_xyz shape:
        (n_segments, 2, 3)
    """
    segments_plot = segments_xyz[:, :, [0, 2, 1]]
    return segments_plot


def sample_indices(indices, max_count, seed=42):
    """
    从索引数组中抽样，避免三维图过密。
    """
    indices = np.asarray(indices)

    if max_count is None:
        return indices

    if max_count <= 0:
        return np.array([], dtype=int)

    if len(indices) <= max_count:
        return indices

    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_count, replace=False)
    sampled = np.sort(sampled)

    return sampled


def set_axes_equal_3d(ax):
    """
    让 matplotlib 3D 坐标轴按照真实比例显示，避免几何形状被拉伸。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_dominant_neighbor_direction(order_name):
    """
    对于 C-order flatten，最后一个轴变化最快。
    因此一维向量中连续相邻点最容易集中在 order_name 的最后一个方向。

    例如：
        xyz -> z
        yzx -> x
        xzy -> y
    """
    return order_name[-1]


def get_plot_direction_vector_for_physical_direction(physical_direction):
    """
    根据物理邻接方向，返回 Matplotlib 绘图坐标中的方向向量。

    物理坐标：
        X：水平
        Y：垂直
        Z：水平

    绘图坐标：
        plot_x = physical X
        plot_y = physical Z
        plot_z = physical Y

    因此：
        physical X -> plot [1, 0, 0]
        physical Y -> plot [0, 0, 1]
        physical Z -> plot [0, 1, 0]
    """
    direction_map = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 1.0]),
        'z': np.array([0.0, 1.0, 0.0])
    }

    if physical_direction not in direction_map:
        raise ValueError("physical_direction must be 'x', 'y', or 'z'.")

    return direction_map[physical_direction]


def add_dominant_direction_arrow(
    ax,
    coords_plot,
    dominant_direction,
    order_name,
    color_map,
    arrow_length_ratio=0.35,
    arrow_line_width=5.5,
    label_offset_ratio=0.03
):
    """
    在三维图中添加一个清晰、显眼的主邻接方向箭头。

    设计目标：
    - 不把箭头埋在点云内部；
    - 把箭头放在图的前下方 / 边缘区域；
    - 用与主邻接方向一致的颜色；
    - 更符合论文图或展示图中的“示意箭头”风格。

    coords_plot:
        已经是绘图坐标 [physical X, physical Z, physical Y]
    """
    coord_min = coords_plot.min(axis=0)
    coord_max = coords_plot.max(axis=0)
    coord_range = coord_max - coord_min
    max_range = np.max(coord_range)

    xmin, ymin, zmin = coord_min
    xmax, ymax, zmax = coord_max
    dx, dy, dz = coord_range

    direction_vector = get_plot_direction_vector_for_physical_direction(
        dominant_direction
    )

    direction_axis = int(np.argmax(np.abs(direction_vector)))
    direction_axis_range = coord_range[direction_axis]

    if direction_axis_range <= 0:
        direction_axis_range = max_range

    # 箭头长度基于当前方向对应坐标轴的尺寸，避免超出图框
    arrow_length = arrow_length_ratio * direction_axis_range
    label_offset = label_offset_ratio * max_range

    # 将箭头放在图的前下方或边缘区域，而不是点云内部
    if dominant_direction == 'x':
        # X 方向箭头：图前下方，从左往右
        arrow_start = np.array([
            xmin + 0.08 * dx,
            ymin + 0.06 * dy,
            zmin + 0.08 * dz
        ])

    elif dominant_direction == 'z':
        # Z 方向箭头：图前下方，沿 physical Z 水平方向
        arrow_start = np.array([
            xmin + 0.08 * dx,
            ymin + 0.08 * dy,
            zmin + 0.08 * dz
        ])

    elif dominant_direction == 'y':
        # Y 方向箭头：竖直向上，放在左前下角
        arrow_start = np.array([
            xmin + 0.10 * dx,
            ymin + 0.08 * dy,
            zmin + 0.05 * dz
        ])

    else:
        raise ValueError("dominant_direction must be 'x', 'y', or 'z'.")

    arrow_end = arrow_start + arrow_length * direction_vector

    physical_direction_name_map = {
        'x': 'Physical X direction',
        'y': 'Physical Y vertical direction',
        'z': 'Physical Z direction'
    }

    label_text = (
        f"Dominant adjacency\n"
        f"{order_name.upper()} -> {physical_direction_name_map[dominant_direction]}"
    )

    ax.quiver(
        arrow_start[0],
        arrow_start[1],
        arrow_start[2],
        direction_vector[0],
        direction_vector[1],
        direction_vector[2],
        length=arrow_length,
        normalize=True,
        color=color_map[dominant_direction],
        linewidth=arrow_line_width,
        arrow_length_ratio=0.20
    )

    text_position = arrow_end + label_offset * direction_vector

    ax.text(
        text_position[0],
        text_position[1],
        text_position[2],
        label_text,
        color=color_map[dominant_direction],
        fontsize=10,
        fontweight='bold'
    )


def visualize_vectorization_adjacency_3d(
    coords_ordered,
    index_sequence,
    adjacency_stats,
    order_name,
    save_dir,
    max_points_to_plot=30000,
    max_true_edges_per_direction=8000,
    max_non_neighbor_jumps=0,
    point_size=5,
    line_width=0.6,
    seed=42
):
    """
    将一维展开中相邻的元素对，映射回三维物理空间进行可视化。

    重要物理设定：
    - X 和 Z 是水平方向；
    - Y 是垂直方向。

    因此绘图时使用：
    - 图中 X 轴 = 物理 X；
    - 图中 Y 轴 = 物理 Z；
    - 图中 Z 轴 = 物理 Y，也就是竖直方向。
    """
    os.makedirs(save_dir, exist_ok=True)

    pair_indices_by_type = classify_1d_adjacent_pairs(index_sequence)
    dominant_direction = get_dominant_neighbor_direction(order_name)

    color_map = {
        'x': 'red',
        'y': 'green',
        'z': 'blue',
        'non_neighbor': 'gray'
    }

    label_map = {
        'x': (
            f"X horizontal true neighbors: "
            f"{adjacency_stats['X_neighbor_count_in_1D']} "
            f"({adjacency_stats['X_ratio_in_all_1D_pairs']:.4f})"
        ),
        'y': (
            f"Y vertical true neighbors: "
            f"{adjacency_stats['Y_neighbor_count_in_1D']} "
            f"({adjacency_stats['Y_ratio_in_all_1D_pairs']:.4f})"
        ),
        'z': (
            f"Z horizontal true neighbors: "
            f"{adjacency_stats['Z_neighbor_count_in_1D']} "
            f"({adjacency_stats['Z_ratio_in_all_1D_pairs']:.4f})"
        ),
        'non_neighbor': 'Non-3D-neighbor jumps'
    }

    # 将物理坐标 [X, Y, Z] 转换为绘图坐标 [X, Z, Y]
    coords_plot = convert_physical_coords_to_plot_coords(coords_ordered)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制全部网格点，若点数过多则抽样
    all_point_indices = np.arange(coords_plot.shape[0])
    sampled_all_points = sample_indices(
        all_point_indices,
        max_points_to_plot,
        seed=seed
    )

    ax.scatter(
        coords_plot[sampled_all_points, 0],
        coords_plot[sampled_all_points, 1],
        coords_plot[sampled_all_points, 2],
        c='lightgray',
        s=max(point_size * 0.5, 1),
        alpha=0.20,
        label='Grid points'
    )

    # 绘制 X/Y/Z 方向真实邻接边
    for direction in ['x', 'y', 'z']:
        pair_ids = pair_indices_by_type[direction]

        sampled_pair_ids = sample_indices(
            pair_ids,
            max_true_edges_per_direction,
            seed=seed
        )

        if len(sampled_pair_ids) == 0:
            continue

        segments_physical = np.stack(
            [
                coords_ordered[sampled_pair_ids],
                coords_ordered[sampled_pair_ids + 1]
            ],
            axis=1
        )

        segments_plot = convert_physical_segments_to_plot_segments(
            segments_physical
        )

        # 如果是当前展开方式下的主邻接方向，就画得更明显
        if direction == dominant_direction:
            current_line_width = line_width * 2.4
            current_alpha = 0.95
            current_point_size = point_size * 2.0
        else:
            current_line_width = line_width
            current_alpha = 0.45
            current_point_size = point_size

        line_collection = Line3DCollection(
            segments_plot,
            colors=color_map[direction],
            linewidths=current_line_width,
            alpha=current_alpha
        )

        ax.add_collection3d(line_collection)

        highlighted_point_ids = np.unique(
            np.concatenate([sampled_pair_ids, sampled_pair_ids + 1])
        )

        ax.scatter(
            coords_plot[highlighted_point_ids, 0],
            coords_plot[highlighted_point_ids, 1],
            coords_plot[highlighted_point_ids, 2],
            c=color_map[direction],
            s=current_point_size,
            alpha=current_alpha,
            label=label_map[direction]
        )

    # 可选：绘制一维相邻但三维不相邻的跳跃边
    if max_non_neighbor_jumps is not None and max_non_neighbor_jumps > 0:
        non_pair_ids = pair_indices_by_type['non_neighbor']

        sampled_non_pair_ids = sample_indices(
            non_pair_ids,
            max_non_neighbor_jumps,
            seed=seed
        )

        if len(sampled_non_pair_ids) > 0:
            non_segments_physical = np.stack(
                [
                    coords_ordered[sampled_non_pair_ids],
                    coords_ordered[sampled_non_pair_ids + 1]
                ],
                axis=1
            )

            non_segments_plot = convert_physical_segments_to_plot_segments(
                non_segments_physical
            )

            non_line_collection = Line3DCollection(
                non_segments_plot,
                colors=color_map['non_neighbor'],
                linewidths=max(line_width * 0.5, 0.2),
                alpha=0.12
            )

            ax.add_collection3d(non_line_collection)

    dominant_name_map = {
        'x': 'X horizontal direction',
        'y': 'Y vertical direction',
        'z': 'Z horizontal direction'
    }

    title_text = (
        f"3D Vectorization Adjacency - Order {order_name.upper()}\n"
        f"Dominant adjacent direction under C-order flatten: "
        f"{dominant_name_map[dominant_direction]}\n"
        f"True 3D-neighbor ratio in 1D pairs = "
        f"{adjacency_stats['True_3D_neighbor_ratio_in_all_1D_pairs']:.6f}, "
        f"Captured total 3D edges ratio = "
        f"{adjacency_stats['Captured_total_3D_edges_ratio']:.6f}"
    )

    ax.set_title(title_text, fontsize=12)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")

    ax.set_xlim(coords_plot[:, 0].min(), coords_plot[:, 0].max())
    ax.set_ylim(coords_plot[:, 1].min(), coords_plot[:, 1].max())
    ax.set_zlim(coords_plot[:, 2].min(), coords_plot[:, 2].max())

    set_axes_equal_3d(ax)

    # 添加明显的主邻接方向箭头
    if SHOW_DOMINANT_DIRECTION_ARROW:
        add_dominant_direction_arrow(
            ax=ax,
            coords_plot=coords_plot,
            dominant_direction=dominant_direction,
            order_name=order_name,
            color_map=color_map,
            arrow_length_ratio=ADJACENCY_ARROW_LENGTH_RATIO,
            arrow_line_width=ADJACENCY_ARROW_LINE_WIDTH,
            label_offset_ratio=ADJACENCY_ARROW_LABEL_OFFSET_RATIO
        )

    # 让视角更容易看出 X-Z 水平面和 Y 竖直方向
    ax.view_init(elev=22, azim=-55)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        loc='upper right',
        fontsize=8
    )

    plt.tight_layout()

    save_path = os.path.join(
        save_dir,
        f"vectorization_adjacency_3d_{order_name}_physical_Y_vertical_with_clear_arrow.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved physical 3D vectorization adjacency visualization with clear arrow to:")
    print(save_path)


def visualize_all_orders_adjacency_3d(
    allowed_orders,
    x_coords_unique,
    y_coords_unique,
    z_coords_unique,
    Nx,
    Ny,
    Nz,
    base_save_dir
):
    """
    对所有展开方式绘制三维邻接可视化图。

    注意：
    图中物理 Y 轴为竖直方向。
    """
    adjacency_3d_dir = os.path.join(
        base_save_dir,
        "vectorization_adjacency_3d_plots_all_orders_Y_vertical_with_clear_arrow"
    )

    os.makedirs(adjacency_3d_dir, exist_ok=True)

    for order in allowed_orders:
        print(f"\nGenerating physical 3D adjacency visualization for order '{order}'...")

        idx_seq = build_ordered_grid_indices(
            Nx,
            Ny,
            Nz,
            order
        )

        coords_ordered = build_ordered_coords(
            x_coords_unique,
            y_coords_unique,
            z_coords_unique,
            order
        )

        stats = compute_vector_adjacency_stats(
            index_sequence=idx_seq,
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            order_name=order
        )

        visualize_vectorization_adjacency_3d(
            coords_ordered=coords_ordered,
            index_sequence=idx_seq,
            adjacency_stats=stats,
            order_name=order,
            save_dir=adjacency_3d_dir,
            max_points_to_plot=MAX_POINTS_TO_PLOT_3D,
            max_true_edges_per_direction=MAX_TRUE_EDGES_PER_DIRECTION,
            max_non_neighbor_jumps=MAX_NON_NEIGHBOR_JUMPS,
            point_size=ADJACENCY_POINT_SIZE,
            line_width=ADJACENCY_LINE_WIDTH,
            seed=SEED
        )


# ======================================
# 3. Main Analysis and Execution
# ======================================

# --- 用户输入选择展开方式 ---
allowed_orders = ['xyz', 'yzx', 'xzy']

chosen_order = ""

while chosen_order not in allowed_orders:
    chosen_order = input("请选择数据展开方式 ('xyz', 'yzx', or 'xzy'): ").lower().strip()

    if chosen_order not in allowed_orders:
        print("输入无效，请重新输入。")

SAVE_DIR = os.path.join(BASE_SAVE_DIR, f'POD_run_{chosen_order}')
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n已选择展开方式: '{chosen_order}'. 结果将保存在: {SAVE_DIR}")


# ======================================
# Optional: Compare adjacency ratios of all vectorization orders
# ======================================
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


# ======================================
# 新增：绘制所有展开方式的三维邻接关系图
# ======================================
if PLOT_3D_ADJACENCY and PLOT_ALL_ORDERS_3D_ADJACENCY:
    visualize_all_orders_adjacency_3d(
        allowed_orders=allowed_orders,
        x_coords_unique=x_coords_unique,
        y_coords_unique=y_coords_unique,
        z_coords_unique=z_coords_unique,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        base_save_dir=BASE_SAVE_DIR
    )


# ======================================
# Step A: 根据选择的展开方式准备数据
# ======================================
print(f"\nPreparing data for order '{chosen_order}'...")

# all_snapshots_tensor 的基础轴顺序是 (x, y, z, sample)
axis_to_tensor_axis = {
    'x': 0,
    'y': 1,
    'z': 2
}

# 根据 chosen_order 自动生成转置顺序
spatial_permutation = tuple(axis_to_tensor_axis[axis] for axis in chosen_order)

permuted_tensor = np.transpose(
    all_snapshots_tensor,
    spatial_permutation + (3,)
)

# 生成与 permuted_tensor.reshape 后的行顺序严格一致的坐标
coords = build_ordered_coords(
    x_coords_unique,
    y_coords_unique,
    z_coords_unique,
    chosen_order
)

# 生成当前展开顺序下的一维向量空间索引序列
index_sequence = build_ordered_grid_indices(
    Nx,
    Ny,
    Nz,
    chosen_order
)

# 计算当前展开顺序下的一维相邻元素对应真实 x/y/z 邻接比例
adjacency_stats = compute_vector_adjacency_stats(
    index_sequence=index_sequence,
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    order_name=chosen_order
)

print_adjacency_stats(adjacency_stats)

# 保存当前展开方式的空间邻接统计结果
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


# ======================================
# 新增：单独保存当前展开方式的三维邻接图到当前 run 文件夹
# ======================================
if PLOT_3D_ADJACENCY:
    visualize_vectorization_adjacency_3d(
        coords_ordered=coords,
        index_sequence=index_sequence,
        adjacency_stats=adjacency_stats,
        order_name=chosen_order,
        save_dir=SAVE_DIR,
        max_points_to_plot=MAX_POINTS_TO_PLOT_3D,
        max_true_edges_per_direction=MAX_TRUE_EDGES_PER_DIRECTION,
        max_non_neighbor_jumps=MAX_NON_NEIGHBOR_JUMPS,
        point_size=ADJACENCY_POINT_SIZE,
        line_width=ADJACENCY_LINE_WIDTH,
        seed=SEED
    )


all_snapshots = permuted_tensor.reshape(N_points, num_samples)

snapshots_train = all_snapshots[:, train_idx]
snapshots_val = all_snapshots[:, val_idx]

print(f"Snapshot matrix shape for this run: {snapshots_train.shape}")


# ======================================
# Step B: POD Analysis
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
# Step C: MLP Training
# ======================================
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


# ======================================
# Step D: Validation & Error Calculation
# ======================================
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


# ======================================
# 4. Final Summary
# ======================================
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


# ======================================
# 5. Save Metrics
# ======================================
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


# 保存一个综合 summary 文件，方便后续论文制表
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
