import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import tensorly as tl
from tensorly.decomposition import tucker
import os
from tqdm import tqdm
import warnings
from scipy.interpolate import griddata
import time

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)


# ==============================================================================
# 1. 数据加载与 POD 空间降维 (与POD-NN脚本完全一致)
# ==============================================================================
class DataAndPODManager:
    def __init__(self, boundary_path, snapshot_dir):
        self.boundary_path = boundary_path
        self.snapshot_dir = snapshot_dir
        self.n_conditions = 89
        self.mean_field = None

    def load_data_and_coords(self):
        """加载边界条件、坐标和所有快照"""
        print("加载边界条件...")
        df_bc = pd.read_csv(self.boundary_path, encoding='utf-16-le', delimiter='\t').astype(float)
        param_cols = ['RACK1', 'RACK2', 'RACK3', 'RACK4', 'RACK5', 'RACK6', 'RACK7', 'RACK8', 'RACK9', 'RACK10',
                      'CRAC1送风温度', 'CRAC1送风速率', 'CRAC2送风温度', 'CRAC2送风速率']
        self.boundary_data = df_bc[param_cols].values
        self.bc_scaler = MinMaxScaler()
        self.boundary_data_normalized = self.bc_scaler.fit_transform(self.boundary_data)

        print("加载所有稀疏温度快照...")
        self.snapshots_matrix = np.zeros(
            (pd.read_csv(os.path.join(self.snapshot_dir, "1.csv")).shape[0], self.n_conditions))
        for i in tqdm(range(1, self.n_conditions + 1), desc="读取CSV"):
            file_path = os.path.join(self.snapshot_dir, f"{i}.csv")
            df = pd.read_csv(file_path, encoding='utf-8')
            if i == 1: self.coords = df[['X (m)', 'Y (m)', 'Z (m)']].values
            self.snapshots_matrix[:, i - 1] = df['Temperature'].values

        # 按空间点归一化
        snapshots_matrix_T = self.snapshots_matrix.T
        self.temp_scaler = MinMaxScaler()
        snapshots_matrix_normalized_T = self.temp_scaler.fit_transform(snapshots_matrix_T)
        self.snapshots_matrix_normalized = snapshots_matrix_normalized_T.T

    def perform_pod_on_training_data(self, train_idx, energy_threshold=0.99):
        """仅在训练数据上执行POD"""
        print("\n正在对训练数据执行POD空间降维...")
        snapshots_train_norm = self.snapshots_matrix_normalized[:, train_idx]

        self.mean_field = np.mean(snapshots_train_norm, axis=1, keepdims=True)
        fluctuations_matrix = snapshots_train_norm - self.mean_field

        U, S, Vh = np.linalg.svd(fluctuations_matrix, full_matrices=False)

        cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        self.r_space = np.searchsorted(cumulative_energy, energy_threshold) + 1
        print(f"为保留 {energy_threshold:.2%} 的能量，选择的空间模态数 (r_space) = {self.r_space}")

        self.U_r = U[:, :self.r_space]
        pod_coefficients = (np.diag(S[:self.r_space]) @ Vh[:self.r_space, :]).T

        print(f"空间基 U_r 形状: {self.U_r.shape}")
        print(f"训练集POD系数矩阵形状: {pod_coefficients.shape}")
        return self.U_r, pod_coefficients


# ==============================================================================
# 2. 神经网络定义与训练器 (与POD-NN脚本完全一致)
# ==============================================================================
class ConditionDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MappingNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=20, hidden_dims=[128, 256, 128]):
        super(MappingNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h));
            layers.append(nn.BatchNorm1d(h));
            layers.append(nn.ReLU());
            layers.append(nn.Dropout(0.3));
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)


class NetworkTrainer:
    def __init__(self, model, train_loader, val_loader=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    def train(self, epochs=2000):
        print(f"开始训练神经网络，设备: {self.device}")
        for epoch in range(epochs):
            self.model.train()
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.6f}")
        print("训练完成！")


# ==============================================================================
# 3. 可视化函数 (与POD-NN脚本完全一致)
# ==============================================================================
def visualize_results(true_temp, reconstructed_temp, coords, case_index, save_dir):
    # ... (此处代码与你修改后的POD-NN脚本中的可视化函数完全相同，为简洁省略) ...
    # 主要修改点：cmap='YlOrRd'
    mask, _ = select_adjacent_planes(coords, 'y', 1.2334, 2)
    if not mask.any(): print(f"Case {case_index}: No points found. Skipping visualization."); return
    axis_map = {'x': (1, 2, 'Y (m)', 'Z (m)'), 'y': (0, 2, 'X (m)', 'Z (m)'), 'z': (0, 1, 'X (m)', 'Y (m)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axis_map['y']
    slice_coords_x, slice_coords_y = coords[mask, ax1_idx], coords[mask, ax2_idx]
    true_plane, recon_plane = true_temp[mask], reconstructed_temp[mask]
    grid_x = np.linspace(slice_coords_x.min(), slice_coords_x.max(), 150)
    grid_y = np.linspace(slice_coords_y.min(), slice_coords_y.max(), 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    true_interp = griddata((slice_coords_x, slice_coords_y), true_plane, (grid_X, grid_Y), method='cubic',
                           fill_value=np.nan)
    recon_interp = griddata((slice_coords_x, slice_coords_y), recon_plane, (grid_X, grid_Y), method='cubic',
                            fill_value=np.nan)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    valid_temps = np.concatenate([true_interp[~np.isnan(true_interp)], recon_interp[~np.isnan(recon_interp)]])
    temp_min, temp_max = valid_temps.min(), valid_temps.max()
    levels = np.linspace(temp_min, temp_max, 50)
    error_field = np.abs(true_interp - recon_interp)
    cs1 = axs[0].contourf(grid_X, grid_Y, true_interp, levels=levels, cmap='jet', extend='both');
    axs[0].set_title(f'True Temperature (Case {case_index})', fontweight='bold');
    fig.colorbar(cs1, ax=axs[0], label='Temperature (°C)')
    cs2 = axs[1].contourf(grid_X, grid_Y, recon_interp, levels=levels, cmap='jet', extend='both');
    axs[1].set_title('Reconstructed Temperature', fontweight='bold');
    fig.colorbar(cs2, ax=axs[1], label='Temperature (°C)')
    cs3 = axs[2].contourf(grid_X, grid_Y, error_field, levels=50, cmap='YlOrRd');
    axs[2].set_title('Absolute Error', fontweight='bold');
    fig.colorbar(cs3, ax=axs[2], label='Error (°C)')
    for ax in axs: ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=2.0)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'case_{case_index}_reconstruction.png');
    plt.savefig(save_path, dpi=200, bbox_inches='tight');
    plt.close(fig)
    print(f"为工况 {case_index} 保存可视化结果至: {save_path}")


def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2};
    axis_idx = axis_map[plane_type];
    unique_values = np.sort(np.unique(coords[:, axis_idx]));
    closest_idx = np.argmin(np.abs(unique_values - target_value));
    start_idx = max(0, closest_idx - num_adjacent);
    end_idx = min(len(unique_values), closest_idx + num_adjacent + 1);
    selected_planes = unique_values[start_idx:end_idx];
    mask = np.isin(coords[:, axis_idx], selected_planes)
    return mask, selected_planes


# ==============================================================================
# 4. 主实验流程
# ==============================================================================
def main():
    # --- 配置 ---
    start_time = time.time()
    boundary_path = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
    snapshot_dir = r"C:\Users\Lenovo\Desktop\condition_data_files"
    save_dir = r'C:\Users\Lenovo\Desktop\PODTucker89_Fixed_Test_Set'
    energy_threshold = 0.99
    r_cond = 15  # Tucker分解中工况基的秩
    epochs = 2000

    # 1. --- 数据加载与划分 ---
    data_manager = DataAndPODManager(boundary_path, snapshot_dir)
    data_manager.load_data_and_coords()

    # 【修改】手动指定训练集和测试集索引
    val_indices_human = [1, 22, 26, 30, 35, 36, 38, 41, 48]
    test_idx = [i - 1 for i in val_indices_human]
    train_idx = [i for i in range(data_manager.n_conditions) if i not in test_idx]
    print(f"\nTrain samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # 2. --- 在训练集上执行POD ---
    U_r, pod_coefficients_train = data_manager.perform_pod_on_training_data(train_idx, energy_threshold)

    # 3. --- 【POD-Tucker核心步骤】在POD系数上进行Tucker分解 ---
    print("\n对训练集POD系数进行Tucker分解...")
    r_space = pod_coefficients_train.shape[1]
    core, factors = tucker(pod_coefficients_train, rank=[r_cond, r_space])
    condition_basis_coeffs_train = factors[0]
    mode_coupling_basis = factors[1]
    print(f"核心张量形状: {core.shape}\n工况基系数形状: {condition_basis_coeffs_train.shape}")

    # 4. --- 训练神经网络 ---
    X_train = data_manager.boundary_data_normalized[train_idx]
    y_train = condition_basis_coeffs_train
    train_loader = DataLoader(ConditionDataset(X_train, y_train), batch_size=16, shuffle=True)

    # 网络输出维度为 r_cond
    network = MappingNetwork(input_dim=14, output_dim=r_cond, hidden_dims=[128, 256, 128])
    trainer = NetworkTrainer(network, train_loader)
    trainer.train(epochs=epochs)

    # 5. --- 在指定的测试集上进行评估和可视化 ---
    print("\n在指定的测试集上进行评估...")
    network.eval()
    all_errors = []

    for i, original_idx in enumerate(test_idx):
        case_index_human = original_idx + 1
        print(f"\n--- 处理测试工况 {i + 1}/{len(test_idx)} (原始工况: {case_index_human}) ---")

        # 获取测试数据
        X_test_single = torch.FloatTensor(data_manager.boundary_data_normalized[original_idx:original_idx + 1])
        true_temp_physical = data_manager.snapshots_matrix[:, original_idx]

        # 神经网络预测工况基系数
        with torch.no_grad():
            predicted_cond_basis = network(X_test_single).numpy()

        # Tucker重构POD系数
        predicted_pod_coeffs = tl.tucker_to_tensor((core, [predicted_cond_basis, mode_coupling_basis]))

        # POD重构物理场
        # 注意：这里需要逆归一化，但POD是在归一化空间中进行的
        # 所以先重构归一化的波动场
        reconstructed_fluctuations_norm = U_r @ predicted_pod_coeffs.T
        # 加上平均场
        reconstructed_snapshot_norm = reconstructed_fluctuations_norm + data_manager.mean_field
        # 逆归一化回物理温度
        reconstructed_temp_physical = data_manager.temp_scaler.inverse_transform(
            reconstructed_snapshot_norm.T).flatten()

        all_errors.append(true_temp_physical - reconstructed_temp_physical)

        # 可视化
        visualize_results(true_temp_physical, reconstructed_temp_physical, data_manager.coords, case_index_human,
                          save_dir)

    # 6. --- 计算并打印最终的全局误差 ---
    all_errors = np.array(all_errors).flatten()
    mae = np.mean(np.abs(all_errors))
    rmse = np.sqrt(np.mean(all_errors ** 2))

    print("\n" + "=" * 50)
    print("  POD-Tucker 模型在整个测试集上的最终性能")
    print("=" * 50)
    print(f"  测试集工况: {val_indices_human}")
    print(f"  平均绝对误差 (MAE): {mae:.4f} °C")
    print(f"  均方根误差 (RMSE): {rmse:.4f} °C")
    print("=" * 50)

    end_time = time.time()
    print(f"\n脚本总执行时间: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    main()

