import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorly as tl
from tensorly.decomposition import tucker
import os
import pickle
from tqdm import tqdm
import warnings
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)




class DataAndPODManager:
    def __init__(self, boundary_path, snapshot_dir):
        self.boundary_path = boundary_path
        self.snapshot_dir = snapshot_dir
        self.n_conditions = 89
        self.mean_field = None

    def load_data(self):
        print("加载边界条件...")
        df_bc = pd.read_csv(self.boundary_path, encoding='utf-16-le', delimiter='\t')
        param_cols = ['RACK1', 'RACK2', 'RACK3', 'RACK4', 'RACK5',
                      'RACK6', 'RACK7', 'RACK8', 'RACK9', 'RACK10',
                      'CRAC1送风温度', 'CRAC1送风速率', 'CRAC2送风温度', 'CRAC2送风速率']
        self.boundary_data = df_bc[param_cols].values
        self.bc_scaler = MinMaxScaler()
        self.boundary_data_normalized = self.bc_scaler.fit_transform(self.boundary_data)


        print("加载稀疏温度快照...")
        snapshots_list = []
        for i in tqdm(range(1, self.n_conditions + 1), desc="读取CSV"):
            file_path = os.path.join(self.snapshot_dir, f"{i}.csv")
            df = pd.read_csv(file_path, encoding='utf-8')
            if i == 1: self.coords = df[['X (m)', 'Y (m)', 'Z (m)']].values
            snapshots_list.append(df['Temperature'].values)
        self.snapshots_matrix = np.column_stack(snapshots_list)  # 形状 [n_points, n_conditions]

        snapshots_matrix_T = self.snapshots_matrix.T
        self.temp_scaler = MinMaxScaler()
        snapshots_matrix_normalized_T = self.temp_scaler.fit_transform(snapshots_matrix_T)
        # 转置回 (n_points, n_conditions)
        self.snapshots_matrix_normalized = snapshots_matrix_normalized_T.T

        return self.boundary_data_normalized, self.snapshots_matrix_normalized, self.coords

    def perform_pod(self, energy_threshold=0.9999):

        print("\n正在执行POD空间降维...")


        self.mean_field = np.mean(self.snapshots_matrix_normalized, axis=1, keepdims=True)
        fluctuations_matrix = self.snapshots_matrix_normalized - self.mean_field


        U, S, Vh = np.linalg.svd(fluctuations_matrix, full_matrices=False)


        cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        self.r_space = np.searchsorted(cumulative_energy, energy_threshold) + 1
        print(f"为保留 {energy_threshold:.2%} 的能量，选择的空间模态数 (r_space) = {self.r_space}")

        self.U_r = U[:, :self.r_space]


        self.pod_coefficients = (np.diag(S[:self.r_space]) @ Vh[:self.r_space, :]).T

        print(f"空间基 U_r 形状: {self.U_r.shape}")
        print(f"POD系数矩阵形状: {self.pod_coefficients.shape}")

        return self.U_r, self.pod_coefficients


class PODTuckerModel:
    def __init__(self, pod_coefficients, boundary_data_normalized, r_cond=15):  # r_cond 默认值设为15
        self.pod_coefficients = pod_coefficients
        self.boundary_data = boundary_data_normalized
        self.r_cond = r_cond

    def decompose_coefficients(self):
        print("\n对POD系数矩阵进行Tucker分解以提取工况基...")
        r_space = self.pod_coefficients.shape[1]
        self.core, self.factors = tucker(self.pod_coefficients, rank=[self.r_cond, r_space])
        self.condition_basis_coeffs = self.factors[0]
        self.mode_coupling_basis = self.factors[1]
        print(f"核心张量形状: {self.core.shape}")
        print(f"工况基系数形状: {self.condition_basis_coeffs.shape}")
        return self.condition_basis_coeffs

    def train_network(self, epochs=300, patience=30):
        print("\n训练神经网络以映射参数到工况基系数...")
        indices = np.arange(self.boundary_data.shape[0])
        train_idx, self.test_idx = train_test_split(indices, test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)
        X_train, y_train = self.boundary_data[train_idx], self.condition_basis_coeffs[train_idx]
        X_val, y_val = self.boundary_data[val_idx], self.condition_basis_coeffs[val_idx]
        self.X_test = self.boundary_data[self.test_idx]
        train_loader = DataLoader(ConditionDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(ConditionDataset(X_val, y_val), batch_size=16, shuffle=False)
        self.network = MappingNetwork(input_dim=14, output_dim=self.r_cond)
        trainer = NetworkTrainer(self.network, train_loader, val_loader)
        trainer.train(epochs=epochs, patience=patience)

    def predict(self, X_new):
        self.network.eval()
        with torch.no_grad():
            new_cond_basis_coeffs = self.network(torch.FloatTensor(X_new)).numpy()
            reconstructed_pod_coeffs = tl.tucker_to_tensor(
                (self.core, [new_cond_basis_coeffs, self.mode_coupling_basis]))
            return reconstructed_pod_coeffs


class ConditionDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class MappingNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=20, hidden_dims=[128, 256, 128]):
        super(MappingNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NetworkTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=15,
                                                              verbose=True)
        self.train_losses, self.val_losses = [], []

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, epochs=300, patience=30):
        print(f"开始训练神经网络，设备: {self.device}")
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证集损失连续 {patience} 次未改善，于 epoch {epoch + 1} 早停。")
                    break
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("训练完成！")


def main():

    boundary_path = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
    snapshot_dir = r"C:\Users\Lenovo\Desktop\condition_data_files"
    data_manager = DataAndPODManager(boundary_path, snapshot_dir)
    boundary_norm, snapshots_norm, coords = data_manager.load_data()
    U_r, pod_coefficients = data_manager.perform_pod(energy_threshold=0.99)

    pod_tucker_model = PODTuckerModel(pod_coefficients, boundary_norm, r_cond=15)
    condition_basis_coeffs = pod_tucker_model.decompose_coefficients()
    pod_tucker_model.train_network()

    print("\n在测试集上评估POD-Tucker模型...")
    X_test = pod_tucker_model.X_test
    test_indices = pod_tucker_model.test_idx

    predicted_pod_coeffs_norm = pod_tucker_model.predict(X_test)
    reconstructed_fluctuations_norm = U_r @ predicted_pod_coeffs_norm.T
    reconstructed_snapshots_norm = reconstructed_fluctuations_norm + data_manager.mean_field

    reconstructed_snapshots_physical_T = data_manager.temp_scaler.inverse_transform(reconstructed_snapshots_norm.T)

    reconstructed_snapshots_physical = reconstructed_snapshots_physical_T.T
    true_snapshots_physical = data_manager.snapshots_matrix[:, test_indices]

    mae = np.mean(np.abs(true_snapshots_physical - reconstructed_snapshots_physical))
    rmse = np.sqrt(np.mean((true_snapshots_physical - reconstructed_snapshots_physical) ** 2))

    print("\n" + "=" * 80)
    print("POD-Tucker 模型最终性能评估:")
    print(f"  测试集平均绝对误差 (MAE): {mae:.4f} °C")
    print(f"  测试集均方根误差 (RMSE): {rmse:.4f} °C")
    print("=" * 80)

    print("\n生成可视化对比图...")
    case_idx_in_test_set = 0
    true_case_original_idx = test_indices[case_idx_in_test_set]
    true_field = true_snapshots_physical[:, case_idx_in_test_set]
    pred_field = reconstructed_snapshots_physical[:, case_idx_in_test_set]
    visualize_sparse_field(true_field, pred_field, coords, true_case_original_idx)


def visualize_sparse_field(true_field, pred_field, coords, case_idx, save_dir="results_pod_tucker"):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print(f"  正在为工况 {case_idx + 1} 生成优化版可视化云图...")

    x_unique, y_unique, z_unique = [np.unique(coords[:, i]) for i in range(3)]
    grid_x, grid_z = np.meshgrid(np.linspace(x_unique.min(), x_unique.max(), 200),
                                 np.linspace(z_unique.min(), z_unique.max(), 300))


    y_slice_val = y_unique[len(y_unique) // 2]

    slice_indices = np.abs(coords[:, 1] - y_slice_val) < 0.2

    points_2d = coords[slice_indices, :][:, [0, 2]]
    true_values_2d = true_field[slice_indices]
    pred_values_2d = pred_field[slice_indices]

    if len(points_2d) < 10:
        print(f"  警告：在 Y={y_slice_val:.2f}m 切面上点数过少({len(points_2d)}个)，无法生成高质量云图。")
        return

    grid_true = griddata(points_2d, true_values_2d, (grid_x, grid_z), method='cubic')
    grid_pred = griddata(points_2d, pred_values_2d, (grid_x, grid_z), method='cubic')
    mask = np.isnan(grid_true)
    grid_error = np.abs(np.nan_to_num(grid_true) - np.nan_to_num(grid_pred))

    grid_error[mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.patch.set_facecolor('white')


    valid_temps = np.concatenate([grid_true[~mask], grid_pred[~mask]])
    vmin, vmax = np.min(valid_temps), np.max(valid_temps)


    for ax in axes:
        ax.set_facecolor('#EAEAF2')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Z (m)', fontsize=12)
        ax.set_aspect('equal', adjustable='box')


    im1 = axes[0].contourf(grid_x, grid_z, grid_true, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True Field (Case {case_idx + 1})", fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')


    im2 = axes[1].contourf(grid_x, grid_z, grid_pred, levels=50, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted Field (POD-Tucker)", fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Temperature (°C)')


    im3 = axes[2].contourf(grid_x, grid_z, grid_error, levels=50, cmap='YlOrRd')
    axes[2].set_title("Absolute Error", fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Error (°C)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'case_{case_idx + 1}_comparison_masked.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  优化版可视化结果已保存至: {save_path}")


if __name__ == "__main__":
    main()
