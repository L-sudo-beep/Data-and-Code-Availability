import pandas as pd
import numpy as np
from scipy.linalg import svd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import random


SEED = 42
ENERGY_THRESHOLD = 0.99
HIDDEN_LAYERS = [128, 64]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
TRAIN_SPLIT = 0.9
PLANE_TYPE = 'y'
PLANE_VALUE = 1.2334
NUM_ADJACENT_PLANES = 2
VAL_INDEX = 3
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t')
bc_df = bc_df.astype(float)
conditions = bc_df.values
num_samples = conditions.shape[0]
assert num_samples == 89, "Boundary conditions should have 89 rows"


scaler_conditions = MinMaxScaler()
conditions = scaler_conditions.fit_transform(conditions)


temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]


indices = list(range(num_samples))
train_idx, val_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)
print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")


first_df = pd.read_csv(temp_files[0])
coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
N_points = coords.shape[0]


print("Unique X values:", np.sort(np.unique(coords[:, 0])))
print("Unique Y values:", np.sort(np.unique(coords[:, 1])))
print("Unique Z values:", np.sort(np.unique(coords[:, 2])))


snapshots_train = np.zeros((N_points, len(train_idx)))
for i, idx in enumerate(train_idx):
    df = pd.read_csv(temp_files[idx])
    snapshots_train[:, i] = df['Temperature'].values

mean_temp = np.mean(snapshots_train, axis=1)[:, np.newaxis]


fluctuations = snapshots_train - mean_temp

U, S, Vt = svd(fluctuations, full_matrices=False)

energy = np.cumsum(S ** 2) / np.sum(S ** 2)

K = np.argmax(energy >= ENERGY_THRESHOLD) + 1
print(f"Selected K: {K} modes to retain {ENERGY_THRESHOLD * 100:.2f}% energy")

modes = U[:, :K]


coeffs_train = modes.T @ fluctuations
coeffs_train = coeffs_train.T
scaler_coeffs = MinMaxScaler()
coeffs_train = scaler_coeffs.fit_transform(coeffs_train)
conditions_train = conditions[train_idx, :]

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



input_size = 14
output_size = K
model = MLP(input_size, output_size, HIDDEN_LAYERS)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

X_train = torch.tensor(conditions_train, dtype=torch.float32)
y_train = torch.tensor(coeffs_train, dtype=torch.float32)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")
snapshots_val = np.zeros((N_points, len(val_idx)))
for i, idx in enumerate(val_idx):
    df = pd.read_csv(temp_files[idx])
    snapshots_val[:, i] = df['Temperature'].values

conditions_val = conditions[val_idx, :]
val_i = VAL_INDEX % len(val_idx)
condition_val = torch.tensor(conditions_val[val_i:val_i + 1], dtype=torch.float32)
true_temp = snapshots_val[:, val_i]


model.eval()
with torch.no_grad():
    pred_coeffs_norm = model(condition_val).numpy()

pred_coeffs = scaler_coeffs.inverse_transform(pred_coeffs_norm).flatten()  # (K,)

reconstructed = mean_temp.flatten() + modes @ pred_coeffs

mse = np.mean((true_temp - reconstructed) ** 2)
print(f"Validation MSE for sample {val_i} (original index {val_idx[val_i]}): {mse:.6f}")

def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]
    unique_values = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_values - target_value))
    start_idx = max(0, closest_idx - num_adjacent)
    end_idx = min(len(unique_values), closest_idx + num_adjacent + 1)
    selected_planes = unique_values[start_idx:end_idx]
    print(f"Selected {len(selected_planes)} planes for {plane_type}-axis:")
    print(f"Target value: {target_value}")
    print(f"Selected plane values: {selected_planes}")

    mask = np.zeros(coords.shape[0], dtype=bool)
    for plane_val in selected_planes:
        plane_mask = np.abs(coords[:, axis_idx] - plane_val) < 1e-10  # 很小的容差
        mask |= plane_mask

    return mask, selected_planes



mask, selected_planes = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
if PLANE_TYPE == 'x':
    slice_coords_x = coords[mask, 1]  # Y
    slice_coords_y = coords[mask, 2]  # Z
    xlabel = 'Y (m)'
    ylabel = 'Z (m)'
elif PLANE_TYPE == 'y':
    slice_coords_x = coords[mask, 0]  # X
    slice_coords_y = coords[mask, 2]  # Z
    xlabel = 'X (m)'
    ylabel = 'Z (m)'
else:  # 'z'
    slice_coords_x = coords[mask, 0]  # X
    slice_coords_y = coords[mask, 1]  # Y
    xlabel = 'X (m)'
    ylabel = 'Y (m)'

true_plane = true_temp[mask]
recon_plane = reconstructed[mask]

num_points_on_planes = mask.sum()
print(f"Number of points on selected planes: {num_points_on_planes}")

if num_points_on_planes == 0:
    print(f"No points found on the selected planes. Check your data.")
else:
    from scipy.interpolate import griddata
    grid_resolution = 100
    x_min, x_max = slice_coords_x.min(), slice_coords_x.max()
    y_min, y_max = slice_coords_y.min(), slice_coords_y.max()

    grid_x = np.linspace(x_min, x_max, grid_resolution)
    grid_y = np.linspace(y_min, y_max, grid_resolution)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    try:
        true_interp = griddata(
            (slice_coords_x, slice_coords_y), true_plane,
            (grid_X, grid_Y), method='linear', fill_value=np.nan
        )
        recon_interp = griddata(
            (slice_coords_x, slice_coords_y), recon_plane,
            (grid_X, grid_Y), method='linear', fill_value=np.nan
        )

        temp_min = min(true_plane.min(), recon_plane.min())
        temp_max = max(true_plane.max(), recon_plane.max())
        print(f"Unified color range: [{temp_min:.3f}, {temp_max:.3f}]")


        levels = np.linspace(temp_min, temp_max, 50)
        error_field = true_interp - recon_interp
        error_min = np.nanmin(error_field)
        error_max = np.nanmax(error_field)
        error_abs_max = max(abs(error_min), abs(error_max))
        error_levels = np.linspace(-error_abs_max, error_abs_max, 50)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        sc1 = axs[0].scatter(slice_coords_x, slice_coords_y, c=true_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f'True Temperature (Scatter)\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.colorbar(sc1, ax=axs[0], shrink=0.8)

        cs1 = axs[1].contourf(grid_X, grid_Y, true_interp, levels=levels,
                              cmap='jet', vmin=temp_min, vmax=temp_max)
        axs[1].set_title('True Temperature (Interpolated)')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        fig.colorbar(cs1, ax=axs[1], shrink=0.8)
        cs2 = axs[2].contourf(grid_X, grid_Y, recon_interp, levels=levels,
                              cmap='jet', vmin=temp_min, vmax=temp_max)
        axs[2].set_title('Reconstructed Temperature (Interpolated)')
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)
        fig.colorbar(cs2, ax=axs[2], shrink=0.8)

        cs3 = axs[3].contourf(grid_X, grid_Y, error_field, levels=error_levels,
                              cmap='RdBu_r', vmin=-error_abs_max, vmax=error_abs_max)
        axs[3].set_title(f'Error Field (True - Reconstructed)\nRange: [{error_min:.4f}, {error_max:.4f}]')
        axs[3].set_xlabel(xlabel)
        axs[3].set_ylabel(ylabel)
        fig.colorbar(cs3, ax=axs[3], shrink=0.8)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR,
                                 f'validation_interpolated_{PLANE_TYPE}_{PLANE_VALUE}_val{val_i}_adj{NUM_ADJACENT_PLANES}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved interpolated visualization to: {save_path}")
        plt.show()

    except Exception as e:
        print(f"Interpolation failed: {e}")
        print("Falling back to scatter plot...")
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        temp_min = min(true_plane.min(), recon_plane.min())
        temp_max = max(true_plane.max(), recon_plane.max())
        print(f"Unified color range for scatter plot: [{temp_min:.3f}, {temp_max:.3f}]")
        error_plane = true_plane - recon_plane
        error_abs_max = max(abs(error_plane.min()), abs(error_plane.max()))

        sc1 = axs[0].scatter(slice_coords_x, slice_coords_y, c=true_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[0].set_title(f'True Temperature\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        fig.colorbar(sc1, ax=axs[0], shrink=0.8)

        sc2 = axs[1].scatter(slice_coords_x, slice_coords_y, c=recon_plane,
                             cmap='jet', s=10, vmin=temp_min, vmax=temp_max)
        axs[1].set_title(f'Reconstructed Temperature\nPlanes: {PLANE_TYPE}∈{selected_planes}')
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)
        fig.colorbar(sc2, ax=axs[1], shrink=0.8)
        sc3 = axs[2].scatter(slice_coords_x, slice_coords_y, c=error_plane,
                             cmap='RdBu_r', s=10, vmin=-error_abs_max, vmax=error_abs_max)
        axs[2].set_title(f'Error Field (True - Reconstructed)')
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel(ylabel)
        fig.colorbar(sc3, ax=axs[2], shrink=0.8)
        axs[3].hist(error_plane, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axs[3].set_title(f'Error Distribution\nMean: {error_plane.mean():.4f}, Std: {error_plane.std():.4f}')
        axs[3].set_xlabel('Temperature Error')
        axs[3].set_ylabel('Frequency')
        axs[3].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR,
                                 f'validation_scatter_{PLANE_TYPE}_{PLANE_VALUE}_val{val_i}_adj{NUM_ADJACENT_PLANES}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter visualization to: {save_path}")
        plt.show()

plane_mse = np.mean((true_plane - recon_plane) ** 2)
plane_mae = np.mean(np.abs(true_plane - recon_plane))
plane_max_error = np.max(np.abs(true_plane - recon_plane))

print(f"\n=== Error Statistics on Selected Planes ===")
print(f"MSE: {plane_mse:.6f}")
print(f"MAE: {plane_mae:.6f}")
print(f"Max Absolute Error: {plane_max_error:.6f}")
print(f"Temperature range - True: [{true_plane.min():.3f}, {true_plane.max():.3f}]")
print(f"Temperature range - Reconstructed: [{recon_plane.min():.3f}, {recon_plane.max():.3f}]")
print("\nTask complete. Modify parameters as needed for further runs.")