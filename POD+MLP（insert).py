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
ENERGY_THRESHOLD = 0.995
HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DATA_DIR = r'C:\Users\Lenovo\Desktop\insert'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
PLANE_TYPE = 'y'
PLANE_VALUE = 1.26
NUM_ADJACENT_PLANES = 0
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-insert'

# ======================================
# Set Seeds for Reproducibility
# ======================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
for i in tqdm(range(num_samples)):
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
print(f"Selected K={K} modes for {ENERGY_THRESHOLD*100:.1f}% energy")

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
    def forward(self, x): return self.net(x)

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
for epoch in range(EPOCHS):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss={loss.item():.6f}")

# ======================================
# Visualization and Evaluation
# ======================================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]
    unique_vals = np.sort(np.unique(coords[:, axis_idx]))
    closest = np.argmin(np.abs(unique_vals - target_value))
    low = max(0, closest - num_adjacent)
    high = min(len(unique_vals), closest + num_adjacent + 1)
    sel = unique_vals[low:high]
    mask = np.isin(coords[:, axis_idx], sel)
    return mask

def visualize_results(true_temp, pred_temp, coords, case_index, vmax_shared=None):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not mask.any(): return None
    x, z = coords[mask, 0], coords[mask, 2]
    t_true, t_pred = true_temp[mask], pred_temp[mask]
    grid_x, grid_z = np.linspace(x.min(), x.max(), 150), np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z)
    Tt = griddata((x, z), t_true, (GX, GZ), method='cubic')
    Tp = griddata((x, z), t_pred, (GX, GZ), method='cubic')
    err = np.abs(Tt - Tp)
    vmax = vmax_shared if vmax_shared else np.nanmax(err)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap in zip(axs, [Tt, Tp, err],
                                     [f"True (Case {case_index})", "Reconstructed", "Absolute Error"],
                                     ["jet", "jet", "YlOrRd"]):
        vmin = 0 if title == "Absolute Error" else np.nanmin(data)
        vmax_plot = vmax if title == "Absolute Error" else np.nanmax(data)
        cs = ax.contourf(GX, GZ, data, levels=50, cmap=cmap, vmin=vmin, vmax=vmax_plot)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, f"case_{case_index}_reconstruction.png"), dpi=200)
    plt.close(fig)
    return np.nanmax(err)

# ======================================
# Validation
# ======================================
print("\nValidating on test set...")
model.eval()
all_errors = np.zeros((len(val_idx), N_points))
mae_list, rmse_list = [], []
global_error_max = 0

for i, idx in enumerate(val_idx):
    cond = torch.tensor(conditions_val_norm[i:i+1], dtype=torch.float32)
    with torch.no_grad():
        pred_coeff = scaler_coeffs.inverse_transform(model(cond).numpy()).flatten()
    rec_temp = mean_temp.flatten() + modes @ pred_coeff
    true_temp = snapshots_val[:, i]
    err = true_temp - rec_temp
    all_errors[i, :] = err
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    mae_list.append(mae)
    rmse_list.append(rmse)
    vmax_now = visualize_results(true_temp, rec_temp, coords, idx + 1)
    global_error_max = max(global_error_max, vmax_now if vmax_now else 0)
    print(f"Case {idx+1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C")

# Save global max error
np.save(os.path.join(SAVE_DIR, "global_error_max.npy"), global_error_max)

# ======================================
# Results summary & plots
# ======================================
df = pd.DataFrame({
    "Case": [i + 1 for i in val_idx],
    "MAE": mae_list,
    "RMSE": rmse_list
})
df.to_csv(os.path.join(SAVE_DIR, "per_case_metrics.csv"), index=False)

plt.figure(figsize=(8, 4))
plt.plot(df["Case"], df["MAE"], '-o', label="MAE")
plt.plot(df["Case"], df["RMSE"], '-s', label="RMSE")
plt.xlabel("Case")
plt.ylabel("Error (°C)")
plt.title("POD+MLP Per-case Error Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "mae_rmse_compare.png"), dpi=200)
plt.close()

print("\n===== Summary =====")
print(f"Mean MAE = {np.mean(mae_list):.4f} °C")
print(f"Mean RMSE = {np.mean(rmse_list):.4f} °C")
print(f"Saved results to: {SAVE_DIR}")
print(f"Execution time: {time.time() - start_time:.2f} s")
