# -*- coding: utf-8 -*-
"""
POD-based ROM for steady temperature fields with pluggable regressors:
- ResMLP (baseline)
- 1D CNN
- Tiny Transformer

Features:
- Strict train-only normalization for inputs (no leakage)
- Reconstruction consistency loss in POD subspace (works for all three models)
- AdamW + Cosine LR + Early stopping + Grad clipping
- Validation recon visualizations; optional training error maps
"""

import os
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import svd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==============================
# Configurable Parameters
# ==============================
SEED = 42
ENERGY_THRESHOLD = 0.99            # POD retained energy
LEARNING_RATE = 1e-3
EPOCHS = 2000
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-3
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 100
CONSISTENCY_ALPHA = 0.1            # weight for reconstruction consistency loss
CONSISTENCY_N_PTS = 5000           # sampled spatial points per step (adjust to memory)

# Model switch: "resmlp" | "cnn1d" | "transformer"
MODEL_TYPE = "cnn1d"         # <<< 设置你要测试的模型

# Data I/O
DATA_DIR = r'C:\Users\Lenovo\Desktop\condition_data_files'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD89_Fixed_Test_Set'

# Visualization slice
PLANE_TYPE = 'y'                   # 'x' | 'y' | 'z'
PLANE_VALUE = 1.2334
NUM_ADJACENT_PLANES = 2

# Validation set (human-readable indices starting at 1)
VAL_INDICES_HUMAN = [8, 18, 28, 38, 48, 58, 68, 78, 88]

# Train error maps toggle
EXPORT_TRAIN_ERROR_MAPS = False     # 若要导出训练集误差云图，改为 True

# ==============================
# Reproducibility
# ==============================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================
# Utils: Plane selection & Visualization
# ==============================
def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[plane_type]
    unique_values = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_values - target_value))
    start_idx = max(0, closest_idx - num_adjacent)
    end_idx = min(len(unique_values), closest_idx + num_adjacent + 1)
    selected_planes = unique_values[start_idx:end_idx]
    mask = np.isin(coords[:, axis_idx], selected_planes)
    return mask, selected_planes


def visualize_results(true_temp, reconstructed_temp, coords, case_index, save_dir,
                      plane_type='y', plane_value=1.2334, num_adjacent=2):
    mask, selected_planes = select_adjacent_planes(coords, plane_type, plane_value, num_adjacent)
    if not mask.any():
        print(f"Case {case_index}: No points on selected planes. Skipping visualization.")
        return

    axis_map = {'x': (1, 2, 'Y (m)', 'Z (m)'), 'y': (0, 2, 'X (m)', 'Z (m)'), 'z': (0, 1, 'X (m)', 'Y (m)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axis_map[plane_type]
    slice_coords_x, slice_coords_y = coords[mask, ax1_idx], coords[mask, ax2_idx]
    true_plane, recon_plane = true_temp[mask], reconstructed_temp[mask]

    grid_x = np.linspace(slice_coords_x.min(), slice_coords_x.max(), 150)
    grid_y = np.linspace(slice_coords_y.min(), slice_coords_y.max(), 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    true_interp = griddata((slice_coords_x, slice_coords_y), true_plane, (grid_X, grid_Y),
                           method='cubic', fill_value=np.nan)
    recon_interp = griddata((slice_coords_x, slice_coords_y), recon_plane, (grid_X, grid_Y),
                            method='cubic', fill_value=np.nan)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    temp_min = np.nanmin([true_interp, recon_interp])
    temp_max = np.nanmax([true_interp, recon_interp])
    levels = np.linspace(temp_min, temp_max, 50)

    err_field = np.abs(true_interp - recon_interp)

    cs1 = axs[0].contourf(grid_X, grid_Y, true_interp, levels=levels, cmap='jet')
    axs[0].set_title(f'True Temperature (Case {case_index})')
    fig.colorbar(cs1, ax=axs[0])

    cs2 = axs[1].contourf(grid_X, grid_Y, recon_interp, levels=levels, cmap='jet')
    axs[1].set_title('Reconstructed Temperature')
    fig.colorbar(cs2, ax=axs[1])

    cs3 = axs[2].contourf(grid_X, grid_Y, err_field, levels=50, cmap='YlOrRd')
    axs[2].set_title('Absolute Error')
    fig.colorbar(cs3, ax=axs[2])

    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{MODEL_TYPE}_case_{case_index}_reconstruction.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def visualize_error_only(true_temp, reconstructed_temp, coords, case_index, save_dir,
                         plane_type='y', plane_value=1.2334, num_adjacent=2, title_prefix='Error'):
    mask, _ = select_adjacent_planes(coords, plane_type, plane_value, num_adjacent)
    if not mask.any():
        print(f"Case {case_index}: No points on selected planes. Skipping error plot.")
        return

    axis_map = {'x': (1, 2, 'Y (m)', 'Z (m)'), 'y': (0, 2, 'X (m)', 'Z (m)'), 'z': (0, 1, 'X (m)', 'Y (m)')}
    ax1_idx, ax2_idx, xlabel, ylabel = axis_map[plane_type]
    slice_coords_x, slice_coords_y = coords[mask, ax1_idx], coords[mask, ax2_idx]

    err_plane = (reconstructed_temp - true_temp)[mask]  # signed; change to abs() if desired

    grid_x = np.linspace(slice_coords_x.min(), slice_coords_x.max(), 150)
    grid_y = np.linspace(slice_coords_y.min(), slice_coords_y.max(), 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    err_interp = griddata((slice_coords_x, slice_coords_y), err_plane, (grid_X, grid_Y),
                          method='cubic', fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmax = np.nanmax(np.abs(err_interp))
    cs = ax.contourf(grid_X, grid_Y, err_interp, levels=50, cmap='bwr', vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Reconstruction Error (°C)')

    ax.set_title(f'{title_prefix} (Case {case_index})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{MODEL_TYPE}_case_{case_index}_error.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved error map: {save_path}")


# ==============================
# Models
# ==============================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    def forward(self, x):
        return x + self.net(x)

class ResMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden=256, depth=4):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(depth)])
        self.out = nn.Linear(hidden, output_size)
    def forward(self, x):
        h = self.inp(x)
        h = self.blocks(h)
        return self.out(h)

class CNN1DHead(nn.Module):
    """
    Treat the 14 input features as a 1D sequence of length L=feat_dim with 1 channel.
    """
    def __init__(self, feat_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # length -> ceil(L/2)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # global pooling -> [B, 128, 1]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),            # -> [B, 128]
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        # x: [B, feat_dim]
        x = x.unsqueeze(1)  # [B, 1, L]
        h = self.net(x)     # [B, 128, 1]
        out = self.fc(h)    # [B, out_dim]
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x):
        # x: [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # broadcast

class TinyTransformer(nn.Module):
    """
    Treat each scalar feature as a token (length=L=feat_dim, feature_dim=1).
    Project to d_model, add sinusoidal PE, run encoder, mean-pool, predict K.
    """
    def __init__(self, feat_dim, out_dim, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.token_proj = nn.Linear(1, d_model)  # shared across positions
        self.posenc = PositionalEncoding(d_model, max_len=feat_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: [B, feat_dim]
        x = x.unsqueeze(-1)          # [B, L, 1]
        h = self.token_proj(x)       # [B, L, d_model]
        h = self.posenc(h)           # add PE
        h = self.encoder(h)          # [B, L, d_model]
        h = self.norm(h)
        h = h.mean(dim=1)            # mean-pool over tokens -> [B, d_model]
        out = self.head(h)           # [B, out_dim]
        return out


def build_model(model_type, input_dim, out_dim):
    model_type = model_type.lower()
    if model_type == "resmlp":
        return ResMLP(input_size=input_dim, output_size=out_dim, hidden=256, depth=4)
    elif model_type == "cnn1d":
        return CNN1DHead(feat_dim=input_dim, out_dim=out_dim)
    elif model_type == "transformer":
        return TinyTransformer(feat_dim=input_dim, out_dim=out_dim, d_model=128, nhead=4, num_layers=2, dim_ff=256)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")


# ==============================
# Main
# ==============================
def main():
    start_time = time.time()

    # -------- Load BCs
    bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
    conditions = bc_df.values
    num_samples, n_feat = conditions.shape
    print(f"Loaded {num_samples} boundary-condition rows with {n_feat} features.")

    # Validation split
    val_idx = [i - 1 for i in VAL_INDICES_HUMAN]
    all_indices = list(range(num_samples))
    train_idx = [i for i in all_indices if i not in val_idx]
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)} (human indices: {VAL_INDICES_HUMAN})")

    # -------- Strict train-only normalization for conditions (no leakage)
    scaler_conditions = MinMaxScaler()
    scaler_conditions.fit(conditions[train_idx])
    conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
    conditions_val_norm   = scaler_conditions.transform(conditions[val_idx])

    # -------- Load snapshots
    temp_files = [os.path.join(DATA_DIR, f"{i}.csv") for i in range(1, num_samples + 1)]
    print("Reading coordinates and all snapshots...")
    first_df = pd.read_csv(temp_files[0])
    coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
    N_points = coords.shape[0]

    all_snapshots = np.zeros((N_points, num_samples))
    for i in tqdm(range(num_samples)):
        df = pd.read_csv(temp_files[i])
        all_snapshots[:, i] = df['Temperature'].values

    snapshots_train = all_snapshots[:, train_idx]
    snapshots_val = all_snapshots[:, val_idx]

    # -------- POD on training data
    print("\nPerforming POD on training data...")
    mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
    fluctuations = snapshots_train - mean_temp
    U, S, Vt = svd(fluctuations, full_matrices=False)

    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    K = int(np.argmax(energy >= ENERGY_THRESHOLD) + 1)
    print(f"Selected K={K} modes to retain {ENERGY_THRESHOLD * 100:.2f}% energy")

    modes = U[:, :K]                           # [N_points, K]
    coeffs_train = (modes.T @ fluctuations).T  # [N_train, K]

    # validation projection (no leakage)
    coeffs_val = (modes.T @ (snapshots_val - mean_temp)).T  # [N_val, K]

    # Standardize coefficients (fit ONLY on training coefficients)
    scaler_coeffs = StandardScaler()
    coeffs_train_norm = scaler_coeffs.fit_transform(coeffs_train)
    coeffs_val_norm   = scaler_coeffs.transform(coeffs_val)

    # Tensors
    X_train = torch.tensor(conditions_train_norm, dtype=torch.float32).to(device)
    y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32).to(device)
    X_val   = torch.tensor(conditions_val_norm, dtype=torch.float32).to(device)
    y_val   = torch.tensor(coeffs_val_norm, dtype=torch.float32).to(device)

    # For consistency loss
    train_indices_tensor = torch.tensor(np.arange(len(train_idx)), dtype=torch.long)
    modes_t = torch.tensor(modes, dtype=torch.float32, device=device)                  # [N_points, K]
    mean_t  = torch.tensor(mean_temp.flatten(), dtype=torch.float32, device=device)    # [N_points]
    snapshots_train_t = torch.tensor(snapshots_train, dtype=torch.float32, device=device)  # [N_points, N_train]

    # Dataloader
    dataset = TensorDataset(X_train, y_train, train_indices_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -------- Build model
    print(f"\n=== Training {MODEL_TYPE.upper()} with consistency loss ===")
    model = build_model(MODEL_TYPE, input_dim=n_feat, out_dim=K).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val = float('inf')
    bad_epochs = 0
    best_state = None

    # Scaler params as device tensors for inverse standardization
    sc_scale = torch.tensor(scaler_coeffs.scale_, dtype=torch.float32, device=device)  # [K]
    sc_mean  = torch.tensor(scaler_coeffs.mean_,  dtype=torch.float32, device=device)  # [K]

    for epoch in range(EPOCHS):
        model.train()
        for bx, by, bcol in loader:
            optimizer.zero_grad()
            pred_norm = model(bx)                    # [B, K]
            loss_coeff = criterion(pred_norm, by)

            # inverse standardize to coefficients
            pred_coeff = pred_norm * sc_scale + sc_mean   # [B, K]

            # random subset of spatial points for reconstruction loss
            n_pts = min(CONSISTENCY_N_PTS, modes_t.shape[0])
            idx = torch.randint(0, modes_t.shape[0], (n_pts,), device=device)
            modes_sub = modes_t.index_select(0, idx)      # [n_pts, K]
            mean_sub  = mean_t.index_select(0, idx)       # [n_pts]

            # reconstruct temperature for batch
            T_hat = mean_sub[:, None] + modes_sub @ pred_coeff.T       # [n_pts, B]
            T_true = snapshots_train_t.index_select(0, idx)[:, bcol]   # [n_pts, B]

            loss_rec = torch.mean((T_hat - T_true) ** 2)

            loss = loss_coeff + CONSISTENCY_ALPHA * loss_rec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        # validation on coefficient loss
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  train_coeff_loss={loss_coeff.item():.6e}  "
                  f"train_rec_loss={loss_rec.item():.6e}  val_coeff_loss={val_loss:.6e}")

        # early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopped at epoch {epoch+1}. Best val_coeff_loss={best_val:.6e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -------- Predict on validation (coeff → reconstruct)
    model.eval()
    with torch.no_grad():
        pred_norm_val = model(X_val)                                   # [N_val, K]
        pred_coeff_val = pred_norm_val * sc_scale + sc_mean            # inverse standardize
        pred_coeff_val = pred_coeff_val.detach().cpu().numpy()         # [N_val, K]

    recon_val = mean_temp.flatten()[:, None] + modes @ pred_coeff_val.T  # [N_points, N_val]

    # -------- Evaluate & visualize
    print("\nEvaluating on validation set & saving visualizations...")
    all_errors = []

    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, original_idx in enumerate(val_idx):
        case_index_human = original_idx + 1
        true_temp = snapshots_val[:, i]
        reconstructed = recon_val[:, i]
        all_errors.append(true_temp - reconstructed)

        visualize_results(true_temp, reconstructed, coords, case_index_human, SAVE_DIR,
                          plane_type=PLANE_TYPE, plane_value=PLANE_VALUE, num_adjacent=NUM_ADJACENT_PLANES)

    all_errors = np.array(all_errors).flatten()
    mae = np.mean(np.abs(all_errors))
    rmse = np.sqrt(np.mean(all_errors ** 2))

    print("\n" + "=" * 60)
    print(f"  Model: {MODEL_TYPE.upper()}")
    print("  Final Performance on the Validation/Test Set")
    print("=" * 60)
    print(f"  Test Set Indices (Human Readable): {VAL_INDICES_HUMAN}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} °C")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} °C")
    print("=" * 60)

    # (Optional) training-set error maps
    if EXPORT_TRAIN_ERROR_MAPS:
        print("\nGenerating training-set error maps...")
        # Using predicted coeffs on training set
        with torch.no_grad():
            pred_norm_tr = model(X_train)                                   # [N_train, K]
            pred_coeff_tr = (pred_norm_tr * sc_scale + sc_mean).cpu().numpy()
        recon_train_model = mean_temp.flatten()[:, None] + modes @ pred_coeff_tr.T  # [N_points, N_train]

        train_err_dir_model = os.path.join(SAVE_DIR, f"{MODEL_TYPE}_Train_Errors_ModelPred")
        for j, orig_idx in enumerate(train_idx):
            case_index_human = orig_idx + 1
            true_temp = snapshots_train[:, j]
            recon_temp = recon_train_model[:, j]
            visualize_error_only(true_temp, recon_temp, coords, case_index_human, train_err_dir_model,
                                 plane_type=PLANE_TYPE, plane_value=PLANE_VALUE, num_adjacent=NUM_ADJACENT_PLANES,
                                 title_prefix=f'{MODEL_TYPE.upper()} Prediction Error')

    elapsed = time.time() - start_time
    print(f"\nTotal script execution time: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
