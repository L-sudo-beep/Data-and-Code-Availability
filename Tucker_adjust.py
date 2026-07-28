"""
Data Center Temperature Field Reconstruction
Tucker Decomposition + NN Mapping (Fixed Geometry Version)
Author: Legend Co., Ltd. - Mia's helper
Date: 2025-10-10

Key Fixes:
1) Use raw physical coordinates (meters) for interpolation & plotting (no re-scaling).
2) Auto-infer tensor shape from unique coordinate arrays (remove hard-coded 126×190×229).
3) Enforce cross-case coordinate consistency by sorting each CSV by (X, Y, Z) before stacking.
4) Physical Y-slice selection (default 1.77 m) with nearest-layer picking.
"""

import os
import pickle
import warnings
from typing import Tuple, List

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

from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter

# -------------------------- Global Config --------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

warnings.filterwarnings("ignore")

ROOM_SIZE = (6.6, 4.5, 9.0)  # (X, Y, Z) in meters - used only for titles
TARGET_Y_SLICE_M = 1.77      # choose nearest Y layer to this physical height (meters)
SAVE_DIR = "temperature_results"  # figures & CSV output directory

# -------------------------- 1. Data Loader --------------------------

class DataCenterDataLoader:
    """
    - Loads boundary conditions (89 × 14)
    - Loads 89 CSV snapshots; each CSV contains columns: X (m), Y (m), Z (m), Temperature
    - Ensures cross-case coordinate consistency by sorting each CSV by (X, Y, Z)
    - Builds a 4D tensor with mask:
        tensor.shape = (Nx, Ny, Nz, N_cases)
        mask.shape   = (Nx, Ny, Nz, N_cases)
      where Nx = len(unique X), etc.
    """

    def __init__(self, boundary_path: str, snapshot_dir: str, n_conditions: int = 89):
        self.boundary_path = boundary_path
        self.snapshot_dir = snapshot_dir
        self.n_conditions = n_conditions

        self.boundary_data = None
        self.boundary_data_normalized = None
        self.scaler = None

        self.coords = None
        self.x_unique = None
        self.y_unique = None
        self.z_unique = None

        self.snapshots_matrix = None  # (N_points, N_cases)
        self.tensor = None
        self.mask = None
        self.n_points = 0

    # ---------- Boundary conditions ----------

    def load_boundary_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading boundary conditions...")
        df = pd.read_csv(self.boundary_path, encoding='utf-16-le', delimiter='\t')

        param_cols = [
            'RACK1', 'RACK2', 'RACK3', 'RACK4', 'RACK5',
            'RACK6', 'RACK7', 'RACK8', 'RACK9', 'RACK10',
            'CRAC1送风温度', 'CRAC1送风速率', 'CRAC2送风温度', 'CRAC2送风速率'
        ]
        self.boundary_data = df[param_cols].values
        print(f"Boundary shape: {self.boundary_data.shape}")

        self.scaler = MinMaxScaler()
        self.boundary_data_normalized = self.scaler.fit_transform(self.boundary_data)
        return self.boundary_data, self.boundary_data_normalized

    # ---------- Snapshots (temperature CSVs) ----------

    @staticmethod
    def _read_and_sort_csv(file_path: str) -> pd.DataFrame:
        """Read CSV and sort rows by (X, Y, Z) to establish a canonical ordering."""
        df = pd.read_csv(file_path, encoding='utf-8')
        # Normalize column names if necessary
        cols_map = {c: c.strip() for c in df.columns}
        df = df.rename(columns=cols_map)
        expected = ['X (m)', 'Y (m)', 'Z (m)', 'Temperature']
        for col in expected:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in {file_path}")

        df_sorted = df.sort_values(by=['X (m)', 'Y (m)', 'Z (m)'], kind='mergesort').reset_index(drop=True)
        return df_sorted

    def load_snapshots(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading temperature snapshots...")
        snapshots_list: List[np.ndarray] = []
        coords_reference = None

        for i in range(1, self.n_conditions + 1):
            file_path = os.path.join(self.snapshot_dir, f"{i}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Snapshot CSV not found: {file_path}")

            df_sorted = self._read_and_sort_csv(file_path)
            coords = df_sorted[['X (m)', 'Y (m)', 'Z (m)']].to_numpy(dtype=np.float64)
            temps = df_sorted['Temperature'].to_numpy(dtype=np.float32)

            if coords_reference is None:
                coords_reference = coords
            else:
                # ensure same coordinates across cases (after sorting)
                if not np.allclose(coords_reference, coords, atol=1e-9):
                    raise ValueError(
                        f"Coordinates in {i}.csv differ from case 1 after sorting. "
                        "Please verify identical grids across cases."
                    )

            snapshots_list.append(temps)

        self.coords = coords_reference
        self.n_points = self.coords.shape[0]
        self.snapshots_matrix = np.column_stack(snapshots_list)  # (N_points, N_cases)

        # Unique coordinates along each axis (physical meters)
        self.x_unique = np.unique(self.coords[:, 0])
        self.y_unique = np.unique(self.coords[:, 1])
        self.z_unique = np.unique(self.coords[:, 2])

        print(f"Snapshots matrix: {self.snapshots_matrix.shape} "
              f"(N_points={self.n_points}, N_cases={len(snapshots_list)})")
        print(f"Temperature range: [{self.snapshots_matrix.min():.2f}, {self.snapshots_matrix.max():.2f}] °C")
        print(f"X range: {self.x_unique.min():.3f} ~ {self.x_unique.max():.3f} m, "
              f"Y range: {self.y_unique.min():.3f} ~ {self.y_unique.max():.3f} m, "
              f"Z range: {self.z_unique.min():.3f} ~ {self.z_unique.max():.3f} m")
        return self.coords, self.snapshots_matrix

    # ---------- Build tensor & mask ----------

    def build_tensor_with_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Building 4D tensor + mask (auto shape from unique coords)...")
        if self.coords is None or self.snapshots_matrix is None:
            raise RuntimeError("Call load_snapshots() before building the tensor.")

        background_temp = self.snapshots_matrix.min()

        Nx, Ny, Nz = len(self.x_unique), len(self.y_unique), len(self.z_unique)
        tensor = np.full((Nx, Ny, Nz, self.n_conditions), background_temp, dtype=np.float32)
        mask = np.zeros((Nx, Ny, Nz, self.n_conditions), dtype=np.float32)

        # mappings
        x_to_idx = {x: i for i, x in enumerate(self.x_unique)}
        y_to_idx = {y: i for i, y in enumerate(self.y_unique)}
        z_to_idx = {z: i for i, z in enumerate(self.z_unique)}

        x_arr, y_arr, z_arr = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        for p in range(self.n_points):
            xi = x_to_idx[x_arr[p]]
            yi = y_to_idx[y_arr[p]]
            zi = z_to_idx[z_arr[p]]
            tensor[xi, yi, zi, :] = self.snapshots_matrix[p, :]
            mask[xi, yi, zi, :] = 1.0

        self.tensor, self.mask = tensor, mask
        print(f"Tensor shape: {tensor.shape}; valid points per case ~ {int(mask[..., 0].sum())}")
        return tensor, mask

# -------------------------- 2. Tucker Decomposition --------------------------

class TuckerDecomposer:
    def __init__(self, tensor: np.ndarray, ranks: List[int] = None):
        self.tensor = tensor
        self.ranks = ranks or [15, 15, 15, 20]
        self.core = None
        self.factors = None
        self.condition_coefficients = None

    def perform(self):
        print(f"\nTucker decomposition with ranks={self.ranks} ...")
        # Use TensorLy's tucker (HOSVD) directly; tensor is dense here
        self.core, self.factors = tucker(self.tensor, rank=self.ranks, init='svd')
        self.condition_coefficients = self.factors[3]  # shape: (N_cases, r4)
        print(f"Core shape: {self.core.shape}")
        for i, U in enumerate(self.factors, 1):
            print(f"Factor {i} shape: {U.shape}")
        return self.core, self.factors, self.condition_coefficients

    def reconstruct_tensor(self, condition_coeff: np.ndarray = None) -> np.ndarray:
        """Reconstruct using possibly new condition coefficients (N_cases, r4)."""
        if condition_coeff is None:
            condition_coeff = self.condition_coefficients
        # Replace the 4th factor with provided coefficients
        factors_new = [self.factors[0], self.factors[1], self.factors[2], condition_coeff]
        reconstructed = tl.tucker_to_tensor((self.core, factors_new))
        return reconstructed

    @staticmethod
    def evaluate(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> dict:
        valid = mask > 0
        mae = np.abs(original[valid] - reconstructed[valid]).mean()
        rmse = np.sqrt(((original[valid] - reconstructed[valid]) ** 2).mean())
        rel = mae / (np.abs(original[valid]).mean() + 1e-8)
        return {"MAE": float(mae), "RMSE": float(rmse), "Relative_Error": float(rel)}

# -------------------------- 3. Neural Network --------------------------

class ConditionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MappingNetwork(nn.Module):
    def __init__(self, input_dim=14, output_dim=20, hidden_dims=(128, 256, 256, 128), dropout=0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class NetworkTrainer:
    def __init__(self, model, train_loader, val_loader, lr=1e-3,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=10, verbose=True)
        self.train_losses, self.val_losses = [], []

    def _epoch(self, train=True):
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        total = 0.0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                if train: self.opt.zero_grad()
                yhat = self.model(X)
                loss = self.crit(yhat, y)
                if train:
                    loss.backward()
                    self.opt.step()
                total += loss.item()
        return total / len(loader)

    def train(self, epochs=300, early_stop=30, ckpt='best_model.pth'):
        best = float('inf'); patience = 0
        print(f"\nTraining NN on {self.device} ...")
        for e in range(epochs):
            tr = self._epoch(True)
            va = self._epoch(False)
            self.train_losses.append(tr); self.val_losses.append(va)
            self.sch.step(va)
            if (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - Train {tr:.6f}  Val {va:.6f}")
            if va < best - 1e-8:
                best = va; patience = 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopped at epoch {e+1}")
                    break
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print("NN training done.")

# -------------------------- 4. Interpolation & Visualization --------------------------

class TemperatureFieldInterpolator:
    """Use raw physical coordinates (meters) for slice extraction & interpolation."""
    def __init__(self, x_unique: np.ndarray, y_unique: np.ndarray, z_unique: np.ndarray):
        self.xu, self.yu, self.zu = x_unique, y_unique, z_unique

    def nearest_y_index(self, y_target_m: float) -> int:
        return int(np.argmin(np.abs(self.yu - y_target_m)))

    def extract_slice_data(self, temp_field_3d: np.ndarray, mask_3d: np.ndarray, y_index: int):
        slice_mask = mask_3d[:, y_index, :]
        ii, kk = np.where(slice_mask > 0)
        points = np.column_stack([self.xu[ii], self.zu[kk]])  # (N,2) in meters
        values = temp_field_3d[ii, y_index, kk]
        y_phys = self.yu[y_index]
        return points, values, y_phys

    @staticmethod
    def interpolate_slice(points: np.ndarray, values: np.ndarray,
                          method='cubic', points_per_meter=50, smooth_sigma=1.5):
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        z_min, z_max = points[:, 1].min(), points[:, 1].max()
        n_x = max(int((x_max - x_min) * points_per_meter) + 1, 2)
        n_z = max(int((z_max - z_min) * points_per_meter) + 1, 2)

        grid_x = np.linspace(x_min, x_max, n_x)
        grid_z = np.linspace(z_min, z_max, n_z)
        GX, GZ = np.meshgrid(grid_x, grid_z, indexing='ij')

        grid = griddata(points, values, (GX, GZ), method=method)
        # Fill NaN (near boundaries) using nearest
        nan_m = np.isnan(grid)
        if nan_m.any():
            grid[nan_m] = griddata(points, values, (GX[nan_m], GZ[nan_m]), method='nearest')

        if smooth_sigma and smooth_sigma > 0:
            grid = gaussian_filter(grid, sigma=smooth_sigma)

        return grid_x, grid_z, grid  # note: grid shape (n_x, n_z)

def visualize_temperature_slice(true_3d, pred_3d, mask_3d,
                                interpolator: TemperatureFieldInterpolator,
                                case_idx: int, y_index: int,
                                out_dir: str = SAVE_DIR):
    os.makedirs(out_dir, exist_ok=True)

    pts_true, vals_true, y_phys = interpolator.extract_slice_data(true_3d, mask_3d, y_index)
    pts_pred, vals_pred, _ = interpolator.extract_slice_data(pred_3d, mask_3d, y_index)

    gx, gz, T_true = interpolator.interpolate_slice(pts_true, vals_true, method='cubic', points_per_meter=50)
    _,  _,  T_pred = interpolator.interpolate_slice(pts_pred, vals_pred, method='cubic', points_per_meter=50)

    err = np.abs(T_true - T_pred)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())

    im0 = axes[0].contourf(gx, gz, T_true.T, levels=60, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True Temperature Field\nCase {case_idx+1}, Y={y_phys:.2f} m", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Z (m)"); axes[0].set_aspect('equal'); axes[0].grid(alpha=0.3, ls='--')
    plt.colorbar(im0, ax=axes[0], label="Temperature (°C)")

    im1 = axes[1].contourf(gx, gz, T_pred.T, levels=60, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicted Temperature Field\nCase {case_idx+1}, Y={y_phys:.2f} m", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Z (m)"); axes[1].set_aspect('equal'); axes[1].grid(alpha=0.3, ls='--')
    plt.colorbar(im1, ax=axes[1], label="Temperature (°C)")

    im2 = axes[2].contourf(gx, gz, err.T, levels=60, cmap='YlOrRd')
    axes[2].set_title(f"Absolute Error\nCase {case_idx+1}, Y={y_phys:.2f} m", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("X (m)"); axes[2].set_ylabel("Z (m)"); axes[2].set_aspect('equal'); axes[2].grid(alpha=0.3, ls='--')
    plt.colorbar(im2, ax=axes[2], label="Error (°C)")

    mae  = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    mx   = float(np.max(err))

    fig.suptitle(
        f"Temperature Field Reconstruction  |  Room: {ROOM_SIZE[0]}×{ROOM_SIZE[1]}×{ROOM_SIZE[2]} m\n"
        f"MAE={mae:.3f}°C  RMSE={rmse:.3f}°C  Max={mx:.3f}°C",
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"case_{case_idx+1}_Y{y_phys:.2f}m.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved slice figure: {out_path}")
    return mae, rmse, mx

# -------------------------- 5. Main Pipeline --------------------------

def main():
    # ======= Paths (EDIT THESE) =======
    boundary_path = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
    snapshot_dir  = r"C:\Users\Lenovo\Desktop\condition_data_files"

    # ======= Step 1: Load data =======
    loader = DataCenterDataLoader(boundary_path, snapshot_dir, n_conditions=89)
    bc_raw, bc_norm = loader.load_boundary_conditions()
    coords, snaps = loader.load_snapshots()
    tensor, mask = loader.build_tensor_with_mask()

    # ======= Step 2: Tucker =======
    ranks = [15, 15, 15, 20]
    decomposer = TuckerDecomposer(tensor, ranks=ranks)
    core, factors, cond_coeff = decomposer.perform()

    recon_all = decomposer.reconstruct_tensor()  # same factors
    tucker_metrics = TuckerDecomposer.evaluate(tensor, recon_all, mask)
    print(f"\nTucker reconstruction metrics: {tucker_metrics}")

    # ======= Step 3: Train/Val/Test split =======
    idx_all = np.arange(cond_coeff.shape[0])  # N_cases
    train_idx, test_idx = train_test_split(idx_all, test_size=0.1, random_state=SEED)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.1, random_state=SEED)

    X_train, y_train = bc_norm[train_idx], cond_coeff[train_idx]
    X_val,   y_val   = bc_norm[val_idx],   cond_coeff[val_idx]
    X_test,  y_test  = bc_norm[test_idx],  cond_coeff[test_idx]

    train_loader = DataLoader(ConditionDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader   = DataLoader(ConditionDataset(X_val, y_val),   batch_size=16, shuffle=False)

    # ======= Step 4: Train NN =======
    model = MappingNetwork(input_dim=14, output_dim=ranks[3], hidden_dims=(128, 256, 256, 128))
    trainer = NetworkTrainer(model, train_loader, val_loader)
    trainer.train(epochs=300, early_stop=30, ckpt='best_model.pth')

    # save training curve
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(trainer.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer.val_losses,   label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training & Validation Loss'); plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'training_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ======= Step 5: Test inference =======
    model.eval()
    device = trainer.device
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()

    # ======= Step 6: Reconstruct & Evaluate per-case =======
    results = []
    for i, case_idx in enumerate(test_idx):
        coeff_pred = y_pred[i:i+1]  # shape (1, r4)
        factors_pred = [factors[0], factors[1], factors[2], coeff_pred]
        temp_pred = tl.tucker_to_tensor((core, factors_pred))[:, :, :, 0]  # (Nx,Ny,Nz)
        temp_true = tensor[:, :, :, case_idx]
        m = mask[:, :, :, case_idx] > 0

        mae = float(np.mean(np.abs(temp_true[m] - temp_pred[m])))
        rmse = float(np.sqrt(np.mean((temp_true[m] - temp_pred[m])**2)))
        re = float(mae / (np.mean(np.abs(temp_true[m])) + 1e-8))
        print(f"Case {case_idx+1}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, RE={re:.2%}")
        results.append({"Case": int(case_idx+1), "MAE": mae, "RMSE": rmse, "Relative_Error": re})

    # ======= Step 7: Visualization with physical coordinates =======
    interpolator = TemperatureFieldInterpolator(loader.x_unique, loader.y_unique, loader.z_unique)
    y_index = interpolator.nearest_y_index(TARGET_Y_SLICE_M)

    vis_errors = []
    for i, case_idx in enumerate(test_idx):
        coeff_pred = y_pred[i:i+1]
        factors_pred = [factors[0], factors[1], factors[2], coeff_pred]
        temp_pred_vis = tl.tucker_to_tensor((core, factors_pred))[:, :, :, 0]
        temp_true_vis = tensor[:, :, :, case_idx]
        mask_vis = mask[:, :, :, case_idx]
        mae, rmse, mx = visualize_temperature_slice(
            temp_true_vis, temp_pred_vis, mask_vis, interpolator, case_idx, y_index, out_dir=SAVE_DIR
        )
        vis_errors.append({"Case": int(case_idx+1), "MAE": mae, "RMSE": rmse, "Max_Error": mx})

    # ======= Step 8: Save artifacts =======
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(SAVE_DIR, 'test_results.csv'), index=False)

    vis_df = pd.DataFrame(vis_errors)
    vis_df.to_csv(os.path.join(SAVE_DIR, 'visualization_errors.csv'), index=False)

    stash = {
        "model_state_dict": model.state_dict(),
        "tucker_core": core,
        "tucker_factors": factors,
        "condition_coefficients": cond_coeff,
        "scaler": loader.scaler,
        "ranks": ranks,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "coords": coords,
        "x_unique": loader.x_unique, "y_unique": loader.y_unique, "z_unique": loader.z_unique,
        "room_size": ROOM_SIZE,
        "tucker_metrics": tucker_metrics,
    }
    with open(os.path.join(SAVE_DIR, "experiment_results.pkl"), "wb") as f:
        pickle.dump(stash, f)

    print("\n================ SUMMARY ================")
    print(f"Room size (X×Y×Z): {ROOM_SIZE[0]} × {ROOM_SIZE[1]} × {ROOM_SIZE[2]} m")
    print(f"Tucker ranks: {ranks}")
    print(f"Tucker MAE={tucker_metrics['MAE']:.4f} °C, RMSE={tucker_metrics['RMSE']:.4f} °C")
    print(f"Test mean MAE={results_df['MAE'].mean():.4f} °C, RMSE={results_df['RMSE'].mean():.4f} °C, "
          f"RE={results_df['Relative_Error'].mean():.2%}")
    print(f"Y-slice used: {loader.y_unique[y_index]:.3f} m (nearest to {TARGET_Y_SLICE_M} m)")
    print(f"All figures & CSV saved under: {os.path.abspath(SAVE_DIR)}")

# -------------------------- Entrypoint --------------------------

if __name__ == "__main__":
    main()
