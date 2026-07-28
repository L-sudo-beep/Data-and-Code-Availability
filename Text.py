"""
Data Center Temperature Field Reconstruction (Fixed)
- Tucker with mask (preferred) or 3D interpolation fallback
- Uses raw physical coordinates (meters) end-to-end
- Auto-infers tensor shape from unique coords
- Physical Y-slice selection (nearest to TARGET_Y_SLICE_M)

Author: Legend Co., Ltd. - Mia's helper
Date: 2025-10-10
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
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# -------------------------- Global Config --------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")

ROOM_SIZE = (6.6, 4.5, 9.0)   # (X, Y, Z) meters (for titles only)
TARGET_Y_SLICE_M = 1.77       # choose nearest Y layer to this physical height
SAVE_DIR = "temperature_results"  # outputs

# Fallback dense-grid step (used only if mask unsupported)
GRID_STEP = (0.1, 0.1, 0.1)   # (dx, dy, dz) meters

# -------------------------- 1. Data Loader --------------------------

class DataCenterDataLoader:
    """
    - Loads boundary conditions (89 × 14)
    - Loads 89 CSV snapshots (columns: X (m), Y (m), Z (m), Temperature)
    - Sorts each CSV by (X, Y, Z) to ensure canonical order
    - Builds sparse tensor + mask (observed voxels only)
    """

    def __init__(self, boundary_path: str, snapshot_dir: str, n_conditions: int = 89):
        self.boundary_path = boundary_path
        self.snapshot_dir = snapshot_dir
        self.n_conditions = n_conditions

        self.boundary_data = None
        self.boundary_data_normalized = None
        self.scaler = None

        self.coords = None
        self.snapshots_matrix = None  # (N_points, N_cases)
        self.n_points = 0

        self.x_unique = None
        self.y_unique = None
        self.z_unique = None

        self.tensor = None
        self.mask = None

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
        df = pd.read_csv(file_path, encoding='utf-8')
        cols_map = {c: c.strip() for c in df.columns}
        df = df.rename(columns=cols_map)
        expected = ['X (m)', 'Y (m)', 'Z (m)', 'Temperature']
        for col in expected:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in {file_path}")
        return df.sort_values(by=['X (m)', 'Y (m)', 'Z (m)'], kind='mergesort').reset_index(drop=True)

    def load_snapshots(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading temperature snapshots...")
        snapshots_list: List[np.ndarray] = []
        coords_ref = None

        for i in range(1, self.n_conditions + 1):
            file_path = os.path.join(self.snapshot_dir, f"{i}.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Snapshot CSV not found: {file_path}")
            df_sorted = self._read_and_sort_csv(file_path)
            coords = df_sorted[['X (m)', 'Y (m)', 'Z (m)']].to_numpy(dtype=np.float64)
            temps = df_sorted['Temperature'].to_numpy(dtype=np.float32)

            if coords_ref is None:
                coords_ref = coords
            else:
                if not np.allclose(coords_ref, coords, atol=1e-9):
                    raise ValueError(f"Coordinates in {i}.csv differ from case 1 even after sorting.")

            snapshots_list.append(temps)

        self.coords = coords_ref
        self.n_points = self.coords.shape[0]
        self.snapshots_matrix = np.column_stack(snapshots_list)

        # Unique coordinates
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

    # ---------- Sparse tensor + mask ----------

    def build_tensor_with_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Building 4D tensor + mask (auto shape from unique coords)...")
        Nx, Ny, Nz = len(self.x_unique), len(self.y_unique), len(self.z_unique)
        tensor = np.zeros((Nx, Ny, Nz, self.n_conditions), dtype=np.float32)   # values at observed voxels
        mask   = np.zeros((Nx, Ny, Nz, self.n_conditions), dtype=np.float32)   # 1: observed, 0: missing

        x_to_idx = {x: i for i, x in enumerate(self.x_unique)}
        y_to_idx = {y: i for i, y in enumerate(self.y_unique)}
        z_to_idx = {z: i for i, z in enumerate(self.z_unique)}

        x_arr, y_arr, z_arr = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        for p in range(self.n_points):
            xi = x_to_idx[x_arr[p]]
            yi = y_to_idx[y_arr[p]]
            zi = z_to_idx[z_arr[p]]
            tensor[xi, yi, zi, :] = self.snapshots_matrix[p, :]
            mask[xi, yi, zi, :]   = 1.0

        self.tensor, self.mask = tensor, mask
        print(f"Tensor shape: {tensor.shape}; valid points per case ~ {int(mask[..., 0].sum())}")
        return tensor, mask

# -------------------------- 2. Tucker Decomposition --------------------------

class TuckerDecomposer:
    def __init__(self, tensor: np.ndarray, mask: np.ndarray = None, ranks: List[int] = None):
        self.tensor = tensor
        self.mask = mask
        self.ranks = ranks or [15, 15, 15, 20]
        self.core = None
        self.factors = None
        self.condition_coefficients = None
        self.used_fallback = False  # mark if mask unsupported

    def perform(self):
        print(f"\nTucker decomposition with ranks={self.ranks} ...")
        # Prefer masked Tucker (so missing voxels don't bias the model)
        try:
            self.core, self.factors = tucker(self.tensor, rank=self.ranks, mask=self.mask, init='svd')
        except TypeError:
            try:
                self.core, self.factors = tucker(self.tensor, ranks=self.ranks, mask=self.mask, init='svd')
            except TypeError:
                raise RuntimeError(
                    "Your TensorLy version does not support masked Tucker. "
                    "Please run the 'dense interpolation fallback' path in main()."
                )
        self.condition_coefficients = self.factors[3]
        print(f"Core shape: {self.core.shape}")
        for i, U in enumerate(self.factors, 1):
            print(f"Factor {i} shape: {U.shape}")
        return self.core, self.factors, self.condition_coefficients

    def perform_on_dense(self):
        """Use ONLY if you created a dense tensor (no missing -> mask all ones)."""
        print(f"\nTucker (dense) with ranks={self.ranks} ...")
        try:
            self.core, self.factors = tucker(self.tensor, rank=self.ranks, init='svd')
        except TypeError:
            self.core, self.factors = tucker(self.tensor, rank=self.ranks, init='svd')
        self.condition_coefficients = self.factors[3]
        print(f"Core shape: {self.core.shape}")
        for i, U in enumerate(self.factors, 1):
            print(f"Factor {i} shape: {U.shape}")
        return self.core, self.factors, self.condition_coefficients

    def reconstruct_tensor(self, condition_coeff: np.ndarray = None) -> np.ndarray:
        if condition_coeff is None:
            condition_coeff = self.condition_coefficients
        factors_new = [self.factors[0], self.factors[1], self.factors[2], condition_coeff]
        return tl.tucker_to_tensor((self.core, factors_new))

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
        best = float('inf'); pat = 0
        print(f"\nTraining NN on {self.device} ...")
        for e in range(epochs):
            tr = self._epoch(True)
            va = self._epoch(False)
            self.train_losses.append(tr); self.val_losses.append(va)
            self.sch.step(va)
            if (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - Train {tr:.6f}  Val {va:.6f}")
            if va < best - 1e-8:
                best = va; pat = 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                pat += 1
                if pat >= early_stop:
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

    def extract_slice_data(self, temp3d: np.ndarray, mask3d: np.ndarray, y_index: int):
        slice_mask = mask3d[:, y_index, :]
        ii, kk = np.where(slice_mask > 0)
        points = np.column_stack([self.xu[ii], self.zu[kk]])  # (N, 2) meters
        values = temp3d[ii, y_index, kk]
        y_phys = self.yu[y_index]
        return points, values, y_phys

    @staticmethod
    def interpolate_slice(points: np.ndarray, values: np.ndarray,
                          method='cubic', ppm=50, smooth_sigma=1.5):
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        z_min, z_max = points[:, 1].min(), points[:, 1].max()
        n_x = max(int((x_max - x_min) * ppm) + 1, 2)
        n_z = max(int((z_max - z_min) * ppm) + 1, 2)
        gx = np.linspace(x_min, x_max, n_x)
        gz = np.linspace(z_min, z_max, n_z)
        GX, GZ = np.meshgrid(gx, gz, indexing='ij')
        grid = griddata(points, values, (GX, GZ), method=method)
        nanm = np.isnan(grid)
        if nanm.any():
            grid[nanm] = griddata(points, values, (GX[nanm], GZ[nanm]), method='nearest')
        if smooth_sigma and smooth_sigma > 0:
            grid = gaussian_filter(grid, sigma=smooth_sigma)
        return gx, gz, grid

def visualize_temperature_slice(true_3d, pred_3d, mask_3d,
                                interpolator: TemperatureFieldInterpolator,
                                case_idx: int, y_index: int,
                                out_dir: str = SAVE_DIR):
    os.makedirs(out_dir, exist_ok=True)

    pts_true, vals_true, y_phys = interpolator.extract_slice_data(true_3d, mask_3d, y_index)
    pts_pred, vals_pred, _ = interpolator.extract_slice_data(pred_3d, mask_3d, y_index)

    gx, gz, T_true = interpolator.interpolate_slice(pts_true, vals_true, method='cubic', ppm=50)
    _,  _,  T_pred = interpolator.interpolate_slice(pts_pred, vals_pred, method='cubic', ppm=50)
    err = np.abs(T_true - T_pred)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    vmin = min(T_true.min(), T_pred.min()); vmax = max(T_true.max(), T_pred.max())

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

# -------------------------- 5. Dense interpolation fallback --------------------------

def build_dense_grid_tensor(coords: np.ndarray, snapshots_matrix: np.ndarray,
                            grid_step: Tuple[float, float, float] = GRID_STEP):
    """Interpolate each case from scattered (X,Y,Z) to a dense regular grid."""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

    xs = np.arange(x_min, x_max + 1e-9, grid_step[0])
    ys = np.arange(y_min, y_max + 1e-9, grid_step[1])
    zs = np.arange(z_min, z_max + 1e-9, grid_step[2])

    Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_grid = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    n_cases = snapshots_matrix.shape[1]
    tensor_dense = np.empty((len(xs), len(ys), len(zs), n_cases), dtype=np.float32)

    print(f"Interpolating to dense grid: {len(xs)}×{len(ys)}×{len(zs)} ...")
    pts_scatter = coords  # (N_points, 3)

    for j in range(n_cases):
        vals = snapshots_matrix[:, j]
        Tg = griddata(pts_scatter, vals, pts_grid, method='linear')
        nanm = np.isnan(Tg)
        if nanm.any():
            Tg[nanm] = griddata(pts_scatter, vals, pts_grid[nanm], method='nearest')
        tensor_dense[..., j] = Tg.reshape(len(xs), len(ys), len(zs)).astype(np.float32)

    mask_dense = np.ones_like(tensor_dense, dtype=np.float32)
    return tensor_dense, mask_dense, xs, ys, zs

# -------------------------- 6. Main Pipeline --------------------------

def main():
    # ======= Paths (EDIT THESE) =======
    boundary_path = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"
    snapshot_dir  = r"C:\Users\Lenovo\Desktop\condition_data_files"

    # ======= Step 1: Load data =======
    loader = DataCenterDataLoader(boundary_path, snapshot_dir, n_conditions=89)
    bc_raw, bc_norm = loader.load_boundary_conditions()
    coords, snaps = loader.load_snapshots()
    tensor, mask = loader.build_tensor_with_mask()

    # ======= Step 2: Tucker (masked if supported) =======
    ranks = [15, 15, 15, 20]
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        decomposer = TuckerDecomposer(tensor, mask, ranks=ranks)
        core, factors, cond_coeff = decomposer.perform()
        dense_mode = False
        x_axis, y_axis, z_axis = loader.x_unique, loader.y_unique, loader.z_unique
    except RuntimeError as e:
        print("\nMasked Tucker unsupported in your TensorLy. Falling back to dense 3D interpolation...")
        tensor_dense, mask_dense, xs, ys, zs = build_dense_grid_tensor(coords, snaps, GRID_STEP)
        decomposer = TuckerDecomposer(tensor_dense, mask_dense, ranks=ranks)
        core, factors, cond_coeff = decomposer.perform_on_dense()
        dense_mode = True
        x_axis, y_axis, z_axis = xs, ys, zs
        # For downstream consistency:
        tensor, mask = tensor_dense, mask_dense
        loader.x_unique, loader.y_unique, loader.z_unique = xs, ys, zs

    recon_all = decomposer.reconstruct_tensor()
    tucker_metrics = TuckerDecomposer.evaluate(tensor, recon_all, mask)
    print(f"\nTucker reconstruction metrics: {tucker_metrics}")

    # ======= Step 3: Train/Val/Test split =======
    idx_all = np.arange(cond_coeff.shape[0])
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
        coeff_pred = y_pred[i:i+1]
        factors_pred = [factors[0], factors[1], factors[2], coeff_pred]
        temp_pred = tl.tucker_to_tensor((core, factors_pred))[:, :, :, 0]
        temp_true = tensor[:, :, :, case_idx]
        m = mask[:, :, :, case_idx] > 0

        mae = float(np.mean(np.abs(temp_true[m] - temp_pred[m])))
        rmse = float(np.sqrt(np.mean((temp_true[m] - temp_pred[m])**2)))
        re = float(mae / (np.mean(np.abs(temp_true[m])) + 1e-8))
        print(f"Case {case_idx+1}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, RE={re:.2%}")
        results.append({"Case": int(case_idx+1), "MAE": mae, "RMSE": rmse, "Relative_Error": re})

    # ======= Step 7: Visualization =======
    interpolator = TemperatureFieldInterpolator(x_axis, y_axis, z_axis)
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
        "room_size": ROOM_SIZE,
        "tucker_metrics": tucker_metrics,
        "dense_fallback": dense_mode,
        "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
    }
    with open(os.path.join(SAVE_DIR, "experiment_results.pkl"), "wb") as f:
        pickle.dump(stash, f)

    print("\n================ SUMMARY ================")
    print(f"Room size: {ROOM_SIZE[0]} × {ROOM_SIZE[1]} × {ROOM_SIZE[2]} m")
    print(f"Tucker ranks: {ranks}")
    print(f"Tucker MAE={tucker_metrics['MAE']:.4f} °C, RMSE={tucker_metrics['RMSE']:.4f} °C")
    print(f"Test mean -> MAE={results_df['MAE'].mean():.4f} °C, "
          f"RMSE={results_df['RMSE'].mean():.4f} °C, RE={results_df['Relative_Error'].mean():.2%}")
    print(f"Y-slice used: {y_axis[y_index]:.3f} m (nearest to {TARGET_Y_SLICE_M} m)")
    print(f"Dense fallback used: {dense_mode}")
    print(f"Outputs: {os.path.abspath(SAVE_DIR)}")

# -------------------------- Entrypoint --------------------------

if __name__ == "__main__":
    main()
