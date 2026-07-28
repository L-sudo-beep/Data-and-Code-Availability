#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tucker + Transformer (Leak-free HOSVD) — Residual Alpha + Energy-Weighted Loss
-------------------------------------------------------------------------------
- HOSVD 仅在训练集上拟合；测试集严格留出
- Tucker 系数 alpha 通过（A^T A + λI)^(-1)A^T vec(P) 得到，λ=1e-6（更保守释放方差）
- 训练目标：delta = alpha - alpha_mu（只学残差，推理时加回均值）
- 损失：对每一维系数以 ||A[:,k]||_2 进行能量加权的 MSE，避免收缩到均值
- 可视化色标与 POD+Transformer 完全一致（jet / YlOrRd，联合 min/max）

Author: (your name)
Date: 2025-10-19
"""

import os
import sys
import math
import json
import time
import warnings
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from matplotlib.colors import Normalize

# Torch (Transformer)
import torch
import torch.nn as nn
import torch.optim as optim

# Optional interpolation
try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -------------------------- User Config --------------------------

# Paths
PARAMS_PATH   = r"C:\Users\Lenovo\Desktop\Boundary_Conditions.csv"  # UTF-16 LE
SNAPSHOT_DIR  = r"C:\Users\Lenovo\Desktop\insert"                   # 1.csv..89.csv (UTF-8)

# Grid and data info
NX, NY, NZ = 75, 51, 103
N_CASES    = 89
Y_SLICE    = 1.23  # meters

# Adaptive rank selection (spatial modes on TRAIN only)
USE_ADAPTIVE_RANKS = True
ENERGY_THRESH = 0.99
FALLBACK_RANKS = (15, 15, 15, 20)

# Centering and scaling
CENTER_ALONG_N = True              # remove mean over cases before HOSVD (per split)
SCALE_INPUTS = True
SCALE_OUTPUT_COEFF = True          # applies to delta labels

# Alpha solve
ALPHA_RIDGE_LAMBDA = 1e-6          # smaller λ -> richer variance, less shrink to mean

# Transformer hyperparameters (smaller & stabler)
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 128
DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 1800
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 220
MAX_GRAD_NORM = 1.0

RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization
PLOT_METHOD = "contourf"   # "contourf" | "imshow"
SAVE_ERROR_MAP = True

# Output directory
FIG_DIR = "./figures_tucker_transformer_residual"
os.makedirs(FIG_DIR, exist_ok=True)

# Fixed TEST indices (1-based -> 0-based) — 与 POD+Transformer 比较一致
TEST_IDX_ONE_BASED = [8, 9, 20, 21, 58, 68, 72, 76, 84]

# -------------------------- IO & Grid --------------------------

def read_params_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16", sep=None, engine="python")
    if df.shape[0] == N_CASES + 1:
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[0] != N_CASES:
        warnings.warn(f"[WARN] Parameter CSV has {df.shape[0]} rows, expected {N_CASES}. Proceeding.")
    if df.shape[1] < 14:
        raise ValueError("Parameter CSV must contain 14 columns.")
    return df.iloc[:, :14]

def read_one_snapshot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    need = {"X (m)", "Y (m)", "Z (m)", "Temperature"}
    if not need.issubset(df.columns):
        raise ValueError(f"Snapshot {path} missing columns {need}")
    return df

def build_grid_from_df(df: pd.DataFrame):
    xs = np.sort(df["X (m)"].unique())
    ys = np.sort(df["Y (m)"].unique())
    zs = np.sort(df["Z (m)"].unique())
    if (len(xs), len(ys), len(zs)) != (NX, NY, NZ):
        raise ValueError(f"Grid size mismatch: {(len(xs), len(ys), len(zs))} vs {(NX,NY,NZ)}")
    return xs, ys, zs

def df_to_grid_values(df, xs, ys, zs):
    ix = np.searchsorted(xs, df["X (m)"].to_numpy())
    iy = np.searchsorted(ys, df["Y (m)"].to_numpy())
    iz = np.searchsorted(zs, df["Z (m)"].to_numpy())
    temp = df["Temperature"].to_numpy()
    grid = np.empty((NX, NY, NZ), dtype=np.float64)
    grid[ix, iy, iz] = temp
    return grid

# -------------------------- Tensor algebra --------------------------

def mode_n_unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)

def n_mode_product(T: np.ndarray, U: np.ndarray, mode: int) -> np.ndarray:
    T_perm = np.moveaxis(T, mode, 0)
    out = U @ T_perm.reshape(T_perm.shape[0], -1)
    new_shape = (U.shape[0],) + T_perm.shape[1:]
    out = out.reshape(new_shape)
    return np.moveaxis(out, 0, mode)

def select_rank_by_energy(X: np.ndarray, thr: float):
    U, S, VT = randomized_svd(X, n_components=min(X.shape), random_state=RANDOM_STATE)
    energy = (S**2); cum = np.cumsum(energy) / np.sum(energy)
    r = int(np.searchsorted(cum, thr) + 1)
    r = max(2, min(r, U.shape[1]))
    return U[:, :r], r

def hosvd_adaptive(T: np.ndarray, energy_thr: float, fallback_ranks: Tuple[int,int,int,int]):
    try:
        X0 = mode_n_unfold(T, 0); Ux, rx = select_rank_by_energy(X0, energy_thr)
        X1 = mode_n_unfold(T, 1); Uy, ry = select_rank_by_energy(X1, energy_thr)
        X2 = mode_n_unfold(T, 2); Uz, rz = select_rank_by_energy(X2, energy_thr)
        X3 = mode_n_unfold(T, 3); Un, rn = select_rank_by_energy(X3, energy_thr)
        ranks = (rx, ry, rz, rn)
        if any(r < 2 for r in ranks):
            raise RuntimeError("Very small adaptive ranks")
    except Exception:
        rx, ry, rz, rn = fallback_ranks
        Ux, *_ = randomized_svd(mode_n_unfold(T, 0), n_components=rx, random_state=RANDOM_STATE)
        Uy, *_ = randomized_svd(mode_n_unfold(T, 1), n_components=ry, random_state=RANDOM_STATE)
        Uz, *_ = randomized_svd(mode_n_unfold(T, 2), n_components=rz, random_state=RANDOM_STATE)
        Un, *_ = randomized_svd(mode_n_unfold(T, 3), n_components=rn, random_state=RANDOM_STATE)
        ranks = (rx, ry, rz, rn)

    G = T.copy()
    G = n_mode_product(G, Ux.T, 0)
    G = n_mode_product(G, Uy.T, 1)
    G = n_mode_product(G, Uz.T, 2)
    G = n_mode_product(G, Un.T, 3)
    return (G, [Ux, Uy, Uz, Un]), ranks

# -------------------------- Alpha targets (ridge) --------------------------

def build_A(G_trunc: np.ndarray):
    rx, ry, rz, rn = G_trunc.shape
    A = G_trunc.reshape(rx*ry*rz, rn)  # columns are vec(G[...,k])
    return A  # (m, rn)

def ridge_pinv(A: np.ndarray, lam: float):
    AtA = A.T @ A
    r = AtA.shape[0]
    M = np.linalg.solve(AtA + lam*np.eye(r), A.T)  # (rn, m)
    return M

def project_to_spatial(Tn: np.ndarray, Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray) -> np.ndarray:
    P = n_mode_product(Tn, Ux.T, 0)
    P = n_mode_product(P, Uy.T, 1)
    P = n_mode_product(P, Uz.T, 2)
    return P  # (rx,ry,rz)

def compute_alpha_batch(T_used: np.ndarray,
                        idx_list: np.ndarray,
                        Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray,
                        A: np.ndarray, M: np.ndarray) -> np.ndarray:
    alphas = []
    for n in idx_list:
        Pn = project_to_spatial(T_used[..., n], Ux, Uy, Uz).reshape(-1)  # (m,)
        alpha_n = M @ Pn  # (rn,)
        alphas.append(alpha_n)
    return np.asarray(alphas)  # (len(idx_list), rn)

def reconstruct_from_alpha(G_trunc: np.ndarray,
                           Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray,
                           A: np.ndarray,
                           alpha: np.ndarray,
                           T_mean: Optional[np.ndarray]) -> np.ndarray:
    vecP = A @ alpha
    rx, ry, rz, rn = G_trunc.shape
    P = vecP.reshape(rx, ry, rz)
    T1 = n_mode_product(P, Ux, 0)
    T2 = n_mode_product(T1, Uy, 1)
    T3 = n_mode_product(T2, Uz, 2)
    if T_mean is not None:
        T3 = T3 + T_mean[..., 0]
    return T3

# -------------------------- Transformer --------------------------

class TransformerRegressor(nn.Module):
    def __init__(self, num_features: int, output_size: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.1,
                 energy_weights: Optional[np.ndarray]=None):
        super().__init__()
        self.num_features = num_features
        self.value_embed = nn.Linear(1, d_model)
        self.type_embed  = nn.Embedding(num_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="relu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_size))
        nn.init.normal_(self.type_embed.weight, mean=0.0, std=0.02)

        if energy_weights is None:
            self.register_buffer("ew", torch.ones(output_size))
        else:
            ew = energy_weights.astype(np.float32)
            ew = ew / (ew.mean() + 1e-12)  # 归一化
            self.register_buffer("ew", torch.from_numpy(ew))

    def forward(self, x):
        B, F = x.shape
        v = self.value_embed(x.unsqueeze(-1))
        idx = torch.arange(F, device=x.device).unsqueeze(0).expand(B, -1)
        t = self.type_embed(idx)
        z = self.encoder(v + t)
        z = z.mean(dim=1)
        y = self.head(z)
        return y

def weighted_mse(pred, target, weights):
    # weights: (D,)  — 每个维度的能量权重
    diff = pred - target
    return ((diff**2) * weights).mean()

def train_transformer(Xtr, Ytr, Xva, Yva, out_dim, energy_weights, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    model = TransformerRegressor(
        num_features=Xtr.shape[1], output_size=out_dim,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_ff=DIM_FF, dropout=DROPOUT, energy_weights=energy_weights
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    Yva_t = torch.tensor(Yva, dtype=torch.float32, device=DEVICE)

    best_val = float("inf")
    best_state = None
    last_improve = 0

    for ep in range(1, EPOCHS+1):
        model.train()
        idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), BATCH_SIZE):
            sel = idx[i:i+BATCH_SIZE]
            xb = Xtr_t[sel]; yb = Ytr_t[sel]
            opt.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb, model.ew)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = weighted_mse(model(Xva_t), Yva_t, model.ew).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            last_improve = ep

        if ep - last_improve >= EARLY_STOP_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model

# -------------------------- Plotting (match POD style) --------------------------

def upsample_if_needed(xs, zs, Txz):
    if (not SCIPY_OK) or (Txz.shape[0] < 3) or (Txz.shape[1] < 3):
        return xs, zs, Txz
    xi = np.linspace(xs.min(), xs.max(), len(xs))
    zi = np.linspace(zs.min(), zs.max(), len(zs))
    sp = RectBivariateSpline(xs, zs, Txz)
    Txz_hi = sp(xi, zi)
    return xi, zi, Txz_hi

def plot_slice_pair(xs, zs, T_true_slice, T_pred_slice, y_val, out_path, method="contourf"):
    temp_min = min(T_true_slice.min(), T_pred_slice.min())
    temp_max = max(T_true_slice.max(), T_pred_slice.max())
    levels_lin = np.linspace(temp_min, temp_max, 50)

    Xg, Zg = np.meshgrid(xs, zs, indexing="ij")
    fig = plt.figure(figsize=(18, 5), dpi=150)
    plt.suptitle(f"Y={y_val:.3f} m | POD-style colormap (jet / YlOrRd)")

    ax1 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 1)
    if method == "contourf":
        c1 = ax1.contourf(Xg, Zg, T_true_slice, levels=levels_lin, cmap='jet')
    else:
        extent = (xs.min(), xs.max(), zs.min(), zs.max())
        c1 = ax1.imshow(T_true_slice.T, origin="lower", extent=extent, aspect="auto",
                        vmin=temp_min, vmax=temp_max, cmap='jet', interpolation="bicubic")
    ax1.set_title("True Temperature"); ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
    ax1.set_aspect('equal', adjustable='box'); fig.colorbar(c1, ax=ax1)

    ax2 = fig.add_subplot(1, 3 if SAVE_ERROR_MAP else 2, 2)
    if method == "contourf":
        c2 = ax2.contourf(Xg, Zg, T_pred_slice, levels=levels_lin, cmap='jet')
    else:
        extent = (xs.min(), xs.max(), zs.min(), zs.max())
        c2 = ax2.imshow(T_pred_slice.T, origin="lower", extent=extent, aspect="auto",
                        vmin=temp_min, vmax=temp_max, cmap='jet', interpolation="bicubic")
    ax2.set_title("Reconstructed Temperature"); ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)")
    ax2.set_aspect('equal', adjustable='box'); fig.colorbar(c2, ax=ax2)

    if SAVE_ERROR_MAP:
        err = np.abs(T_true_slice - T_pred_slice)
        ax3 = fig.add_subplot(1, 3, 3)
        if method == "contourf":
            c3 = ax3.contourf(Xg, Zg, err, levels=50, cmap='YlOrRd')
        else:
            extent = (xs.min(), xs.max(), zs.min(), zs.max())
            c3 = ax3.imshow(err.T, origin="lower", extent=extent, aspect="auto",
                            cmap='YlOrRd', interpolation="bicubic")
        ax3.set_title("Absolute Error"); ax3.set_xlabel("X (m)"); ax3.set_ylabel("Z (m)")
        ax3.set_aspect('equal', adjustable='box'); fig.colorbar(c3, ax=ax3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# -------------------------- Main --------------------------

def main():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("=== Load params & snapshots ===")
    params_df = read_params_csv(PARAMS_PATH)
    X_params = params_df.to_numpy(np.float64)

    # fixed test set
    idx_test = np.array([i-1 for i in TEST_IDX_ONE_BASED], dtype=int)
    idx_all  = np.arange(N_CASES)
    idx_train_all = np.setdiff1d(idx_all, idx_test, assume_unique=False)

    # grid from first file
    df0 = read_one_snapshot_csv(os.path.join(SNAPSHOT_DIR, "1.csv"))
    xs, ys, zs = build_grid_from_df(df0)
    y_slice_index = int(np.argmin(np.abs(ys - Y_SLICE)))
    print(f"Y slice ≈ {Y_SLICE} m -> index {y_slice_index}, y={ys[y_slice_index]:.4f}")

    # load all
    T_all = np.empty((NX, NY, NZ, N_CASES), dtype=np.float64)
    T_all[..., 0] = df_to_grid_values(df0, xs, ys, zs)
    for i in range(2, N_CASES+1):
        p = os.path.join(SNAPSHOT_DIR, f"{i}.csv")
        dfi = read_one_snapshot_csv(p)
        T_all[..., i-1] = df_to_grid_values(dfi, xs, ys, zs)
        if i % 10 == 0 or i == N_CASES:
            print(f"  Loaded {i}/{N_CASES}")

    # mean-centering per SPLIT (train mean only)
    train_mean = None
    if CENTER_ALONG_N:
        train_mean = T_all[..., idx_train_all].mean(axis=3, keepdims=True)   # train mean
        T_train_used = T_all[..., idx_train_all] - train_mean
        T_test_used  = T_all[..., idx_test] - train_mean
    else:
        T_train_used = T_all[..., idx_train_all]
        T_test_used  = T_all[..., idx_test]

    print("=== HOSVD on TRAIN only ===")
    (G_train, [Ux, Uy, Uz, Un_train]), ranks = hosvd_adaptive(T_train_used, ENERGY_THRESH, FALLBACK_RANKS)
    rx, ry, rz, rn_base = ranks
    print(f"TRAIN ranks: rx={rx}, ry={ry}, rz={rz}, rn={rn_base}, core={G_train.shape}")

    # rn grid
    rn_grid = sorted(set([max(2, rn_base-4), rn_base, min(G_train.shape[3], rn_base+4)]))
    print(f"rn candidates: {rn_grid}")

    # split TRAIN into train_sub / val_sub for rn selection
    train_sub, val_sub = train_test_split(idx_train_all, test_size=0.15, random_state=RANDOM_STATE, shuffle=True)

    # input scaler on train_sub
    x_scaler_sel = StandardScaler().fit(X_params[train_sub]) if SCALE_INPUTS else None

    best = {"rn": None, "val_mae": float("inf"), "bundle": None}

    for rn_sel in rn_grid:
        print(f"\n=== rn={rn_sel} -> build alpha labels (ridge) & residual targets ===")
        G_trunc = G_train[..., :rn_sel]
        A = build_A(G_trunc)
        M = ridge_pinv(A, ALPHA_RIDGE_LAMBDA)

        # energy weights per coef: ||A[:,k]||_2
        ew = np.linalg.norm(A, axis=0) + 1e-12

        # map from global case index to local index in T_train_used (0..len(train)-1)
        idx_map = {idx: i for i, idx in enumerate(idx_train_all)}
        train_local = np.array([idx_map[i] for i in train_sub], dtype=int)
        val_local   = np.array([idx_map[i] for i in val_sub], dtype=int)

        # alpha for train_sub / val_sub
        alphas_train_sub = compute_alpha_batch(T_train_used, train_local, Ux, Uy, Uz, A, M)
        alphas_val_sub   = compute_alpha_batch(T_train_used, val_local,   Ux, Uy, Uz, A, M)

        # residual targets
        alpha_mu = alphas_train_sub.mean(axis=0, keepdims=True)  # (1, rn_sel)
        deltas_train = alphas_train_sub - alpha_mu
        deltas_val   = alphas_val_sub   - alpha_mu

        # inputs
        Xtr_raw = X_params[train_sub]; Xva_raw = X_params[val_sub]
        if SCALE_INPUTS:
            Xtr = x_scaler_sel.transform(Xtr_raw)
            Xva = x_scaler_sel.transform(Xva_raw)
        else:
            Xtr, Xva = Xtr_raw, Xva_raw

        # output scalers（对 delta 而不是 alpha）
        if SCALE_OUTPUT_COEFF:
            y_scaler_sel = StandardScaler().fit(deltas_train)
            Ytr = y_scaler_sel.transform(deltas_train)
            Yva = y_scaler_sel.transform(deltas_val)
        else:
            y_scaler_sel = None
            Ytr, Yva = deltas_train, deltas_val

        # train transformer for rn_sel
        model_sel = train_transformer(Xtr, Ytr, Xva, Yva, out_dim=rn_sel, energy_weights=ew, seed=RANDOM_STATE)

        # 验证：预测 delta，再 + alpha_mu，重构到场
        model_sel.eval()
        with torch.no_grad():
            d_val_pred_s = model_sel(torch.tensor(Xva, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        d_val_pred = y_scaler_sel.inverse_transform(d_val_pred_s) if SCALE_OUTPUT_COEFF else d_val_pred_s
        alpha_val_pred = alpha_mu + d_val_pred

        # reconstruct & MAE
        maes = []
        for j, case_idx in enumerate(val_sub):
            T_pred = reconstruct_from_alpha(G_trunc, Ux, Uy, Uz, A, alpha_val_pred[j],
                                            T_mean=train_mean if CENTER_ALONG_N else None)
            T_true = T_all[..., case_idx]
            maes.append(mean_absolute_error(T_true.ravel(), T_pred.ravel()))
        mae_val_mean = float(np.mean(maes))
        print(f"  rn={rn_sel}: val MAE={mae_val_mean:.4f}")

        if mae_val_mean < best["val_mae"] - 1e-9:
            best.update({"rn": rn_sel, "val_mae": mae_val_mean,
                         "bundle": (G_trunc.copy(), A.copy(), ew.copy(), x_scaler_sel, y_scaler_sel, alpha_mu.copy())})

    rn_best = best["rn"]
    print(f"\n>>> Selected rn_best={rn_best} (val MAE={best['val_mae']:.4f})")

    # ---------- retrain on FULL TRAIN with rn_best ----------
    G_best, A_best, ew_best, x_scaler, y_scaler, alpha_mu = best["bundle"]
    M_best = ridge_pinv(A_best, ALPHA_RIDGE_LAMBDA)

    # alpha on FULL TRAIN
    train_local_full = np.arange(T_train_used.shape[3], dtype=int)
    alphas_train_full = compute_alpha_batch(T_train_used, train_local_full, Ux, Uy, Uz, A_best, M_best)
    deltas_train_full = alphas_train_full - alpha_mu  # (N_train, rn_best)

    # inputs
    X_train_raw = X_params[idx_train_all]
    X_test_raw  = X_params[idx_test]
    if SCALE_INPUTS:
        if x_scaler is None: x_scaler = StandardScaler().fit(X_train_raw)
        X_train = x_scaler.transform(X_train_raw)
        X_test  = x_scaler.transform(X_test_raw)
    else:
        X_train, X_test = X_train_raw, X_test_raw

    # outputs (delta)
    if SCALE_OUTPUT_COEFF:
        if y_scaler is None: y_scaler = StandardScaler().fit(deltas_train_full)
        Y_train = y_scaler.transform(deltas_train_full)
    else:
        Y_train = deltas_train_full

    # inner val for early stopping
    Xtr, Xva, Ytr, Yva = train_test_split(X_train, Y_train, test_size=0.1,
                                          random_state=RANDOM_STATE, shuffle=True)
    model = train_transformer(Xtr, Ytr, Xva, Yva, out_dim=rn_best, energy_weights=ew_best, seed=RANDOM_STATE)

    # predict test deltas -> alphas
    model.eval()
    with torch.no_grad():
        d_test_s = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    d_test = y_scaler.inverse_transform(d_test_s) if SCALE_OUTPUT_COEFF else d_test_s
    Y_pred_alpha = alpha_mu + d_test  # (n_test, rn_best)

    # reconstruct test fields (using TRAIN-only bases and mean)
    preds, gts = [], []
    for j, case_idx in enumerate(idx_test):
        T_pred = reconstruct_from_alpha(G_best, Ux, Uy, Uz, A_best, Y_pred_alpha[j],
                                        T_mean=train_mean if CENTER_ALONG_N else None)
        preds.append(T_pred)
        gts.append(T_all[..., case_idx])
    preds = np.stack(preds, axis=-1)
    gts   = np.stack(gts,   axis=-1)

    # metrics
    mae_list, rmse_list, mae_slice_list, rmse_slice_list = [], [], [], []
    for k in range(preds.shape[-1]):
        y_true = gts[..., k].ravel()
        y_hat  = preds[..., k].ravel()
        mae_list.append(mean_absolute_error(y_true, y_hat))
        rmse_list.append(math.sqrt(mean_squared_error(y_true, y_hat)))

        y_true_s = gts[:, y_slice_index, :, k].ravel()
        y_hat_s  = preds[:, y_slice_index, :, k].ravel()
        mae_slice_list.append(mean_absolute_error(y_true_s, y_hat_s))
        rmse_slice_list.append(math.sqrt(mean_squared_error(y_true_s, y_hat_s)))

    print(f"[Full 3D]   MAE={np.mean(mae_list):.4f}, RMSE={np.mean(rmse_list):.4f}")
    print(f"[Y={ys[y_slice_index]:.3f} m] MAE={np.mean(mae_slice_list):.4f}, RMSE={np.mean(rmse_slice_list):.4f}")

    # save summary
    summary = {
        "train_hosvd_only": True,
        "ranks_train": {"rx": int(rx), "ry": int(ry), "rz": int(rz), "rn_adaptive": int(rn_base), "rn_best": int(rn_best)},
        "alpha_ridge_lambda": ALPHA_RIDGE_LAMBDA,
        "residual_learning": True,
        "energy_weighted_loss": True,
        "transformer": {"d_model": D_MODEL, "nhead": NHEAD, "layers": NUM_LAYERS, "dim_ff": DIM_FF,
                        "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                        "epochs": EPOCHS, "batch": BATCH_SIZE, "patience": EARLY_STOP_PATIENCE},
        "center_along_N": bool(CENTER_ALONG_N),
        "scale_inputs": bool(SCALE_INPUTS),
        "scale_output_coeff_on_delta": bool(SCALE_OUTPUT_COEFF),
        "test_indices_1b": TEST_IDX_ONE_BASED,
        "test_indices_0b": [int(x) for x in idx_test.tolist()],
        "full3d_mae_each": [float(x) for x in mae_list],
        "full3d_rmse_each": [float(x) for x in rmse_list],
        "slice_mae_each": [float(x) for x in mae_slice_list],
        "slice_rmse_each": [float(x) for x in rmse_slice_list],
        "full3d_mae_mean": float(np.mean(mae_list)),
        "full3d_rmse_mean": float(np.mean(rmse_list)),
        "slice_mae_mean": float(np.mean(mae_slice_list)),
        "slice_rmse_mean": float(np.mean(rmse_slice_list)),
    }
    with open(os.path.join(FIG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # plots
    print("=== Plot Y-slice maps (GT vs Pred; +Error) ===")
    for k, case_idx in enumerate(idx_test):
        T_true_slice = gts[:, y_slice_index, :, k]
        T_pred_slice = preds[:, y_slice_index, :, k]
        xs_plot, zs_plot, T_true_plot = upsample_if_needed(xs, zs, T_true_slice)
        _,       _,       T_pred_plot = upsample_if_needed(xs, zs, T_pred_slice)

        out_path = os.path.join(FIG_DIR, f"yslice_case_{case_idx+1}_Tucker_Transformer_residual.png")
        plot_slice_pair(xs_plot, zs_plot, T_true_plot, T_pred_plot, ys[y_slice_index],
                        out_path, method=PLOT_METHOD)
        print(f"  Saved {out_path}")

    print("Done. Outputs in:", os.path.abspath(FIG_DIR))

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
