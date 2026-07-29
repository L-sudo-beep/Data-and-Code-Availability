#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import platform
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import griddata
from scipy.linalg import svd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


SEED = 42
ENERGY_THRESHOLD = 0.99
DCR_SWEEP_THRESHOLDS = [0.90, 0.95, 0.97, 0.99, 0.995, 0.999]
DCR_COUNT_MEAN_FIELD = True
DCR_COUNT_MLP_WEIGHTS = False
DCR_BYTES_PER_VALUE = 8

HIDDEN_LAYERS = [128, 256, 128]
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 16
DROPOUT = 0.30

USE_CUDA_IF_AVAILABLE = True

TIMING_WARMUP_RUNS = 5
TIMING_INFERENCE_REPEATS = 200
TIMING_RECONSTRUCTION_WARMUP_RUNS = 1
TIMING_RECONSTRUCTION_REPEATS = 5
TIMING_TOTAL_ONLINE_WARMUP_RUNS = 1
TIMING_TOTAL_ONLINE_REPEATS = 5

DATA_DIR = r'C:\Users\Lenovo\Desktop\insert'
BC_FILE = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
SAVE_DIR = r'C:\Users\Lenovo\Desktop\POD-1.26'

PLANE_TYPE = 'y'
PLANE_VALUE = 2.52
NUM_ADJACENT_PLANES = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    'cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu'
)


def sync_device():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)


def hardware_info():
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'device_used': str(DEVICE),
        'cuda_available': bool(torch.cuda.is_available()),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def rank_from_energy_singular_values(S, threshold):
    threshold = float(np.clip(threshold, 0.0, 1.0))
    energy = S ** 2
    if np.sum(energy) <= 0:
        return 1
    cumulative = np.cumsum(energy) / np.sum(energy)
    return max(1, min(int(np.searchsorted(cumulative, threshold) + 1), len(S)))


def count_mlp_parameters(input_dim, output_dim, hidden_layers):
    total = 0
    prev = input_dim
    for h in hidden_layers:
        total += prev * h + h
        total += 2 * h
        prev = h
    total += prev * output_dim + output_dim
    return int(total)


def compute_pod_dcr_counts(n_points, n_cases, K, input_dim, hidden_layers):
    original = int(n_points * n_cases)
    mean_entries = int(n_points) if DCR_COUNT_MEAN_FIELD else 0
    mode_entries = int(n_points * K)
    coeff_entries = int(n_cases * K)
    mlp_entries = (
        count_mlp_parameters(input_dim, K, hidden_layers)
        if DCR_COUNT_MLP_WEIGHTS else 0
    )
    compressed = mean_entries + mode_entries + coeff_entries + mlp_entries
    return {
        'original_entries': original,
        'compressed_entries': int(compressed),
        'mean_entries': mean_entries,
        'mode_entries': mode_entries,
        'coeff_entries': coeff_entries,
        'mlp_entries': mlp_entries,
        'DCR': float(original / compressed),
        'compression_percent': float((1.0 - compressed / original) * 100.0),
        'original_MB_float64': float(original * DCR_BYTES_PER_VALUE / 1024 ** 2),
        'compressed_MB_float64': float(compressed * DCR_BYTES_PER_VALUE / 1024 ** 2),
    }


def build_dcr_sweep(S, thresholds, n_points, n_train, n_all, input_dim):
    rows = []
    for thr in sorted(set(float(x) for x in thresholds)):
        K = rank_from_energy_singular_values(S, thr)
        train = compute_pod_dcr_counts(n_points, n_train, K, input_dim, HIDDEN_LAYERS)
        all_cases = compute_pod_dcr_counts(n_points, n_all, K, input_dim, HIDDEN_LAYERS)
        rows.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'energy_threshold': thr,
            'K': K,
            'train_original_entries': train['original_entries'],
            'train_compressed_entries': train['compressed_entries'],
            'train_mean_entries': train['mean_entries'],
            'train_mode_entries': train['mode_entries'],
            'train_coeff_entries': train['coeff_entries'],
            'train_mlp_entries': train['mlp_entries'],
            'DCR_train_pod': train['DCR'],
            'compression_percent_train_pod': train['compression_percent'],
            'train_original_MB_float64': train['original_MB_float64'],
            'train_compressed_MB_float64': train['compressed_MB_float64'],
            'all_original_entries': all_cases['original_entries'],
            'all_compressed_entries': all_cases['compressed_entries'],
            'all_mean_entries': all_cases['mean_entries'],
            'all_mode_entries': all_cases['mode_entries'],
            'all_coeff_entries': all_cases['coeff_entries'],
            'all_mlp_entries': all_cases['mlp_entries'],
            'DCR_all_projected_coeff': all_cases['DCR'],
            'compression_percent_all_projected_coeff': all_cases['compression_percent'],
            'all_original_MB_float64': all_cases['original_MB_float64'],
            'all_compressed_MB_float64': all_cases['compressed_MB_float64'],
        })
    return pd.DataFrame(rows)


def save_dcr_outputs(df, current_threshold):
    df.to_csv(os.path.join(SAVE_DIR, 'dcr_sweep.csv'), index=False, encoding='utf-8-sig')
    df.to_csv(os.path.join(SAVE_DIR, 'dcr_vs_energy_threshold.csv'), index=False, encoding='utf-8-sig')
    current = df[np.isclose(df['energy_threshold'], current_threshold)]
    if current.empty:
        current = df.iloc[[0]]
    current.to_csv(os.path.join(SAVE_DIR, 'dcr_current.csv'), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(9, 5))
    plt.plot(df['energy_threshold'], df['DCR_train_pod'], '-o', label='Train POD representation')
    plt.plot(df['energy_threshold'], df['DCR_all_projected_coeff'], '-s', label='All projected coefficients')
    plt.xlabel('Energy Threshold')
    plt.ylabel('Data Compression Ratio, DCR')
    plt.title('POD DCR vs Energy Threshold')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'dcr_vs_energy_threshold.png'), dpi=200)
    plt.close()
    return current.iloc[0]

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def reconstruct_from_normalized_coeff(pred_coeff_norm, scaler_coeffs, modes, mean_temp):
    coeff = scaler_coeffs.inverse_transform(
        np.asarray(pred_coeff_norm, dtype=np.float64).reshape(1, -1)
    ).ravel()
    return mean_temp.ravel() + modes @ coeff


def benchmark_inference(model, normalized_conditions):
    model.eval()
    case_times = []
    with torch.no_grad():
        for row in normalized_conditions:
            x = torch.as_tensor(row.reshape(1, -1), dtype=torch.float32, device=DEVICE)
            for _ in range(TIMING_WARMUP_RUNS):
                _ = model(x)
            sync_device()
            times = []
            for _ in range(TIMING_INFERENCE_REPEATS):
                sync_device()
                t0 = time.perf_counter()
                _ = model(x)
                sync_device()
                times.append(time.perf_counter() - t0)
            case_times.append(np.mean(times))
    return np.asarray(case_times, dtype=float)


def benchmark_reconstruction(pred_coeff_norm_all, scaler_coeffs, modes, mean_temp):
    case_times = []
    for coeff_norm in pred_coeff_norm_all:
        for _ in range(TIMING_RECONSTRUCTION_WARMUP_RUNS):
            _ = reconstruct_from_normalized_coeff(coeff_norm, scaler_coeffs, modes, mean_temp)
        times = []
        for _ in range(TIMING_RECONSTRUCTION_REPEATS):
            t0 = time.perf_counter()
            _ = reconstruct_from_normalized_coeff(coeff_norm, scaler_coeffs, modes, mean_temp)
            times.append(time.perf_counter() - t0)
        case_times.append(np.mean(times))
    return np.asarray(case_times, dtype=float)


def full_online_predict(raw_condition, scaler_conditions, model, scaler_coeffs, modes, mean_temp):
    cond_norm = scaler_conditions.transform(np.asarray(raw_condition).reshape(1, -1))
    x = torch.as_tensor(cond_norm, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred_norm = model(x)
    sync_device()
    pred_norm = pred_norm.detach().cpu().numpy()[0]
    return reconstruct_from_normalized_coeff(pred_norm, scaler_coeffs, modes, mean_temp)


def benchmark_total_online(raw_conditions, scaler_conditions, model, scaler_coeffs, modes, mean_temp):
    model.eval()
    case_times = []
    for row in raw_conditions:
        for _ in range(TIMING_TOTAL_ONLINE_WARMUP_RUNS):
            _ = full_online_predict(row, scaler_conditions, model, scaler_coeffs, modes, mean_temp)
        times = []
        for _ in range(TIMING_TOTAL_ONLINE_REPEATS):
            sync_device()
            t0 = time.perf_counter()
            _ = full_online_predict(row, scaler_conditions, model, scaler_coeffs, modes, mean_temp)
            sync_device()
            times.append(time.perf_counter() - t0)
        case_times.append(np.mean(times))
    return np.asarray(case_times, dtype=float)


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        'mean_seconds': float(np.mean(values)),
        'std_seconds': float(np.std(values)),
        'min_seconds': float(np.min(values)),
        'max_seconds': float(np.max(values)),
        'mean_ms': float(np.mean(values) * 1000),
        'std_ms': float(np.std(values) * 1000),
        'min_ms': float(np.min(values) * 1000),
        'max_ms': float(np.max(values) * 1000),
    }


def save_timing_outputs(test_cases, decomposition_time, training_time,
                        inference_times, reconstruction_times, online_times):
    inf = latency_summary(inference_times)
    rec = latency_summary(reconstruction_times)
    online = latency_summary(online_times)
    total_offline = decomposition_time + training_time

    aggregate = pd.DataFrame([{
        'Method': 'POD',
        'Device': str(DEVICE),
        'Decomposition time (s)': decomposition_time,
        'DNN training time (s)': training_time,
        'Total offline time (s)': total_offline,
        'DNN inference time per case - mean (ms)': inf['mean_ms'],
        'DNN inference time per case - std (ms)': inf['std_ms'],
        'Field reconstruction time per case - mean (ms)': rec['mean_ms'],
        'Field reconstruction time per case - std (ms)': rec['std_ms'],
        'Total online prediction time per case - mean (ms)': online['mean_ms'],
        'Total online prediction time per case - std (ms)': online['std_ms'],
    }])
    aggregate.to_csv(os.path.join(SAVE_DIR, 'computational_time_metrics.csv'),
                     index=False, encoding='utf-8-sig')

    per_case = pd.DataFrame({
        'Case': test_cases,
        'DNN inference time (ms)': inference_times * 1000,
        'Field reconstruction time (ms)': reconstruction_times * 1000,
        'Total online prediction time (ms)': online_times * 1000,
    })
    per_case.to_csv(os.path.join(SAVE_DIR, 'computational_time_per_case.csv'),
                    index=False, encoding='utf-8-sig')

    return {
        'decomposition_time_seconds': float(decomposition_time),
        'dnn_training_time_seconds': float(training_time),
        'total_offline_time_seconds': float(total_offline),
        'dnn_inference_time_per_case': inf,
        'field_reconstruction_time_per_case': rec,
        'total_online_prediction_time_per_case': online,
    }

def select_adjacent_planes(coords, plane_type, target_value, num_adjacent):
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[plane_type]
    unique_vals = np.sort(np.unique(coords[:, axis_idx]))
    closest_idx = np.argmin(np.abs(unique_vals - target_value))
    low = max(0, closest_idx - num_adjacent)
    high = min(len(unique_vals), closest_idx + num_adjacent + 1)
    return np.isin(coords[:, axis_idx], unique_vals[low:high])


def visualize_results(true_temp, pred_temp, coords, case_index):
    mask = select_adjacent_planes(coords, PLANE_TYPE, PLANE_VALUE, NUM_ADJACENT_PLANES)
    if not np.any(mask):
        return None
    pts = coords[mask]
    x, z = pts[:, 0], pts[:, 2]
    grid_x = np.linspace(x.min(), x.max(), 150)
    grid_z = np.linspace(z.min(), z.max(), 150)
    GX, GZ = np.meshgrid(grid_x, grid_z)
    Tt = griddata((x, z), true_temp[mask], (GX, GZ), method='cubic')
    Tp = griddata((x, z), pred_temp[mask], (GX, GZ), method='cubic')
    err = np.abs(Tt - Tp)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Temperature Field Reconstruction - Case {case_index}', fontsize=16)
    items = [
        (axs[0], Tt, 'True Temperature', 'jet', np.nanmin(Tt), np.nanmax(Tt)),
        (axs[1], Tp, 'Reconstructed Temperature', 'jet', np.nanmin(Tt), np.nanmax(Tt)),
        (axs[2], err, 'Absolute Error', 'YlOrRd', 0, np.nanmax(err)),
    ]
    for ax, data, title, cmap, vmin, vmax in items:
        if np.all(np.isnan(data)):
            continue
        cs = ax.contourf(GX, GZ, data, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(cs, ax=ax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(SAVE_DIR, f'case_{case_index}_reconstruction.png'), dpi=200)
    plt.close(fig)
    return float(np.nanmax(err))

def main():
    total_start = time.perf_counter()
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'Using device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    bc_df = pd.read_csv(BC_FILE, encoding='utf-16', sep='\t').astype(float)
    conditions = bc_df.values
    num_samples = conditions.shape[0]
    val_indices_human = [8, 9, 20, 21, 58, 68, 72, 76, 84]
    val_idx = [i - 1 for i in val_indices_human]
    train_idx = [i for i in range(num_samples) if i not in val_idx]

    scaler_conditions = MinMaxScaler().fit(conditions[train_idx])
    conditions_train_norm = scaler_conditions.transform(conditions[train_idx])
    conditions_val_norm = scaler_conditions.transform(conditions[val_idx])

    temp_files = [os.path.join(DATA_DIR, f'{i}.csv') for i in range(1, num_samples + 1)]
    first_df = pd.read_csv(temp_files[0])
    coords = first_df[['X (m)', 'Y (m)', 'Z (m)']].values
    n_points = coords.shape[0]
    all_snapshots = np.zeros((n_points, num_samples), dtype=np.float64)
    print('Loading all snapshots...')
    for i in tqdm(range(num_samples), desc='Loading Snapshots'):
        all_snapshots[:, i] = pd.read_csv(temp_files[i])['Temperature'].values
    snapshots_train = all_snapshots[:, train_idx]
    snapshots_val = all_snapshots[:, val_idx]

    print('\nPerforming POD analysis...')
    t0 = time.perf_counter()
    mean_temp = np.mean(snapshots_train, axis=1, keepdims=True)
    fluctuations = snapshots_train - mean_temp
    U, S, Vt = svd(fluctuations, full_matrices=False)
    K = rank_from_energy_singular_values(S, ENERGY_THRESHOLD)
    modes = U[:, :K]
    coeffs_train = (modes.T @ fluctuations).T
    decomposition_time = time.perf_counter() - t0
    print(f'Selected K={K} for {ENERGY_THRESHOLD * 100:.1f}% energy')
    print(f'Decomposition time: {decomposition_time:.6f} s')

    thresholds = sorted(set([ENERGY_THRESHOLD] + DCR_SWEEP_THRESHOLDS))
    dcr_df = build_dcr_sweep(S, thresholds, n_points, len(train_idx), num_samples, conditions.shape[1])
    current_dcr = save_dcr_outputs(dcr_df, ENERGY_THRESHOLD)
    print('\nPOD DCR sweep table:')
    print(dcr_df[[
        'energy_threshold', 'K', 'DCR_train_pod',
        'compression_percent_train_pod', 'DCR_all_projected_coeff',
        'compression_percent_all_projected_coeff'
    ]].to_string(index=False))


    scaler_coeffs = MinMaxScaler().fit(coeffs_train)
    coeffs_train_norm = scaler_coeffs.transform(coeffs_train)

    # DNN
    model = MLP(conditions.shape[1], K, HIDDEN_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    X_train = torch.tensor(conditions_train_norm, dtype=torch.float32)
    y_train = torch.tensor(coeffs_train_norm, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(DEVICE.type == 'cuda')
    )

    print('\nTraining MLP...')
    model.train()
    sync_device()
    t0 = time.perf_counter()
    for epoch in tqdm(range(EPOCHS), desc='Training MLP'):
        for bx, by in loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 500 == 0:
            print(f'Epoch {epoch + 1}/{EPOCHS} | Loss={loss.item():.6f}')
    sync_device()
    training_time = time.perf_counter() - t0
    print(f'DNN training time: {training_time:.6f} s')

    model.eval()
    with torch.no_grad():
        val_tensor = torch.as_tensor(conditions_val_norm, dtype=torch.float32, device=DEVICE)
        pred_coeff_norm_all = model(val_tensor)
        sync_device()
        pred_coeff_norm_all = pred_coeff_norm_all.cpu().numpy()

    print('\nBenchmarking computational time...')
    inference_times = benchmark_inference(model, conditions_val_norm)
    reconstruction_times = benchmark_reconstruction(
        pred_coeff_norm_all, scaler_coeffs, modes, mean_temp
    )
    online_times = benchmark_total_online(
        conditions[val_idx], scaler_conditions, model, scaler_coeffs, modes, mean_temp
    )
    timing = save_timing_outputs(
        val_indices_human, decomposition_time, training_time,
        inference_times, reconstruction_times, online_times
    )

    print('\n================ COMPUTATIONAL TIME ================')
    print(f'Decomposition time                     : {decomposition_time:.6f} s')
    print(f'DNN training time                     : {training_time:.6f} s')
    print('DNN inference time per case           : '
          f"{timing['dnn_inference_time_per_case']['mean_ms']:.6f} ± "
          f"{timing['dnn_inference_time_per_case']['std_ms']:.6f} ms")
    print('Field reconstruction time per case    : '
          f"{timing['field_reconstruction_time_per_case']['mean_ms']:.6f} ± "
          f"{timing['field_reconstruction_time_per_case']['std_ms']:.6f} ms")
    print('Total online prediction time per case : '
          f"{timing['total_online_prediction_time_per_case']['mean_ms']:.6f} ± "
          f"{timing['total_online_prediction_time_per_case']['std_ms']:.6f} ms")
    print('====================================================')

    all_true = np.zeros((n_points, len(val_idx)))
    all_pred = np.zeros((n_points, len(val_idx)))
    mae_list, rmse_list, rrmse_list = [], [], []
    global_error_max = 0.0
    print('\nValidating test cases...')
    for i, idx in enumerate(tqdm(val_idx, desc='Validating Cases')):
        rec_temp = reconstruct_from_normalized_coeff(
            pred_coeff_norm_all[i], scaler_coeffs, modes, mean_temp
        )
        true_temp = snapshots_val[:, i]
        all_true[:, i] = true_temp
        all_pred[:, i] = rec_temp
        err = true_temp - rec_temp
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        rrmse = np.linalg.norm(err) / (np.linalg.norm(true_temp) + 1e-9)
        mae_list.append(mae)
        rmse_list.append(rmse)
        rrmse_list.append(rrmse)
        print(f'Case {idx + 1:2d}: MAE={mae:.4f} °C, RMSE={rmse:.4f} °C, rRMSE={rrmse:.4f}')
        vmax = visualize_results(true_temp, rec_temp, coords, idx + 1)
        if vmax is not None:
            global_error_max = max(global_error_max, vmax)
    np.save(os.path.join(SAVE_DIR, 'global_error_max.npy'), global_error_max)

    selected_points = [
        113159, 118413, 118310, 113158, 118516,
        113056, 118207, 113262, 113055, 112953
    ]
    rrmse_n = {}
    for point_idx in selected_points:
        if point_idx >= n_points:
            continue
        true_series = all_true[point_idx, :]
        pred_series = all_pred[point_idx, :]
        value = np.linalg.norm(true_series - pred_series) / (np.linalg.norm(true_series) + 1e-9)
        rrmse_n[str(point_idx)] = float(value)

    metrics_df = pd.DataFrame({
        'Case': [i + 1 for i in val_idx],
        'MAE (°C)': mae_list,
        'RMSE (°C)': rmse_list,
        'rRMSE': rrmse_list,
    })
    metrics_df.to_csv(os.path.join(SAVE_DIR, 'per_case_metrics.csv'),
                      index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Case'], metrics_df['MAE (°C)'], '-o', label='MAE (°C)')
    plt.plot(metrics_df['Case'], metrics_df['RMSE (°C)'], '-s', label='RMSE (°C)')
    plt.plot(metrics_df['Case'], metrics_df['rRMSE'], '-^', label='rRMSE (relative)')
    plt.xlabel('Case Number')
    plt.ylabel('Error Value')
    plt.title('POD+MLP Per-Case Error Metrics')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_metrics_compare.png'), dpi=200)
    plt.close()

    total_script_time = time.perf_counter() - total_start
    summary = {
        'model_type': 'POD_MLP',
        'energy_threshold': float(ENERGY_THRESHOLD),
        'selected_K': int(K),
        'hardware': hardware_info(),
        'timing_definition': {
            'decomposition_time': 'Mean field + centering + SVD + rank selection + modes + training coefficient projection.',
            'dnn_training_time': 'Optimizer/training loop only.',
            'dnn_inference_time': 'Forward propagation only with an already normalized input.',
            'field_reconstruction_time': 'Coefficient inverse scaling + full-field reconstruction + mean-field restoration.',
            'total_online_prediction_time': 'Input scaling + tensor conversion + DNN inference + coefficient inverse scaling + field reconstruction.',
            'excluded': 'Data loading, plotting, metric calculation, and file writing.',
        },
        'timing_settings': {
            'inference_warmup_runs': TIMING_WARMUP_RUNS,
            'inference_repeats': TIMING_INFERENCE_REPEATS,
            'reconstruction_warmup_runs': TIMING_RECONSTRUCTION_WARMUP_RUNS,
            'reconstruction_repeats': TIMING_RECONSTRUCTION_REPEATS,
            'total_online_warmup_runs': TIMING_TOTAL_ONLINE_WARMUP_RUNS,
            'total_online_repeats': TIMING_TOTAL_ONLINE_REPEATS,
        },
        'computational_time': timing,
        'dcr_current': {
            'energy_threshold': float(current_dcr['energy_threshold']),
            'K': int(current_dcr['K']),
            'DCR_train_pod': float(current_dcr['DCR_train_pod']),
            'compression_percent_train_pod': float(current_dcr['compression_percent_train_pod']),
            'DCR_all_projected_coeff': float(current_dcr['DCR_all_projected_coeff']),
            'compression_percent_all_projected_coeff': float(current_dcr['compression_percent_all_projected_coeff']),
            'train_original_entries': int(current_dcr['train_original_entries']),
            'train_compressed_entries': int(current_dcr['train_compressed_entries']),
            'all_original_entries': int(current_dcr['all_original_entries']),
            'all_compressed_entries': int(current_dcr['all_compressed_entries']),
        },
        'test_indices_one_based': val_indices_human,
        'metrics_summary': {
            'mean_MAE': float(np.mean(mae_list)),
            'mean_RMSE': float(np.mean(rmse_list)),
            'mean_rRMSE': float(np.mean(rrmse_list)),
            'MAE_each': [float(x) for x in mae_list],
            'RMSE_each': [float(x) for x in rmse_list],
            'rRMSE_each': [float(x) for x in rrmse_list],
            'rRMSE_n_pre_selected_points': rrmse_n,
        },
        'mlp_params': {
            'hidden_layers': HIDDEN_LAYERS,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'dropout': DROPOUT,
            'normalization_layer': 'BatchNorm1d',
        },
        'total_script_execution_time_seconds': float(total_script_time),
    }
    with open(os.path.join(SAVE_DIR, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\nSaved files include:')
    print('  computational_time_metrics.csv')
    print('  computational_time_per_case.csv')
    print('  metrics_summary.json')
    print('  per_case_metrics.csv')
    print('  dcr_current.csv')
    print('  dcr_sweep.csv')
    print(f'\nTotal script execution time: {total_script_time:.2f} s')
    print(f'Results saved to: {SAVE_DIR}')


if __name__ == '__main__':
    main()
