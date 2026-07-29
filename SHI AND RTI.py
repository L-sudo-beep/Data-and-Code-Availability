import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\Lenovo\Desktop\condition_data_files"
SAVE_PATH = r"C:\Users\Lenovo\Desktop\SHI_RTI_results.csv"

N_CASES = 89

rack_inlet_regions = [
    {
        "name": "rack_inlet_face_1",
        "xmin": 2.8, "xmax": 2.8,
        "ymin": 0.69, "ymax": 2.74,
        "zmin": 3.0, "zmax": 6.0,
    },
    {
        "name": "rack_inlet_face_2",
        "xmin": 3.8, "xmax": 3.8,
        "ymin": 0.69, "ymax": 2.74,
        "zmin": 3.0, "zmax": 6.0,
    },
]

rack_outlet_regions = [
    {
        "name": "rack_outlet_face_1",
        "xmin": 1.78, "xmax": 1.78,
        "ymin": 0.69, "ymax": 2.74,
        "zmin": 3.0, "zmax": 6.0,
    },
    {
        "name": "rack_outlet_face_2",
        "xmin": 4.9, "xmax": 4.9,
        "ymin": 0.69, "ymax": 2.74,
        "zmin": 3.0, "zmax": 6.0,
    },
]

ac_supply_regions = [
    {
        "name": "ac_supply_1",
        "xmin": 3.0, "xmax": 4.0,
        "ymin": 0.6, "ymax": 0.6,
        "zmin": 7.15, "zmax": 7.5,
    },
    {
        "name": "ac_supply_2",
        "xmin": 2.7, "xmax": 3.6,
        "ymin": 0.6, "ymax": 0.6,
        "zmin": 1.5, "zmax": 1.9,
    },
]

ac_return_regions = [
    {
        "name": "ac_return_1",
        "xmin": 2.4, "xmax": 4.2,
        "ymin": 2.40, "ymax": 2.55,
        "zmin": 1.4, "zmax": 2.2,
    },
    {
        "name": "ac_return_2",
        "xmin": 2.5, "xmax": 4.3,
        "ymin": 2.40, "ymax": 2.55,
        "zmin": 6.8, "zmax": 7.5,
    },
]


def read_snapshot_csv(file_path):
    """读取单个工况 CSV 文件"""
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(file_path)


def axis_region_mask(values, vmin, vmax, tol=1e-8):
    arr = values.to_numpy(dtype=float)

    lo = min(vmin, vmax)
    hi = max(vmin, vmax)
    if np.isclose(lo, hi, atol=tol):
        target = 0.5 * (lo + hi)

        unique_vals = np.sort(np.unique(arr))
        nearest_val = unique_vals[np.argmin(np.abs(unique_vals - target))]

        mask = np.isclose(arr, nearest_val, atol=tol)
        return mask

    mask = (arr >= lo - tol) & (arr <= hi + tol)
    return mask


def region_mean_temperature(df, region):

    required_cols = {"X (m)", "Y (m)", "Z (m)", "Temperature"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV 文件缺少必要列。需要列名为: {required_cols}，"
            f"当前文件列名为: {list(df.columns)}"
        )

    x_mask = axis_region_mask(df["X (m)"], region["xmin"], region["xmax"])
    y_mask = axis_region_mask(df["Y (m)"], region["ymin"], region["ymax"])
    z_mask = axis_region_mask(df["Z (m)"], region["zmin"], region["zmax"])

    mask = x_mask & y_mask & z_mask

    selected = df.loc[mask, "Temperature"]

    if selected.empty:
        raise ValueError(
            f"No grid points found in region: {region['name']}\n"
            f"Region setting: {region}"
        )

    return selected.mean()


results = []

for case_id in range(1, N_CASES + 1):
    file_path = os.path.join(DATA_DIR, f"{case_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = read_snapshot_csv(file_path)

    rack_inlet_temps = [
        region_mean_temperature(df, region)
        for region in rack_inlet_regions
    ]

    rack_outlet_temps = [
        region_mean_temperature(df, region)
        for region in rack_outlet_regions
    ]
    ac_supply_temps = [
        region_mean_temperature(df, region)
        for region in ac_supply_regions
    ]

    ac_return_temps = [
        region_mean_temperature(df, region)
        for region in ac_return_regions
    ]

    T_in_avg = np.mean(rack_inlet_temps)
    T_out_avg = np.mean(rack_outlet_temps)
    T_sup = np.mean(ac_supply_temps)
    T_ret = np.mean(ac_return_temps)

    numerator_shi = np.sum(np.array(rack_inlet_temps) - T_sup)
    denominator_shi = np.sum(np.array(rack_outlet_temps) - T_sup)

    SHI = numerator_shi / (denominator_shi + 1e-12)

    RTI = (T_ret - T_sup) / (T_out_avg - T_in_avg + 1e-12) * 100.0

    row = {
        "Case": case_id,
        "T_supply_avg": T_sup,
        "T_return_avg": T_ret,
        "T_rack_inlet_avg": T_in_avg,
        "T_rack_outlet_avg": T_out_avg,
        "SHI": SHI,
        "RTI_percent": RTI,
    }


    for i, temp in enumerate(rack_inlet_temps, start=1):
        row[f"Rack_inlet_face_{i}_T"] = temp


    for i, temp in enumerate(rack_outlet_temps, start=1):
        row[f"Rack_outlet_face_{i}_T"] = temp


    for i, temp in enumerate(ac_supply_temps, start=1):
        row[f"AC_supply_{i}_T"] = temp


    for i, temp in enumerate(ac_return_temps, start=1):
        row[f"AC_return_{i}_T"] = temp

    results.append(row)

    print(f"Case {case_id} finished.")


results_df = pd.DataFrame(results)

results_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print("\nSHI and RTI results saved to:")
print(SAVE_PATH)

print("\nSummary:")
print(results_df[["SHI", "RTI_percent"]].describe())
