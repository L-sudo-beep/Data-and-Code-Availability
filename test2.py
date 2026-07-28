import pandas as pd
import numpy as np

# --- 请将 'your/path/to/Boundary_Conditions.csv' 替换为你的实际文件路径 ---
file_path = r'C:\Users\Lenovo\Desktop\Boundary_Conditions.csv'
# ---

# 读取工况文件
try:
    df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
except Exception as e:
    print(f"尝试使用 utf-16 读取失败: {e}")
    print("正在尝试使用其他编码读取...")
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8', on_bad_lines='skip')

# 清理列名，并定位到CRAC送风速率列
df.columns = df.columns.str.strip()
crac1_col_name = df.columns[11]
crac2_col_name = df.columns[13]

# 创建一个新的DataFrame用于显示，包含原始索引和两列CRAC速率
# 原始索引就是我们需要的 0-based index
df_display = pd.DataFrame({
    'Original_Index': df.index,
    crac1_col_name: df[crac1_col_name],
    crac2_col_name: df[crac2_col_name]
})

# --- 分别排序并显示 ---

# 1. 按CRAC1送风速率排序（升序，最低的在最前面）
df_sorted_by_crac1 = df_display.sort_values(by=crac1_col_name, ascending=True)

# 2. 按CRAC2送风速率排序（升序，最低的在最前面）
df_sorted_by_crac2 = df_display.sort_values(by=crac2_col_name, ascending=True)


# --- 最终结果 ---
print("="*60)
print("     专为“冷却系统故障模拟”设计的数据集选择助手")
print("="*60)
print("操作指南:")
print("1. 查看下面两个列表，它们分别按CRAC1和CRAC2的送风速率排序。")
print("2. 从'Sorted by CRAC1'列表的顶部，挑选出4-5个'Original_Index'。")
print("3. 从'Sorted by CRAC2'列表的顶部，挑选出4-5个'Original_Index'。")
print("4. 将这两组索引合并（去除重复），构成你的最终测试集。")
print("   例如：test_idx = [idx1, idx2, ...]")
print("="*60)


# 设置pandas显示选项，以确保所有行都被打印出来
pd.set_option('display.max_rows', None)

print("\n\n--- Sorted by CRAC1 Speed (Ascending) ---")
print(df_sorted_by_crac1)

print("\n\n--- Sorted by CRAC2 Speed (Ascending) ---")
print(df_sorted_by_crac2)

pd.reset_option('display.max_rows')

print("\n"+"="*60)
print("完成选择后，请将你整理好的 test_idx 列表应用到你的实验脚本中。")
print("="*60)

