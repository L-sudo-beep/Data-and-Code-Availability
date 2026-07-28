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

# 清理列名
df.columns = df.columns.str.strip()
rack_cols = df.columns[:10]

# 计算左右两侧的平均功率和偏向度
p_left = df[rack_cols[0:5]].mean(axis=1)
p_right = df[rack_cols[5:10]].mean(axis=1)
df['bias'] = p_right - p_left

# 创建一个新的DataFrame用于显示，包含原始索引
# 原始索引就是我们需要的 0-based index
df_display = pd.DataFrame({
    'Original_Index': df.index,
    'Bias_(P_right - P_left)': df['bias']
})

# 根据偏向度对显示表进行排序（降序）
df_sorted_display = df_display.sort_values(by='Bias_(P_right - P_left)', ascending=False)

# --- 最终结果 ---
print("="*60)
print("     所有工况的“右侧加热”偏向度排序 (从高到低)")
print(" “Bias”值越大，代表该工况越偏向于 RACK6-RACK10 侧加热")
print("="*60)
print("请从下面列表的顶部，手动选择9个'Original_Index'作为你的测试集。\n")

# 设置pandas显示选项，以确保所有行都被打印出来
pd.set_option('display.max_rows', None)

print(df_sorted_display)

pd.reset_option('display.max_rows')

print("\n"+"="*60)
print("操作指南:")
print("1. 查看上面列表的第一列 'Original_Index'。")
print("2. 挑选出 bias 值最高的9个索引。")
print("3. 将这9个数字手动整理成一个列表，例如：test_idx = [idx1, idx2, ...]")
print("4. 将这个 test_idx 列表应用到你的实验脚本中。")
print("="*60)
