import matplotlib.pyplot as plt
import numpy as np

# =======================
# 输入实验数据
# =======================

# 测试集工况编号
cases = [8, 9, 20, 21, 58, 68, 72, 76, 84]

# POD+MLP 实验结果
mae_pod = [1.2247, 1.0483, 1.2138, 1.0216, 1.2211, 0.9413, 2.4486, 1.7650, 1.6170]
rmse_pod = [1.6585, 1.5407, 1.9713, 1.6611, 1.7562, 1.6230, 3.2088, 2.3304, 2.0313]

# Tucker+MLP 实验结果
mae_tucker = [1.2032, 0.9383, 1.0873, 0.8526, 1.1591, 0.7559, 3.1891, 2.7002, 1.6848]
rmse_tucker = [1.7309, 1.3014, 1.5217, 1.2150, 1.5788, 1.1225, 4.2878, 3.5190, 1.9693]

# =======================
# 绘制 MAE 对比折线图（含标注）
# =======================
plt.figure(figsize=(9, 5), dpi=150)
plt.plot(cases, mae_pod, 'o-', color='tab:blue', label='POD + MLP')
plt.plot(cases, mae_tucker, 's--', color='tab:red', label='Tucker + MLP')

# 添加数据标注
for x, y in zip(cases, mae_pod):
    plt.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='tab:blue')
for x, y in zip(cases, mae_tucker):
    plt.text(x, y - 0.1, f'{y:.2f}', ha='center', va='top', fontsize=8, color='tab:red')

plt.title('Comparison of MAE for Each Test Case', fontsize=12)
plt.xlabel('Test Case Index', fontsize=10)
plt.ylabel('Mean Absolute Error (°C)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

# =======================
# 绘制 RMSE 对比折线图（含标注）
# =======================
plt.figure(figsize=(9, 5), dpi=150)
plt.plot(cases, rmse_pod, 'o-', color='tab:blue', label='POD + MLP')
plt.plot(cases, rmse_tucker, 's--', color='tab:red', label='Tucker + MLP')

# 添加数据标注
for x, y in zip(cases, rmse_pod):
    plt.text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=8, color='tab:blue')
for x, y in zip(cases, rmse_tucker):
    plt.text(x, y - 0.1, f'{y:.2f}', ha='center', va='top', fontsize=8, color='tab:red')

plt.title('Comparison of RMSE for Each Test Case', fontsize=12)
plt.xlabel('Test Case Index', fontsize=10)
plt.ylabel('Root Mean Square Error (°C)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
