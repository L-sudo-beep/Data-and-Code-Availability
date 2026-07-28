import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 🔷 1. 输入你的真实误差数据（长度 = 10）
# ======================================================

# 示例数据 —— 请替换成你的真实误差值
# pod_error    = np.array([19.21, 18, 18.38, 19.34, 17.84, 19.46, 18.51, 19.19, 19.52, 19.53])
# tucker_error = np.array([15.7, 14.99, 15.05, 15.78, 14.71, 15.6, 14.97, 14.8, 15.61, 15.53])
pod_error    = np.array([18.82, 17.77, 18.08, 18.9, 17.58, 19.02, 18.16, 18.7, 19.07, 19.07])
tucker_error = np.array([16.16, 14.87, 15.07, 16.15, 14.58, 16.07, 14.98, 15.57, 15.94, 16.02])


# ======================================================
# 🔷 2. 配置论文风格
# ======================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# ======================================================
# 🔷 3. 绘制柱状图
# ======================================================

points = [f"Point {i}" for i in range(1, 11)]
x = np.arange(len(points))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

# ------------------------------------------------------
# 🔵 使用 0–255 的 RGB 配色方案（请按需替换）
# ------------------------------------------------------
# 示例：POD 使用 (206, 208, 231)，Tucker 使用 (150, 180, 255)
colors_rgb = [
    (70, 130, 180),  # 深蓝色（Deep Blue，经典稳重的蓝色）
    (198, 40, 40)  # Tucker
]

# 归一化到 0–1（Matplotlib 要求）
colors = [(r/255, g/255, b/255) for r, g, b in colors_rgb]

# 绘制柱状图
ax.bar(x - bar_width/2, pod_error,    width=bar_width, color=colors[0], label="POD")
ax.bar(x + bar_width/2, tucker_error, width=bar_width, color=colors[1], label="Tucker")
ax.set_ylim(0, 20)
ax.set_yticks(np.arange(0, 21, 2))
# 标签与标题
ax.set_ylabel("Error (%)")
ax.set_xlabel("Point Index")
#ax.set_title("Error Comparison at 10 Hotspot Points")

# 坐标刻度
ax.set_xticks(x)
ax.set_xticklabels(points)

# 网格线
#ax.grid(axis="y", linestyle="--", alpha=0.5)

# 图例
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.10))

plt.tight_layout()

# 保存图像（按需取消注释）
# plt.savefig("hotspot_error_comparison.png", dpi=300)
# plt.savefig("hotspot_error_comparison.pdf", dpi=300)

plt.show()
