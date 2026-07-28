import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ==================== 可配置参数 ====================

# 1. 图片和层级设置
image_dir = r"C:\Users\Lenovo\Desktop\POD-images"
image_files = [
    os.path.join(image_dir, "1.26.png"),
    os.path.join(image_dir, "1.53.png"),
    os.path.join(image_dir, "1.98.png"),
    os.path.join(image_dir, "2.07.png"),  # 假设图片名为 2.25.png
    os.path.join(image_dir, "2.52.png")
]
# 这是垂直方向 Y 轴的位置
y_levels = [1.26, 1.53, 1.98, 2.25, 2.52]

# 2. 水平面的尺寸
X_LENGTH = 9.0  # X方向长度 (m)
Z_LENGTH = 6.6  # Z方向长度 (m)

# 3. 立体感偏移量：每个切片在Z轴（深度）方向上错开的距离
OFFSET_PER_SLICE = 0.8

# 4. 视图角度
ELEVATION = 20  # 仰角
AZIMUTH = -75  # 方位角

# 5. 性能优化：通过降低步长来加快绘图速度，对显示效果影响很小
# 如果图片分辨率很高，设置1会非常慢
R_STRIDE = 5
C_STRIDE = 5

# ==================== 代码实现 ====================

# 检查文件是否存在
print("--- 正在检查文件 ---")
all_files_found = True
for f in image_files:
    if not os.path.exists(f):
        print(f"❌ 找不到文件：{f}")
        all_files_found = False
    else:
        print(f"✅ 找到文件：{f}")

if not all_files_found:
    print("\n[错误] 部分图片文件未找到，请检查路径和文件名。程序已终止。")
    exit()

print("\n--- 开始绘制 ---")
# 创建3D图形和坐标轴
fig = plt.figure(figsize=(8, 7))  # 稍微调整画布大小以获得更好的视觉效果
ax = fig.add_subplot(111, projection='3d')

# 遍历图片和对应的Y轴高度
for i, (img_file, y_level) in enumerate(zip(image_files, y_levels)):
    # 读取图片作为纹理
    img = mpimg.imread(img_file)

    # 获取图片尺寸
    # 注意：mpimg读取的shape是(高度, 宽度)，对应我们这里的Z和X方向
    nz_pixels, nx_pixels, _ = img.shape

    # 创建水平面网格 (X-Z平面)
    x_coords = np.linspace(0, X_LENGTH, nx_pixels)
    z_coords = np.linspace(0, Z_LENGTH, nz_pixels)
    X_grid, Z_grid = np.meshgrid(x_coords, z_coords)

    # 需求3: 增加立体感，在Z轴方向上增加偏移
    Z_grid_offset = Z_grid + i * OFFSET_PER_SLICE

    # 需求2: 绘制表面，垂直轴是Y轴
    # ax.plot_surface(X, Y, Z, ...)
    # 我们把 Y轴数据 传给 Z 参数，把 Z轴数据 传给 Y 参数
    ax.plot_surface(
        X_grid,  # X 坐标
        Z_grid_offset,  # Z 坐标 (带偏移)
        np.full_like(X_grid, y_level),  # Y 坐标 (恒定高度)
        facecolors=img,  # 使用图片作为颜色
        shade=False,  # 不进行光照计算，直接使用图片颜色
        rstride=R_STRIDE,  # 行步长（性能优化）
        cstride=C_STRIDE  # 列步长（性能优化）
    )

# --- 坐标轴和背景美化 ---

# 需求2: 设置正确的坐标轴标签
ax.set_xlabel('X (m)', fontsize=12)
ax.set_zlabel('Y (m)', fontsize=12)  # 垂直轴现在是Y轴
ax.set_ylabel('Z (m)', fontsize=12)  # 深度轴现在是Z轴

# 需求1: 去掉背景网格和面板颜色
ax.grid(False)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# 设置坐标轴刻度位置为y_levels
ax.set_zticks(y_levels)

# 调整视图角度
ax.view_init(elev=ELEVATION, azim=AZIMUTH)

# 调整布局以防止标签重叠
plt.tight_layout()
plt.show()
print("--- 绘制完成 ---")

