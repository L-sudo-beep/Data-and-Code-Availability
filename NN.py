import os

# 定义目标路径
base_path = r"C:\Users\Lenovo\Desktop\工况"

# 如果主文件夹不存在，先创建它
if not os.path.exists(base_path):
    os.makedirs(base_path)
    print(f"主文件夹已创建: {base_path}")

# 循环创建从 281 到 536 的文件夹
# range(281, 537) 意味着从 281 开始，到 537 结束（不包含 537）
for i in range(281, 537):
    folder_name = str(i)
    folder_path = os.path.join(base_path, folder_name)

    # 检查文件夹是否已存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"已创建: {folder_path}")
    else:
        print(f"已存在: {folder_path}")

print("所有文件夹创建完成。")
