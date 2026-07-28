import os
from PIL import Image


def process_journal_images(input_folder, output_folder, min_width=1063, target_dpi=300):
    """
    批量处理图片以符合期刊要求：
    1. DPI 设置为 300
    2. 宽度至少为 min_width (默认1063)
    """

    # 如果输出目录不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出目录: {output_folder}")

    # 支持的文件格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not files:
        print("未在输入目录中找到支持的图片文件。")
        return

    print(f"找到 {len(files)} 张图片，开始处理...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)

        # 为了兼容性，建议输出为 TIFF 格式（期刊最常用的无损格式），也可以改为 .jpg 或 .png
        # 这里默认保留原后缀，或者你可以强制改为 .tif
        output_filename = os.path.splitext(filename)[0] + "_processed" + os.path.splitext(filename)[1]
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                # 1. 检查并调整像素尺寸 (Pixels)
                current_w, current_h = img.size

                if current_w < min_width:
                    # 计算缩放比例，保持纵横比
                    ratio = min_width / current_w
                    new_h = int(current_h * ratio)

                    # 使用 LANCZOS 滤镜进行高质量重采样（放大）
                    img = img.resize((min_width, new_h), Image.Resampling.LANCZOS)
                    print(f"[{filename}] 宽度 ({current_w}px) 小于标准，已调整为 ({min_width}x{new_h}px)")
                else:
                    print(f"[{filename}] 宽度 ({current_w}px) 符合标准，无需调整像素尺寸。")

                # 2. 保存并设定 DPI
                # 注意：DPI 是元组 (x_dpi, y_dpi)
                # 对于 JPG，需要设置 quality；对于 TIFF，通常使用 lzw 压缩

                ext = os.path.splitext(filename)[1].lower()

                if ext in ['.tif', '.tiff']:
                    img.save(output_path, dpi=(target_dpi, target_dpi), compression="tiff_lzw")
                elif ext in ['.jpg', '.jpeg']:
                    img.save(output_path, dpi=(target_dpi, target_dpi), quality=95, subsampling=0)
                else:  # PNG
                    img.save(output_path, dpi=(target_dpi, target_dpi))

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print(f"\n所有处理完成！图片已保存至: {output_folder}")


# ================= 配置区域 =================

# 请将此处改为你存放原始图片的文件夹路径
my_input_folder = r"C:\Users\Lenovo\Desktop\input"

# 请将此处改为你想保存处理后图片的文件夹路径
my_output_folder = r"C:\Users\Lenovo\Desktop\output"

# 运行函数
if __name__ == "__main__":
    # 如果文件夹不存在，你可以手动创建文件夹并将图片放入 input_images
    if not os.path.exists(my_input_folder):
        os.makedirs(my_input_folder)
        print(f"请将你的图片放入 '{my_input_folder}' 文件夹中，然后重新运行脚本。")
    else:
        process_journal_images(my_input_folder, my_output_folder)
