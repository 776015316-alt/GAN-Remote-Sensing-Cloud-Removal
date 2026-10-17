import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def resize_images(input_dir, output_dir, target_size=(256, 256)):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 遍历输入目录中的所有文件
    for file_path in tqdm(Path(input_dir).iterdir()):
        if file_path.suffix.lower() in supported_formats:
            try:
                # 打开图片
                img = Image.open(file_path)

                # 调整尺寸（使用高质量下采样）
                img_resized = img.resize(target_size, Image.LANCZOS)

                # 构建输出路径
                output_path = Path(output_dir) / file_path.name

                # 保存图片（保持原始格式）
                img_resized.save(output_path)

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")


if __name__ == "__main__":
    # 设置路径
    input_folder = "C:/Users/Administrator/Desktop/result/512/RICE2/Dense_HyA/epoch_0001"
    output_folder = "C:/Users/Administrator/Desktop/result/256/RICE2/Dense_HyA"

    resize_images(input_folder, output_folder)