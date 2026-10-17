from PIL import Image
import numpy as np
import os


def split_tif_to_png(input_path, output_dir, tile_size=512):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取TIFF图像
    img = Image.open(input_path)
    img_array = np.array(img)

    # 获取图像尺寸
    height, width = img_array.shape[:2]

    # 计算分块数量
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # 遍历分块
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # 计算裁剪范围
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # 裁剪并保存
            tile = img.crop((left, upper, right, lower))
            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(input_path))[0]}_tile{y}_{x}.png"
            )
            tile.save(output_path)
            print(f"Saved: {output_path}")


# 使用示例
split_tif_to_png("C:/Users/Administrator/Desktop/envi_data/114.4_30.5_1008/1.tif", "D:/SpA-GAN_for_cloud_removal/data/WuHan")
