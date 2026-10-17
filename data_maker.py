import cv2
import math
import numpy as np
import random
import os


def add_fog_to_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Failed to read image {img_path}")
        return

    # 转换为浮点型并进行归一化
    img_f = img.astype(np.float32) / 255.0
    row, col = img.shape[:2]

    # 随机参数设置
    A = 0.5  # 雾的亮度系数
    beta = random.uniform(0.1, 0.2)  # 雾浓度系数
    size = random.uniform(20, math.sqrt(max(row, col)))  # 雾化范围
    center = (random.randint(0, row - 1), random.randint(0, col - 1))  # 雾化中心

    # 创建坐标网格 (向量化计算)
    x, y = np.mgrid[0:row, 0:col].astype(np.float32)

    # 计算距离场
    d = -0.04 * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) + size
    td = np.exp(-beta * d)[..., np.newaxis]  # 增加通道维度

    # 应用雾效公式
    img_f = img_f * td + A * (1 - td)

    # 处理输出图像
    output_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)  # 约束范围并转换类型

    # 保存结果
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, output_img)


def process_images_in_folder(folder_path, output_folder):
    # 自动创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 遍历处理图片
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            add_fog_to_image(img_path, output_folder)


if __name__ == '__main__':
    input_folder = os.path.abspath('C:\\Users\\Administrator\\Desktop\\patches')
    output_folder = os.path.abspath('C:\\Users\\Administrator\\Desktop\\ftygh')

    print(f"Processing images in: {input_folder}")
    process_images_in_folder(input_folder, output_folder)
    print(f"Foggy images saved to: {output_folder}")
