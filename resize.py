import cv2
import os

def resize_images(input_folder, output_folder, target_size=(512, 512)):
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):  # 处理JPEG格式图片
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # 如果图像读取失败，跳过该文件
            if img is None:
                print(f"Warning: Unable to read image '{filename}', skipping.")
                continue

            # Resize图像
            resized_img = cv2.resize(img, target_size)

            # 构造输出路径，保存为PNG格式
            output_filename = os.path.splitext(filename)[0] + '.png'  # 保留原始文件名，修改扩展名为PNG
            output_path = os.path.join(output_folder, output_filename)

            # 保存resize后的图像为PNG格式
            cv2.imwrite(output_path, resized_img)

            print(f"Resized '{filename}' to {target_size} and saved as '{output_filename}'")


if __name__ == '__main__':
    input_folder = '/media/xiaoqingm/63C354536EA822D7/experiments/CloudRemoval/SpA-GAN_for_cloud_removal/data/OSM16/ground_truth1024'  # 替换成您的输入文件夹路径
    output_folder = '/media/xiaoqingm/63C354536EA822D7/experiments/CloudRemoval/SpA-GAN_for_cloud_removal/data/OSM16/ground_truth'  # 替换成您的输出文件夹路径
    resize_images(input_folder, output_folder)