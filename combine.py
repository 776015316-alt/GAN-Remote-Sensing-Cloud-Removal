import os
import shutil


def merge_datasets(source_dirs, target_dir):

    # 创建目标目录结构
    os.makedirs(os.path.join(target_dir, "cloudy_image"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "ground_truth"), exist_ok=True)

    # 计数器用于生成唯一文件名
    counter = 1

    for dataset_dir in source_dirs:
        # 遍历每个数据集的 cloudy_image
        cloudy_src = os.path.join(dataset_dir, "cloudy_image")
        for filename in os.listdir(cloudy_src):
            if filename.endswith((".png", ".jpg", ".tif")):  # 支持常见图像格式
                # 生成新文件名（保留原始格式）
                new_name = f"merged_{counter}{os.path.splitext(filename)[1]}"
                shutil.copy2(
                    os.path.join(cloudy_src, filename),
                    os.path.join(target_dir, "cloudy_image", new_name)
                )

                # 同步处理对应的 ground_truth
                gt_src = os.path.join(dataset_dir, "ground_truth")
                gt_file = os.path.join(gt_src, filename)
                if os.path.exists(gt_file):
                    shutil.copy2(
                        gt_file,
                        os.path.join(target_dir, "ground_truth", new_name)
                    )
                else:
                    print(f"Warning: Missing ground truth for {filename}")

                counter += 1


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    SOURCE_DATASETS = ["./data/RICE_DATASET/RICE1", "./data/RICE_DATASET/RICE2"]
    TARGET_DIR = "./data/RICE_DATASET/RICE_MERGED"

    # 执行合并
    merge_datasets(SOURCE_DATASETS, TARGET_DIR)
    print(f"数据集已合并至 {TARGET_DIR}，共处理 {len(os.listdir(os.path.join(TARGET_DIR, 'cloudy_image')))} 对图像")