import os
import random
import shutil


def create_dataset():
    # 定义源目录和目标目录
    source_dir = './data/RICE_DATASET'
    target_cloudy = 'cloudy_image'
    target_gt = 'ground_truth'

    # 创建目标目录
    os.makedirs(target_cloudy, exist_ok=True)
    os.makedirs(target_gt, exist_ok=True)

    # 要处理的子文件夹列表
    subfolders = ['RICE1', 'RICE2', 'OSM16', 'WuHan']

    # 初始化全局计数器
    counter = 1

    for sub in subfolders:
        # 构建子文件夹路径
        cloudy_dir = os.path.join(source_dir, sub, 'cloudy_image')
        gt_dir = os.path.join(source_dir, sub, 'ground_truth')

        # 获取所有云雾图像文件
        cloudy_files = [f for f in os.listdir(cloudy_dir)
                        if os.path.isfile(os.path.join(cloudy_dir, f))]

        # 检查文件数量是否足够
        if len(cloudy_files) < 250:
            raise ValueError(f"子文件夹 {sub} 中只有 {len(cloudy_files)} 个文件，不足250个")

        # 随机选择250个文件
        selected = random.sample(cloudy_files, 250)

        for filename in selected:
            # 构建云雾图像完整路径
            src_cloudy = os.path.join(cloudy_dir, filename)

            # 查找对应的地面真实图像
            base_name = os.path.splitext(filename)[0]
            gt_matches = [f for f in os.listdir(gt_dir)
                          if os.path.splitext(f)[0] == base_name]

            if not gt_matches:
                continue  # 跳过没有对应地面真实图像的情况

            # 构建地面真实图像路径
            gt_filename = gt_matches[0]
            src_gt = os.path.join(gt_dir, gt_filename)

            # 生成新文件名（保留原始扩展名）
            ext_cloudy = os.path.splitext(filename)[1]
            ext_gt = os.path.splitext(gt_filename)[1]
            new_cloudy_name = f"{counter}{ext_cloudy}"
            new_gt_name = f"{counter}{ext_gt}"

            # 构建目标路径
            dst_cloudy = os.path.join(target_cloudy, new_cloudy_name)
            dst_gt = os.path.join(target_gt, new_gt_name)

            # 复制文件
            shutil.copy2(src_cloudy, dst_cloudy)
            shutil.copy2(src_gt, dst_gt)

            # 递增计数器
            counter += 1

    print(f"数据集创建完成！共处理了{counter - 1}对图像")


if __name__ == "__main__":
    create_dataset()