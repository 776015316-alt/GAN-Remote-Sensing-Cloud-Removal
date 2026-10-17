import os
import random

# 图片文件夹路径
images_folder = '/media/xiaoqingm/63C354536EA822D7/experiments/CloudRemoval/SpA-GAN_for_cloud_removal/data/OSM16/ground_truth'

# 获取所有图片文件名列表
image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]

# 打乱文件名列表的顺序
random.shuffle(image_files)

# 计算分割点，按8:2的比例分割
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

# 分割训练集和测试集
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# 写入训练集文件列表
with open('train_list.txt', 'w') as f:
    for filename in train_files:
        f.write(filename + '\n')

# 写入测试集文件列表
with open('test_list.txt', 'w') as f:
    for filename in test_files:
        f.write(filename + '\n')

print('训练集和测试集文件列表已生成。')