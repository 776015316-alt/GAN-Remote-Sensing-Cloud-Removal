import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import os


def calculate_and_plot(file_path, output_dir, max_epoch_limit=150):
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return

    entries = re.findall(r'\{.*?\}', content, re.DOTALL)
    data = []
    for entry in entries:
        try:
            data.append(json.loads(entry))
        except json.JSONDecodeError as e:
            print(f"JSON解析错误：{e}")
            return

    groups = defaultdict(lambda: {'gen_sum': 0.0, 'dis_sum': 0.0, 'count': 0})
    for item in data:
        epoch = item['epoch']
        if epoch <= max_epoch_limit:  # 过滤超过限制的epoch
            groups[epoch]['gen_sum'] += item['gen/loss']
            groups[epoch]['dis_sum'] += item['dis/loss']
            groups[epoch]['count'] += 1

    epochs = sorted(groups.keys())
    gen_losses = []
    dis_losses = []

    # 准备x轴刻度
    max_epoch = max(epochs) if epochs else 0
    if max_epoch >= 25:
        xticks = []
        current = 25
        while current <= max_epoch:
            xticks.append(current)
            current += 25
        # 确保包含最后一个epoch
        if not xticks or xticks[-1] != max_epoch:
            xticks.append(max_epoch)
    else:
        xticks = epochs.copy()  # 当epoch数小于25时显示所有刻度

    for epoch in epochs:
        count = groups[epoch]['count']
        avg_gen = groups[epoch]['gen_sum'] / count
        avg_dis = groups[epoch]['dis_sum'] / count
        gen_losses.append(avg_gen)
        dis_losses.append(avg_dis)
        print(f"Epoch {epoch}: Gen Loss {avg_gen:.4f}, Dis Loss {avg_dis:.4f}")

    # 绘制生成器损失图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_losses, marker='o', linestyle='-', label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss Trend')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(xticks)
    plt.xlim(0, max_epoch_limit)  # 设置x轴显示范围
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gen_loss.pdf'))
    plt.close()

    # 绘制判别器损失图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, dis_losses, marker='s', color='orange', linestyle='-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss Trend')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(xticks)
    plt.xlim(0, max_epoch_limit)  # 设置x轴显示范围
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dis_loss.pdf'))
    plt.close()

    # 绘制合并损失图
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_losses, marker='o', linestyle='-', label='Generator Loss')
    plt.plot(epochs, dis_losses, marker='s', color='orange', linestyle='-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Loss Trends')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(xticks)
    plt.xlim(0, max_epoch_limit)  # 设置x轴显示范围
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_loss.pdf'))
    plt.close()

    print(f"\n图表已保存至：{output_dir}")


# 使用示例（新增max_epoch_limit参数）
calculate_and_plot('./RICE1_2_ALLMERGED_results/000034/log.txt', 'training_plots', max_epoch_limit=150)