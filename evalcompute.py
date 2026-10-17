import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as SSIM
from piq import fsim as FSIM, psnr as PIQ_PSNR, ssim as PIQ_SSIM, KID
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Normalize, ToTensor, Resize, Compose
from torch.autograd import Variable


# 配置参数（根据实际情况修改）
class Config:
    cuda = torch.cuda.is_available()
    width = 256
    height = 256
    batch_size = 1  # 因为是逐个处理图像对


config = Config()

# 初始化Inception模型
inception = inception_v3(
    weights=Inception_V3_Weights.IMAGENET1K_V1,
    transform_input=False
).eval()
if config.cuda:
    inception = inception.cuda()

# 预处理管道
preprocess = Compose([
    Resize((config.height, config.width)),
    ToTensor(),
])


# 加载图像文件夹
def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = preprocess(img)
            images.append(img)
    return torch.stack(images)


# 加载真实和生成图像
real_images = load_images('.\\data\\zizhi\\ground_truth')
fake_images = load_images('.\\data\\zizhi\\gt\\epoch_0001')

assert len(real_images) == len(fake_images), "图像数量不匹配"

# 初始化指标
kid_metric = KID()
all_real_features = []
all_fake_features = []
metrics = {
    'mse': 0,
    'psnr': 0,
    'ssim': 0,
    'fsim': 0,
    'cc': 0
}

# 计算指标
for real, fake in zip(real_images, fake_images):
    # 转换为batch格式
    real = real.unsqueeze(0)
    fake = fake.unsqueeze(0)

    if config.cuda:
        real = real.cuda()
        fake = fake.cuda()

    # 计算常规指标
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(fake, real)
        psnr = PIQ_PSNR(fake, real, data_range=1.0)
        ssim = PIQ_SSIM(fake, real, data_range=1.0)
        fsim_val = FSIM(fake, real)

        # 计算CC
        out_flat = fake.view(1, -1)
        t_flat = real.view(1, -1)
        mu_out = torch.mean(out_flat)
        mu_t = torch.mean(t_flat)
        cov = torch.mean((out_flat - mu_out) * (t_flat - mu_t))
        std_out = torch.std(out_flat)
        std_t = torch.std(t_flat)
        cc = cov / (std_out * std_t + 1e-8)

        # 累加指标
        metrics['mse'] += mse.item()
        metrics['psnr'] += psnr.item()
        metrics['ssim'] += ssim.item()
        metrics['fsim'] += fsim_val.item()
        metrics['cc'] += cc.item()

        # 提取特征用于KID
        normalize = Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        real_norm = normalize(real)
        fake_norm = normalize(fake)

        real_feat = inception(real_norm).view(real.size(0), -1)
        fake_feat = inception(fake_norm).view(fake.size(0), -1)

        all_real_features.append(real_feat.cpu())
        all_fake_features.append(fake_feat.cpu())

# 计算KID
all_real_features = torch.cat(all_real_features, dim=0)
all_fake_features = torch.cat(all_fake_features, dim=0)
kid_value = kid_metric(all_fake_features, all_real_features).item()

# 计算平均值
num_images = len(real_images)
avg_metrics = {k: v / num_images for k, v in metrics.items()}
avg_metrics['kid'] = kid_value

# 输出结果
print("Evaluation Results:")
print(f"Average MSE: {avg_metrics['mse']:.4f}")
print(f"Average PSNR: {avg_metrics['psnr']:.4f} dB")
print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
print(f"Average KID: {avg_metrics['kid']:.4f}")
print(f"Average FSIM: {avg_metrics['fsim']:.4f}")
print(f"Average CC: {avg_metrics['cc']:.4f}")