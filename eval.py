import numpy as np
import torch
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable
from utils import save_image
from skimage.util.dtype import dtype_range
from piq import fsim as FSIM, psnr as PIQ_PSNR, ssim as PIQ_SSIM, KID
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import Normalize


def test(config, test_data_loader, gen, criterionMSE, epoch):
    # 初始化Inception特征提取模型
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).eval()

    if config.cuda:
        inception = inception.cuda()

    avg_mse = 0
    avg_psnr = 0
    avg_ssim = 0
    avg_kid = 0
    avg_fsim = 0
    avg_cc = 0
    kid_metric = KID()

    all_real_images = []
    all_fake_images = []

    for i, batch in enumerate(test_data_loader):
        x, t = Variable(batch[0]), Variable(batch[1])
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)
        att, out = gen(x)

        # 可视化保存部分
        if epoch % config.snapshot_interval == 0:
            h, w, c = 1, 3, 3
            width, height = config.width, config.height
            allim = np.zeros((h, w, c, width, height))
            x_ = x.cpu().numpy()[0]
            t_ = t.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            t_rgb = t_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255
            allim = allim.transpose(0, 3, 1, 4, 2).reshape((h * height, w * width, c))
            save_image(config.out_dir, allim, i, epoch)

        out_clamped = torch.clamp(out[:, :3], 0, 1)
        t_clamped = torch.clamp(t[:, :3], 0, 1)
        all_real_images.append(t_clamped)
        all_fake_images.append(out_clamped)

        # 常规指标计算
        mse = criterionMSE(out_clamped, t_clamped)
        psnr = PIQ_PSNR(out_clamped, t_clamped, data_range=1.0)
        ssim = PIQ_SSIM(out_clamped, t_clamped, data_range=1.0)
        fsim_val = FSIM(out_clamped, t_clamped)

        out_flat = out_clamped.view(out_clamped.size(0), -1)
        t_flat = t_clamped.view(t_clamped.size(0), -1)
        mu_out = torch.mean(out_flat, dim=1, keepdim=True)
        mu_t = torch.mean(t_flat, dim=1, keepdim=True)
        cov = torch.mean((out_flat - mu_out) * (t_flat - mu_t), dim=1)
        std_out = torch.std(out_flat, dim=1, unbiased=False)
        std_t = torch.std(t_flat, dim=1, unbiased=False)
        cc = cov / (std_out * std_t + 1e-8)
        batch_cc = cc.mean().item()

        avg_mse += mse.item()
        avg_psnr += psnr.item()
        avg_ssim += ssim.item()
        avg_fsim += fsim_val.item()
        avg_cc += batch_cc

    # 特征提取与KID计算
    with torch.no_grad():
        all_real = torch.cat(all_real_images, dim=0)
        all_fake = torch.cat(all_fake_images, dim=0)

        # 归一化到Inception要求的范围[-1, 1]
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        real_normalized = normalize(all_real)
        fake_normalized = normalize(all_fake)

        # 提取特征（使用Inception的中间层）
        if config.cuda:
            real_features = inception(real_normalized.cuda()).view(all_real.size(0), -1)
            fake_features = inception(fake_normalized.cuda()).view(all_fake.size(0), -1)
        else:
            real_features = inception(real_normalized).view(all_real.size(0), -1)
            fake_features = inception(fake_normalized).view(all_fake.size(0), -1)

    kid_value = kid_metric(fake_features, real_features)
    avg_kid = kid_value.item()

    # 平均指标计算
    num_batches = len(test_data_loader)
    avg_mse /= num_batches
    avg_psnr /= num_batches
    avg_ssim /= num_batches
    avg_fsim /= num_batches
    avg_cc /= num_batches

    print("===> Avg. MSE: {:.4f}".format(avg_mse))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim))
    print("===> Avg. KID: {:.4f}".format(avg_kid))
    print("===> Avg. FSIM: {:.4f}".format(avg_fsim))
    print("===> Avg. CC: {:.4f}".format(avg_cc))

    return {
        'epoch': epoch,
        'mse': avg_mse,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'kid': avg_kid,
        'fsim': avg_fsim,
        'cc': avg_cc
    }