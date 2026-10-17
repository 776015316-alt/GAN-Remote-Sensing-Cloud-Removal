import json
import os
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt


class LogReport():
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []
        os.makedirs(log_dir, exist_ok=True)  # 自动创建目录

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        gen_loss = []
        dis_loss = []

        for l in self.log_:
            epoch.append(l['epoch'])
            gen_loss.append(l['gen/loss'])
            dis_loss.append(l['dis/loss'])

        epoch = np.asarray(epoch)
        gen_loss = np.asarray(gen_loss)
        dis_loss = np.asarray(dis_loss)

        plt.figure(figsize=(10, 5))
        plt.plot(epoch, gen_loss, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Loss Trend')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_gen.pdf'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epoch, dis_loss, label='Discriminator Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss Trend')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_dis.pdf'))
        plt.close()


class TestReport():
    def __init__(self, log_dir, log_name='log_test'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []
        os.makedirs(log_dir, exist_ok=True)  # 自动创建目录

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        mse = []
        psnr = []
        kid = []
        fsim = []
        ssim = []
        cc = []

        for l in self.log_:
            epoch.append(l['epoch'])
            mse.append(l['mse'])
            psnr.append(l['psnr'])
            kid.append(l['kid'])
            fsim.append(l['fsim'])
            ssim.append(l['ssim'])
            cc.append(l['cc'])

        epoch = np.asarray(epoch)
        mse = np.asarray(mse)
        psnr = np.asarray(psnr)
        kid = np.asarray(kid)
        fsim = np.asarray(fsim)
        ssim = np.asarray(ssim)
        cc = np.asarray(cc)

        # 绘制 MSE
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, mse, label='MSE', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE Trend')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_mse.pdf'))
        plt.close()

        # 绘制 PSNR
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, psnr, label='PSNR', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('PSNR Trend')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_psnr.pdf'))
        plt.close()

        # 绘制 KID
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, kid, label='KID', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('KID')
        plt.title('KID Trend')
        plt.ylim(0, 0.5)  # KID值通常较小，范围根据实际情况调整
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_kid.pdf'))
        plt.close()

        # 绘制 FSIM
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, fsim, label='FSIM', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('FSIM')
        plt.title('FSIM Trend')
        plt.ylim(0.7, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_fsim.pdf'))
        plt.close()

        # 绘制 SSIM
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, ssim, label='SSIM', color='teal')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('SSIM Trend')
        plt.ylim(0.7, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_ssim.pdf'))
        plt.close()

        # 绘制相关系数
        plt.figure(figsize=(10, 5))
        plt.plot(epoch, cc, label='Correlation Coefficient', color='magenta')
        plt.xlabel('Epoch')
        plt.ylabel('CC')
        plt.title('Correlation Coefficient Trend')
        plt.ylim(0.7, 1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(self.log_dir, 'graph_cc.pdf'))
        plt.close()