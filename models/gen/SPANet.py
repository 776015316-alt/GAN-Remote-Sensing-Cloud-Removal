import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models.models_utils import weights_init, print_network

###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class DRDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DRDB, self).__init__()
        # 多尺度空洞卷积 + 密集连接
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, in_channels, 1)  # 1x1卷积融合
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        out = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        return out + x  # 残差连接

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=True):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)
        self.conv1 = conv1x1(self.out_channels, self.out_channels)
        self.conv2 = conv1x1(self.out_channels * 4, self.out_channels)
        self.conv3 = conv1x1(self.out_channels * 4, self.out_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        # 新增通道注意力
        self.channel_attention = ChannelAttention(self.out_channels)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        # 应用通道注意力
        out = out * self.channel_attention(out)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # 方向注意力加权
        if self.attention:
            top_up = top_up * weight[:, 0:1, :, :]
            top_right = top_right * weight[:, 1:2, :, :]
            top_down = top_down * weight[:, 2:3, :, :]
            top_left = top_left * weight[:, 3:4, :, :]

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        out = out * self.channel_attention(out)  # 再次应用通道注意力
        top_up, top_right, top_down, top_left = self.irnn2(out)

        if self.attention:
            top_up = top_up * weight[:, 0:1, :, :]
            top_right = top_right * weight[:, 1:2, :, :]
            top_down = top_down * weight[:, 2:3, :, :]
            top_left = top_left * weight[:, 3:4, :, :]

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmoid(self.conv_out(out))
        return mask

class irnn_layer(nn.Module):
    def __init__(self,in_channels):
        super(irnn_layer,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)

    def forward(self,x):
        _,_,H,W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:,:,:,1:] = F.relu(self.left_weight(x)[:,:,:,:W-1]+x[:,:,:,1:],inplace=False)
        top_right[:,:,:,:-1] = F.relu(self.right_weight(x)[:,:,:,1:]+x[:,:,:,:W-1],inplace=False)
        top_up[:,:,1:,:] = F.relu(self.up_weight(x)[:,:,:H-1,:]+x[:,:,1:,:],inplace=False)
        top_down[:,:,:-1,:] = F.relu(self.down_weight(x)[:,:,1:,:]+x[:,:,:H-1,:],inplace=False)
        return (top_up,top_right,top_down,top_left)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet, self).__init__()
        self.conv_in = nn.Sequential(
            conv3x3(3, 32),
            nn.ReLU(True)
        )
        self.SAM1 = SAM(32, 32, attention=True)
        # 替换 Bottleneck 为 DRDB
        self.res_blocks = nn.ModuleList([DRDB(32) for _ in range(17)])
        self.conv_out = conv3x3(32, 3)

    def forward(self, x):
        out = self.conv_in(x)
        # 前3个DRDB块
        for i in range(3):
            out = F.relu(self.res_blocks[i](out) + out)
        # 第一组SAM（含通道注意力）
        Attention1 = self.SAM1(out)
        for i in range(3, 6):
            out = F.relu(self.res_blocks[i](out) * Attention1 + out)
        # 第二组SAM
        Attention2 = self.SAM1(out)
        for i in range(6, 9):
            out = F.relu(self.res_blocks[i](out) * Attention2 + out)
        # 第三组SAM
        Attention3 = self.SAM1(out)
        for i in range(9, 12):
            out = F.relu(self.res_blocks[i](out) * Attention3 + out)
        # 第四组SAM
        Attention4 = self.SAM1(out)
        for i in range(12, 15):
            out = F.relu(self.res_blocks[i](out) * Attention4 + out)
        # 最后2个DRDB块
        for i in range(15, 17):
            out = F.relu(self.res_blocks[i](out) + out)
        out = self.conv_out(out)
        return Attention4, out

class Generator(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gen = nn.Sequential(OrderedDict([('gen', SPANet())]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)