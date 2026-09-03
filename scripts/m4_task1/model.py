import torch
import torch.nn as nn


# 1. 定义双卷积块：这是U-Net的基础模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. 定义MiniUNet
class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        # --- 编码器 (下采样) ---
        # 每一层：双卷积 -> 最大池化
        self.enc1 = DoubleConv(in_channels, base_channels)  # 32
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)  # 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)  # 128
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)  # 256
        self.pool4 = nn.MaxPool2d(2)

        # --- 瓶颈层 ---
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)  # 512

        # --- 解码器 (上采样) ---
        # 每一层：转置卷积（上采样）-> 与编码器特征拼接 -> 双卷积
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)  # 输入是拼接后的通道数

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # --- 输出层 ---
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # --- 编码器前向传播，并保存各层输出用于跳跃连接 ---
        e1 = self.enc1(x)  # 保存 enc1 的输出
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)  # 保存 enc2 的输出
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)  # 保存 enc3 的输出
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)  # 保存 enc4 的输出
        p4 = self.pool4(e4)

        # --- 瓶颈 ---
        b = self.bottleneck(p4)

        # --- 解码器前向传播，并应用跳跃连接 ---
        d4 = self.up4(b)  # 上采样
        d4 = torch.cat([d4, e4], dim=1)  # 跳跃连接：拼接
        d4 = self.dec4(d4)  # 双卷积

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # --- 输出 ---
        out = self.out_conv(d1)
        return out