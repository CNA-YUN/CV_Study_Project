import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class ISICDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_size=(256, 256)):
        """
        dataframe: 从CSV中筛选出的子集 (例如 df[df['split'] == 'train'])
        transform: 数据增强（翻转、旋转等）
        target_size: 输入尺寸，因为原始图片很大，必须resize
        """
        self.dataframe = dataframe.reset_index(drop=True)  # 重置索引，防止取数出错
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. 从CSV中取出这一行的路径
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        mask_path = row['mask_path']

        # 2. 用PIL打开图片和mask（RGB图，灰度mask）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 'L' 表示灰度图，单通道

        # 3. 基础尺寸调整（因为原始图片尺寸不一，如768x1024）
        # 这里用 torchvision 的 transforms 先做 Resize
        resize_transform = transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask_resize = transforms.Resize(self.target_size,
                                        interpolation=transforms.InterpolationMode.NEAREST)  # mask必须用最近邻，防止出现小数标签

        image = resize_transform(image)
        mask = mask_resize(mask)

        # 4. 手动实现同步增强 (保证 image 和 mask 操作一致)
        # 随机水平翻转 (概率 0.5)
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # 随机旋转 (-15 到 15 度)
        if random.random() < 0.5:  # 也可以直接定义旋转角度，不额外加概率
            angle = random.uniform(-15, 15)
            # 注意：旋转时图像可能会产生黑边，我们用 fill 填充黑色（对于mask填充0）
            image = transforms.functional.rotate(image, angle, fill=0)
            mask = transforms.functional.rotate(mask, angle, fill=0)  # mask的黑色背景填充0

        # 5. 转为 Tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()

        return image, mask
