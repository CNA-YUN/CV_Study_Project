import torch
import torch.nn as nn
import numpy as np


# 1. Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred 是模型的原始输出（logits），需要先经过sigmoid
        pred = torch.sigmoid(pred)
        # 将pred和target展平，方便计算
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


# 2. 组合损失
class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# 计算Dice和IoU
def compute_metrics(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    precision = intersection / (pred.sum() + 1e-6)
    recall = intersection / (target.sum() + 1e-6)

    return dice.item(), iou.item(), precision.item(), recall.item()
