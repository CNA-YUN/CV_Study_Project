"""
自定义 nnU-Net v2 trainer：CPU 资源受限场景下的 1 epoch smoke test。

用途：在仅 CPU（torch 2.13.0+cpu，无 CUDA）的环境中验证
"训练 -> 验证 -> 保存 checkpoint" 链路是否可跑通，
不代表 baseline 的真实分割性能。

通过环境变量 nnUNet_extTrainer 指向本文件所在目录后即可使用：
    nnUNetv2_train 901 3d_fullres 0 -tr nnUNetTrainerSmoke1Epoch -device cpu
"""

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSmoke1Epoch(nnUNetTrainer):
    """与默认 nnUNetTrainer 完全一致，仅缩短训练长度用于 smoke test。"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # ---- 仅以下 4 项与默认 trainer 不同（默认: 1000 / 250 / 50 / 50）----
        self.num_epochs = 1                      # 只跑 1 个 epoch
        self.num_iterations_per_epoch = 5        # 每个 epoch 只取 5 个 batch
        self.num_val_iterations_per_epoch = 2    # 验证只取 2 个 batch
        self.save_every = 1                      # 每 epoch 做一次验证并保存
