import pandas as pd
from torch.utils.data import DataLoader
from dataset import ISICDataset
from model import MiniUNet
from utils import ComboLoss, compute_metrics
import torch
import torch.optim as optim
import random
import numpy as np
import os
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 1. 设置参数（通常从 config 读，这里为了方便直接写死）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # 2. 读取 CSV
    csv_path = 'data/isic_inventory.csv'  # 改成你实际的路径
    df = pd.read_csv(csv_path)

    # 3. 按 split 列划分
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"训练集: {len(train_df)} 张, 验证集: {len(val_df)} 张, 测试集: {len(test_df)} 张")

    # 4. 创建 Dataset 和 DataLoader
    # 注意：训练集需要数据增强，验证和测试集只用 resize 和 totensor
    # 这里我们直接使用 dataset 里的基础 transform，不传额外增强
    train_dataset = ISICDataset(train_df, transform=None)  # 增强已在 __getitem__ 中手动控制
    val_dataset = ISICDataset(val_df, transform=None)
    test_dataset = ISICDataset(test_df, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试时 batch=1 方便记录每个case

    # 5. 初始化模型、损失、优化器
    model = MiniUNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    criterion = ComboLoss()  # 0.5 BCE + 0.5 Dice
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6. 训练循环（30个 epoch）
    best_val_dice = 0.0
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch} Train'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        val_dice_list = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validating'):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                # 计算每个case的Dice
                for i in range(images.size(0)):
                    d, _, _, _ = compute_metrics(preds[i:i + 1], masks[i:i + 1])
                    val_dice_list.append(d)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dice_list)

        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        # 保存最佳模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), 'outputs/isic_unet_experiment/checkpoints/best_model.pth')
            print(f"-> 保存最佳模型，Val Dice: {best_val_dice:.4f}")

    # 7. 测试集评估
    print("\n===== 测试集评估 =====")
    model.load_state_dict(torch.load('outputs/isic_unet_experiment/checkpoints/best_model.pth'))
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            d, i, p, r = compute_metrics(preds, masks)
            test_metrics.append({'dice': d, 'iou': i, 'precision': p, 'recall': r})

    # 保存为 CSV
    result_df = pd.DataFrame(test_metrics)
    result_df.to_csv('outputs/isic_unet_experiment/metrics/case_metrics.csv', index=False)

    # 打印平均指标
    print(f"Test Dice: {result_df['dice'].mean():.4f}, Test IoU: {result_df['iou'].mean():.4f}")
    # 8. 可视化前20张测试集
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(20, 3, figsize=(10, 60))
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= 20: break
            images, masks = images.to(device), masks.to(device)
            pred = torch.sigmoid(model(images)) > 0.5

            # 转 numpy 并显示 (image 是 [1,3,256,256] -> 需要 permute)
            img_np = images.cpu().squeeze(0).permute(1, 2, 0).numpy()
            mask_np = masks.cpu().squeeze(0).squeeze(0).numpy()
            pred_np = pred.cpu().squeeze(0).squeeze(0).numpy()

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Input')
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('GT')
            axes[i, 2].imshow(pred_np, cmap='gray')
            axes[i, 2].set_title('Pred')
    plt.tight_layout()
    plt.savefig('outputs/isic_unet_experiment/visualizations/predictions.png', dpi=150)

if __name__ == '__main__':
    main()