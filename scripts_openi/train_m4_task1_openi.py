import os
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====== 改动1：导入 c2net 上下文 ======
from c2net.context import prepare, upload_output

from dataset import ISICDataset
from model import MiniUNet
from utils import ComboLoss, compute_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # ====== 改动2：初始化 c2net 上下文 ======
    c2net_context = prepare()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    # ====== 改动3：读取 CSV（路径请根据实际存放位置修改） ======
    # 假设你的 CSV 放在代码根目录下的 data/ 文件夹内
    csv_path = 'scripts_openi/isic_inventory.csv'
    df = pd.read_csv(csv_path)

    # ====== 改动4：路径替换 ======
    # CSV 里旧的根目录（注意你的 CSV 中是反斜杠，统一转成正斜杠方便替换）
    OLD_ROOT = "F:/PythonProjects/cv_study_project/data"  # 和你原来一致
    NEW_ROOT = c2net_context.dataset_path  # 这是启智平台挂载的数据集根目录

    # 先统一将反斜杠转为正斜杠（如果 CSV 中有反斜杠）
    df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
    df['mask_path'] = df['mask_path'].str.replace('\\', '/', regex=False)

    # 替换前缀
    df['image_path'] = df['image_path'].str.replace(OLD_ROOT, NEW_ROOT, regex=False)
    df['mask_path'] = df['mask_path'].str.replace(OLD_ROOT, NEW_ROOT, regex=False)

    # 可选：打印一条验证路径
    print(f"替换后的首张图片路径: {df['image_path'].iloc[0]}")
    # 如果文件存在则继续，否则报错（可用于排查）
    # assert os.path.exists(df['image_path'].iloc[0]), "路径不正确，请检查替换规则！"

    # 按 split 划分
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    print(f"训练集: {len(train_df)} 张, 验证集: {len(val_df)} 张, 测试集: {len(test_df)} 张")

    # 创建 Dataset 和 DataLoader
    train_dataset = ISICDataset(train_df, transform=None)
    val_dataset = ISICDataset(val_df, transform=None)
    test_dataset = ISICDataset(test_df, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化模型、损失、优化器
    model = MiniUNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ====== 改动5：输出目录改为 c2net_context.output_path ======
    OUTPUT_DIR = os.path.join(c2net_context.output_path, 'm4_task1_isic_unet_experiment')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)

    # 训练循环
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

        model.eval()
        val_loss = 0.0
        val_dice_list = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validating'):
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                for i in range(images.size(0)):
                    d, _, _, _ = compute_metrics(preds[i:i + 1], masks[i:i + 1])
                    val_dice_list.append(d)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = np.mean(val_dice_list)
        print(
            f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Dice={avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model.pth'))
            print(f"-> 保存最佳模型，Val Dice: {best_val_dice:.4f}")

    # 测试集评估
    print("\n===== 测试集评估 =====")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model.pth')))
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            d, i, p, r = compute_metrics(preds, masks)
            test_metrics.append({'dice': d, 'iou': i, 'precision': p, 'recall': r})

    result_df = pd.DataFrame(test_metrics)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics', 'case_metrics.csv'), index=False)
    print(f"Test Dice: {result_df['dice'].mean():.4f}, Test IoU: {result_df['iou'].mean():.4f}")

    # 可视化前20张测试集
    fig, axes = plt.subplots(20, 3, figsize=(10, 60))
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= 20:
                break
            images, masks = images.to(device), masks.to(device)
            pred = torch.sigmoid(model(images)) > 0.5

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
    plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'predictions.png'), dpi=150)

    # ====== 改动6：回传结果（可选，仅训练任务需要） ======
    upload_output()  # 将 output_path 下的所有文件上传到 OpenI 平台


if __name__ == '__main__':
    main()
