import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import warnings

warnings.filterwarnings("ignore")

# ==================== 0. 固定设置 ====================
SEED = 42
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 256
OUTPUT_DIR = "outputs/m2_task3_pet_unet"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== 1. 数据划分（严格按照文档要求） ====================
def create_splits():
    """生成 train/val/test 划分，保存 split 文件"""
    # 加载官方数据集（仅用于获取文件名和索引）
    full_dataset = datasets.OxfordIIITPet(
        root="./data", split="trainval", target_types="segmentation", download=True
    )
    test_dataset = datasets.OxfordIIITPet(
        root="./data", split="test", target_types="segmentation", download=True
    )

    # 获取文件名并排序
    trainval_filenames = sorted([full_dataset.images[i].stem for i in range(len(full_dataset))])
    test_filenames = sorted([test_dataset.images[i].stem for i in range(len(test_dataset))])

    # 使用独立的 rng(42) 做 permutation
    rng_trainval = np.random.default_rng(42)
    rng_test = np.random.default_rng(42)

    trainval_perm = rng_trainval.permutation(len(trainval_filenames))
    test_perm = rng_test.permutation(len(test_filenames))

    train_names = [trainval_filenames[i] for i in trainval_perm[:1000]]
    val_names = [trainval_filenames[i] for i in trainval_perm[1000:1200]]
    test_names = [test_filenames[i] for i in test_perm[:200]]

    # 保存 split 文件
    with open(os.path.join(OUTPUT_DIR, "split_train.txt"), "w") as f:
        f.write("\n".join(train_names))
    with open(os.path.join(OUTPUT_DIR, "split_val.txt"), "w") as f:
        f.write("\n".join(val_names))
    with open(os.path.join(OUTPUT_DIR, "split_test.txt"), "w") as f:
        f.write("\n".join(test_names))

    return train_names, val_names, test_names


train_names, val_names, test_names = create_splits()
print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")


# ==================== 2. 自定义 Dataset ====================
class PetSegDataset(Dataset):
    def __init__(self, names, split, transform=None, target_transform=None):
        self.names = names
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        # 加载完整数据集（用于获取图像和 mask）
        self.dataset = datasets.OxfordIIITPet(
            root="./data", split=split, target_types="segmentation", download=True
        )
        # 建立文件名到索引的映射
        self.name_to_idx = {self.dataset.images[i].stem: i for i in range(len(self.dataset))}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data_idx = self.name_to_idx[name]
        image, trimap = self.dataset[data_idx]

        # trimap 转二值 mask: foreground = (trimap != 2)
        mask = (trimap != 2).long()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask.float()


# ==================== 3. 数据增强与预处理 ====================
# ImageNet 标准化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# mask 使用最近邻插值
mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

train_dataset = PetSegDataset(train_names, "trainval", train_transform, mask_transform)
val_dataset = PetSegDataset(val_names, "trainval", val_test_transform, mask_transform)
test_dataset = PetSegDataset(test_names, "test", val_test_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ==================== 4. U-Net 模型 ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_convs)):
            x = self.up_convs[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[idx](x)

        return self.final_conv(x)


# ==================== 5. Dice Loss ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


# ==================== 6. 训练函数 ====================
def compute_dice(pred, target, eps=1e-7):
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2. * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)


def train():
    model = UNet().to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss(smooth=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    best_val_dice = 0.0
    log_data = []

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice.extend(compute_dice(outputs, masks).cpu().numpy())
        val_loss /= len(val_loader.dataset)
        mean_val_dice = np.mean(val_dice)

        scheduler.step(mean_val_dice)
        current_lr = optimizer.param_groups[0]['lr']

        log_data.append([epoch, train_loss, val_loss, mean_val_dice, current_lr])
        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {mean_val_dice:.4f} | LR: {current_lr:.6f}")

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  -> Best model saved (Val Dice: {best_val_dice:.4f})")

    # 保存训练日志
    df = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "val_dice", "lr"])
    df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

    # 绘制曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["val_dice"], label="Val Dice", marker='o', color='green')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "curves.png"), dpi=150)
    plt.close()

    return model


# ==================== 7. 测试评估 ====================
def evaluate(model):
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device))
    model.eval()

    results = []
    eps = 1e-7

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            for i in range(images.size(0)):
                pred = preds[i].view(-1)
                target = masks[i].view(-1)
                tp = (pred * target).sum().item()
                fp = (pred * (1 - target)).sum().item()
                fn = ((1 - pred) * target).sum().item()

                dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                iou = (tp + eps) / (tp + fp + fn + eps)
                precision = (tp + eps) / (tp + fp + eps)
                recall = (tp + eps) / (tp + fn + eps)

                results.append({
                    "sample_id": idx * BATCH_SIZE + i,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall
                })

    df = pd.DataFrame(results)
    # 总体统计
    summary = pd.DataFrame({
        "metric": ["dice", "iou", "precision", "recall"],
        "mean": [df["dice"].mean(), df["iou"].mean(), df["precision"].mean(), df["recall"].mean()],
        "std": [df["dice"].std(), df["iou"].std(), df["precision"].std(), df["recall"].std()]
    })

    # 保存
    df.to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "test_metrics_summary.csv"), index=False)

    print("\n=== Test Results ===")
    print(summary.to_string(index=False))
    return df


# ==================== 8. 可视化 ====================
def visualize(model, num_samples=10):
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device))
    model.eval()

    # 获取原始图像（不经过归一化，用于显示）
    raw_dataset = datasets.OxfordIIITPet(
        root="./data", split="test", target_types="segmentation", download=True
    )

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    sample_indices = np.random.choice(len(test_names), num_samples, replace=False)

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            name = test_names[idx]
            data_idx = {raw_dataset.images[i].stem: i for i in range(len(raw_dataset))}[name]

            # 原始图像
            img, trimap = raw_dataset[data_idx]
            img_np = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))

            # 真值 mask
            mask = (trimap != 2).long()
            mask_np = np.array(mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))

            # 预测
            img_tensor = val_test_transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = (torch.sigmoid(output) > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

            # 叠加图
            overlay = img_np.copy().astype(np.float32) / 255.0
            overlay[pred > 0.5, 0] = overlay[pred > 0.5, 0] * 0.5 + 0.5 * 1.0
            overlay[pred > 0.5, 1] = overlay[pred > 0.5, 1] * 0.5
            overlay[pred > 0.5, 2] = overlay[pred > 0.5, 2] * 0.5

            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title("Original")
            axes[row, 0].axis('off')
            axes[row, 1].imshow(mask_np, cmap='gray')
            axes[row, 1].set_title("GT")
            axes[row, 1].axis('off')
            axes[row, 2].imshow(pred, cmap='gray')
            axes[row, 2].set_title("Prediction")
            axes[row, 2].axis('off')
            axes[row, 3].imshow(np.clip(overlay, 0, 1))
            axes[row, 3].set_title("Overlay")
            axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "visualization.png"), dpi=150)
    plt.close()


# ==================== 9. 保存配置 ====================
config = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "image_size": IMAGE_SIZE,
    "model": "UNet",
    "encoder_channels": [64, 128, 256, 512],
    "bottleneck": 1024,
    "loss": "0.5*BCEWithLogitsLoss + 0.5*DiceLoss(smooth=1.0)",
    "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=3, min_lr=1e-6)",
    "device": str(device),
    "train_samples": len(train_names),
    "val_samples": len(val_names),
    "test_samples": len(test_names),
}
with open(os.path.join(OUTPUT_DIR, "config.yaml"), "w") as f:
    yaml.dump(config, f)

# ==================== 10. 主流程 ====================
if __name__ == "__main__":
    model = train()
    evaluate(model)
    visualize(model)
    print(f"\n所有产物已保存至: {OUTPUT_DIR}/")