import os
import random
import json
import time
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

# ==================== OpenI 环境初始化（参考你的 ipynb） ====================
from c2net.context import prepare, upload_output

c2net_context = prepare()

# 注意：OpenI 上的 Oxford-IIIT Pet 数据集可能需要先上传或挂载。
# 假设数据集挂载在 dataset_path 下，这里直接使用 dataset_path 作为根目录
# 如果数据集不存在，torchvision 会自动下载到该目录（需要写权限）
DATA_DIR = c2net_context.dataset_path + '/OxfordPet'  # 若名字不同请修改
OUTPUT_DIR = c2net_context.output_path + '/m2_task3_pet_unet'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 固定设置 ====================
SEED = 42
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Data path: {DATA_DIR}")
print(f"Output path: {OUTPUT_DIR}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ==================== 1. 数据划分（严格按文档要求） ====================
def create_splits():
    """生成 train/val/test 划分，保存 split 文件到 output 目录"""
    # 注意：由于 OpenI 可能不允许在 dataset_path 写入，split 文件保存在 output 目录
    full_trainval = datasets.OxfordIIITPet(
        root=DATA_DIR, split="trainval", target_types="segmentation", download=True
    )
    full_test = datasets.OxfordIIITPet(
        root=DATA_DIR, split="test", target_types="segmentation", download=True
    )

    trainval_filenames = sorted([full_trainval.images[i].stem for i in range(len(full_trainval))])
    test_filenames = sorted([full_test.images[i].stem for i in range(len(full_test))])

    rng_trainval = np.random.default_rng(42)
    rng_test = np.random.default_rng(42)

    trainval_perm = rng_trainval.permutation(len(trainval_filenames))
    test_perm = rng_test.permutation(len(test_filenames))

    train_names = [trainval_filenames[i] for i in trainval_perm[:1000]]
    val_names = [trainval_filenames[i] for i in trainval_perm[1000:1200]]
    test_names = [test_filenames[i] for i in test_perm[:200]]

    # 保存到 output 目录（方便下载查看）
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
        self.dataset = datasets.OxfordIIITPet(
            root=DATA_DIR, split=split, target_types="segmentation", download=True
        )
        self.name_to_idx = {self.dataset.images[i].stem: i for i in range(len(self.dataset))}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data_idx = self.name_to_idx[name]
        image, trimap = self.dataset[data_idx]
        mask = (trimap != 2).long()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask.float()


# ==================== 3. 数据增强 ====================
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
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for feature in features:
            self.encoders.append(DoubleConv(in_channels, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

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


def compute_dice(pred, target, eps=1e-7):
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    return (2. * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)


# ==================== 6. 训练主流程 ====================
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

# 保存日志和曲线
df = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "val_dice", "lr"])
df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
axes[0].set_xlabel("Epoch");
axes[0].set_ylabel("Loss");
axes[0].set_title("Loss Curves");
axes[0].legend();
axes[0].grid(alpha=0.3)
axes[1].plot(df["epoch"], df["val_dice"], label="Val Dice", marker='o', color='green')
axes[1].set_xlabel("Epoch");
axes[1].set_ylabel("Dice");
axes[1].set_title("Validation Dice");
axes[1].legend();
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "curves.png"), dpi=150)
plt.close()

# ==================== 7. 测试评估 ====================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device))
model.eval()
results = []
eps = 1e-7
with torch.no_grad():
    for images, masks in test_loader:
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
            results.append({"dice": dice, "iou": iou, "precision": precision, "recall": recall})

df_test = pd.DataFrame(results)
df_test.to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), index=False)
summary = pd.DataFrame({
    "metric": ["dice", "iou", "precision", "recall"],
    "mean": [df_test["dice"].mean(), df_test["iou"].mean(), df_test["precision"].mean(), df_test["recall"].mean()],
    "std": [df_test["dice"].std(), df_test["iou"].std(), df_test["precision"].std(), df_test["recall"].std()]
})
summary.to_csv(os.path.join(OUTPUT_DIR, "test_metrics_summary.csv"), index=False)
print("\n=== Test Results ===")
print(summary.to_string(index=False))

# ==================== 8. 保存配置并回传 ====================
config = {
    "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
    "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "image_size": IMAGE_SIZE,
    "model": "UNet", "device": str(device),
}
with open(os.path.join(OUTPUT_DIR, "config.yaml"), "w") as f:
    yaml.dump(config, f)

print(f"\n所有产物已保存至: {OUTPUT_DIR}")
# 如果训练任务，可调用 upload_output() 回传
# upload_output()