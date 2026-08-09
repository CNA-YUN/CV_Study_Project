import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from PIL import Image
import warnings
from _init_ import BASE_ROOT
warnings.filterwarnings("ignore")

# ==================== 本地路径配置（请务必修改为你的实际路径） ====================
BASE_DIR = BASE_ROOT  # 你的项目根目录
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据集存放目录
WEIGHT_DIR = os.path.join(BASE_DIR, "outputs")  # 从OpenI下载的权重目录
OUTPUT_DIR = os.path.join(WEIGHT_DIR, "m2_task5_error_analysis")  # 错误分析结果输出目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 固定设置 ====================
SEED = 42
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ==================== 第一部分：分类错误分析（任务2：ResNet18） ====================
def load_cifar10_test_indices():
    """生成与任务2完全一致的测试集索引（每类100张，种子42）"""
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    test_indices = []
    for class_id in range(10):
        idx = np.where(np.array(full_test.targets) == class_id)[0]
        np.random.seed(SEED)
        np.random.shuffle(idx)
        test_indices.extend(idx[:100])
    return test_indices


def create_resnet18_cifar():
    """定义与任务2完全相同的ResNet18结构（适配32x32）"""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def analyze_classification():
    print("\n" + "=" * 50)
    print("开始分类错误分析 (ResNet18)...")

    # 1. 加载模型权重
    model = create_resnet18_cifar().to(DEVICE)
    weight_path = os.path.join(WEIGHT_DIR, "m2_task2_cifar10", "ResNet18_best.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"分类权重不存在: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 2. 准备测试数据
    test_indices = load_cifar10_test_indices()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transform)
    test_subset = Subset(full_test, test_indices)
    loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 推理
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title("ResNet18 Confusion Matrix (CIFAR-10 Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_confusion_matrix.png"), dpi=150)
    plt.close()
    print("✅ 混淆矩阵已保存")

    # 5. 错误样本拼图（每类至少5个）
    error_indices = [i for i, (p, t) in enumerate(zip(all_preds, all_labels)) if p != t]
    class_errors = {c: [] for c in range(10)}
    for i in error_indices:
        class_errors[all_labels[i]].append(i)

    selected_errors = []
    for c in range(10):
        if len(class_errors[c]) >= 5:
            selected_errors.extend(np.random.choice(class_errors[c], 5, replace=False).tolist())
        else:
            selected_errors.extend(class_errors[c])
    np.random.shuffle(selected_errors)

    # 加载原始图像（未归一化）用于显示
    raw_test = datasets.CIFAR10(root=DATA_DIR, train=False, transform=transforms.ToTensor())
    raw_subset = Subset(raw_test, test_indices)

    n = len(selected_errors)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, idx in enumerate(selected_errors):
        img, _ = raw_subset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        true_label = CIFAR10_CLASSES[all_labels[idx]]
        pred_label = CIFAR10_CLASSES[all_preds[idx]]
        axes[i].imshow(img)
        axes[i].set_title(f"ID:{idx} T:{true_label}\nP:{pred_label}", fontsize=9, color='red')
        axes[i].axis('off')
    for j in range(len(selected_errors), len(axes)):
        axes[j].axis('off')
    plt.suptitle("CIFAR-10 Error Cases (≥5 per class)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_error_cases.png"), dpi=150)
    plt.close()
    print("✅ 错误样本拼图已保存")

    # 6. 保存CSV
    df = pd.DataFrame({
        'sample_id': selected_errors,
        'true_label': [all_labels[i] for i in selected_errors],
        'true_label_name': [CIFAR10_CLASSES[all_labels[i]] for i in selected_errors],
        'pred_label': [all_preds[i] for i in selected_errors],
        'pred_label_name': [CIFAR10_CLASSES[all_preds[i]] for i in selected_errors]
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "classification_error_cases.csv"), index=False)
    print("✅ 分类错误CSV已保存")
    return selected_errors, all_labels, all_preds


# ==================== 第二部分：分割错误分析（任务3：U-Net） ====================
# 复制任务3的模型定义
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
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[idx](x)
        return self.final_conv(x)


def load_pet_test_names():
    """优先读取split_test.txt，若不存在则重新生成（与任务3逻辑一致）"""
    split_path = os.path.join(WEIGHT_DIR, "m2_task3_pet_unet", "split_test.txt")
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        print(f"从 {split_path} 读取测试集，共 {len(names)} 个样本")
        return names

    # 如果文件不存在，按任务3规则重新生成（使用全部文件随机抽取200）
    print("⚠️ 未找到 split_test.txt，正在重新生成测试集划分（与任务3逻辑一致）...")
    full_dataset = datasets.OxfordIIITPet(root=DATA_DIR, split="test", target_types="segmentation", download=True)
    test_filenames = sorted([full_dataset.images[i].stem for i in range(len(full_dataset))])
    rng_test = np.random.default_rng(42)
    test_perm = rng_test.permutation(len(test_filenames))
    test_names = [test_filenames[i] for i in test_perm[:200]]
    return test_names


def analyze_segmentation():
    print("\n" + "=" * 50)
    print("开始分割错误分析 (U-Net)...")

    # 1. 加载模型权重
    model = UNet().to(DEVICE)
    weight_path = os.path.join(WEIGHT_DIR, "m2_task3_pet_unet", "best_model.pth")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"分割权重不存在: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 2. 准备测试数据
    test_names = load_pet_test_names()
    IMAGE_SIZE = 256
    image_dir = os.path.join(DATA_DIR, "oxford-iiit-pet", "images")
    trimap_dir = os.path.join(DATA_DIR, "oxford-iiit-pet", "annotations", "trimaps")

    # 如果本地没有数据集，自动下载
    if not os.path.exists(image_dir):
        print("本地未找到Oxford Pet数据集，正在下载...")
        _ = datasets.OxfordIIITPet(root=DATA_DIR, split="test", target_types="segmentation", download=True)
        image_dir = os.path.join(DATA_DIR, "oxford-iiit-pet", "images")
        trimap_dir = os.path.join(DATA_DIR, "oxford-iiit-pet", "annotations", "trimaps")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    images_list, true_masks_list, pred_masks_list, sample_ids = [], [], [], []

    with torch.no_grad():
        for name in test_names:
            # 读图
            img_path = os.path.join(image_dir, name + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, name + '.png')
            img_pil = Image.open(img_path).convert('RGB')
            # 读trimap转mask
            trimap_pil = Image.open(os.path.join(trimap_dir, name + '.png'))
            mask_array = (np.array(trimap_pil) != 2).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_array, mode='L')

            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            mask_tensor = mask_transform(mask_pil).unsqueeze(0).to(DEVICE)

            # 推理
            output = model(img_tensor)
            pred = (torch.sigmoid(output) > 0.5).float()

            # 保存用于显示 (反归一化)
            img_display = img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = img_display * std + mean
            img_display = np.clip(img_display, 0, 1)

            images_list.append(img_display)
            true_masks_list.append(mask_tensor.cpu().squeeze(0).squeeze(0).numpy())
            pred_masks_list.append(pred.cpu().squeeze(0).squeeze(0).numpy())
            sample_ids.append(name)

    # 3. 计算指标
    metrics_list = []
    for true, pred in zip(true_masks_list, pred_masks_list):
        eps = 1e-7
        pred_bin = (pred > 0.5).astype(np.uint8)
        true_bin = (true > 0.5).astype(np.uint8)
        tp = np.sum((pred_bin == 1) & (true_bin == 1))
        fp = np.sum((pred_bin == 1) & (true_bin == 0))
        fn = np.sum((pred_bin == 0) & (true_bin == 1))
        tn = np.sum((pred_bin == 0) & (true_bin == 0))
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        metrics_list.append({'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'dice': dice, 'iou': iou, 'precision': precision,
                             'recall': recall})

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.insert(0, 'sample_id', sample_ids)

    # 4. 生成四组三色误差图
    def overlay_error(image, true_mask, pred_mask):
        overlay = image.copy().astype(np.float32)
        pred_bin = pred_mask > 0.5
        true_bin = true_mask > 0.5
        tp = pred_bin & true_bin
        fp = pred_bin & (~true_bin)
        fn = (~pred_bin) & true_bin
        alpha = 0.5
        overlay[tp, 0] = overlay[tp, 0] * (1 - alpha) + 0 * alpha
        overlay[tp, 1] = overlay[tp, 1] * (1 - alpha) + 255 * alpha
        overlay[tp, 2] = overlay[tp, 2] * (1 - alpha) + 0 * alpha
        overlay[fp, 0] = overlay[fp, 0] * (1 - alpha) + 255 * alpha
        overlay[fp, 1] = overlay[fp, 1] * (1 - alpha) + 0 * alpha
        overlay[fp, 2] = overlay[fp, 2] * (1 - alpha) + 0 * alpha
        overlay[fn, 0] = overlay[fn, 0] * (1 - alpha) + 0 * alpha
        overlay[fn, 1] = overlay[fn, 1] * (1 - alpha) + 0 * alpha
        overlay[fn, 2] = overlay[fn, 2] * (1 - alpha) + 255 * alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    dice_list = metrics_df['dice'].values
    fp_list = metrics_df['fp'].values
    fn_list = metrics_df['fn'].values
    indices = list(range(len(sample_ids)))

    sorted_dice = np.argsort(dice_list)
    highest5 = sorted_dice[-5:][::-1]
    lowest5 = sorted_dice[:5]
    sorted_fp = np.argsort(fp_list)[-5:][::-1]
    sorted_fn = np.argsort(fn_list)[-5:][::-1]

    def plot_group(indices_subset, title, filename, metric_name='Dice'):
        n = len(indices_subset)
        cols = 5
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if n > 1 else [axes]
        for i, idx in enumerate(indices_subset):
            img = images_list[idx]
            true = true_masks_list[idx]
            pred = pred_masks_list[idx]
            overlay_img = overlay_error(img, true, pred)
            axes[i].imshow(overlay_img)
            if metric_name == 'Dice':
                val = dice_list[idx]
            elif metric_name == 'FP':
                val = fp_list[idx]
            elif metric_name == 'FN':
                val = fn_list[idx]
            else:
                val = 0
            axes[i].set_title(f"ID:{sample_ids[idx]}\n{metric_name}={val:.4f}", fontsize=9)
            axes[i].axis('off')
        for j in range(len(indices_subset), len(axes)):
            axes[j].axis('off')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()

    plot_group(highest5, "Seg - Top 5 Dice", "seg_top5_dice.png", 'Dice')
    plot_group(lowest5, "Seg - Bottom 5 Dice", "seg_bottom5_dice.png", 'Dice')
    plot_group(sorted_fp, "Seg - Top 5 FP", "seg_top5_fp.png", 'FP')
    plot_group(sorted_fn, "Seg - Top 5 FN", "seg_top5_fn.png", 'FN')
    print("✅ 四组分割三色误差图已保存")

    # 5. 保存分割指标CSV
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "segmentation_metrics.csv"), index=False)
    print("✅ 分割指标CSV已保存")
    return sample_ids, images_list, true_masks_list, pred_masks_list, metrics_df


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 运行分类分析
    analyze_classification()
    # 运行分割分析
    analyze_segmentation()

    print("\n" + "=" * 50)
    print(f"🎉 所有错误分析产物已保存至: {OUTPUT_DIR}")
    print("请提交以下文件：")
    print("  - classification_confusion_matrix.png")
    print("  - classification_error_cases.png")
    print("  - classification_error_cases.csv")
    print("  - seg_top5_dice.png, seg_bottom5_dice.png, seg_top5_fp.png, seg_top5_fn.png")
    print("  - segmentation_metrics.csv")
    print("  - error_analysis.py (本脚本)")
    print("  - 300-500字分析报告 (参考下方模板)")