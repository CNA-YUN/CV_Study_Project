import os
import random
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import yaml
import warnings

warnings.filterwarnings("ignore")

# ==================== OpenI 环境初始化 ====================
from c2net.context import prepare, upload_output

c2net_context = prepare()

# 复用 CIFAR-10 数据集路径（与你的 Notebook 保持一致）
DATA_DIR = c2net_context.dataset_path + '/CIFAR-10'
BASE_OUTPUT = c2net_context.output_path + '/m2_task4_transfer'
os.makedirs(BASE_OUTPUT, exist_ok=True)

# ==================== 固定设置 ====================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
INPUT_SIZE = 224
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Data path: {DATA_DIR}")
print(f"Output path: {BASE_OUTPUT}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==================== 1. 数据划分（生成 cifar10_split.json） ====================
def create_cifar10_split():
    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    split = {"train": [], "val": [], "test": []}

    rng_train = np.random.default_rng(42)
    for class_id in range(10):
        idx = np.where(np.array(full_train.targets) == class_id)[0]
        perm = rng_train.permutation(idx)
        split["train"].extend(perm[:500].tolist())
        split["val"].extend(perm[500:600].tolist())

    rng_test = np.random.default_rng(42)
    for class_id in range(10):
        idx = np.where(np.array(full_test.targets) == class_id)[0]
        perm = rng_test.permutation(idx)
        split["test"].extend(perm[:100].tolist())

    split_path = os.path.join(BASE_OUTPUT, "cifar10_split.json")
    with open(split_path, "w") as f:
        json.dump(split, f)
    return split


split = create_cifar10_split()


# ==================== 2. 数据加载 ====================
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, transform=train_transform)
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, transform=val_test_transform)

    train_subset = Subset(full_train, split["train"])
    val_subset = Subset(full_train, split["val"])
    test_subset = Subset(full_test, split["test"])

    return (DataLoader(train_subset, BATCH_SIZE, shuffle=True, num_workers=4),
            DataLoader(val_subset, BATCH_SIZE, shuffle=False, num_workers=4),
            DataLoader(test_subset, BATCH_SIZE, shuffle=False, num_workers=4))


# ==================== 3. 模型创建 ====================
def create_model(strategy):
    if strategy == "scratch":
        model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, NUM_CLASSES)

    if strategy == "linear_probe":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 4. 训练单一策略 ====================
def train_strategy(strategy):
    print(f"\n{'=' * 50}\nTraining Strategy: {strategy.upper()}\n{'=' * 50}")
    output_dir = os.path.join(BASE_OUTPUT, strategy)
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = create_model(strategy).to(device)
    trainable_params = count_trainable_params(model)
    print(f"Trainable params: {trainable_params:,}")

    if strategy == "full_finetune":
        optimizer = optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc, best_val_f1, best_epoch = 0.0, 0.0, 0
    log_data = []
    start_time = time.time()
    peak_memory = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        current_lr = optimizer.param_groups[0]['lr']
        log_data.append([epoch, train_loss, val_loss, train_acc, val_acc, val_f1, current_lr])

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
            best_val_acc, best_val_f1, best_epoch = val_acc, val_f1, epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        scheduler.step()
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1024 ** 2)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    total_time = time.time() - start_time
    df = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "lr"])
    df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

    # 绘制曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch");
    axes[0].set_ylabel("Loss");
    axes[0].set_title(f"{strategy} - Loss");
    axes[0].legend();
    axes[0].grid(alpha=0.3)
    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Acc", marker='s')
    axes[1].set_xlabel("Epoch");
    axes[1].set_ylabel("Accuracy");
    axes[1].set_title(f"{strategy} - Accuracy");
    axes[1].legend();
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curves.png"), dpi=150)
    plt.close()

    # 测试评估
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(10)]).plot(cmap='Blues')
    plt.title(f"{strategy} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    config = {
        "strategy": strategy, "seed": SEED, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "trainable_params": trainable_params, "best_epoch": best_epoch,
        "test_accuracy": test_acc, "test_macro_precision": test_prec,
        "test_macro_recall": test_rec, "test_macro_f1": test_f1,
        "total_train_time_s": total_time, "peak_gpu_memory_mb": peak_memory,
    }
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    return config


# ==================== 5. 运行三种策略 ====================
strategies = ["scratch", "linear_probe", "full_finetune"]
all_results = [train_strategy(s) for s in strategies]

# ==================== 6. 生成对比表 ====================
comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df[[
    "strategy", "trainable_params", "best_epoch", "test_accuracy",
    "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "total_train_time_s", "peak_gpu_memory_mb"
]]
comparison_df.to_csv(os.path.join(BASE_OUTPUT, "transfer_comparison.csv"), index=False)
print("\n" + "=" * 60)
print("TRANSFER LEARNING COMPARISON RESULTS")
print("=" * 60)
print(comparison_df.to_string(index=False))
print(f"\n所有产物已保存至: {BASE_OUTPUT}")
upload_output()