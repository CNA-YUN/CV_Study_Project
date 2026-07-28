import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score
from _init_ import BASE_ROOT, DATA_DIR
import warnings

warnings.filterwarnings("ignore")

# ==================== 0. 固定设置 ====================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 32
NUM_CLASSES = 10
OUTPUT_DIR = BASE_ROOT / "outputs/cifar10_task2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 固定随机种子
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

# ==================== 1. 数据加载与固定划分 ====================
# CIFAR-10 标准归一化参数
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# 训练增强：仅随机水平翻转
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# 下载原始数据集
full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

# 按类别划分训练/验证集（来自官方训练集）
train_indices, val_indices = [], []
for class_id in range(NUM_CLASSES):
    idx = np.where(np.array(full_train.targets) == class_id)[0]
    np.random.seed(SEED)  # 确保每个类别 shuffle 一致
    np.random.shuffle(idx)
    train_indices.extend(idx[:500])  # 每类 500
    val_indices.extend(idx[500:600])  # 每类 100

# 按类别划分测试集（来自官方测试集）
test_indices = []
for class_id in range(NUM_CLASSES):
    idx = np.where(np.array(full_test.targets) == class_id)[0]
    np.random.seed(SEED)
    np.random.shuffle(idx)
    test_indices.extend(idx[:100])  # 每类 100

train_subset = Subset(full_train, train_indices)
val_subset = Subset(full_train, val_indices)
test_subset = Subset(full_test, test_indices)

# 为 Subset 单独指定 transform（需要重写 dataset 的 transform）
train_subset.dataset.transform = train_transform
val_subset.dataset.transform = val_test_transform
test_subset.dataset.transform = val_test_transform

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"训练集: {len(train_subset)}, 验证集: {len(val_subset)}, 测试集: {len(test_subset)}")


# ==================== 2. 模型定义 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_resnet18():
    model = models.resnet18(weights=None)
    # 适配 32x32 输入：修改 stem
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 移除最大池化，保持尺寸
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


# ==================== 3. 训练与验证函数 ====================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


# ==================== 4. 训练循环（封装为函数） ====================
def train_model(model, model_name):
    print(f"\n========== 开始训练 {model_name} ==========")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log_data = []
    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        log_data.append([epoch, train_loss, val_loss, train_acc, val_acc, current_lr])
        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")

    # 保存日志
    df = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"])
    csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_training_log.csv")
    df.to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion)

    # 计算测试集 Macro F1
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\n>>> {model_name} Test Results: Accuracy = {test_acc:.4f}, Macro F1 = {macro_f1:.4f}")

    return model, df, all_preds, all_labels, test_acc, macro_f1


# ==================== 5. 运行两个模型 ====================
models_dict = {
    "SimpleCNN": SimpleCNN(),
    "ResNet18": create_resnet18(),
}

test_results = {}
for name, model in models_dict.items():
    model, df, preds, labels, acc, f1 = train_model(model, name)
    test_results[name] = {"acc": acc, "f1": f1, "preds": preds, "labels": labels}


# ==================== 6. 绘制训练曲线 ====================
def plot_curves(model_name):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f"{model_name}_training_log.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Acc", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_curves.png"), dpi=150)
    plt.close()


for name in models_dict.keys():
    plot_curves(name)


# ==================== 7. 绘制错误分类样本图（取前20个） ====================
def plot_error_cases(model_name, preds, labels, num_samples=20):
    # 获取错误样本索引
    errors = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    if len(errors) < num_samples:
        print(f"Warning: Only {len(errors)} errors found for {model_name}, using all.")
        selected = errors
    else:
        selected = errors[:num_samples]

    # 获取对应的图像（需反归一化）
    test_dataset = test_subset.dataset
    test_dataset.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])  # 保持原样获取tensor
    # 但我们需要原始图，为了反归一化，我们重新加载不带归一化的transform
    raw_test = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
    raw_test_subset = Subset(raw_test, test_indices)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i, idx in enumerate(selected):
        img_tensor, _ = raw_test_subset[idx]  # 原始 [0,1] 范围
        # 反归一化（仅用于显示）
        img = img_tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color='red')
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(len(selected), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{model_name} - Error Cases (Top {len(selected)})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_error_cases.png"), dpi=150)
    plt.close()


for name in models_dict.keys():
    plot_error_cases(name, test_results[name]["preds"], test_results[name]["labels"])

# ==================== 8. 终端打印最终结果 ====================
print("\n" + "=" * 50)
print("FINAL TEST RESULTS (Best Checkpoint)")
print("=" * 50)
for name, res in test_results.items():
    print(f"{name:12s} | Test Acc: {res['acc']:.4f} | Macro F1: {res['f1']:.4f}")
print(f"\n所有产物已保存至: {OUTPUT_DIR}/")
print("请提交: 训练代码, training_log.csv (2个), best.pth (2个), curves.png (2张), error_cases.png (2张)")
