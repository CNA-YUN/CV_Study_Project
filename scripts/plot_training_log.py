import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts._init_ import BASE_ROOT,OUTPUT_DIR,DATA_DIR




# 读取日志
df = pd.read_csv(OUTPUT_DIR/'training_log.csv')
epochs = df['epoch'].values

# ---------- 1. Loss 曲线 ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, df['train_loss'], 'b-', label='Train Loss', linewidth=2)
ax.plot(epochs, df['val_loss'], 'r-', label='Val Loss', linewidth=2)

# 最佳验证 loss（最小值）
best_idx = df['val_loss'].idxmin()
best_epoch = df.loc[best_idx, 'epoch']
best_val_loss = df.loc[best_idx, 'val_loss']
ax.plot(best_epoch, best_val_loss, 'ro', markersize=10, label=f'Best Val Loss (Epoch {best_epoch})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training and Validation Loss', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'loss_curve.png', dpi=300)
plt.close()

# ---------- 2. Accuracy 曲线 ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, df['train_acc'] * 100, 'b-', label='Train Acc (%)', linewidth=2)
ax.plot(epochs, df['val_acc'] * 100, 'r-', label='Val Acc (%)', linewidth=2)

# 最佳验证 accuracy（最大值）
best_idx = df['val_acc'].idxmax()
best_epoch = df.loc[best_idx, 'epoch']
best_val_acc = df.loc[best_idx, 'val_acc'] * 100
ax.plot(best_epoch, best_val_acc, 'ro', markersize=10, label=f'Best Val Acc (Epoch {best_epoch})')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Training and Validation Accuracy', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'accuracy_curve.png', dpi=300)
plt.close()

# ---------- 3. Learning Rate 曲线 ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, df['lr'], 'g-', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Learning Rate Schedule', fontsize=14)
ax.grid(True, alpha=0.3)
# 学习率曲线不标注红点，如果需要也可以标出变化点，但题目要求“最佳 epoch 用红点标注”，通常指验证集最佳，学习率一般不标。
# 但为了统一，我们可标出学习率变化的位置，这里略过。
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'lr_curve.png', dpi=300)
plt.close()

print("三张曲线图已生成: loss_curve.png, accuracy_curve.png, lr_curve.png")
