import pandas as pd
import numpy as np
from pathlib import Path



BASE_ROOT = Path.cwd().parent
OUTPUT_DIR = BASE_ROOT/'outputs'
np.random.seed(42)
epochs = 20

# 模拟训练 loss：从 2.0 逐渐下降到 0.3，加一点噪声
train_loss = 2.0 * np.exp(-0.15 * np.arange(epochs)) + 0.1 * np.random.randn(epochs)
train_loss = np.clip(train_loss, 0.1, 2.5)

# 模拟验证 loss：先降后升（过拟合信号）
val_loss = 1.8 * np.exp(-0.12 * np.arange(epochs)) + 0.05 * np.arange(epochs) * 0.08 + 0.1 * np.random.randn(epochs)
val_loss = np.clip(val_loss, 0.2, 2.5)

# 模拟训练 accuracy：从 30% 升到 95%
train_acc = 0.3 + 0.65 * (1 - np.exp(-0.18 * np.arange(epochs))) + 0.02 * np.random.randn(epochs)
train_acc = np.clip(train_acc, 0.25, 0.98)

# 模拟验证 accuracy：先升到 75% 左右，然后持平或略降
val_acc = 0.3 + 0.45 * (1 - np.exp(-0.15 * np.arange(epochs))) - 0.003 * np.arange(epochs) + 0.02 * np.random.randn(epochs)
val_acc = np.clip(val_acc, 0.25, 0.85)

# 模拟学习率：前 10 个 epoch 固定 0.001，后 10 个衰减到 0.0001
lr = np.ones(epochs) * 0.001
lr[10:] = 0.001 * np.exp(-0.2 * np.arange(10))

# 构建 DataFrame
df = pd.DataFrame({
    'epoch': np.arange(1, epochs+1),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'lr': lr
})

df.to_csv(OUTPUT_DIR/'training_log.csv', index=False)
print(f"模拟日志已生成: {OUTPUT_DIR}/training_log.csv")