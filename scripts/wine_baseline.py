import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from _init_ import BASE_ROOT,DATA_DIR,OUTPUT_DIR
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略收敛警告（可选）
# ==================== 0. 固定设置 ====================
RANDOM_STATE = 42
TEST_SIZE = 0.3
output_dir = OUTPUT_DIR/'wine_baseline'
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 加载数据与固定划分 ====================
data = load_wine()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

# 关键：三个模型共用这一个划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}\n")

# ==================== 2. 定义模型字典 ====================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "SVC": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

# ==================== 3. 训练与评估循环 ====================
results = []
model_infos = []  # 存储每个模型的预测结果，用于绘图
for name, clf in models.items():
    print(f"正在训练 {name} ...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 获取预测概率（用于OvR ROC-AUC）
    # 注意：SVC设置了probability=True，此处安全
    y_proba = clf.predict_proba(X_test)

    # ---- 计算指标 ----
    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    # OvR (One-vs-Rest) ROC-AUC，计算macro平均
    roc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Macro Precision": round(macro_prec, 4),
        "Macro Recall": round(macro_rec, 4),
        "Macro F1": round(macro_f1, 4),
        "OvR ROC-AUC": round(roc_ovr, 4),
    })

    # ---- 保存混淆矩阵（存入字典供后续统一绘图） ----
    cm = confusion_matrix(y_test, y_pred)
    model_infos.append({
        "name": name,
        "cm": cm,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "clf": clf,
    })

# ==================== 4. 保存 metrics.csv ====================
df_metrics = pd.DataFrame(results)
df_metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
print("\n指标结果已保存至 metrics.csv：")
print(df_metrics.to_string(index=False))

# ==================== 5. 绘制 confusion_matrix.png ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, info in zip(axes, model_infos):
    disp = ConfusionMatrixDisplay(confusion_matrix=info["cm"], display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(info["name"], fontsize=13)
    ax.grid(False)
plt.suptitle("Wine Dataset - Confusion Matrices (Test Set)", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("confusion_matrix.png 已生成")

# ==================== 6. 绘制 roc_ovr.png（宏观平均ROC曲线） ====================
plt.figure(figsize=(8, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for info, color in zip(model_infos, colors):
    y_proba = info["y_proba"]
    # 计算宏观平均ROC曲线：对每个类别计算fpr, tpr，然后插值平均
    fpr_grid = np.linspace(0.0, 1.0, 100)
    mean_tpr = 0.0
    n_classes = len(class_names)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
    mean_tpr /= n_classes
    # 让曲线从(0,0)开始，到(1,1)结束
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    plt.plot(
        fpr_grid, mean_tpr, color=color, lw=2,
        label=f"{info['name']} (OvR Macro AUC = {auc_ovr:.4f})"
    )

# 对角线
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance level (AUC=0.5)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.title("One-vs-Rest (OvR) Macro-average ROC Curves", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_ovr.png"), dpi=150)
plt.close()
print("roc_ovr.png 已生成")

print(f"\n所有任务产物已保存至: {output_dir}/")
print("请提交: wine_baseline.py, metrics.csv, confusion_matrix.png, roc_ovr.png")
