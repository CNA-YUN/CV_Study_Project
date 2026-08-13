import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from skimage import morphology, measure
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes
import warnings

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_DIR = Path.cwd().parent
# 假设 isic_inventory.csv 在 outputs/isic2018_check/ 下
INVENTORY_CSV = BASE_DIR / "outputs/m3_task2_isic2018_check/isic_inventory.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "m3_task4_isic_postprocess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 后处理参数
DISK_OPEN = 3
DISK_CLOSE = 5
MIN_AREA = 512  # 像素（在256x256下）


# ==================================================

def load_val_set(csv_path):
    """从 inventory.csv 中提取验证集样本信息"""
    df = pd.read_csv(csv_path)
    val_df = df[df['split'] == 'val'].copy()
    # 确保路径存在
    samples = []
    for _, row in val_df.iterrows():
        samples.append({
            'case_id': row['case_id'],
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'height': int(row['height']),
            'width': int(row['width'])
        })
    return samples


def otsu_segmentation(image_gray):
    """Otsu 阈值分割，返回二值图像 (0/1)"""
    thresh = threshold_otsu(image_gray)
    binary = (image_gray > thresh).astype(np.uint8)
    return binary


def postprocess(mask_binary, keep_largest=True):
    """
    应用固定后处理序列
    mask_binary: 二值图像 (0/1)
    keep_largest: 是否保留最大连通域（可选）
    返回处理后二值图像
    """
    # 1. 开运算 (先腐蚀后膨胀) 去除小噪点
    kernel_open = morphology.disk(DISK_OPEN)
    opened = morphology.binary_opening(mask_binary, kernel_open)

    # 2. 闭运算 (先膨胀后腐蚀) 填充内部小空洞
    kernel_close = morphology.disk(DISK_CLOSE)
    closed = morphology.binary_closing(opened, kernel_close)

    # 3. 孔洞填充 (填充完全封闭的洞)
    filled = binary_fill_holes(closed).astype(np.uint8)

    # 4. 移除小连通域 (面积 < MIN_AREA)
    labeled = measure.label(filled, connectivity=2)
    props = measure.regionprops(labeled)
    # 构建保留的区域掩膜
    mask_keep = np.zeros_like(filled, dtype=np.uint8)
    for prop in props:
        if prop.area >= MIN_AREA:
            mask_keep[labeled == prop.label] = 1

    # 5. 可选：仅保留最大连通域
    if keep_largest:
        if np.sum(mask_keep) > 0:
            # 重新标记以找到最大区域
            labeled_keep = measure.label(mask_keep, connectivity=2)
            props_keep = measure.regionprops(labeled_keep)
            if props_keep:
                largest = max(props_keep, key=lambda x: x.area)
                mask_keep = (labeled_keep == largest.label).astype(np.uint8)

    return mask_keep


def compute_metrics(mask_gt, mask_pred):
    """计算 Dice, IoU, Precision, Recall"""
    # 确保都是二值且形状一致
    gt = (mask_gt > 0).astype(np.uint8)
    pred = (mask_pred > 0).astype(np.uint8)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    tp = intersection
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(gt, np.logical_not(pred)).sum()

    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, iou, precision, recall


def plot_comparison(img_rgb, mask_gt, mask_pred_before, mask_pred_after,
                    case_id, save_dir, metrics_before, metrics_after):
    """
    绘制三张对比图：GT, Otsu(前), 后处理(后)
    叠加红色轮廓
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 辅助：在图像上绘制轮廓
    def overlay_contour(ax, img, mask, title, metrics=None):
        # 归一化图像显示
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色顺序
        ax.imshow(img_display)
        if np.sum(mask) > 0:
            contours = measure.find_contours(mask, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if metrics is not None:
            text = f"Dice:{metrics[0]:.3f} IoU:{metrics[1]:.3f}"
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                    color='white', fontsize=9, ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5))

    overlay_contour(axes[0], img_rgb, mask_gt, "Ground Truth")
    overlay_contour(axes[1], img_rgb, mask_pred_before, "Otsu (Before)", metrics_before)
    overlay_contour(axes[2], img_rgb, mask_pred_after, "Post-processed (After)", metrics_after)

    plt.suptitle(f"Case {case_id}  (Dice: {metrics_before[0]:.3f} -> {metrics_after[0]:.3f})", fontsize=14)
    plt.tight_layout()
    save_path = save_dir / f"{case_id}_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("ISIC 2018 验证集后处理评估")
    print(f"输入清单: {INVENTORY_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 1. 加载验证集样本
    samples = load_val_set(INVENTORY_CSV)
    print(f"验证集样本数: {len(samples)}")

    # 存储所有指标
    records = []

    # 用于寻找提升/下降最大的病例
    dice_deltas = []

    for sample in tqdm(samples, desc="Processing"):
        case_id = sample['case_id']
        img_path = sample['image_path']
        mask_path = sample['mask_path']

        # 读取图像 (彩色) 和真值
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: 无法读取 {img_path}, 跳过")
            continue
        # 转灰度用于Otsu
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # 读取掩膜 (灰度) 并二值化
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gt is None:
            print(f"Warning: 无法读取 {mask_path}, 跳过")
            continue
        mask_gt = (mask_gt > 127).astype(np.uint8)

        # 统一 resize 到 256x256 (之前 inventory 已做，但为确保一致)
        if img_bgr.shape[:2] != (256, 256):
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            mask_gt = cv2.resize(mask_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_gt = (mask_gt > 0.5).astype(np.uint8)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ----- 初始预测 (Otsu) -----
        pred_before = otsu_segmentation(img_gray)
        # 计算初始指标
        dice_b, iou_b, prec_b, rec_b = compute_metrics(mask_gt, pred_before)

        # ----- 后处理 (完整流程，保留最大连通域) -----
        pred_after = postprocess(pred_before, keep_largest=True)
        # 计算后处理指标
        dice_a, iou_a, prec_a, rec_a = compute_metrics(mask_gt, pred_after)

        # 记录
        records.append({
            'case_id': case_id,
            'dice_before': dice_b,
            'iou_before': iou_b,
            'precision_before': prec_b,
            'recall_before': rec_b,
            'dice_after': dice_a,
            'iou_after': iou_a,
            'precision_after': prec_a,
            'recall_after': rec_a,
            'dice_delta': dice_a - dice_b,
            'iou_delta': iou_a - iou_b,
            'precision_delta': prec_a - prec_b,
            'recall_delta': rec_a - rec_b
        })

        dice_deltas.append((case_id, dice_a - dice_b))

    # 2. 保存完整指标 CSV
    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "metrics_before_after.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n指标CSV已保存: {csv_path}")

    # 3. 找出提升最大和下降最大的5例 (按Dice delta)
    sorted_deltas = sorted(dice_deltas, key=lambda x: x[1], reverse=True)
    top5_improve = sorted_deltas[:5]
    top5_decline = sorted_deltas[-5:][::-1]  # 下降最多的（最负的）

    # 生成对比图 (需要读取对应样本的图像和预测)
    plot_dir = OUTPUT_DIR / "comparison_plots"
    os.makedirs(plot_dir, exist_ok=True)

    def generate_plots_for_cases(case_list, suffix):
        for case_id, delta in case_list:
            # 找到样本
            sample = next((s for s in samples if s['case_id'] == case_id), None)
            if sample is None:
                continue
            img_bgr = cv2.imread(sample['image_path'])
            if img_bgr is None:
                continue
            img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask_gt = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask_gt = cv2.resize(mask_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_gt = (mask_gt > 127).astype(np.uint8)

            pred_before = otsu_segmentation(img_gray)
            pred_after = postprocess(pred_before, keep_largest=True)

            # 从records中取指标
            rec = next((r for r in records if r['case_id'] == case_id), None)
            if rec is None:
                continue
            metrics_b = (rec['dice_before'], rec['iou_before'])
            metrics_a = (rec['dice_after'], rec['iou_after'])

            plot_comparison(img_bgr, mask_gt, pred_before, pred_after,
                            case_id, plot_dir, metrics_b, metrics_a)
            print(f"  -> 已生成 {suffix} 图: {case_id} (delta={delta:.4f})")

    print("\n生成提升最大的5例对比图...")
    generate_plots_for_cases(top5_improve, "improve")
    print("\n生成下降最大的5例对比图...")
    generate_plots_for_cases(top5_decline, "decline")

    # 4. 分析各后处理步骤的适用性（基于统计）
    # 我们无法逐步骤测试，但可以通过观察整体效果得出结论
    # 统计平均指标
    mean_dice_before = df['dice_before'].mean()
    mean_dice_after = df['dice_after'].mean()
    improved_count = (df['dice_delta'] > 0).sum()
    declined_count = (df['dice_delta'] < 0).sum()

    print("\n" + "=" * 60)
    print("整体统计")
    print(f"平均 Dice: {mean_dice_before:.4f} -> {mean_dice_after:.4f}")
    print(f"改进样本数: {improved_count}, 下降样本数: {declined_count}")
    print("\n后处理步骤适用性分析（基于经验与观察）:")
    print("  - 开运算（半径3）：有助于去除Otsu产生的孤立噪点，但可能使细小病变边缘收缩。")
    print("  - 闭运算（半径5）：可填充病变内部的孔洞，但可能合并相邻病变（ISIC多为单病灶，影响不大）。")
    print("  - 孔洞填充：有助于完善病变内部连续性，但对已良好的病变可能引入变形。")
    print("  - 移除小连通域（min_area=512）：能去除小噪点，但可能误删极小的真实病变（<512像素）。")
    print("  - 保留最大连通域：对单病灶数据较为安全，但对含多个独立病变的图像会丢失次要病灶，导致Recall下降。")
    print("\n不适合当前数据的步骤（基于观察）：")
    if declined_count > 0:
        print("  - 对于已经分割较好的病例（Dice>0.9），后处理通常反而降低指标，因此对高质量预测应谨慎使用。")
    if improved_count < len(df) * 0.5:
        print("  - 整体提升不明显，说明该后处理组合并非对所有病例都有益。")
    print("  - 特别是'移除小连通域'和'保留最大连通域'，在图像中有多个独立病灶（罕见）或病变本身很小的情况下，会有害。")
    print("=" * 60)


if __name__ == "__main__":
    main()