import os
import random
import csv
import numpy as np
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
from _init_ import BASE_ROOT

# ==================== 配置区域（请根据实际路径修改）====================
# 假设你的数据集解压后结构如下：
# data/
#   ISIC2018_Task1_Training_Input/     # 存放所有 .jpg 原图
#   ISIC2018_Task1_Training_GroundTruth/ # 存放所有 _segmentation.png

IMAGE_DIR = BASE_ROOT / "data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
MASK_DIR = BASE_ROOT / "data/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth"
OUTPUT_DIR = BASE_ROOT / "outputs/m3_task2_isic2018_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 固定参数
TOTAL_PAIRS = 1400
TRAIN_NUM = 1000
VAL_NUM = 200
TEST_NUM = 200
RANDOM_SEED = 42
RESIZE_SIZE = 256

# 异常判定阈值（可根据实际情况调整）
FG_RATIO_EMPTY = 0.0
FG_RATIO_TOO_SMALL = 0.005  # 小于 0.5% 视为异常小
FG_RATIO_TOO_LARGE = 0.85  # 大于 85% 视为异常大


# ====================================================================


def load_file_pairs():
    """扫描文件夹，按 ID 配对图像和掩膜，按文件名排序后截取前 TOTAL_PAIRS 对"""
    # 获取所有图像文件（.jpg）
    img_paths = sorted(glob(str(IMAGE_DIR / "*.jpg")))
    pairs = []
    for img_path in img_paths:
        # 提取 ID：例如 ISIC_1234567
        case_id = Path(img_path).stem
        # 构造对应的掩膜路径
        mask_path = MASK_DIR / f"{case_id}_segmentation.png"
        if mask_path.exists():
            pairs.append((case_id, img_path, str(mask_path)))
        else:
            print(f"警告: 未找到 {case_id} 对应的掩膜文件，跳过")

    # 按 case_id 排序（已通过 img_paths 排序保证，但再显式排序一次确保安全）
    pairs.sort(key=lambda x: x[0])

    # 截取前 TOTAL_PAIRS 对
    if len(pairs) < TOTAL_PAIRS:
        print(f"警告: 总配对数量 ({len(pairs)}) 少于要求的 {TOTAL_PAIRS}，将使用全部 {len(pairs)} 对")
        return pairs
    return pairs[:TOTAL_PAIRS]


def split_dataset(pairs):
    """随机种子划分数据集"""
    random.seed(RANDOM_SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    train_set = shuffled[:TRAIN_NUM]
    val_set = shuffled[TRAIN_NUM:TRAIN_NUM + VAL_NUM]
    test_set = shuffled[TRAIN_NUM + VAL_NUM:TRAIN_NUM + VAL_NUM + TEST_NUM]
    return train_set, val_set, test_set


def resize_image(image, mask, target_size=RESIZE_SIZE):
    """统一缩放到 target_size x target_size"""
    h, w = image.shape[:2]
    # 图像使用双三次插值，掩膜使用最近邻插值保持二值性
    resized_img = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized_img, resized_mask


def analyze_sample(case_id, img_path, mask_path, split):
    """单样本分析，返回指标字典"""
    # 读取图像（保持原始 BGR 通道，OpenCV 默认）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 读取掩膜（灰度图）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜: {mask_path}")

    # 原始尺寸信息
    orig_h, orig_w = img.shape[:2]
    orig_channels = img.shape[2] if len(img.shape) == 3 else 1

    # 统一 Resize
    img_resized, mask_resized = resize_image(img, mask)

    # ---- 提取指标 ----
    # 1. 强度范围（Resize 后）
    intensity_min = float(np.min(img_resized))
    intensity_max = float(np.max(img_resized))

    # 2. Mask 唯一值（注意：Resize 后因插值可能出现 0.5 等中间值，但因使用最近邻，理论上仍为 0 和 1）
    unique_vals = np.unique(mask_resized)
    # 如果掩膜值为 0/255，统一归一化为 0/1 进行比例计算
    if np.all(np.isin(unique_vals, [0, 255])):
        mask_binary = (mask_resized > 127).astype(np.uint8)
    else:
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

    fg_pixels = np.sum(mask_binary)
    total_pixels = RESIZE_SIZE * RESIZE_SIZE
    fg_ratio = fg_pixels / total_pixels

    # 3. 异常标记
    is_empty = (fg_ratio == FG_RATIO_EMPTY)
    is_too_small = (0 < fg_ratio < FG_RATIO_TOO_SMALL)
    is_too_large = (fg_ratio > FG_RATIO_TOO_LARGE)
    is_abnormal = is_empty or is_too_small or is_too_large

    return {
        "case_id": case_id,
        "split": split,
        "image_path": img_path,
        "mask_path": mask_path,
        "height": RESIZE_SIZE,  # 统一尺寸
        "width": RESIZE_SIZE,
        "orig_height": orig_h,
        "orig_width": orig_w,
        "orig_channels": orig_channels,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "mask_unique_values": str(unique_vals.tolist()),
        "fg_ratio": round(fg_ratio, 6),
        "is_empty": is_empty,
        "is_too_small": is_too_small,
        "is_too_large": is_too_large,
        "is_abnormal": is_abnormal,
        # 保留掩膜用于 overlay（后续使用）
        "mask_binary": mask_binary,
        "img_resized": img_resized,
    }


def generate_overlay(img, mask_binary, save_path):
    """生成半透明红色叠加图"""
    # 创建 RGB 叠加层
    overlay = img.copy()
    # 将掩膜区域涂成红色（BGR: (0, 0, 255)）
    overlay[mask_binary == 1] = [0, 0, 255]
    # 混合：原图 0.6 + 红色 0.4
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(save_path, blended)


def main():
    print("=" * 60)
    print("ISIC 2018 图像-标签配对质量检查流程启动")
    print(f"图像目录: {IMAGE_DIR}")
    print(f"掩膜目录: {MASK_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 1. 加载并筛选配对
    print("\n[1] 正在扫描并配对文件...")
    pairs = load_file_pairs()
    print(f"成功配对 {len(pairs)} 组 (目标 {TOTAL_PAIRS})")

    # 2. 划分数据集
    print("\n[2] 按随机种子 42 划分数据集...")
    train_set, val_set, test_set = split_dataset(pairs)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # 3. 逐一分析
    print("\n[3] 开始逐样本分析 (Resize to 256x256)...")
    all_records = []
    abnormal_records = []
    overlay_samples = []

    # 决定哪些样本用于生成 overlay（均匀抽取，总数 20）
    # 按比例分配: Train 14, Val 3, Test 3
    overlay_indices = {
        "train": list(range(0, len(train_set), max(1, len(train_set) // 14)))[:14],
        "val": list(range(0, len(val_set), max(1, len(val_set) // 3)))[:3],
        "test": list(range(0, len(test_set), max(1, len(test_set) // 3)))[:3],
    }

    # 循环处理 Train
    for idx, (case_id, img_p, mask_p) in enumerate(tqdm(train_set, desc="Train")):
        info = analyze_sample(case_id, img_p, mask_p, "train")
        all_records.append(info)
        if info["is_abnormal"]:
            abnormal_records.append(info)
        if idx in overlay_indices["train"]:
            overlay_samples.append(info)

    for idx, (case_id, img_p, mask_p) in enumerate(tqdm(val_set, desc="Val")):
        info = analyze_sample(case_id, img_p, mask_p, "val")
        all_records.append(info)
        if info["is_abnormal"]:
            abnormal_records.append(info)
        if idx in overlay_indices["val"]:
            overlay_samples.append(info)

    for idx, (case_id, img_p, mask_p) in enumerate(tqdm(test_set, desc="Test")):
        info = analyze_sample(case_id, img_p, mask_p, "test")
        all_records.append(info)
        if info["is_abnormal"]:
            abnormal_records.append(info)
        if idx in overlay_indices["test"]:
            overlay_samples.append(info)

    print(f"\n总分析样本数: {len(all_records)}")
    print(f"异常样本数: {len(abnormal_records)}")

    # 4. 保存 CSV: isic_inventory.csv
    print("\n[4] 保存 CSV 文件...")
    inventory_path = OUTPUT_DIR / "isic_inventory.csv"
    # 定义 CSV 列（严格按照要求）
    fieldnames = [
        "case_id", "split", "image_path", "mask_path",
        "height", "width", "fg_ratio",
        # 附加字段（便于调阅，但不影响验收）
        "orig_height", "orig_width", "orig_channels",
        "intensity_min", "intensity_max", "mask_unique_values",
        "is_empty", "is_too_small", "is_too_large", "is_abnormal"
    ]
    with open(inventory_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            # 只保留字段名中的键
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"  -> 清单已保存: {inventory_path}")

    # 5. 保存异常样本表
    if abnormal_records:
        abnormal_path = OUTPUT_DIR / "abnormal_samples.csv"
        with open(abnormal_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in abnormal_records:
                row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print(f"  -> 异常样本表已保存: {abnormal_path}")
    else:
        print("  -> 未发现异常样本，跳过异常表生成")

    # 6. 生成 20 张 Overlay 图
    print("\n[5] 生成 20 张 Overlay 叠加图...")
    overlay_dir = OUTPUT_DIR / "overlay_plots"
    os.makedirs(overlay_dir, exist_ok=True)

    # 如果 overlay_samples 不足 20，补充一些正常样本
    if len(overlay_samples) < 20:
        # 从所有记录中按顺序补充
        for r in all_records:
            if len(overlay_samples) >= 20:
                break
            if r not in overlay_samples:
                overlay_samples.append(r)

    for i, info in enumerate(overlay_samples[:20]):
        save_name = f"overlay_{i + 1:02d}_{info['case_id']}.png"
        save_path = overlay_dir / save_name
        generate_overlay(info["img_resized"], info["mask_binary"], str(save_path))
        print(f"  -> {save_name} 已生成")

    print("\n" + "=" * 60)
    print("✅ 所有检查任务完成！")
    print(f"请提交以下文件至验收目录：")
    print(f"  1. {__file__} (本脚本)")
    print(f"  2. {inventory_path}")
    print(f"  3. {abnormal_path if abnormal_records else '(无异常样本, 无需提交)'}")
    print(f"  4. {overlay_dir} 文件夹中的所有 20 张 PNG 图片")
    print("=" * 60)


if __name__ == "__main__":
    main()
