#!/usr/bin/env python3
"""
CIFAR-10 子集预处理脚本
从 CSV 中读取子集，按种子抽取 N 张，resize 并保存 RGB/灰度/均衡化版本。
"""

import os
import argparse
import logging
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torchvision.datasets as datasets
from tqdm import tqdm


# -------------------- 配置日志 --------------------
def setup_logging(save_dir):
    log_file = Path(save_dir) / 'preprocess.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('preprocess')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件 handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# -------------------- 图像处理 --------------------
def process_image(img_pil, size):
    """返回 RGB, 灰度(RGB格式), 均衡化(RGB格式)"""
    img_rgb = img_pil.resize((size, size), Image.BILINEAR)
    img_gray = img_rgb.convert('L')
    img_eq = ImageOps.equalize(img_gray)
    # 转回 RGB 便于保存为彩色图像（视觉上仍为灰度）
    img_gray_rgb = img_gray.convert('RGB')
    img_eq_rgb = img_eq.convert('RGB')
    return img_rgb, img_gray_rgb, img_eq_rgb


# -------------------- 生成对比拼图 --------------------
def create_comparison_grid(images, labels, class_names, size, save_path, n_samples=5):
    """选取前 n_samples 张（或随机），生成 4×5 对比拼图"""
    # 如果总样本少于 n_samples，则取全部
    n = min(n_samples, len(images))
    selected_indices = random.sample(range(len(images)), n)  # 随机选 n 张

    # 每个子图尺寸
    thumb_size = (size // 2, size // 2)  # 缩略图缩小一半以节省空间

    # 准备每行的四个图：rgb, gray, eq, label
    rows = []
    for idx in selected_indices:
        img_rgb, img_gray, img_eq = process_image(images[idx], size)
        # 缩略图
        rgb_thumb = img_rgb.resize(thumb_size, Image.BILINEAR)
        gray_thumb = img_gray.resize(thumb_size, Image.BILINEAR)
        eq_thumb = img_eq.resize(thumb_size, Image.BILINEAR)

        # 创建标签图片（纯白背景 + 文字）
        label_img = Image.new('RGB', thumb_size, color=(255, 255, 255))
        # 用 PIL 绘制文字（需要 ImageDraw）
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(label_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        cls_name = class_names[labels[idx]] if labels is not None else ''
        text = f"Class: {cls_name}"
        draw.text((5, 5), text, fill=(0, 0, 0), font=font)
        draw.text((5, 30), "RGB", fill=(0, 0, 0), font=font)
        draw.text((5, 55), "Gray", fill=(0, 0, 0), font=font)
        draw.text((5, 80), "Equalized", fill=(0, 0, 0), font=font)

        # 水平拼接四个图（RGB, Gray, Eq, Label）
        row = Image.new('RGB', (thumb_size[0] * 4, thumb_size[1]))
        row.paste(rgb_thumb, (0, 0))
        row.paste(gray_thumb, (thumb_size[0], 0))
        row.paste(eq_thumb, (thumb_size[0] * 2, 0))
        row.paste(label_img, (thumb_size[0] * 3, 0))
        rows.append(row)

    # 垂直拼接所有行
    total_height = sum(r.height for r in rows)
    grid = Image.new('RGB', (rows[0].width, total_height))
    y_offset = 0
    for r in rows:
        grid.paste(r, (0, y_offset))
        y_offset += r.height

    grid.save(save_path)


# -------------------- 主函数 --------------------
def main():
    BASE_DIR = Path.cwd().parent
    DATA_DIR = BASE_DIR / 'data'
    parser = argparse.ArgumentParser(description='CIFAR-10 预处理')
    parser.add_argument('--input_csv', type=str, default=BASE_DIR / 'outputs/cifar10_inventory.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--save_dir', type=str, default=BASE_DIR / 'outputs/m1_task3_preprocess',
                        help='输出根目录')
    parser.add_argument('--image_size', type=int, default=224,
                        help='resize 后的图像尺寸')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # 创建保存目录
    save_root = Path(args.save_dir)
    rgb_dir = save_root / 'rgb'
    gray_dir = save_root / 'gray'
    eq_dir = save_root / 'equalized'
    for d in [rgb_dir, gray_dir, eq_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 日志
    logger = setup_logging(save_root)
    logger.info("开始预处理任务")
    logger.info(f"参数: input_csv={args.input_csv}, save_dir={args.save_dir}, "
                f"image_size={args.image_size}, seed={args.seed}")

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. 读取 CSV
    logger.info(f"读取 CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df = df.sort_values('index').reset_index(drop=True)
    logger.info(f"CSV 包含 {len(df)} 条记录")

    # 2. 加载原始 CIFAR-10 训练集（假设 data 目录存在）
    logger.info("加载 CIFAR-10 训练集图像...")
    trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
    all_images = [trainset[i][0] for i in range(len(trainset))]
    logger.info(f"成功加载 {len(all_images)} 张图像")

    # 3. 从子集中抽取 200 张
    subset_indices = df['index'].tolist()
    selected = random.sample(subset_indices, 200)
    logger.info(f"抽取 {len(selected)} 张图像 (种子 {args.seed})")

    # 获取图像和标签
    selected_images = [all_images[idx] for idx in selected]
    selected_labels = [trainset[idx][1] for idx in selected]
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 4. 处理并保存
    logger.info("开始处理图像...")
    start_time = time.time()
    for i, (img, label) in enumerate(tqdm(zip(selected_images, selected_labels),
                                          total=len(selected_images), desc="Processing")):
        # 处理
        rgb, gray, eq = process_image(img, args.image_size)
        # 文件名
        fname = f"{class_names[label]}_{i:04d}.png"
        rgb.save(rgb_dir / fname)
        gray.save(gray_dir / fname)
        eq.save(eq_dir / fname)

        # 每 50 张记录一次日志
        if (i + 1) % 50 == 0:
            logger.info(f"已处理 {i + 1} 张")

    elapsed = time.time() - start_time
    logger.info(f"处理完成，耗时 {elapsed:.2f} 秒")
    logger.info(f"RGB 图像保存至 {rgb_dir}")
    logger.info(f"灰度图像保存至 {gray_dir}")
    logger.info(f"均衡化图像保存至 {eq_dir}")

    # 5. 生成对比拼图
    logger.info("生成对比拼图...")
    grid_path = save_root / 'comparison_grid.png'
    create_comparison_grid(selected_images, selected_labels, class_names,
                           args.image_size, grid_path, n_samples=5)
    logger.info(f"对比拼图已保存至 {grid_path}")

    logger.info("所有任务完成！")


if __name__ == '__main__':
    main()
