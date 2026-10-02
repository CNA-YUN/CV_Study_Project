# -*- coding: utf-8 -*-
"""
任务 4.3.3 —— MedSAM 点/框提示医学分割比较（本地版）

数据集：ISIC 2018，复用 outputs/m3_task2_isic2018_check/isic_inventory.csv 中
        seed=42 划分出的 split="test" 共 200 例。
固定模型：MedSAM（medsam_vit_b.pth，SAM ViT-B 结构）。
固定提示（均在【原图坐标】生成）：
        - GT box：真值 mask 外接框向四周各扩展 10 像素，并裁剪到图像边界。
        - 前景中心点：距离变换最深点（cv2.distanceTransform 取 argmax，保证落在前景内部）。
        - 负点 ×4：背景距离图上贪心取"距前景最远"的点，每取一个用半径 r = 0.5×最远背景距离
                   的圆形邻域抑制，保证 4 点围绕病灶分散（通常落在图像四角）。
输入尺寸：长边 resize 到 1024（保持长宽比）→ 右下 padding 到 1024×1024 → 逐图 min-max 归一化。
          注意：这里用 ResizeLongestSide（SAM 官方变换），不是官方脚本里的直接 resize 到方形。
推理：每张图只做 1 次 image_encoder，point / box / point+box 三种提示复用同一 embedding；
      multimask_output=False（MedSAM 训练口径）。
评估：mask 还原到原图尺寸后与原尺寸 GT 计算 Dice / IoU。

【重要声明 —— 交互式上界设置】
本实验的 GT box 由真值 mask 外接框生成，前景中心点与负点同样由真值推导，
因此三种设置均属于 Oracle Prompt / 交互式上界协议：它衡量的是"提示质量最优时
MedSAM 的能力上限"，并不是完全自动分割的性能。真实自动流程需要用检测器生成
候选框（用检测框替代 GT box），其指标通常会显著低于本表中 box 行的数值。

用法：
    python medsam_prompt_isic.py --limit 5 --make-vis          # CPU 冒烟
    python medsam_prompt_isic.py --device cuda:0 --make-vis    # GPU 全量 200 张
    python medsam_prompt_isic.py --resume                      # 断点续跑
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage import measure  # noqa: E402
from tqdm import tqdm  # noqa: E402

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_ROOT = Path(__file__).resolve().parents[2]          # .../cv_study_project
MEDSAM_ROOT = BASE_ROOT / "external" / "MedSAM"          # SAM 源码所在仓库根目录
sys.path.insert(0, str(MEDSAM_ROOT))

from segment_anything import sam_model_registry            # noqa: E402
from segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402

DEFAULT_INVENTORY = BASE_ROOT / "outputs" / "m3_task2_isic2018_check" / "isic_inventory.csv"
DEFAULT_IMAGE_DIR = (BASE_ROOT / "data" / "ISIC2018_Task1-2_Training_Input"
                     / "ISIC2018_Task1-2_Training_Input")
DEFAULT_MASK_DIR = (BASE_ROOT / "data" / "ISIC2018_Task1_Training_GroundTruth"
                    / "ISIC2018_Task1_Training_GroundTruth")
DEFAULT_CKPT = BASE_ROOT / "ckpts" / "medsam_vit_b.pth"
OUTPUT_DIR = BASE_ROOT / "outputs" / "m4_task3_medsam_prompt_isic"

# ==================== 固定超参（对应任务书，不要随意改）====================
TARGET_SIZE = 1024          # 长边 resize 目标 / padding 后的方形边长
BOX_EXPAND_PX = 10          # GT box 四周扩展像素
NUM_NEG_POINTS = 4          # 负点个数
MASK_THRESHOLD = 0.5        # 概率图二值化阈值
PROMPT_MODES = ["point", "box", "point+box"]

FIELDNAMES = [
    "case_id", "split", "prompt_mode",
    "orig_h", "orig_w",
    "gt_box", "pos_point", "neg_points",
    "dice", "iou",
    "gt_area", "pred_area",
    "infer_time_sec",
    "image_path", "mask_path",
]


# ==================== 数据加载 ====================
def load_test_cases(inventory_path, split="test"):
    """从清单 CSV 取出指定 split 的 case_id（保持 CSV 中的顺序，可复现）。"""
    df = pd.read_csv(inventory_path)
    sub = df[df["split"] == split].reset_index(drop=True)
    return [(str(r.case_id), str(r.split)) for r in sub.itertuples()]


def index_dir(directory, exts, is_mask=False):
    """扫描目录建立 case_id -> 文件路径 的索引，避免依赖 CSV 里的绝对路径。"""
    idx = {}
    for p in sorted(Path(directory).glob("*")):
        if p.suffix.lower() not in exts:
            continue
        key = p.stem.replace("_segmentation", "") if is_mask else p.stem
        idx[key] = str(p)
    return idx


# ==================== 提示生成（原图坐标，一律返回 float32）====================
def get_gt_box(mask, expand=BOX_EXPAND_PX):
    """真值 mask 外接框向四周各扩展 expand 像素，并裁剪到图像边界。返回 [x0,y0,x1,y1]。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(int(xs.min()) - expand, 0)
    y0 = max(int(ys.min()) - expand, 0)
    x1 = min(int(xs.max()) + expand, w - 1)
    y1 = min(int(ys.max()) + expand, h - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def get_center_point(mask):
    """前景中心点：距离变换最深点，保证一定落在前景内部。返回 [x, y]。"""
    fg = (mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return None
    dt = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dt)
    if max_val <= 0:
        return None
    return np.array([float(max_loc[0]), float(max_loc[1])], dtype=np.float32)


def _greedy_farthest_points(dt, k, radius):
    """在距离图上贪心取 k 个最远点，每取一个点以 radius 做圆形邻域抑制。"""
    work = dt.copy()
    pts = []
    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        pts.append([float(x), float(y)])
        cv2.circle(work, (x, y), int(radius), -1.0, -1)
    return pts


def get_negative_points(mask, k=NUM_NEG_POINTS):
    """
    背景中距前景最远的 k 个负点：在背景距离图上贪心取最大值，
    每取一个点用圆形邻域抑制，保证 k 个点分散（通常落在图像四角）。

    抑制半径取"最远背景距离的一半"：这样半径与前景的尺度自适应，
    能够强制 4 个点围绕病灶分散；若半径过大导致凑不满 k 个点，则逐级减半放宽。
    （注意：若用固定的短边 10% 作半径，在病灶偏居一侧时 4 个点会沿距离脊线
      堆叠在同一条边上，形成单侧强负提示，实测会把 mask 压小。）
    返回 (k, 2) 的 [x, y] 数组。
    """
    bg = (mask == 0).astype(np.uint8)
    if bg.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dt = cv2.distanceTransform(bg, cv2.DIST_L2, 5)   # 每个背景像素到最近前景的距离
    max_dist = float(dt.max())
    if max_dist <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    radius = max(1.0, 0.5 * max_dist)
    pts = []
    while True:
        pts = _greedy_farthest_points(dt, k, int(round(radius)))
        if len(pts) == k or radius <= 1.0:
            break
        radius = max(1.0, radius * 0.5)
    return np.array(pts, dtype=np.float32)


# ==================== 预处理 / 推理 ====================
def preprocess(img_rgb, target=TARGET_SIZE):
    """
    长边 resize 到 target（保持长宽比）→ 右下 padding 到 target×target → 整块 min-max 归一化。
    返回: tensor (1,3,target,target)、(resize 后未填充的 h,w)、ResizeLongestSide 变换器
    """
    resize_tf = ResizeLongestSide(target)
    resized = resize_tf.apply_image(img_rgb)             # 要求 H×W×C 的 uint8，内部走 torchvision
    h, w = resized.shape[:2]
    arr = np.zeros((target, target, 3), dtype=np.float32)
    arr[:h, :w] = resized.astype(np.float32)
    arr = (arr - arr.min()) / np.clip(arr.max() - arr.min(), a_min=1e-8, a_max=None)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, (h, w), resize_tf


@torch.no_grad()
def medsam_predict(model, embedding, orig_hw, resized_hw, resize_tf,
                   box=None, points=None, labels=None, device="cpu"):
    """
    用同一份 image embedding 跑一次 mask decoder。
    注意：prompt_encoder 内部已对 points/boxes 做了 +0.5（Shift to center of pixel），
          调用方不要再手动加 0.5；坐标必须是 float32。
    """
    pe = model.prompt_encoder

    boxes_t = None
    if box is not None:
        b1024 = resize_tf.apply_boxes(box.reshape(1, 4).astype(np.float32), orig_hw)
        boxes_t = torch.as_tensor(b1024, dtype=torch.float, device=device)

    points_t = None
    if points is not None and len(points) > 0:
        c1024 = resize_tf.apply_coords(points.astype(np.float32), orig_hw)   # (N,2)
        points_t = (
            torch.as_tensor(c1024, dtype=torch.float, device=device).unsqueeze(0),   # (1,N,2)
            torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long,
                            device=device).unsqueeze(0),                             # (1,N)
        )

    sparse_emb, dense_emb = pe(points=points_t, boxes=boxes_t, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=pe.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )

    prob = torch.sigmoid(low_res_logits)                                   # (1,1,256,256)
    prob = F.interpolate(prob, size=(TARGET_SIZE, TARGET_SIZE),
                         mode="bilinear", align_corners=False)
    prob = prob[0, 0, :resized_hw[0], :resized_hw[1]].cpu().numpy()        # 裁掉右下 padding
    H, W = orig_hw
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)        # cv2 是 (宽, 高)
    pred = (prob > MASK_THRESHOLD).astype(np.uint8)
    return pred, prob


def dice_iou(pred, gt, eps=1e-6):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    dice = (2.0 * inter + eps) / (p.sum() + g.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def load_medsam(ckpt_path, device):
    """严格加载 MedSAM 权重；若 key 不匹配则打印诊断信息后报错，不静默 strict=False。"""
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"MedSAM 权重不存在: {ckpt_path}")
    try:
        model = sam_model_registry["vit_b"](checkpoint=ckpt_path)   # 内部 load_state_dict(strict=True)
    except RuntimeError as e:
        state = torch.load(ckpt_path, map_location="cpu")
        keys = list(state.keys())[:10]
        print("[ERROR] MedSAM 权重 strict 加载失败：", e)
        print("[ERROR] 权重顶层 key 前 10 个：", keys)
        print("[ERROR] 请检查 ckpt 是否为 SAM ViT-B 结构；禁止静默使用 strict=False。")
        raise
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MedSAM ViT-B 加载完成 | 参数量 {n_params/1e6:.1f}M | device={device}")
    return model


# ==================== 可视化 ====================
def fmt_box(box):
    return "" if box is None else f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"


def fmt_points(pts):
    return ";".join(f"{int(p[0])},{int(p[1])}" for p in pts)


def draw_prompts(ax, box, pos, neg):
    if box is not None:
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2))
    if pos is not None:
        ax.scatter([pos[0]], [pos[1]], marker="*", c="lime", s=220,
                   edgecolors="black", linewidths=1.0, zorder=5)
    if neg is not None and len(neg) > 0:
        ax.scatter(neg[:, 0], neg[:, 1], marker="x", c="red", s=110,
                   linewidths=2.5, zorder=5)


def save_overlay(img_rgb, gt, preds, metrics, box, pos, neg, case_id, save_path):
    """四面板叠加图：① 提示+GT ② 点预测 ③ 框预测 ④ 点+框预测（均叠加 GT 轮廓）。"""
    contours = measure.find_contours(gt.astype(float), 0.5)
    H, W = gt.shape
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))

    def paint(ax, title, pred=None, dice=None, iou=None, show_prompt=True):
        ax.imshow(img_rgb)
        if pred is not None:
            rgba = np.zeros((*pred.shape, 4), dtype=float)
            rgba[pred > 0] = (251 / 255, 252 / 255, 30 / 255, 0.55)   # 官方 MedSAM 黄色
            ax.imshow(rgba)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color="lime", lw=1.6, ls="--")  # GT 轮廓
        if show_prompt:
            draw_prompts(ax, box, pos, neg)
        sub = f"Dice={dice:.4f}  IoU={iou:.4f}" if dice is not None else "GT contour"
        ax.set_title(f"{title}\n{sub}", fontsize=12)
        ax.axis("off")

    panels = [
        ("Prompt + GT (box+10px, 1 pos, 4 neg)", None),
        ("Point prompt", "point"),
        ("Box prompt", "box"),
        ("Point + Box prompt", "point+box"),
    ]
    for ax, (title, mode) in zip(axes, panels):
        if mode is None:
            paint(ax, title, show_prompt=True)
            continue
        score = metrics.get(mode)
        if score is None:        # 该模式被 --resume 跳过
            paint(ax, title, show_prompt=True)
        else:
            paint(ax, title, pred=preds.get(mode), dice=score[0], iou=score[1])

    fig.suptitle(f"{case_id}   ({H}x{W})", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ==================== 主流程 ====================
def parse_args():
    p = argparse.ArgumentParser(description="MedSAM 点/框提示 ISIC 2018 分割比较")
    p.add_argument("--inventory", type=str, default=str(DEFAULT_INVENTORY))
    p.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    p.add_argument("--mask-dir", type=str, default=str(DEFAULT_MASK_DIR))
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--medsam-root", type=str, default=str(MEDSAM_ROOT))
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 例，0 表示全部")
    p.add_argument("--num-vis", type=int, default=20, help="叠加图组数")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda:0")
    p.add_argument("--num-threads", type=int, default=0, help="CPU 线程数，0 表示不设置")
    p.add_argument("--resume", action="store_true", help="按 (case_id, prompt_mode) 跳过已完成项")
    p.add_argument("--make-vis", action="store_true", help="生成叠加图")
    p.add_argument("--save-masks", action="store_true", help="额外保存预测 mask（png）")
    return p.parse_args()


def run_evaluation(args):
    """核心评测流程（本地版与 OpenI 版共用入口）。"""
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "overlays"
    mask_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "medsam_metrics.csv"
    summary_path = out_dir / "medsam_metrics_summary.csv"

    print("=" * 70)
    print("任务 4.3.3 MedSAM 点/框提示医学分割比较")
    print(f"清单: {args.inventory}  split={args.split}")
    print(f"图像目录: {args.image_dir}")
    print(f"掩膜目录: {args.mask_dir}")
    print(f"权重: {args.checkpoint}")
    print(f"输出: {out_dir}")
    print("=" * 70)

    # 1. 数据与模型
    cases = load_test_cases(args.inventory, args.split)
    if args.limit > 0:
        cases = cases[:args.limit]
    img_index = index_dir(args.image_dir, {".jpg", ".jpeg", ".png"}, is_mask=False)
    mask_index = index_dir(args.mask_dir, {".png", ".jpg", ".jpeg"}, is_mask=True)
    print(f"[data] 待处理 {len(cases)} 例 | 图像索引 {len(img_index)} | 掩膜索引 {len(mask_index)}")

    device = args.device
    model = load_medsam(args.checkpoint, device)

    # 2. 断点续跑
    done = set()
    file_exists = csv_path.exists()
    if args.resume and file_exists:
        old = pd.read_csv(csv_path)
        done = set(zip(old["case_id"].astype(str), old["prompt_mode"].astype(str)))
        print(f"[resume] 已完成 {len(done)} 条，将跳过")

    # 3. 均匀抽取可视化样本
    n_cases = len(cases)
    if args.make_vis and n_cases > 0:
        stride = max(1, n_cases // max(1, args.num_vis))
        vis_positions = set(range(0, n_cases, stride)[:args.num_vis])
    else:
        vis_positions = set()
    if args.make_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    fout = open(csv_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    skipped = 0
    pbar = tqdm(enumerate(cases), total=n_cases, desc="MedSAM inference", unit="case")
    for idx, (case_id, split_name) in pbar:
        img_path = img_index.get(case_id)
        mask_path = mask_index.get(case_id)
        need_vis = idx in vis_positions
        if img_path is None or mask_path is None:
            print(f"[warn] {case_id} 缺图像或掩膜，跳过")
            skipped += 1
            continue
        if not need_vis and all((case_id, m) in done for m in PROMPT_MODES):
            continue

        # --- 读图 ---
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[warn] 无法读取 {img_path}，跳过")
            skipped += 1
            continue
        if img_bgr.ndim == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"[warn] 无法读取 {mask_path}，跳过")
            skipped += 1
            continue
        gt = (gt > 127).astype(np.uint8)
        if gt.shape != (H, W):
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            gt = (gt > 127).astype(np.uint8)
        if gt.sum() == 0:
            print(f"[warn] {case_id} 真值掩膜为空，跳过")
            skipped += 1
            continue

        # --- 提示 ---
        box = get_gt_box(gt)
        pos = get_center_point(gt)
        neg = get_negative_points(gt)
        if box is None or pos is None:
            print(f"[warn] {case_id} 提示生成失败，跳过")
            skipped += 1
            continue
        points = np.vstack([pos, neg]) if len(neg) > 0 else pos.reshape(1, 2)
        labels = np.array([1] + [0] * len(neg), dtype=np.int64)

        # --- 预处理 + 一次编码 ---
        tensor, resized_hw, resize_tf = preprocess(img_rgb)
        t0 = time.perf_counter()
        with torch.no_grad():
            embedding = model.image_encoder(tensor.to(device))
        enc_time = time.perf_counter() - t0

        preds, metrics = {}, {}
        for mode in PROMPT_MODES:
            if not need_vis and (case_id, mode) in done:
                continue
            if mode == "point":
                b, p_, l_ = None, points, labels
            elif mode == "box":
                b, p_, l_ = box, None, None
            else:
                b, p_, l_ = box, points, labels

            t1 = time.perf_counter()
            pred, _ = medsam_predict(model, embedding, (H, W), resized_hw, resize_tf,
                                     box=b, points=p_, labels=l_, device=device)
            mode_time = time.perf_counter() - t1
            dice, iou = dice_iou(pred, gt)

            preds[mode] = pred
            metrics[mode] = (dice, iou)

            if (case_id, mode) not in done:
                writer.writerow({
                    "case_id": case_id,
                    "split": split_name,
                    "prompt_mode": mode,
                    "orig_h": H,
                    "orig_w": W,
                    "gt_box": fmt_box(box),
                    "pos_point": f"{int(pos[0])},{int(pos[1])}",
                    "neg_points": fmt_points(neg),
                    "dice": round(dice, 6),
                    "iou": round(iou, 6),
                    "gt_area": int(gt.sum()),
                    "pred_area": int(pred.sum()),
                    "infer_time_sec": round(mode_time, 4),
                    "image_path": img_path,
                    "mask_path": mask_path,
                })
                fout.flush()
            if args.save_masks:
                cv2.imwrite(str(mask_dir / f"{case_id}_{mode.replace('+', 'and')}.png"),
                            pred * 255)

        pbar.set_postfix({m: f"{metrics[m][0]:.3f}" for m in metrics})

        # --- 叠加图 ---
        if need_vis:
            order = len([p for p in vis_dir.glob("*.png")]) + 1
            save_overlay(img_rgb, gt, preds, metrics, box, pos, neg, case_id,
                         vis_dir / f"overlay_{order:02d}_{case_id}.png")

    fout.close()

    # 4. 汇总
    df = pd.read_csv(csv_path)
    rows = []
    for mode in PROMPT_MODES:
        sub = df[df["prompt_mode"] == mode]
        if len(sub) == 0:
            continue
        rows.append({
            "prompt_mode": mode,
            "n": len(sub),
            "dice_mean": sub["dice"].mean(),
            "dice_std": sub["dice"].std(),
            "dice_median": sub["dice"].median(),
            "iou_mean": sub["iou"].mean(),
            "iou_std": sub["iou"].std(),
            "iou_median": sub["iou"].median(),
            "dice_lt_0.5_count": int((sub["dice"] < 0.5).sum()),
            "mean_infer_time_sec": sub["infer_time_sec"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"完成 {len(df)} 条记录（跳过 {skipped} 例）")
    print(summary.to_string(index=False,
                            formatters={c: "{:.4f}".format for c in summary.columns
                                        if c.startswith(("dice_", "iou_", "mean_"))}))
    print(f"\n逐例指标: {csv_path}")
    print(f"汇总指标: {summary_path}")
    if args.make_vis:
        print(f"叠加图:   {vis_dir} （{len(list(vis_dir.glob('*.png')))} 组）")
    print("""
【交互式上界声明】GT box 由真值 mask 外接框 ±10px 生成，前景中心点与负点同样由真值推导，
因此本表三种提示均属于 Oracle Prompt / 交互式上界协议，衡量的是"提示最优时 MedSAM 的
能力上限"，并非完全自动分割性能。真实自动流程需以检测器候选框替代 GT box，指标会更低。""")
    print("=" * 70)


def main():
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
