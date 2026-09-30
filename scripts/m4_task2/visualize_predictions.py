"""
可视化 fold 0 验证集的预测结果（GT 轮廓 + 预测轮廓叠加在原始 CT 上），
并输出每个病例的 Dice 统计（CSV / 柱状图）。

用法：
    python visualize_predictions.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nnunet_env import (  # noqa: E402
    DATASET_FOLDER, FOLD, METRICS_DIR, MODEL_FOLDER, PRED_DIR, PREP_FOLDER, VIZ_DIR,
)

VAL_IMAGES = PRED_DIR / "val_images"
VAL_GT = PRED_DIR / "val_gt"
VAL_PREDS = PRED_DIR / "val_preds"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.sum(a & b) / denom) if denom > 0 else float("nan")


def pick_slice(gt: np.ndarray) -> int:
    """选择 GT 面积最大的轴向切片。"""
    areas = gt.sum(axis=(1, 2))
    return int(np.argmax(areas)) if areas.max() > 0 else gt.shape[0] // 2


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(VAL_PREDS.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"未找到预测结果: {VAL_PREDS}")

    rows = []
    for pred_file in pred_files:
        case = pred_file.name.replace(".nii.gz", "")
        img = nib.load(VAL_IMAGES / f"{case}_0000.nii.gz").get_fdata()
        gt = nib.load(VAL_GT / f"{case}.nii.gz").get_fdata()
        pred = nib.load(pred_file).get_fdata()

        d = dice(pred, gt)
        rows.append({"case": case, "dice": d,
                     "gt_voxels": int((gt > 0).sum()),
                     "pred_voxels": int((pred > 0).sum())})

        z = pick_slice(gt)
        sl_img, sl_gt, sl_pred = img[z], gt[z], pred[z]
        lo, hi = np.percentile(sl_img, [1, 99])
        sl_img = np.clip(sl_img, lo, hi)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        ax[0].imshow(sl_img, cmap="gray")
        ax[0].set_title(f"{case}  CT (axial z={z})")
        ax[1].imshow(sl_img, cmap="gray")
        if sl_gt.max() > 0:
            ax[1].contour(sl_gt, levels=[0.5], colors="lime", linewidths=1.5)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(sl_img, cmap="gray")
        if sl_gt.max() > 0:
            ax[2].contour(sl_gt, levels=[0.5], colors="lime", linewidths=1.5)
        if sl_pred.max() > 0:
            ax[2].contour(sl_pred, levels=[0.5], colors="red", linewidths=1.5)
        ax[2].set_title(f"Prediction (3D Dice={d:.4f})")
        for a in ax:
            a.axis("off")
        plt.suptitle(f"Dataset901_SpleenStudy | {MODEL_FOLDER.name} | fold {FOLD} | {case}", fontsize=12)
        plt.tight_layout()
        out = VIZ_DIR / f"{case}_pred_overlay.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[viz] {out.name}  dice={d:.4f}")

    csv_path = METRICS_DIR / "per_case_dice.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "dice", "gt_voxels", "pred_voxels"])
        w.writeheader()
        w.writerows(rows)
    mean_dice = float(np.nanmean([r["dice"] for r in rows]))
    print(f"[viz] 平均 Dice = {mean_dice:.4f}（n={len(rows)}），CSV -> {csv_path}")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([r["case"] for r in rows], [r["dice"] for r in rows], color="#4C72B0")
    ax.axhline(mean_dice, ls="--", color="crimson", label=f"mean = {mean_dice:.4f}")
    ax.set_ylabel("Dice (spleen)")
    ax.set_title(f"Per-case Dice — fold {FOLD} — smoke test (未完整训练)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "per_case_dice.png", dpi=130)
    plt.close()

    stats = {
        "model_folder": str(MODEL_FOLDER),
        "fold": FOLD,
        "num_cases": len(rows),
        "mean_dice": mean_dice,
        "per_case": rows,
        "note": "CPU 环境 1 epoch smoke test，未完成完整训练，该数值不代表 nnU-Net baseline 性能",
    }
    (METRICS_DIR / "per_case_dice.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[viz] 可视化目录: {VIZ_DIR}")


if __name__ == "__main__":
    main()
