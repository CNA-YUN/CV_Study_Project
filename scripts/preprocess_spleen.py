import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from scipy.ndimage import zoom
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
BASE_ROOT = Path.cwd().parent
DATA_ROOT = BASE_ROOT / "data" / "Task09_Spleen" / "Task09_Spleen"
OUTPUT_DIR = BASE_ROOT / "outputs" / "msd_spleen_preprocess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 固定预处理参数
TARGET_SPACING = (1.5, 1.5, 1.5)  # (Z, Y, X) 统一各向同性
CT_MIN, CT_MAX = -125, 275
NORM_MIN, NORM_MAX = 0.0, 1.0
NUM_CASES = 10


# ==============================================

def resample_volume(volume, current_spacing, target_spacing, order):
    """
    使用 scipy.ndimage.zoom 进行体素重采样
    volume: 形状 (Z, Y, X)
    current_spacing: (sz, sy, sx)
    target_spacing: (tz, ty, tx)
    order: 3 三线性（图像），0 最近邻（标签）
    """
    if current_spacing == target_spacing:
        return volume, 1.0

    # 计算缩放因子 (新体素 / 旧体素)
    factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    # 确保输出形状不为零
    new_shape = np.round(np.array(volume.shape) * np.array(factors)).astype(int)
    # zoom 接受 shape 或 factor，用 factor 更安全，但避免因浮点误差导致形状偏差
    # 使用 map_coordinates 更精确，但 zoom 配合 order 简单高效
    resampled = zoom(volume, factors, order=order)
    return resampled, factors


def get_foreground_center(label, axis):
    """获取包含前景的质心切片（轴 0=Z, 1=Y, 2=X）"""
    if axis == 0:
        proj = np.sum(label, axis=(1, 2))
    elif axis == 1:
        proj = np.sum(label, axis=(0, 2))
    elif axis == 2:
        proj = np.sum(label, axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")
    if np.sum(proj) == 0:
        return label.shape[axis] // 2
    indices = np.arange(len(proj))
    return int(round(np.average(indices, weights=proj)))


def plot_before_after(original_img, original_lbl, processed_img, processed_lbl,
                      spacing_orig, spacing_proc, case_name, save_dir):
    """
    绘制处理前后同一轴向切片的 overlay 对比图
    为了保证对比的是同一个解剖位置，计算物理 Z 坐标对应的索引
    """
    # 1. 在预处理后的图像上找到病灶中心切片 (物理坐标)
    z_proc = get_foreground_center(processed_lbl, axis=0)
    # 计算该切片的物理 Z 坐标 (单位: mm)
    z_physical = z_proc * spacing_proc[0]

    # 2. 计算该物理坐标在原始图像中对应的切片索引
    z_orig = int(round(z_physical / spacing_orig[0]))
    z_orig = np.clip(z_orig, 0, original_img.shape[0] - 1)

    # 3. 提取切片 (均为 (Y, X) 形状)
    orig_slice_img = original_img[z_orig, :, :]
    orig_slice_lbl = original_lbl[z_orig, :, :]
    proc_slice_img = processed_img[z_proc, :, :]
    proc_slice_lbl = processed_lbl[z_proc, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 辅助绘图函数
    def plot_overlay(ax, img, lbl, title, spacing):
        # 归一化显示 (图像强度可能不是标准 0-1，做 99% 截断)
        img_display = np.clip(img, 0, np.percentile(img, 99))
        if img_display.max() - img_display.min() > 1e-6:
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        ax.imshow(img_display, cmap='gray')
        if np.sum(lbl) > 0:
            from skimage import measure
            contours = measure.find_contours(lbl, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plot_overlay(axes[0], orig_slice_img, orig_slice_lbl,
                 f"Before (Z={z_orig}, Sp={spacing_orig[0]:.2f}mm)", spacing_orig)
    plot_overlay(axes[1], proc_slice_img, proc_slice_lbl,
                 f"After (Z={z_proc}, Sp={spacing_proc[0]:.2f}mm)", spacing_proc)

    plt.suptitle(f"{case_name} - Physical Z = {z_physical:.1f} mm", fontsize=14)
    plt.tight_layout()
    save_path = save_dir / f"{case_name}_comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()


def process_single_case(img_path, lbl_path, case_idx, target_spacing):
    """
    处理单个病例：重采样 -> Clip -> 归一化 -> 保存 -> 返回元数据
    """
    # ----- 1. 加载数据 (保持 Nibabel 原生顺序 (X, Y, Z)，重采样完再转置) -----
    # 注意：为了仿射矩阵更新的便利性，我们在 (X,Y,Z) 空间下重采样，最后保存时也保持 (X,Y,Z)
    img_nib = nib.load(img_path)
    lbl_nib = nib.load(lbl_path)

    img_data = img_nib.get_fdata().astype(np.float32)  # (X, Y, Z)
    lbl_data = lbl_nib.get_fdata().astype(np.uint8)  # (X, Y, Z)

    # 提取原始空间参数
    affine = img_nib.affine.copy()
    orig_spacing_xyz = img_nib.header.get_zooms()  # (sx, sy, sz)

    # 目标 spacing 顺序为 (Z, Y, X)，为了重采样需转成 (X, Y, Z)
    target_xyz = (target_spacing[2], target_spacing[1], target_spacing[0])

    # ----- 2. 重采样 (Resample) -----
    # 计算缩放因子 (当前spacing -> 目标spacing)
    factors_xyz = [orig_spacing_xyz[i] / target_xyz[i] for i in range(3)]

    # 图像：三线性插值 (order=3)
    img_resampled = zoom(img_data, factors_xyz, order=3)
    # 标签：最近邻插值 (order=0) 确保离散性
    lbl_resampled = zoom(lbl_data, factors_xyz, order=0)

    # 由于最近邻插值可能产生浮点数，强制取整并截断为 0/1
    lbl_resampled = np.round(lbl_resampled).astype(np.uint8)
    lbl_resampled = np.clip(lbl_resampled, 0, 1)

    # 更新仿射矩阵以匹配新体素网格
    # new_affine[:3,:3] = old_affine[:3,:3] @ diag(target/original)
    scale_matrix = np.diag([target_xyz[i] / orig_spacing_xyz[i] for i in range(3)])
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ scale_matrix

    # ----- 3. CT 窗位裁剪与归一化 -----
    img_clipped = np.clip(img_resampled, CT_MIN, CT_MAX)
    img_normalized = (img_clipped - CT_MIN) / (CT_MAX - CT_MIN)
    img_normalized = np.clip(img_normalized, NORM_MIN, NORM_MAX).astype(np.float32)

    # ----- 4. 转置为 (Z, Y, X) 方便保存和可视化 -----
    # Nibabel 保存时要求 (X,Y,Z)，但我们的可视化习惯用 (Z,Y,X)，这里保存时转回去
    # 为了一致性，我们将最终输出保存为 (X, Y, Z) 顺序，这样符合 NIfTI 标准
    img_final = img_normalized  # 保持 (X, Y, Z)
    lbl_final = lbl_resampled  # 保持 (X, Y, Z)

    # ----- 5. 保存为 NIfTI 文件 -----
    case_name = f"spleen_{case_idx:03d}"
    img_out_path = OUTPUT_DIR / "images" / f"{case_name}_preprocessed.nii.gz"
    lbl_out_path = OUTPUT_DIR / "labels" / f"{case_name}_preprocessed.nii.gz"
    os.makedirs(OUTPUT_DIR / "images", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "labels", exist_ok=True)

    img_out = nib.Nifti1Image(img_final, new_affine)
    lbl_out = nib.Nifti1Image(lbl_final, new_affine)
    nib.save(img_out, img_out_path)
    nib.save(lbl_out, lbl_out_path)

    # ----- 6. 生成前后对比图 (在 (Z,Y,X) 下操作，需要转置) -----
    # 获取转置后的数据用于可视化
    img_orig_zxy = np.transpose(img_data, (2, 1, 0))
    lbl_orig_zxy = np.transpose(lbl_data, (2, 1, 0))
    img_proc_zxy = np.transpose(img_final, (2, 1, 0))
    lbl_proc_zxy = np.transpose(lbl_final, (2, 1, 0))

    # 原始 spacing (转换为 Z,Y,X)
    orig_spacing_zxy = (orig_spacing_xyz[2], orig_spacing_xyz[1], orig_spacing_xyz[0])
    proc_spacing_zxy = target_spacing

    plot_dir = OUTPUT_DIR / "comparison_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_before_after(
        img_orig_zxy, lbl_orig_zxy,
        img_proc_zxy, lbl_proc_zxy,
        orig_spacing_zxy, proc_spacing_zxy,
        case_name, plot_dir
    )

    # ----- 7. 返回元数据 -----
    metadata = {
        "case_id": case_name,
        "original_shape_xyz": str(img_data.shape),
        "final_shape_xyz": str(img_final.shape),
        "original_spacing_xyz": str(orig_spacing_xyz),
        "target_spacing_xyz": str(target_xyz),
        "intensity_before_min": float(np.min(img_data)),
        "intensity_before_max": float(np.max(img_data)),
        "intensity_after_min": float(np.min(img_normalized)),
        "intensity_after_max": float(np.max(img_normalized)),
        "label_unique_after": str(np.unique(lbl_final).tolist()),
        "image_path": str(img_out_path),
        "label_path": str(lbl_out_path),
    }
    return metadata


def main():
    print("=" * 60)
    print("MSD Spleen 预处理 Pipeline (Z/Y/X 各向同性)")
    print(f"目标 Spacing: {TARGET_SPACING} mm")
    print(f"CT Clip: [{CT_MIN}, {CT_MAX}] -> Norm [{NORM_MIN}, {NORM_MAX}]")
    print("=" * 60)

    # 1. 取前 10 个训练病例
    images_dir = DATA_ROOT / "imagesTr"
    labels_dir = DATA_ROOT / "labelsTr"
    if not images_dir.exists():
        raise FileNotFoundError(f"请检查数据集路径: {DATA_ROOT}")

    image_files = sorted(glob(str(images_dir / "*.nii.gz")))[:NUM_CASES]
    print(f"找到 {len(image_files)} 个病例，将处理前 {NUM_CASES} 个")

    all_metadata = []

    for idx, img_path in enumerate(image_files, start=1):
        basename = Path(img_path).stem.replace('.nii', '')
        lbl_path = labels_dir / f"{basename}.nii.gz"
        if not lbl_path.exists():
            lbl_path = labels_dir / f"{basename}.nii"
        if not lbl_path.exists():
            print(f"警告: 未找到标签 {lbl_path}，跳过")
            continue

        print(f"\n处理病例 {idx}: {basename}")
        meta = process_single_case(img_path, lbl_path, idx, TARGET_SPACING)
        all_metadata.append(meta)
        print(f"  -> 完成，新形状: {meta['final_shape_xyz']}")
        print(f"  -> 标签唯一值: {meta['label_unique_after']}")

    # 2. 保存 metadata CSV
    df = pd.DataFrame(all_metadata)
    csv_path = OUTPUT_DIR / "preprocess_metadata.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("✅ 预处理全部完成！")
    print(f"元数据 CSV: {csv_path}")
    print(f"预处理图像: {OUTPUT_DIR / 'images'}")
    print(f"预处理标签: {OUTPUT_DIR / 'labels'}")
    print(f"对比图: {OUTPUT_DIR / 'comparison_plots'}")
    print("\n验收重点自检:")
    print("  1. 标签唯一值应为 [0, 1] -> 已强制约束")
    print("  2. 图像与标签通过同一新仿射矩阵保存 -> 无错位")
    print("=" * 60)


if __name__ == "__main__":
    main()