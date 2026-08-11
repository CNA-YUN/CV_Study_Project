import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from pydicom.data import get_testdata_files
import glob
from pathlib import Path
import warnings
from matplotlib.patches import Rectangle
from skimage import measure

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_ROOT = Path.cwd().parent  # 当前工作目录，你也可以改成绝对路径
DATA_ROOT = BASE_ROOT / "data" / "Task09_Spleen" / "Task09_Spleen"  # MSD 数据集根目录
OUTPUT_DIR = BASE_ROOT / "outputs" / "m3_task1_dicom_nifti"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"数据根目录: {DATA_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")


def plot_three_views(image, label, case_name, title_prefix="", is_2d=False):
    """
    绘制 axial / coronal / sagittal 三视图（论文级配图，含物理比例和方向标注）

    参数：
        image: 3D (Z,Y,X) 或 2D 数组
        label: 3D (Z,Y,X) 或 2D 标签
        case_name: 病例名称
        title_prefix: 前缀（如 "NIfTI"）
        is_2d: 若为 True，则所有视图显示同一张切片（用于 DICOM）
    """
    # ---- 处理 2D 单切片 ----
    if is_2d or image.ndim == 2:
        if image.ndim == 2:
            img_slice = image
            lbl_slice = label if label.ndim == 2 else label[0]
        else:
            mid = image.shape[0] // 2
            img_slice = image[mid]
            lbl_slice = label[mid] if label.ndim == 3 else label

        # 生成三张不同旋转的视图（仅示意，无物理比例）
        slices = [
            (img_slice, lbl_slice, "Axial (Z)"),
            (np.rot90(img_slice, k=1), np.rot90(lbl_slice, k=1), "Coronal (rotated)"),
            (np.rot90(img_slice, k=-1), np.rot90(lbl_slice, k=-1), "Sagittal (rotated)"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (img_s, lbl_s, title) in zip(axes, slices):
            img_display = np.clip(img_s, 0, np.percentile(img_s, 99))
            if img_display.max() - img_display.min() > 1e-6:
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
            ax.imshow(img_display, cmap='gray')
            if np.sum(lbl_s) > 0:
                contours = measure.find_contours(lbl_s, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            ax.set_title(title)
            ax.axis('off')
        plt.suptitle(f"{title_prefix} {case_name} (2D)", fontsize=14)
        plt.tight_layout()
        # 保存路径沿用你的 OUTPUT_DIR
        save_path = OUTPUT_DIR / f"{case_name}_three_views.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> 三视图已保存: {save_path}")
        return

    # ---- 以下是 3D 体数据的专业配图 ----
    # 注意：这里假设 image 和 label 的 shape 为 (Z, Y, X)
    # 修正 get_foreground_center 的轴映射（此处内联实现，避免依赖外部错误函数）
    def get_center(vol, lbl, axis):
        # axis: 0=Z(轴向), 1=Y(冠状), 2=X(矢状)
        if axis == 0:
            proj = np.sum(lbl, axis=(1, 2))
        elif axis == 1:
            proj = np.sum(lbl, axis=(0, 2))
        elif axis == 2:
            proj = np.sum(lbl, axis=(0, 1))
        else:
            raise ValueError
        if np.sum(proj) == 0:
            return vol.shape[axis] // 2
        indices = np.arange(len(proj))
        return int(round(np.average(indices, weights=proj)))

    z_c = get_center(image, label, 0)
    y_c = get_center(image, label, 1)
    x_c = get_center(image, label, 2)

    # 提取切片
    axial_img = image[z_c, :, :]  # (Y, X)
    axial_lbl = label[z_c, :, :]
    coronal_img = image[:, y_c, :]  # (Z, X)
    coronal_lbl = label[:, y_c, :]
    sagittal_img = image[:, :, x_c]  # (Z, Y)
    sagittal_lbl = label[:, :, x_c]

    # 获取 spacing
    # 在函数内部尝试从全局获取。
    try:
        # 尝试从全局变量中获取 spacing（需你在主程序中定义 global_spacing）
        spacing = global_spacing
    except NameError:
        # 若未定义，则使用单位间距（不拉伸）
        spacing = (1.0, 1.0, 1.0)
        print("警告: 未设置 spacing，使用默认 1,1,1，图像比例可能失真。")

    sz, sy, sx = spacing  # (Z, Y, X)

    # 计算物理范围 (extent)
    # 轴向: 形状 (Y, X) -> extent = [X_min, X_max, Y_min, Y_max]
    x_min, x_max = 0, sx * axial_img.shape[1]
    y_min, y_max = 0, sy * axial_img.shape[0]
    z_min, z_max = 0, sz * coronal_img.shape[0]

    extent_axial = [x_min, x_max, y_min, y_max]
    extent_coronal = [x_min, x_max, z_min, z_max]
    extent_sagittal = [y_min, y_max, z_min, z_max]

    # ---- 创建画布 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # 辅助函数：标准化显示
    def norm_img(img):
        img = np.clip(img, 0, np.percentile(img, 99))
        if img.max() - img.min() > 1e-6:
            return (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    # ---- (A) 轴向 ----
    ax = axes[0]
    ax.imshow(norm_img(axial_img), cmap='gray', extent=extent_axial, origin='upper')
    # 病灶半透明叠加
    if np.sum(axial_lbl) > 0:
        ax.imshow(axial_lbl, cmap='autumn_r', extent=extent_axial, origin='upper', alpha=0.3, vmin=0.5, vmax=1)
        ax.contour(axial_lbl, levels=[0.5], colors='red', linewidths=0.8, extent=extent_axial, origin='upper')
    # 方向标注 (R在左, L在右)
    ax.text(0.02, 0.98, 'R', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='left',
            va='top')
    ax.text(0.98, 0.98, 'L', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='right',
            va='top')
    ax.text(0.02, 0.02, 'P', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='left',
            va='bottom')
    ax.text(0.98, 0.02, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='right',
            va='bottom')
    ax.set_title(f'Axial (Z={z_c}, {sz:.2f}mm)', fontsize=14)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')

    # ---- (B) 冠状 ----
    ax = axes[1]
    ax.imshow(norm_img(coronal_img), cmap='gray', extent=extent_coronal, origin='upper')
    if np.sum(coronal_lbl) > 0:
        ax.imshow(coronal_lbl, cmap='autumn_r', extent=extent_coronal, origin='upper', alpha=0.3, vmin=0.5, vmax=1)
        ax.contour(coronal_lbl, levels=[0.5], colors='red', linewidths=0.8, extent=extent_coronal, origin='upper')
    ax.text(0.02, 0.98, 'R', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='left',
            va='top')
    ax.text(0.98, 0.98, 'L', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='right',
            va='top')
    ax.text(0.02, 0.02, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='left',
            va='bottom')
    ax.text(0.98, 0.02, 'H', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='right',
            va='bottom')
    ax.set_title(f'Coronal (Y={y_c}, {sy:.2f}mm)', fontsize=14)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_aspect('equal')

    # ---- (C) 矢状 ----
    ax = axes[2]
    ax.imshow(norm_img(sagittal_img), cmap='gray', extent=extent_sagittal, origin='upper')
    if np.sum(sagittal_lbl) > 0:
        ax.imshow(sagittal_lbl, cmap='autumn_r', extent=extent_sagittal, origin='upper', alpha=0.3, vmin=0.5, vmax=1)
        ax.contour(sagittal_lbl, levels=[0.5], colors='red', linewidths=0.8, extent=extent_sagittal, origin='upper')
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='left',
            va='top')
    ax.text(0.98, 0.98, 'P', transform=ax.transAxes, fontsize=14, fontweight='bold', color='yellow', ha='right',
            va='top')
    ax.text(0.02, 0.02, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='left',
            va='bottom')
    ax.text(0.98, 0.02, 'H', transform=ax.transAxes, fontsize=14, fontweight='bold', color='cyan', ha='right',
            va='bottom')
    ax.set_title(f'Sagittal (X={x_c}, {sx:.2f}mm)', fontsize=14)
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_aspect('equal')

    plt.suptitle(f"{title_prefix} {case_name}", fontsize=14)
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{case_name}_three_views.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> 三视图已保存: {save_path}")


# ==================== 1. DICOM 处理 ====================
def process_dicom():
    print("\n" + "=" * 50)
    print("处理 DICOM 文件: CT_small.dcm")
    print("=" * 50)

    # 获取测试文件
    dcm_files = get_testdata_files("CT_small.dcm")
    if not dcm_files:
        raise FileNotFoundError("无法获取 CT_small.dcm 测试文件")
    dcm_path = dcm_files[0]
    print(f"DICOM 文件路径: {dcm_path}")

    # 读取 DICOM
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array  # 2D 单切片

    # 提取元数据
    metadata = {
        "case_name": "CT_small",
        "format": "DICOM",
        "shape": str(pixel_array.shape),
        "dtype": str(pixel_array.dtype),
        "intensity_min": float(np.min(pixel_array)),
        "intensity_max": float(np.max(pixel_array)),
        "spacing": str(getattr(ds, 'PixelSpacing', 'N/A')),
        "slice_thickness": str(getattr(ds, 'SliceThickness', 'N/A')),
        "modality": str(getattr(ds, 'Modality', 'N/A')),
        "origin": "N/A (DICOM uses ImagePositionPatient)",
        "direction": "N/A (DICOM uses ImageOrientationPatient)",
        "affine": "N/A (not applicable for DICOM)",
        "shape_consistent": "N/A (single slice)",
        "affine_consistent": "N/A (single slice)",
    }
    print(f"元数据: {metadata}")

    # 构造伪标签（手动标记中心区域为前景，以便显示轮廓）
    h, w = pixel_array.shape
    pseudo_label = np.zeros_like(pixel_array, dtype=np.uint8)
    # 在图像中心画一个矩形作为模拟前景
    pseudo_label[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1

    # 生成三视图（2D 模式，所有视图显示同一张切片）
    plot_three_views(pixel_array, pseudo_label, "CT_small_dcm", title_prefix="DICOM", is_2d=True)

    return metadata


# ==================== 2. NIfTI 处理 ====================
def process_nifti_case(image_path, label_path, case_idx):
    print(f"\n处理 NIfTI 病例 {case_idx}: {Path(image_path).name}")

    # 读取图像和标签
    img_nib = nib.load(image_path)
    lbl_nib = nib.load(label_path)

    # 获取像素数据 (nibabel 默认是 (X, Y, Z)，我们转置为 (Z, Y, X) 方便切片)
    image_data = img_nib.get_fdata().astype(np.float32)
    label_data = lbl_nib.get_fdata().astype(np.uint8)

    # 转置为 (Z, Y, X)
    image_data = np.transpose(image_data, (2, 1, 0))
    label_data = np.transpose(label_data, (2, 1, 0))

    # 提取空间信息
    affine = img_nib.affine
    header = img_nib.header
    pixdim = header.get_zooms()  # (X, Y, Z) 方向的 spacing
    # 转置后 spacing 对应 (Z, Y, X)
    spacing = (pixdim[2], pixdim[1], pixdim[0])  # (Z, Y, X)
    global global_spacing
    global_spacing = spacing
    # 提取 origin 和 direction（从 affine 矩阵计算）
    origin = affine[:3, 3]  # 物理原点
    # direction cosines (旋转矩阵)
    direction = affine[:3, :3]

    # 转置 direction 以匹配 (Z, Y, X)
    direction_zxy = direction[[2, 1, 0], :][:, [2, 1, 0]]

    metadata = {
        "case_name": f"Spleen_case_{case_idx:02d}",
        "format": "NIfTI",
        "shape": str(image_data.shape),  # (Z, Y, X)
        "dtype": str(image_data.dtype),
        "intensity_min": float(np.min(image_data)),
        "intensity_max": float(np.max(image_data)),
        "spacing": str(spacing),
        "origin": str(origin),
        "direction": str(direction_zxy),
        "affine": str(affine),
        "label_shape": str(label_data.shape),
        "label_dtype": str(label_data.dtype),
        "label_unique_values": str(np.unique(label_data)),
    }
    print(f"元数据: {metadata}")

    # 生成三视图
    plot_three_views(image_data, label_data, f"Spleen_case_{case_idx:02d}", title_prefix="NIfTI")

    # 检查图像和标签空间是否一致
    img_affine = img_nib.affine
    lbl_affine = lbl_nib.affine
    is_same_shape = image_data.shape == label_data.shape
    is_same_affine = np.allclose(img_affine, lbl_affine)
    metadata["shape_consistent"] = str(is_same_shape)
    metadata["affine_consistent"] = str(is_same_affine)

    print(f"  -> 图像与标签形状一致: {is_same_shape}")
    print(f"  -> 图像与标签 affine 一致: {is_same_affine}")

    return metadata


# ==================== 3. 主程序入口 ====================
if __name__ == "__main__":
    all_metadata = []

    # ---- 3.1 处理 DICOM ----
    dcm_meta = process_dicom()
    all_metadata.append(dcm_meta)

    # ---- 3.2 处理 NIfTI (MSD Task09 Spleen 前 3 例) ----
    # 查找 imagesTr 目录下的所有 .nii.gz 文件并按文件名排序
    images_dir = DATA_ROOT / "imagesTr"
    labels_dir = DATA_ROOT / "labelsTr"
    global_spacing = (1, 1, 1)
    if not images_dir.exists() or not labels_dir.exists():
        print("\n" + "=" * 50)
        print("⚠️ 警告: 未找到 MSD Task09 Spleen 数据集！")
        print(f"请确保将数据集放在: {DATA_ROOT}")
        print("下载地址: http://medicaldecathlon.com/")
        print("解压后应有 imagesTr/ 和 labelsTr/ 文件夹。")
        print("=" * 50)
        # 为了演示，我们仍然可以继续，但会提示缺少 NIfTI 部分
        print("跳过 NIfTI 处理，仅 DICOM 结果可用。")
    else:
        # 获取所有图像文件并排序
        image_files = sorted(glob.glob(str(images_dir / "*.nii.gz")))
        # 取前 3 个
        selected_images = image_files[:3]

        if len(selected_images) < 3:
            print(f"警告: 仅找到 {len(selected_images)} 个病例，少于 3 个。")

        for idx, img_path in enumerate(selected_images, start=1):
            # 构造对应的标签路径
            basename = Path(img_path).stem.replace('.nii', '')  # 去掉 .nii.gz 后缀
            # 注意：MSD 文件名如 "spleen_1.nii.gz"，标签为 "spleen_1.nii.gz"
            # 标签文件在 labelsTr 下，文件名相同
            lbl_path = labels_dir / (basename + ".nii.gz")
            if not lbl_path.exists():
                # 尝试 .nii
                lbl_path = labels_dir / (basename + ".nii")
            if not lbl_path.exists():
                print(f"错误: 找不到标签文件 {lbl_path}")
                continue

            # 处理
            meta = process_nifti_case(img_path, lbl_path, idx)
            all_metadata.append(meta)

    # ---- 3.3 保存所有元数据到 metadata.csv ----
    df = pd.DataFrame(all_metadata)
    csv_path = OUTPUT_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 50)
    print(f"✅ 所有元数据已保存至: {csv_path}")
    print(f"✅ 三视图图片已保存至: {OUTPUT_DIR}")
    print("\n请提交以下文件：")
    print("  - read_medical_image.py (本脚本)")
    print(f"  - {csv_path}")
    print("  - 所有 *_three_views.png 图片")
    print("\n务必在报告/周报中说明：")
    print("  - 对于 NIfTI 病例，图像和标签的 shape/affine 是否一致")
    print("  - 若不一致，可能的原因是什么（如重采样、配准问题）")
