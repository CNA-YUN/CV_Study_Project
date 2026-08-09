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

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
BASE_ROOT = Path.cwd().parent  # 当前工作目录，你也可以改成绝对路径
DATA_ROOT = BASE_ROOT / "data" / "Task09_Spleen" / "Task09_Spleen"  # MSD 数据集根目录
OUTPUT_DIR = BASE_ROOT / "outputs" / "m3_task1_dicom_nifti"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"数据根目录: {DATA_ROOT}")
print(f"输出目录: {OUTPUT_DIR}")


# ==================== 通用工具函数 ====================
def get_foreground_center(volume, label, axis):
    """
    获取包含前景（label>0）的中心切片坐标
    axis: 0=冠状(Y轴投影), 1=矢状(X轴投影), 2=轴向(Z轴投影)
    """
    # 计算每个切片上是否有前景
    if axis == 0:  # 冠状面，沿 Y 轴投影
        proj = np.sum(label, axis=(1, 2))
    elif axis == 1:  # 矢状面，沿 X 轴投影
        proj = np.sum(label, axis=(0, 2))
    elif axis == 2:  # 轴向面，沿 Z 轴投影
        proj = np.sum(label, axis=(0, 1))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    # 找到有前景的切片索引
    fg_indices = np.where(proj > 0)[0]
    if len(fg_indices) == 0:
        # 如果没有前景，取体积中间切片
        return volume.shape[axis] // 2
    # 取前景区域的中间索引
    return fg_indices[len(fg_indices) // 2]


def plot_three_views(image, label, case_name, title_prefix="", is_2d=False):
    """
    绘制 axial / coronal / sagittal 三视图
    image: 3D numpy 数组 (Z, Y, X)
    label: 3D numpy 数组
    is_2d: 如果为 True，则所有视图都显示同一张切片（用于 DICOM 单切片）
    """
    if is_2d or image.ndim == 2:
        # 如果是 2D 图像，所有视图显示同一张切片的不同旋转/转置
        if image.ndim == 2:
            img_slice = image
            lbl_slice = label if label.ndim == 2 else label[0]
        else:
            # 取中间切片
            mid = image.shape[0] // 2
            img_slice = image[mid]
            lbl_slice = label[mid] if label.ndim == 3 else label

        # 生成三张图：轴向显示原图，冠状显示旋转90度，矢状显示旋转-90度
        slices = [
            (img_slice, lbl_slice, "Axial (Z)"),
            (np.rot90(img_slice, k=1), np.rot90(lbl_slice, k=1), "Coronal (Y, rotated)"),
            (np.rot90(img_slice, k=-1), np.rot90(lbl_slice, k=-1), "Sagittal (X, rotated)"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (img_s, lbl_s, title) in zip(axes, slices):
            # 归一化显示
            img_display = np.clip(img_s, 0, np.percentile(img_s, 99))
            if img_display.max() - img_display.min() > 1e-6:
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
            ax.imshow(img_display, cmap='gray')
            # 标签轮廓（红色）
            if np.sum(lbl_s) > 0:
                from skimage import measure
                contours = measure.find_contours(lbl_s, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f"{title_prefix} {case_name} (2D DICOM)", fontsize=14)
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{case_name}_three_views.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> 三视图已保存: {save_path}")
        return


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
    print("\n请提交以下文件给老师验收：")
    print("  - read_medical_image.py (本脚本)")
    print(f"  - {csv_path}")
    print("  - 所有 *_three_views.png 图片")
    print("\n务必在报告/周报中说明：")
    print("  - 对于 NIfTI 病例，图像和标签的 shape/affine 是否一致")
    print("  - 若不一致，可能的原因是什么（如重采样、配准问题）")
