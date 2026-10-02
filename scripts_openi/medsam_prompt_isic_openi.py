# -*- coding: utf-8 -*-
"""
任务 4.3.3 —— MedSAM 点/框提示医学分割比较（OpenI / 启智平台版）

与本地版 scripts/m4_task3/medsam_prompt_isic.py 完全共用核心逻辑（提示生成、预处理、
一次编码三次解码、指标、叠加图），本文件只负责：
  1. 通过 c2net 上下文解析平台挂载的数据集目录与输出目录
  2. 自动定位 ISIC 图像目录 / GT 掩膜目录 / 清单 CSV / MedSAM 权重（必要时自动下载）
  3. 组装参数并调用 run_evaluation()
  4. upload_output() 回传结果

【上传要求】在启智平台创建任务时需上传整个仓库目录结构，至少包含：
    scripts/m4_task3/medsam_prompt_isic.py     主逻辑
    scripts_openi/medsam_prompt_isic_openi.py  本入口
    scripts_openi/isic_inventory.csv           seed=42 的划分清单（只用 case_id 与 split）
    external/MedSAM/segment_anything/          SAM 源码（若随数据集上传也可自动找到）
    权重 medsam_vit_b.pth                      放代码目录 / ckpts/ / 数据集目录均可，找不到会自动 wget

【重要声明 —— 交互式上界设置】
GT box 由真值 mask 外接框 ±10px 生成，前景中心点与负点同样由真值推导，三者均属于
Oracle Prompt / 交互式上界协议，衡量的是"提示最优时 MedSAM 的能力上限"，
并非完全自动分割性能。
"""

import subprocess
import sys
from pathlib import Path

import torch
from c2net.context import prepare, upload_output

HERE = Path(__file__).resolve().parent          # .../scripts_openi
REPO_ROOT = HERE.parent                         # 仓库根
c2net_context = prepare()
DATASET_PATH = Path(c2net_context.dataset_path)
OUTPUT_PATH = Path(c2net_context.output_path)

CKPT_NAME = "medsam_vit_b.pth"
CKPT_URLS = [
    "https://hf-mirror.com/spaces/bowang-lab/MedSAM/resolve/main/medsam_vit_b.pth",
    "https://huggingface.co/spaces/bowang-lab/MedSAM/resolve/main/medsam_vit_b.pth",
]


# --------------------------------------------------------------------------- #
# 0. 先把 SAM 源码目录放进 sys.path，再导入主逻辑模块
# --------------------------------------------------------------------------- #
def find_segment_anything_root():
    """定位包含 segment_anything/ 的目录（其父目录需要加入 sys.path）。"""
    candidates = [REPO_ROOT / "external" / "MedSAM"]
    for root in [DATASET_PATH, OUTPUT_PATH, HERE]:
        if root.exists():
            for d in root.rglob("segment_anything"):
                if (d / "build_sam.py").exists():
                    candidates.append(d.parent)
    for c in candidates:
        if (c / "segment_anything" / "build_sam.py").exists():
            return c
    return None


sa_root = find_segment_anything_root()
if sa_root is None:
    sys.exit("[ERROR] 未找到 segment_anything 源码目录，请确认已上传 external/MedSAM/")
sys.path.insert(0, str(sa_root))
sys.path.insert(0, str(HERE.parent / "m4_task3"))

from medsam_prompt_isic import parse_args, run_evaluation  # noqa: E402

print(f"[env] segment_anything: {sa_root}")
print(f"[env] dataset_path: {DATASET_PATH}")
print(f"[env] output_path : {OUTPUT_PATH}")


# --------------------------------------------------------------------------- #
# 1. 自动定位数据与权重
# --------------------------------------------------------------------------- #
def find_dir_most_files(roots, predicate, max_depth=5):
    """在若干根目录搜索"符合 predicate 的文件最多"的目录。"""
    best, best_n = None, 0
    for root in roots:
        if not root or not Path(root).exists():
            continue
        root = Path(root)
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            if len(d.relative_to(root).parts) > max_depth:
                continue
            try:
                n = sum(1 for p in d.iterdir() if p.is_file() and predicate(p))
            except OSError:
                continue
            if n > best_n:
                best, best_n = d, n
    return best, best_n


def is_image(p):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "_segmentation" not in p.name


def is_mask(p):
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name.endswith("_segmentation.png")


def resolve_inventory():
    for cand in [HERE / "isic_inventory.csv",
                 REPO_ROOT / "outputs" / "m3_task2_isic2018_check" / "isic_inventory.csv"]:
        if cand.exists():
            return cand
    for root in [DATASET_PATH, OUTPUT_PATH]:
        if Path(root).exists():
            for p in Path(root).rglob("isic_inventory.csv"):
                return p
    return None


def resolve_ckpt():
    """优先用已上传的权重，找不到则 wget 下载到代码目录。"""
    for cand in [HERE / CKPT_NAME,
                 REPO_ROOT / "ckpts" / CKPT_NAME,
                 Path.cwd() / CKPT_NAME]:
        if cand.exists():
            return cand
    for root in [DATASET_PATH, OUTPUT_PATH]:
        if Path(root).exists():
            for p in Path(root).rglob(CKPT_NAME):
                return p
    dst = HERE / CKPT_NAME
    for url in CKPT_URLS:
        print(f"[ckpt] 未找到本地权重，尝试下载: {url}")
        ret = subprocess.run(["wget", "-q", "-O", str(dst), url], check=False)
        if ret.returncode == 0 and dst.exists() and dst.stat().st_size > 100 * 1024 * 1024:
            print(f"[ckpt] 下载完成: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
            return dst
        print(f"[ckpt] 下载失败(returncode={ret.returncode})，尝试下一个源")
    sys.exit(f"[ERROR] 无法获取 {CKPT_NAME}，请随代码或数据集上传该权重")


img_dir, n_img = find_dir_most_files([DATASET_PATH, OUTPUT_PATH], is_image)
mask_dir, n_mask = find_dir_most_files([DATASET_PATH, OUTPUT_PATH], is_mask)
inventory = resolve_inventory()
ckpt = resolve_ckpt()

if img_dir is None or mask_dir is None or inventory is None:
    sys.exit(f"[ERROR] 数据定位失败: image_dir={img_dir}({n_img}) "
             f"mask_dir={mask_dir}({n_mask}) inventory={inventory}")

print(f"[data] 图像目录: {img_dir} （{n_img} 个文件）")
print(f"[data] 掩膜目录: {mask_dir} （{n_mask} 个文件）")
print(f"[data] 清单 CSV: {inventory}")
print(f"[ckpt] 权重:     {ckpt}")


# --------------------------------------------------------------------------- #
# 2. 组装参数并跑评测
# --------------------------------------------------------------------------- #
args = parse_args()
args.inventory = str(inventory)
args.image_dir = str(img_dir)
args.mask_dir = str(mask_dir)
args.checkpoint = str(ckpt)
args.output_dir = str(OUTPUT_PATH / "m4_task3_medsam_prompt_isic")
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
args.make_vis = True                      # 平台上必须出 20 组叠加图
args.num_threads = 0                      # 平台上不限制线程数
print(f"[run] device={args.device}  output_dir={args.output_dir}")

run_evaluation(args)

# 3. 回传结果
upload_output()
print("[done] 结果已通过 upload_output() 回传")
