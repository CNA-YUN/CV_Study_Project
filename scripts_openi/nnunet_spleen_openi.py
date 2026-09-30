"""
4.3.2 任务 2 —— 启智(OpenI) A100 版 nnU-Net v2 默认 baseline 全流程。

数据集：MSD Task09 Spleen
固定设置：Dataset901_SpleenStudy / 训练集 41 例 / 3d_fullres / fold 0 / 默认 nnUNetPlans
依赖：镜像中已预装（torch(CUDA) + nnunetv2 + nibabel + matplotlib），本脚本不再联网安装。

用法（平台启动命令）：
    bash scripts_openi/run_nnunet_spleen.sh
    # 或
    python scripts_openi/nnunet_spleen_openi.py
    python scripts_openi/nnunet_spleen_openi.py --steps convert verify plan      # 只做数据准备
    python scripts_openi/nnunet_spleen_openi.py --steps train predict evaluate   # 只做训练+评估
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from c2net.context import prepare, upload_output  # noqa: E402

# ----------------------------- 固定设置（勿随意改动） ----------------------------- #
DATASET_ID = 901
DATASET_NAME = "Dataset901_SpleenStudy"
CONFIG = "3d_fullres"
FOLD = 0
NUM_TRAIN = 41
PLANS = "nnUNetPlans"
DEFAULT_TRAINER = "nnUNetTrainer"          # 默认 trainer = 1000 epoch 完整训练
DEFAULT_PP_CONFIGS = ["3d_fullres"]        # 预处理配置（默认只做 baseline 必需的 3d_fullres）

BIN_DIR = Path(sys.executable).parent
PY = sys.executable


# --------------------------------------------------------------------------- #
# 路径 & 运行工具
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.out = Path(output_path) / "m4_task2_nnunet_spleen"
        self.raw = self.out / "nnUNet_raw"
        self.prep = self.out / "nnUNet_preprocessed"
        self.res = self.out / "nnUNet_results"
        self.logs = self.out / "logs"
        self.pred = self.out / "predictions"
        self.viz = self.out / "visualizations"
        self.metrics = self.out / "metrics"
        self.reports = self.out / "reports"
        for d in (self.raw, self.prep, self.res, self.logs, self.pred,
                  self.viz, self.metrics, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        self.dataset_folder = self.raw / DATASET_NAME
        self.prep_folder = self.prep / DATASET_NAME
        self.commands_file = self.out / "commands.txt"

    def env(self) -> dict:
        env = os.environ.copy()
        env["nnUNet_raw"] = str(self.raw)
        env["nnUNet_preprocessed"] = str(self.prep)
        env["nnUNet_results"] = str(self.res)
        env["PYTHONIOENCODING"] = "utf-8"
        return env


def resolve_console_script(name: str):
    exe = BIN_DIR / name
    if exe.exists():
        return [str(exe)]
    found = shutil.which(name)
    if found:
        return [found]
    return None


def run(ctx: Ctx, cmd, log_name: str, env: dict = None, check: bool = True) -> int:
    printable = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 78, flush=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {printable}", flush=True)
    print("=" * 78, flush=True)
    with open(ctx.commands_file, "a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {printable}\n")
    log_path = ctx.logs / f"{log_name}.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# command: {printable}\n\n")
        proc = subprocess.Popen([str(c) for c in cmd], env=env or ctx.env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        fh.write(f"\n# exit code: {proc.returncode}\n# elapsed: {time.time() - t0:.1f} s\n")
    print(f"[done] exit={proc.returncode}, elapsed={time.time() - t0:.1f}s -> {log_path}", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"命令失败 exit={proc.returncode}: {printable}")
    return proc.returncode


# --------------------------------------------------------------------------- #
# step 0: 定位 MSD 源目录（挂载目录为只读，输出一律写入 output_path）
# --------------------------------------------------------------------------- #
def find_msd_root(ctx: Ctx) -> Path:
    def _search(root):
        root = Path(root)
        for j in sorted(root.rglob("dataset.json")):
            p = j.parent
            if (p / "imagesTr").is_dir() and (p / "labelsTr").is_dir():
                return p
        return None

    found = _search(ctx.dataset_path)
    if found is not None:
        return found

    # 兜底：数据集以 tar / zip 形式挂载时先解压到 output_path
    archives = [p for p in sorted(ctx.dataset_path.rglob("*"))
                if p.suffix in (".tar", ".zip", ".gz", ".tgz")]
    for arc in archives:
        dest = ctx.out / "_extracted"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[data] 解压 {arc} -> {dest}")
        if arc.suffix == ".zip":
            with zipfile.ZipFile(arc) as z:
                z.extractall(dest)
        else:
            with tarfile.open(arc) as t:
                t.extractall(dest)
        found = _search(dest)
        if found is not None:
            return found
    raise FileNotFoundError(f"在 {ctx.dataset_path} 下未找到含 imagesTr/labelsTr/dataset.json 的 MSD 目录")


# --------------------------------------------------------------------------- #
# step 1: 转换为 nnU-Net v2 格式 + dataset.json
# --------------------------------------------------------------------------- #
def step_convert(ctx: Ctx, args):
    if (ctx.dataset_folder / "dataset.json").exists() and \
            len(list((ctx.dataset_folder / "imagesTr").glob("*.nii.gz"))) == NUM_TRAIN:
        print(f"[convert] {ctx.dataset_folder} 已存在且完整，跳过转换（避免 ID 冲突断言失败）")
        return
    src = find_msd_root(ctx)
    print(f"[convert] MSD 源目录: {src}")
    # 转换器用文件夹名推断任务名 -> 用符号链接得到 Task09_SpleenStudy
    link_parent = ctx.out / "_msd_src"
    link_parent.mkdir(parents=True, exist_ok=True)
    link = link_parent / "Task09_SpleenStudy"
    if not link.exists():
        os.symlink(src, link, target_is_directory=True)
    print(f"[convert] symlink: {link} -> {src}")

    cmd = resolve_console_script("nnUNetv2_convert_MSD_dataset")
    if cmd is None:  # 兜底：直接调用 API（CLI 的 __main__ 里是硬编码路径，不能用 -m）
        code = ("from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset;"
                f"convert_msd_dataset(r'{link}', {DATASET_ID}, {args.np_pp})")
        cmd = [PY, "-c", code]
    else:
        cmd = cmd + ["-i", str(link), "-overwrite_id", str(DATASET_ID), "-np", str(args.np_pp)]
    run(ctx, cmd, "01_convert_msd_to_nnunet")

    n_img = len(list((ctx.dataset_folder / "imagesTr").glob("*.nii.gz")))
    n_lbl = len(list((ctx.dataset_folder / "labelsTr").glob("*.nii.gz")))
    n_ts = len(list((ctx.dataset_folder / "imagesTs").glob("*.nii.gz")))
    print(f"[convert] imagesTr={n_img}, labelsTr={n_lbl}, imagesTs={n_ts}")
    assert n_img == NUM_TRAIN and n_lbl == NUM_TRAIN, f"训练样本数应为 {NUM_TRAIN}"

    ds = json.loads((ctx.dataset_folder / "dataset.json").read_text(encoding="utf-8"))
    (ctx.out / "dataset.json").write_text(
        json.dumps(ds, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[convert] dataset.json:\n" + json.dumps(ds, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# step 2: 数据完整性检查
# --------------------------------------------------------------------------- #
def step_verify(ctx: Ctx, args):
    code = ("from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity;"
            f"verify_dataset_integrity(r'{ctx.dataset_folder}', {args.np_pp})")
    run(ctx, [PY, "-c", code], "02_verify_dataset_integrity")


# --------------------------------------------------------------------------- #
# step 3: 指纹提取 + 实验规划 + 预处理
# --------------------------------------------------------------------------- #
def step_plan(ctx: Ctx, args):
    cmd = resolve_console_script("nnUNetv2_plan_and_preprocess")
    if cmd is None:
        cmd = [PY, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints"]
    cmd += ["-d", str(DATASET_ID), "--verify_dataset_integrity",
            "-c", *args.pp_configs, "-np", str(args.np_pp), "--no_pbar"]
    run(ctx, cmd, "03_plan_and_preprocess")

    plans_file = ctx.prep_folder / f"{PLANS}.json"
    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    print("[plan] 配置: " + ", ".join(plans["configurations"].keys()))
    cm = plans["configurations"][CONFIG]
    print(f"[plan] {CONFIG}: spacing={cm.get('spacing')}, patch_size={cm.get('patch_size')}, "
          f"batch_size={cm.get('batch_size')}, median_shape={cm.get('median_image_size_in_voxels')}")


# --------------------------------------------------------------------------- #
# step 4: 训练 3d_fullres fold 0
# --------------------------------------------------------------------------- #
def step_train(ctx: Ctx, args):
    cmd = resolve_console_script("nnUNetv2_train")
    if cmd is None:
        cmd = [PY, "-m", "nnunetv2.run.run_training"]
    cmd += [str(DATASET_ID), CONFIG, str(FOLD), "-tr", args.trainer,
            "-p", PLANS, "-device", args.device, "-num_gpus", "1", "--npz"]
    run(ctx, cmd, "04_train_3d_fullres_fold0")
    print(f"[train] 模型目录: {ctx.res / DATASET_NAME / f'{args.trainer}__{PLANS}__{CONFIG}' / f'fold_{FOLD}'}")


# --------------------------------------------------------------------------- #
# step 5: 预测 fold 0 验证集
# --------------------------------------------------------------------------- #
def _prepare_val(ctx: Ctx):
    splits = json.loads((ctx.prep_folder / "splits_final.json").read_text(encoding="utf-8"))
    val_ids = splits[FOLD]["val"]
    val_images = ctx.pred / "val_images"
    val_gt = ctx.pred / "val_gt"
    val_images.mkdir(parents=True, exist_ok=True)
    val_gt.mkdir(parents=True, exist_ok=True)
    for case in val_ids:
        shutil.copy(ctx.dataset_folder / "imagesTr" / f"{case}_0000.nii.gz",
                    val_images / f"{case}_0000.nii.gz")
        shutil.copy(ctx.dataset_folder / "labelsTr" / f"{case}.nii.gz",
                    val_gt / f"{case}.nii.gz")
    print(f"[predict] fold {FOLD} 验证集({len(val_ids)} 例): {val_ids}")
    return val_ids


def step_predict(ctx: Ctx, args):
    val_ids = _prepare_val(ctx)
    val_images = ctx.pred / "val_images"
    val_preds = ctx.pred / "val_preds"
    val_preds.mkdir(parents=True, exist_ok=True)
    model_folder = ctx.res / DATASET_NAME / f"{args.trainer}__{PLANS}__{CONFIG}" / f"fold_{FOLD}"
    chk = "checkpoint_final.pth"
    for cand in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
        if (model_folder / cand).exists():
            chk = cand
            break
    print(f"[predict] 使用 checkpoint: {chk}")

    cmd = resolve_console_script("nnUNetv2_predict")
    if cmd is None:
        raise RuntimeError("未找到 nnUNetv2_predict，请确认 nnunetv2 已正确安装（含 console script）")
    cmd += ["-d", str(DATASET_ID), "-c", CONFIG, "-f", str(FOLD), "-tr", args.trainer,
            "-i", str(val_images), "-o", str(val_preds), "-chk", chk,
            "-device", args.device, "-npp", "3", "-nps", "3", "--disable_progress_bar"]
    run(ctx, cmd, "05_predict_fold0_val")
    print(f"[predict] 预测完成: {len(list(val_preds.glob('*.nii.gz')))} / {len(val_ids)} 例")


# --------------------------------------------------------------------------- #
# step 6: 官方评估
# --------------------------------------------------------------------------- #
def step_evaluate(ctx: Ctx, args):
    val_gt = ctx.pred / "val_gt"
    val_preds = ctx.pred / "val_preds"
    summary = ctx.metrics / "summary.json"
    cmd = resolve_console_script("nnUNetv2_evaluate_folder")
    if cmd is None:
        raise RuntimeError("未找到 nnUNetv2_evaluate_folder，请确认 nnunetv2 已正确安装")
    cmd += [str(val_gt), str(val_preds),
            "-djfile", str(ctx.dataset_folder / "dataset.json"),
            "-pfile", str(ctx.prep_folder / f"{PLANS}.json"),
            "-o", str(summary), "-np", str(args.np_pp)]
    run(ctx, cmd, "06_evaluate_folder")
    print("[evaluate] " + json.dumps(
        json.loads(summary.read_text(encoding="utf-8"))["foreground_mean"], indent=2))


# --------------------------------------------------------------------------- #
# step 7: 预测可视化 + 逐例 Dice
# --------------------------------------------------------------------------- #
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.sum(a & b) / denom) if denom > 0 else float("nan")


def step_visualize(ctx: Ctx, args):
    val_images = ctx.pred / "val_images"
    val_gt = ctx.pred / "val_gt"
    val_preds = ctx.pred / "val_preds"
    pred_files = sorted(val_preds.glob("*.nii.gz"))
    if not pred_files:
        print("[viz] 无预测结果，跳过可视化")
        return

    rows = []
    for pf in pred_files:
        case = pf.name.replace(".nii.gz", "")
        img = nib.load(val_images / f"{case}_0000.nii.gz").get_fdata()
        gt = nib.load(val_gt / f"{case}.nii.gz").get_fdata()
        pr = nib.load(pf).get_fdata()
        d = dice(pr, gt)
        rows.append({"case": case, "dice": d,
                     "gt_voxels": int((gt > 0).sum()), "pred_voxels": int((pr > 0).sum())})

        areas = gt.sum(axis=(1, 2))
        z = int(np.argmax(areas)) if areas.max() > 0 else gt.shape[0] // 2
        lo, hi = np.percentile(img[z], [1, 99])
        sl_img = np.clip(img[z], lo, hi)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        ax[0].imshow(sl_img, cmap="gray")
        ax[0].set_title(f"{case}  CT (axial z={z})")
        ax[1].imshow(sl_img, cmap="gray")
        if gt[z].max() > 0:
            ax[1].contour(gt[z], levels=[0.5], colors="lime", linewidths=1.5)
        ax[1].set_title("Ground Truth")
        ax[2].imshow(sl_img, cmap="gray")
        if gt[z].max() > 0:
            ax[2].contour(gt[z], levels=[0.5], colors="lime", linewidths=1.5)
        if pr[z].max() > 0:
            ax[2].contour(pr[z], levels=[0.5], colors="red", linewidths=1.5)
        ax[2].set_title(f"Prediction (3D Dice={d:.4f})")
        for a in ax:
            a.axis("off")
        plt.suptitle(f"{DATASET_NAME} | {args.trainer}__{PLANS}__{CONFIG} | fold {FOLD} | {case}")
        plt.tight_layout()
        plt.savefig(ctx.viz / f"{case}_pred_overlay.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"[viz] {case}: dice={d:.4f}")

    mean_dice = float(np.nanmean([r["dice"] for r in rows]))
    with open(ctx.metrics / "per_case_dice.csv", "w", encoding="utf-8") as fh:
        fh.write("case,dice,gt_voxels,pred_voxels\n")
        for r in rows:
            fh.write(f"{r['case']},{r['dice']:.6f},{r['gt_voxels']},{r['pred_voxels']}\n")
    (ctx.metrics / "per_case_dice.json").write_text(json.dumps(
        {"model": f"{args.trainer}__{PLANS}__{CONFIG}", "fold": FOLD,
         "num_cases": len(rows), "mean_dice": mean_dice, "per_case": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[viz] 平均 Dice = {mean_dice:.4f} (n={len(rows)})")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([r["case"] for r in rows], [r["dice"] for r in rows], color="#4C72B0")
    ax.axhline(mean_dice, ls="--", color="crimson", label=f"mean = {mean_dice:.4f}")
    ax.set_ylabel("Dice (spleen)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-case Dice — {args.trainer} — fold {FOLD} — A100")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ctx.viz / "per_case_dice.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# step 8: 生成报告
# --------------------------------------------------------------------------- #
def step_report(ctx: Ctx, args):
    dice_json = ctx.metrics / "per_case_dice.json"
    summary_json = ctx.metrics / "summary.json"
    mean_dice = None
    if dice_json.exists():
        mean_dice = json.loads(dice_json.read_text(encoding="utf-8"))["mean_dice"]
    official = ""
    if summary_json.exists():
        official = json.dumps(json.loads(summary_json.read_text(encoding="utf-8"))["foreground_mean"],
                              indent=2, ensure_ascii=False)
    md = f"""# 默认 baseline（启智 A100）— Dataset901_SpleenStudy / 3d_fullres / fold 0

> 仅呈现默认 baseline（未修改 plans / trainer / 数据增强）。修改版结果需另起一节。

## 固定设置
| 项目 | 取值 |
| --- | --- |
| 数据集 | MSD Task09 Spleen（41 训练 / 20 测试，测试无标签） |
| 任务名 | {DATASET_NAME}（ID {DATASET_ID}） |
| 配置 | {CONFIG}，plans = {PLANS}（nnU-Net 自动规划） |
| fold | {FOLD} |
| trainer | {args.trainer} |
| device | {args.device} |
| 预处理配置 | {', '.join(args.pp_configs)}，进程数 {args.np_pp} |

## 结果（fold 0 验证集）
平均 Dice(spleen) = {mean_dice if mean_dice is not None else 'N/A'}

官方 `evaluate_folder` foreground_mean：
```
{official if official else 'N/A'}
```

## 产物
- `dataset.json`、`commands.txt`
- `logs/` 每步日志
- `metrics/summary.json`、`metrics/per_case_dice.csv`
- `visualizations/` 预测叠加图与逐例 Dice 柱状图
- `nnUNet_results/{DATASET_NAME}/{args.trainer}__{PLANS}__{CONFIG}/fold_{FOLD}/`
"""
    (ctx.reports / "baseline_report.md").write_text(md, encoding="utf-8")
    print(f"[report] -> {ctx.reports / 'baseline_report.md'}")


STEPS = {
    "convert": step_convert,
    "verify": step_verify,
    "plan": step_plan,
    "train": step_train,
    "predict": step_predict,
    "evaluate": step_evaluate,
    "visualize": step_visualize,
    "report": step_report,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS.keys()),
                    choices=list(STEPS.keys()) + ["all"])
    ap.add_argument("--trainer", default=DEFAULT_TRAINER,
                    help="默认 nnUNetTrainer(1000 epoch)；时间受限可用 nnUNetTrainer_250epochs")
    ap.add_argument("--device", default=None, help="默认 cuda（不可用时回落 cpu）")
    ap.add_argument("--np-pp", type=int, default=8, help="预处理/验证进程数")
    ap.add_argument("--pp-configs", nargs="+", default=DEFAULT_PP_CONFIGS)
    args = ap.parse_args()

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"[env] torch={torch.__version__}, cuda_available={cuda}, "
              f"gpu={torch.cuda.get_device_name(0) if cuda else '-'}")
        if args.device is None:
            args.device = "cuda" if cuda else "cpu"
    except Exception as e:  # pragma: no cover
        print(f"[env] torch 检查失败: {e}")
        args.device = args.device or "cpu"

    c2net_context = prepare()
    ctx = Ctx(c2net_context.dataset_path, c2net_context.output_path)
    print(f"[ctx] dataset_path = {ctx.dataset_path}")
    print(f"[ctx] output_path  = {ctx.out}")
    (ctx.out / "run_config.json").write_text(json.dumps(
        {"dataset_id": DATASET_ID, "dataset_name": DATASET_NAME, "config": CONFIG,
         "fold": FOLD, "num_train": NUM_TRAIN, "trainer": args.trainer,
         "plans": PLANS, "device": args.device, "np_pp": args.np_pp,
         "pp_configs": args.pp_configs}, indent=2, ensure_ascii=False), encoding="utf-8")

    steps = list(STEPS.keys()) if args.steps == ["all"] else list(args.steps)
    try:
        for name in steps:
            print(f"\n########## STEP: {name} ##########", flush=True)
            STEPS[name](ctx, args)
        print("\n全部步骤完成，开始回传 output ...", flush=True)
    finally:
        # 即使中途失败也回传，保证日志/中间产物能取回
        upload_output()


if __name__ == "__main__":
    main()
