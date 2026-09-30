"""
Task 4.3.2 —— nnU-Net v2 数据准备与默认 baseline 流水线驱动脚本。

数据集：MSD Task09 Spleen（41 例训练 / 20 例测试）
固定设置：Dataset901_SpleenStudy，41 例训练，固定 fold 0，配置 3d_fullres
资源受限替代：本机仅为 CPU（torch 2.13.0+cpu，无 CUDA），
              完整 1000 epoch 的 3d_fullres 训练不可行，
              因此训练阶段使用 1 epoch smoke test（nnUNetTrainerSmoke1Epoch），
              并在报告中明确标注"未完成完整训练"。

用法：
    python run_nnunet_pipeline.py convert     # MSD -> nnU-Net v2 格式 + dataset.json
    python run_nnunet_pipeline.py verify      # 数据完整性检查
    python run_nnunet_pipeline.py plan        # 指纹提取 + 实验规划 + 预处理
    python run_nnunet_pipeline.py train       # 3d_fullres fold 0 训练（1 epoch smoke test）
    python run_nnunet_pipeline.py predict     # 对 fold 0 验证集做预测
    python run_nnunet_pipeline.py evaluate    # 官方 evaluate_folder 评估
    python run_nnunet_pipeline.py all         # 顺序执行以上全部
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nnunet_env import (  # noqa: E402
    CONFIG, DATASET_FOLDER, DATASET_ID, FOLD, LOG_DIR, METRICS_DIR,
    MODEL_FOLDER, NUM_TRAIN, PRED_DIR, PREP_FOLDER, SOURCE_MSD, TASK_DIR,
    TRAINER, VENV_BIN, VENV_PY, build_env, ensure_dirs, run,
)

ensure_dirs()
ENV = build_env()

VAL_IMAGES = PRED_DIR / "val_images"
VAL_GT = PRED_DIR / "val_gt"
VAL_PREDS = PRED_DIR / "val_preds"


# --------------------------------------------------------------------------- #
# step 1: 数据格式转换 + dataset.json
# --------------------------------------------------------------------------- #
def step_convert():
    link_parent = TASK_DIR / "_msd_src"
    link_parent.mkdir(parents=True, exist_ok=True)
    link = link_parent / "Task09_SpleenStudy"
    # MSD 转换脚本用文件夹名推断任务名 -> 用目录联接(junction)得到 Task09_SpleenStudy
    if not link.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(SOURCE_MSD)],
                       check=True, capture_output=True, text=True)
    print(f"[convert] 源目录联接: {link} -> {SOURCE_MSD}")

    run([VENV_BIN / "nnUNetv2_convert_MSD_dataset.exe",
         "-i", link, "-overwrite_id", DATASET_ID, "-np", 4],
        "01_convert_msd_to_nnunet", env=ENV)

    # 校验产物并归档 dataset.json
    ds_json = DATASET_FOLDER / "dataset.json"
    assert ds_json.exists(), f"dataset.json 未生成: {ds_json}"
    n_tr_img = len(list((DATASET_FOLDER / "imagesTr").glob("*.nii.gz")))
    n_tr_lbl = len(list((DATASET_FOLDER / "labelsTr").glob("*.nii.gz")))
    n_ts = len(list((DATASET_FOLDER / "imagesTs").glob("*.nii.gz")))
    print(f"[convert] imagesTr={n_tr_img}, labelsTr={n_tr_lbl}, imagesTs={n_ts}")
    assert n_tr_img == NUM_TRAIN and n_tr_lbl == NUM_TRAIN, "训练样本数不为 41，请检查"
    shutil.copy(ds_json, TASK_DIR / "dataset.json")
    print(f"[convert] dataset.json 已归档 -> {TASK_DIR / 'dataset.json'}")
    print(json.dumps(json.loads(ds_json.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# step 2: 数据完整性检查
# --------------------------------------------------------------------------- #
def step_verify():
    code = (
        "from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity;"
        f"verify_dataset_integrity(r'{DATASET_FOLDER}', 4)"
    )
    run([VENV_PY, "-c", code], "02_verify_dataset_integrity", env=ENV)


# --------------------------------------------------------------------------- #
# step 3: 指纹提取 + 实验规划 + 预处理
# --------------------------------------------------------------------------- #
def step_plan():
    # 注意：-np 1。本机 16GB 内存，多进程并发重采样 CT 会 OOM（已实测 -np 4 失败）。
    # 只预处理基线必需的 3d_fullres；2d / 3d_lowres 可用 `preprocess` 步骤单独补齐。
    run([VENV_BIN / "nnUNetv2_plan_and_preprocess.exe",
         "-d", DATASET_ID, "--verify_dataset_integrity", "-c", CONFIG, "-np", 1, "--no_pbar"],
        "03_plan_and_preprocess", env=ENV)
    plans_file = PREP_FOLDER / "nnUNetPlans.json"
    assert plans_file.exists(), f"plans 文件未生成: {plans_file}"
    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    print("[plan] 生成的配置: " + ", ".join(plans["configurations"].keys()))
    cm = plans["configurations"][CONFIG]
    print(f"[plan] {CONFIG}: spacing={cm.get('spacing')}, patch_size={cm.get('patch_size')}, "
          f"batch_size={cm.get('batch_size')}, median_shape={cm.get('median_image_size_in_voxels')}")


# --------------------------------------------------------------------------- #
# step 3b: 单独补齐其它配置的预处理（可选，非基线必需）
# --------------------------------------------------------------------------- #
def step_preprocess(configs=None, num_processes=1):
    configs = configs or ["2d", "3d_lowres"]
    run([VENV_BIN / "nnUNetv2_preprocess.exe",
         "-d", DATASET_ID, "-plans_name", "nnUNetPlans",
         "-c", *configs, "-np", num_processes, "--no_pbar"],
        "03b_preprocess_extra_configs", env=ENV)


# --------------------------------------------------------------------------- #
# step 4: 训练（3d_fullres, fold 0；CPU 受限 -> 1 epoch smoke test）
# --------------------------------------------------------------------------- #
def step_train():
    run([VENV_BIN / "nnUNetv2_train.exe", DATASET_ID, CONFIG, FOLD,
         "-tr", TRAINER, "-device", "cpu", "--npz"],
        "04_train_3d_fullres_fold0_smoke", env=ENV)
    ckpt = MODEL_FOLDER / "checkpoint_final.pth"
    print(f"[train] checkpoint_final.pth 存在: {ckpt.exists()} ({ckpt})")


# --------------------------------------------------------------------------- #
# step 5: 用 fold 0 模型预测验证集
# --------------------------------------------------------------------------- #
def _prepare_val_folders():
    splits = json.loads((PREP_FOLDER / "splits_final.json").read_text(encoding="utf-8"))
    val_ids = splits[FOLD]["val"]
    print(f"[predict] fold {FOLD} 验证集({len(val_ids)} 例): {val_ids}")
    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_GT.mkdir(parents=True, exist_ok=True)
    for case in val_ids:
        src_img = DATASET_FOLDER / "imagesTr" / f"{case}_0000.nii.gz"
        src_lbl = DATASET_FOLDER / "labelsTr" / f"{case}.nii.gz"
        shutil.copy(src_img, VAL_IMAGES / src_img.name)
        shutil.copy(src_lbl, VAL_GT / src_lbl.name)
    return val_ids


def step_predict():
    val_ids = _prepare_val_folders()
    VAL_PREDS.mkdir(parents=True, exist_ok=True)
    run([VENV_BIN / "nnUNetv2_predict.exe",
         "-d", DATASET_ID, "-c", CONFIG, "-f", FOLD, "-tr", TRAINER,
         "-i", VAL_IMAGES, "-o", VAL_PREDS,
         "-chk", "checkpoint_final.pth", "-device", "cpu",
         "--disable_tta", "-npp", 1, "-nps", 1, "--disable_progress_bar"],
        "05_predict_fold0_val", env=ENV)
    print(f"[predict] 预测结果: {len(list(VAL_PREDS.glob('*.nii.gz')))} / {len(val_ids)} 例")


# --------------------------------------------------------------------------- #
# step 6: 官方评估
# --------------------------------------------------------------------------- #
def step_evaluate():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    run([VENV_BIN / "nnUNetv2_evaluate_folder.exe",
         VAL_GT, VAL_PREDS,
         "-djfile", DATASET_FOLDER / "dataset.json",
         "-pfile", PREP_FOLDER / "nnUNetPlans.json",
         "-o", METRICS_DIR / "summary.json", "-np", 4],
        "06_evaluate_folder", env=ENV)
    summary = json.loads((METRICS_DIR / "summary.json").read_text(encoding="utf-8"))
    print("[evaluate] " + json.dumps(summary["foreground_mean"], indent=2, ensure_ascii=False))


STEPS = {
    "convert": step_convert,
    "verify": step_verify,
    "plan": step_plan,
    "preprocess": step_preprocess,
    "train": step_train,
    "predict": step_predict,
    "evaluate": step_evaluate,
}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args[0] == "preprocess":
        step_preprocess(args[1:], int(args[-1]) if args[-1].isdigit() else 1)
        raise SystemExit(0)
    order = list(STEPS.keys())[:3] + list(STEPS.keys())[4:] if args[0] == "all" else args
    for name in order:
        if name not in STEPS:
            raise SystemExit(f"未知步骤: {name}，可选: {list(STEPS) + ['all']}")
        print(f"\n########## STEP: {name} ##########")
        STEPS[name]()
    print("\n全部步骤完成。日志目录: " + str(LOG_DIR))
