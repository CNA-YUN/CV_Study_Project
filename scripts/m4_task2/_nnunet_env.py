"""Task 4.3.2 公共路径与运行环境配置（MSD Task09 Spleen -> Dataset901_SpleenStudy）。"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- 任务常量（固定设置，不可随意改动）----
DATASET_ID = 901
DATASET_NAME = "Dataset901_SpleenStudy"
CONFIG = "3d_fullres"
FOLD = 0
NUM_TRAIN = 41
TRAINER = "nnUNetTrainerSmoke1Epoch"      # CPU 受限下的 1 epoch smoke test trainer
PLANS = "nnUNetPlans"

# ---- 目录 ----
TASK_DIR = ROOT / "outputs" / "m4_task2_nnunet_spleen"
RAW_DIR = TASK_DIR / "nnUNet_raw"
PREP_DIR = TASK_DIR / "nnUNet_preprocessed"
RES_DIR = TASK_DIR / "nnUNet_results"
LOG_DIR = TASK_DIR / "logs"
PRED_DIR = TASK_DIR / "predictions"
VIZ_DIR = TASK_DIR / "visualizations"
METRICS_DIR = TASK_DIR / "metrics"
REPORT_DIR = TASK_DIR / "reports"

DATASET_FOLDER = RAW_DIR / DATASET_NAME
PREP_FOLDER = PREP_DIR / DATASET_NAME
MODEL_FOLDER = RES_DIR / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{FOLD}"

SOURCE_MSD = ROOT / "data" / "Task09_Spleen" / "Task09_Spleen"

VENV_DIR = ROOT / ".venv_nnunet"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
VENV_BIN = VENV_DIR / "Scripts"
EXT_TRAINER_DIR = Path(__file__).resolve().parent / "external_trainers"


def ensure_dirs():
    for d in (RAW_DIR, PREP_DIR, RES_DIR, LOG_DIR, PRED_DIR, VIZ_DIR, METRICS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def build_env() -> dict:
    """构造供 nnU-Net 各命令使用的环境变量。"""
    ensure_dirs()
    env = os.environ.copy()
    env["nnUNet_raw"] = str(RAW_DIR)
    env["nnUNet_preprocessed"] = str(PREP_DIR)
    env["nnUNet_results"] = str(RES_DIR)
    env["nnUNet_extTrainer"] = str(EXT_TRAINER_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "4"
    return env


def run(cmd, log_name: str, env: dict = None, cwd=None, check: bool = True) -> int:
    """执行命令，实时输出到终端并同步写入日志文件。"""
    ensure_dirs()
    log_path = LOG_DIR / f"{log_name}.log"
    printable = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 78)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] $ {printable}")
    print("=" * 78, flush=True)
    t0 = time.time()
    with open(TASK_DIR / "commands.txt", "a", encoding="utf-8") as cmds:
        cmds.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {printable}\n")
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# command: {printable}\n")
        fh.write(f"# start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env or build_env(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        fh.write(f"\n# exit code: {proc.returncode}\n")
        fh.write(f"# elapsed: {time.time() - t0:.1f} s\n")
    print(f"[done] exit={proc.returncode}, elapsed={time.time() - t0:.1f}s, log -> {log_path}")
    if check and proc.returncode != 0:
        raise RuntimeError(f"命令失败（exit={proc.returncode}）: {printable}")
    return proc.returncode
