#!/usr/bin/env bash
# 4.3.2 任务 2 —— 启智(OpenI) A100 启动脚本
# 依赖已在镜像中预装（torch+CUDA / nnunetv2 / nibabel / matplotlib），此处只做环境自检与启动。
set -euo pipefail

echo "========== Environment check =========="
python -V || python3 -V
nvidia-smi || true

python - <<'PY'
import importlib.metadata as md
import torch

print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0),
          "| mem(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1))
for p in ("nnunetv2", "nibabel", "matplotlib", "SimpleITK"):
    try:
        print(f"{p:<11}:", md.version(p))
    except Exception as e:
        print(f"{p:<11}: NOT FOUND ({e})")
PY

echo "========== Start nnU-Net pipeline =========="
python scripts_openi/nnunet_spleen_openi.py "$@"
