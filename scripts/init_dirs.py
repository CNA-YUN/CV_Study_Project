from os import makedirs
from pathlib import Path
BASE_DIR = Path().cwd().parent
makedirs(BASE_DIR/'data', exist_ok=True)
makedirs(BASE_DIR/'scripts', exist_ok=True)
makedirs(BASE_DIR/'outputs', exist_ok=True)
makedirs(BASE_DIR/'src', exist_ok=True)
makedirs(BASE_DIR/'config', exist_ok=True)
makedirs(BASE_DIR/'result', exist_ok=True)
