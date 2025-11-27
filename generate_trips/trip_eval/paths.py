# trip_eval/paths.py
from pathlib import Path

# 项目根目录：.../OCR_test_2
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"