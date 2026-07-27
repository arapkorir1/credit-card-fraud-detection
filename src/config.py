"""
Central configuration: filesystem paths and project-wide constants.

All other modules and notebooks should import paths from here rather than
hardcoding strings, so the project can be moved/renamed without breaking
anything.
"""

from pathlib import Path

# --- Root ---
# config.py lives at src/config.py, so parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Data ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_FILE = DATA_RAW_DIR / "creditcard.csv"
TRAIN_DATA_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_FILE = DATA_PROCESSED_DIR / "test.csv"

# --- Models ---
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.cbm"           # CatBoost native format
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"  # tuned decision threshold

# --- Reports ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# --- Dataset constants ---
TARGET_COL = "Class"
RAW_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Ensure directories exist at import time ---
for _dir in [
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)