"""
src.data

Handles acquisition and loading of the raw credit card fraud dataset.

Currently implements:
    - download_raw_data(): pulls the dataset from Kaggle via the Kaggle API.

TODO (Phase 4): load_raw_data(), train_test_split_stratified(), save_processed_data()
"""

import subprocess
import sys

from src.config import DATA_RAW_DIR, RAW_DATA_FILE, TARGET_COL

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"


def download_raw_data(force: bool = False) -> None:
    """
    Download the credit card fraud dataset from Kaggle into data/raw/.

    Requires a valid ~/.kaggle/kaggle.json API token (see README setup
    instructions). Shells out to the `kaggle` CLI rather than the kaggle
    Python API, since the CLI handles auth and zip extraction more predictably.

    Args:
        force: if True, re-download even if the raw file already exists.
    """
    if RAW_DATA_FILE.exists() and not force:
        print(
            f"Raw data already exists at {RAW_DATA_FILE}, skipping download. "
            f"Pass force=True to re-download."
        )
        return

    print(f"Downloading {KAGGLE_DATASET} from Kaggle into {DATA_RAW_DIR} ...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_RAW_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "Kaggle download failed. Check that ~/.kaggle/kaggle.json exists, "
            "is valid, and has permissions 600 (see README setup instructions)."
        )

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Download completed but expected file not found at {RAW_DATA_FILE}. "
            f"The dataset's internal filename may have changed — check the "
            f"contents of {DATA_RAW_DIR}."
        )

    print(f"✅ Downloaded successfully: {RAW_DATA_FILE}")
    _print_summary()


def _print_summary() -> None:
    """Quick sanity-check printout after download: shape and class balance."""
    import pandas as pd

    df = pd.read_csv(RAW_DATA_FILE)
    n_total = len(df)
    n_fraud = int(df[TARGET_COL].sum())

    print(f"\nShape: {df.shape}")
    print(f"Fraud cases: {n_fraud:,} / {n_total:,} ({n_fraud / n_total * 100:.3f}%)")


if __name__ == "__main__":
    download_raw_data()