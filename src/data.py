"""
src.data

Handles acquisition, splitting, and I/O of the credit card fraud dataset.

Functions:
    download_raw_data()          - pull the dataset from Kaggle
    load_raw_data()               - read data/raw/creditcard.csv into a DataFrame
    train_test_split_stratified() - stratified split preserving fraud ratio
    save_processed_data()         - persist train/test splits to data/processed/
    load_processed_data()         - read back the persisted splits
"""

import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_DIR,
    RAW_DATA_FILE,
    TARGET_COL,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)

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
    df = pd.read_csv(RAW_DATA_FILE)
    n_total = len(df)
    n_fraud = int(df[TARGET_COL].sum())
    print(f"\nShape: {df.shape}")
    print(f"Fraud cases: {n_fraud:,} / {n_total:,} ({n_fraud / n_total * 100:.3f}%)")


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from disk. Raises if it hasn't been downloaded yet."""
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_FILE} not found. Run `make download-data` first."
        )
    return pd.read_csv(RAW_DATA_FILE)


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split on the target column, preserving the
    ~0.17% fraud ratio in both splits.

    Note: this is a random stratified split, the standard approach for this
    dataset in the literature. A temporal split (train on earlier `Time`,
    test on later `Time`) is arguably more realistic for production fraud
    detection but leaves too few fraud cases in a held-out tail given the
    dataset only spans ~2 days — we use random stratified split here and
    call this trade-off out explicitly rather than pretending it doesn't exist.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Persist train/test splits (features + target recombined) to data/processed/."""
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    print(f"✅ Saved {TRAIN_DATA_FILE} ({len(train_df):,} rows)")
    print(f"✅ Saved {TEST_DATA_FILE} ({len(test_df):,} rows)")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the previously persisted train/test splits."""
    if not TRAIN_DATA_FILE.exists() or not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `make prepare-data` first."
        )
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    download_raw_data()