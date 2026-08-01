"""
src.features

Feature engineering and preprocessing for the credit card fraud dataset.

V1-V28 are already PCA-transformed and anonymized, so no domain-driven
feature engineering is possible on them directly. Feature engineering here
is limited to the two raw columns (Time, Amount):

    - Amount_log: log1p(Amount), since Amount is heavily right-skewed
    - Hour: hour-of-day derived from Time (seconds elapsed), since fraud
      may cluster at certain hours

Also provides:
    - build_preprocessor(): a ColumnTransformer that scales Time/Amount-derived
      columns and passes V1-V28 through untouched (already ~standardized by
      the original PCA)
    - class imbalance handling: get_class_weights(), get_scale_pos_weight(),
      apply_smote()
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.config import RANDOM_STATE

PCA_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
SCALED_COLS = ["Time", "Amount", "Amount_log"]
ENGINEERED_PASSTHROUGH_COLS = ["Hour"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features derived from the raw Time/Amount columns.
    Returns a new dataframe (does not mutate the input).
    """
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    # Time is seconds elapsed since the first transaction in the dataset,
    # which spans ~2 days -> mod 86400 recovers hour-of-day (0-23).
    df["Hour"] = (df["Time"] % 86400) // 3600
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    - RobustScaler on Time, Amount, Amount_log (robust to the extreme
      outliers present in transaction amounts)
    - V1-V28 passed through untouched (already PCA-transformed/roughly
      standardized upstream)
    - Hour passed through untouched (tree models handle a small bounded
      integer range fine; scaling it would only matter for linear/SVM
      models, and even there the effect is minor)
    """
    return ColumnTransformer(
        transformers=[
            ("scaled", RobustScaler(), SCALED_COLS),
            ("pca_passthrough", "passthrough", PCA_FEATURE_COLS),
            ("engineered_passthrough", "passthrough", ENGINEERED_PASSTHROUGH_COLS),
        ],
        remainder="drop",
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return output column names in the order the ColumnTransformer produces them."""
    return SCALED_COLS + PCA_FEATURE_COLS + ENGINEERED_PASSTHROUGH_COLS


def get_class_weights(y: pd.Series) -> dict:
    """
    Compute inverse-frequency class weights: {0: w0, 1: w1}.

    Used for models that accept a class_weight dict (LogisticRegression,
    RandomForest) as an alternative to resampling.
    """
    n_total = len(y)
    n_pos = int(y.sum())
    n_neg = n_total - n_pos
    return {0: n_total / (2 * n_neg), 1: n_total / (2 * n_pos)}


def get_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight = n_negative / n_positive, the convention used
    by XGBoost/LightGBM/CatBoost for binary imbalance.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return n_neg / n_pos


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to the minority (fraud) class.

    IMPORTANT: must only ever be called on TRAINING data, after the
    train/test split, and ideally inside each cross-validation fold (not
    once globally) to avoid leaking synthetic points derived from
    validation-fold neighbors into that fold's evaluation.

    Args:
        X, y: training features/target (already preprocessed/scaled).
        sampling_strategy: minority:majority ratio after resampling. 0.1
            oversamples the minority class to 10% of the majority count —
            a middle ground; full 1:1 balancing tends to overcorrect and
            hurt precision on this dataset.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    return smote.fit_resample(X, y)