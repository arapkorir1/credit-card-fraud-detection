"""
Unit tests for src.features.
"""

import numpy as np
import pandas as pd

from src.features import (
    add_engineered_features,
    apply_smote,
    build_preprocessor,
    get_class_weights,
    get_scale_pos_weight,
)


def test_add_engineered_features_creates_expected_columns(sample_transactions_df):
    df = add_engineered_features(sample_transactions_df)
    assert "Amount_log" in df.columns
    assert "Hour" in df.columns
    assert np.allclose(df["Amount_log"], np.log1p(sample_transactions_df["Amount"]))


def test_add_engineered_features_does_not_mutate_input(sample_transactions_df):
    original_cols = list(sample_transactions_df.columns)
    _ = add_engineered_features(sample_transactions_df)
    assert list(sample_transactions_df.columns) == original_cols


def test_build_preprocessor_output_shape(sample_transactions_df):
    df = add_engineered_features(sample_transactions_df)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    # 28 PCA cols + Time, Amount, Amount_log + Hour = 32 columns
    assert X.shape == (len(df), 32)


def test_get_class_weights_inversely_proportional():
    y = pd.Series([0] * 990 + [1] * 10)
    weights = get_class_weights(y)
    assert weights[1] > weights[0]


def test_get_scale_pos_weight():
    y = pd.Series([0] * 99 + [1] * 1)
    assert get_scale_pos_weight(y) == 99.0


def test_apply_smote_balances_minority_class():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([0] * 190 + [1] * 10)

    X_res, y_res = apply_smote(X, y, sampling_strategy=0.5)

    assert y_res.sum() > y.sum()
    assert (y_res == 0).sum() == (y == 0).sum()