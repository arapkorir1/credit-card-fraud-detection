"""
Unit tests for src.data.
"""

import pandas as pd

from src.data import train_test_split_stratified


def test_train_test_split_stratified_preserves_class_ratio(sample_transactions_df):
    df = pd.concat([sample_transactions_df] * 20, ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        df, test_size=0.25, random_state=42
    )
    assert len(X_train) + len(X_test) == len(df)
    original_ratio = df["Class"].mean()
    train_ratio = y_train.mean()
    assert abs(train_ratio - original_ratio) < 0.05


def test_save_and_load_processed_data_roundtrip(tmp_path, monkeypatch, sample_transactions_df):
    import src.data as data_module

    monkeypatch.setattr(data_module, "TRAIN_DATA_FILE", tmp_path / "train.csv")
    monkeypatch.setattr(data_module, "TEST_DATA_FILE", tmp_path / "test.csv")

    df = pd.concat([sample_transactions_df] * 5, ignore_index=True)
    X_train, X_test, y_train, y_test = data_module.train_test_split_stratified(
        df, test_size=0.2
    )

    data_module.save_processed_data(X_train, X_test, y_train, y_test)
    X_train2, X_test2, y_train2, y_test2 = data_module.load_processed_data()

    assert len(X_train2) == len(X_train)
    assert len(X_test2) == len(X_test)