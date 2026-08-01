"""
Unit tests for src.predict.
"""

import json

import joblib
import pandas as pd
import pytest

from src.features import add_engineered_features, build_preprocessor
from src.models import get_model, save_model
from src.predict import FraudPredictor


@pytest.fixture
def trained_artifacts(tmp_path, sample_transactions_df):
    df = pd.concat([sample_transactions_df] * 20, ignore_index=True)
    df = add_engineered_features(df)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    y = df["Class"]

    model = get_model("catboost", params={"iterations": 50}, scale_pos_weight=3.0)
    model.fit(X, y)

    model_path = tmp_path / "model.cbm"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    threshold_path = tmp_path / "threshold.json"

    save_model(model, path=model_path)
    joblib.dump(preprocessor, preprocessor_path)
    with open(threshold_path, "w") as f:
        json.dump({"threshold": 0.5}, f)

    return model_path, preprocessor_path, threshold_path


def test_fraud_predictor_load_and_predict(trained_artifacts, sample_transactions_df):
    model_path, preprocessor_path, threshold_path = trained_artifacts
    predictor = FraudPredictor.load(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold_path=threshold_path,
    )

    single_row = sample_transactions_df.drop(columns=["Class"]).iloc[[0]]
    result = predictor.predict(single_row)

    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)