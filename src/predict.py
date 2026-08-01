"""
src.predict

Inference-time utilities: load the trained model, preprocessor, and tuned
decision threshold, and produce predictions on new transactions. This is
what api/main.py (Phase 9) wraps in an HTTP endpoint.
"""

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, PREPROCESSOR_FILE, THRESHOLD_FILE
from src.features import add_engineered_features
from src.models import load_model


class FraudPredictor:
    """
    Bundles the fitted preprocessor, trained model, and tuned decision
    threshold into a single inference-ready object.

    Usage:
        predictor = FraudPredictor.load()
        result = predictor.predict(single_row_df)
    """

    def __init__(self, model: Any, preprocessor: Any, threshold: float):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold

    @classmethod
    def load(
        cls,
        model_path=MODEL_FILE,
        preprocessor_path=PREPROCESSOR_FILE,
        threshold_path=THRESHOLD_FILE,
    ) -> "FraudPredictor":
        model = load_model(model_path, model_type="catboost")
        preprocessor = joblib.load(preprocessor_path)
        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]
        return cls(model=model, preprocessor=preprocessor, threshold=threshold)

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        df = add_engineered_features(df)
        return self.preprocessor.transform(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row."""
        X = self._prepare(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict on a single-row (or batch) dataframe and return a
        structured result including the tuned-threshold decision.
        """
        proba = self.predict_proba(df)
        preds = (proba >= self.threshold).astype(int)

        if len(df) == 1:
            return {
                "fraud_probability": float(proba[0]),
                "is_fraud": bool(preds[0]),
                "threshold_used": self.threshold,
            }

        return {
            "fraud_probability": proba.tolist(),
            "is_fraud": preds.astype(bool).tolist(),
            "threshold_used": self.threshold,
        }