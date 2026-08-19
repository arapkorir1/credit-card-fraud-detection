"""
src.predict

Inference-time utilities: load the trained model, preprocessor, and tuned
decision threshold, and produce predictions on new transactions. This is
what api/main.py (Phase 9) wraps in an HTTP endpoint.
"""

import json
from pathlib import Path
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
        summary_path: Path | None = None,
    ) -> "FraudPredictor":
        """
        Load the trained artifacts.

        model_type (CatBoost native vs. joblib) is read from
        training_summary.json (written by src.models.save_training_summary
        during Phase 7's modeling notebook) rather than assumed, since the
        winning model family isn't known ahead of time. If no summary file
        is found alongside model_path (e.g. in older/manually-built
        artifacts), falls back to "catboost" for backward compatibility.
        """
        summary_path = summary_path or (Path(model_path).parent / "training_summary.json")
        model_type = "catboost"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            model_type = summary.get("saved_as", "catboost")

        model = load_model(model_path, model_type=model_type)
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
