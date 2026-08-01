"""
Unit tests for src.evaluate.
"""

import numpy as np

from src.evaluate import compare_thresholds, compute_metrics, find_optimal_threshold


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.01, 0.02, 0.03, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_find_optimal_threshold_returns_valid_threshold():
    rng = np.random.RandomState(0)
    y_true = np.array([0] * 950 + [1] * 50)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=1000), 0, 1)

    threshold, metrics = find_optimal_threshold(y_true, y_proba, beta=1.0)
    assert 0.0 <= threshold <= 1.0
    assert metrics["f1"] > 0


def test_compare_thresholds_returns_expected_columns():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    df = compare_thresholds(y_true, y_proba, thresholds=[0.3, 0.5, 0.7])
    assert list(df.columns) == [
        "threshold", "precision", "recall", "f1", "false_positives", "false_negatives"
    ]
    assert len(df) == 3