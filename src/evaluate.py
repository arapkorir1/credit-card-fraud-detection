"""
src.evaluate

Evaluation metrics and threshold tuning for the fraud detection model.

Given the 0.17% positive class rate, PR-AUC (average precision) is the
primary metric — ROC-AUC is misleadingly optimistic under extreme
imbalance since true negatives dominate the false positive rate.

Provides:
    - compute_metrics(): full metric dict at a given threshold
    - find_optimal_threshold(): sweep thresholds, maximize F-beta
    - compare_thresholds(): side-by-side comparison table
    - save_evaluation_report(): persist metrics to reports/metrics/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import METRICS_DIR


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute the full metric suite at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, beta: float = 1.0
) -> tuple[float, dict]:
    """
    Sweep the precision-recall curve's thresholds and return the one
    maximizing F-beta.

    beta > 1 weights recall higher (catch more fraud, tolerate more false
    positives — often preferred in fraud detection since missed fraud is
    usually costlier than a manual review of a false alarm). We default to
    beta=1.0 here; the evaluation notebook (Phase 8) compares beta=1 vs
    beta=2 explicitly.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve appends a final (precision=1, recall=0) point
    # with no corresponding threshold — drop it to align array lengths.
    precisions, recalls = precisions[:-1], recalls[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)
    f_beta = np.nan_to_num(f_beta)

    best_idx = int(np.argmax(f_beta))
    best_threshold = float(thresholds[best_idx])
    best_metrics = compute_metrics(y_true, y_proba, threshold=best_threshold)

    return best_threshold, best_metrics


def compare_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]
) -> pd.DataFrame:
    """Return a dataframe comparing metrics across a list of candidate thresholds."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_proba, threshold=t)
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "false_positives": m["confusion_matrix"]["false_positive"],
                "false_negatives": m["confusion_matrix"]["false_negative"],
            }
        )
    return pd.DataFrame(rows)


def save_evaluation_report(metrics: dict, model_name: str, filename: str | None = None) -> Path:
    """Save a metrics dict as JSON to reports/metrics/."""
    filename = filename or f"{model_name}_evaluation.json"
    path = METRICS_DIR / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Evaluation report saved to {path}")
    return path