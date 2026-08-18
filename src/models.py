"""
src.models

Model definitions, training, and hyperparameter tuning for fraud detection.

Model families covered:
    - Logistic Regression (linear baseline)
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost

Provides:
    - get_model(): factory returning an unfitted estimator by name
    - train_model(): fit a given model on (X_train, y_train)
    - build_optuna_objective() / run_optuna_study(): PR-AUC-optimized tuning,
      optionally with SMOTE applied inside each CV fold
    - save_model() / load_model(): persistence
    - save_training_summary(): records which model family/strategy won, so
      src.predict can load the right artifact format without hardcoding
      an assumption about which family was chosen
"""

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_STATE
from src.features import get_scale_pos_weight

MODEL_REGISTRY = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
}


def get_model(
    name: str,
    params: dict | None = None,
    scale_pos_weight: float | None = None,
    class_weight: str | dict | None = "balanced",
):
    """
    Factory for an unfitted estimator, with imbalance-aware defaults baked in.

    Args:
        name: one of MODEL_REGISTRY.
        params: hyperparameter overrides (e.g. from Optuna's best_params).
        scale_pos_weight: precomputed n_neg/n_pos ratio; used by the
            boosted-tree families' built-in imbalance handling. Pass 1.0
            (or leave scale_pos_weight unset and pass nothing) to disable it,
            e.g. when SMOTE is handling imbalance instead.
        class_weight: passed to LogisticRegression/RandomForestClassifier.
            Set to None to disable (e.g. when combining with SMOTE, so
            imbalance isn't corrected for twice over).
    """
    params = params or {}

    if name == "logistic_regression":
        return LogisticRegression(
            class_weight=class_weight, max_iter=1000, random_state=RANDOM_STATE, **params
        )

    if name == "random_forest":
        return RandomForestClassifier(
            class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1, **params
        )

    if name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight or 1.0,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            scale_pos_weight=scale_pos_weight or 1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            **params,
        )

    if name == "catboost":
        return CatBoostClassifier(
            scale_pos_weight=scale_pos_weight or 1.0,
            random_state=RANDOM_STATE,
            verbose=False,
            **params,
        )

    raise ValueError(f"Unknown model name '{name}'. Must be one of {MODEL_REGISTRY}.")


def train_model(name: str, X_train, y_train, params: dict | None = None):
    """Instantiate and fit a model by name, computing scale_pos_weight from y_train."""
    spw = get_scale_pos_weight(pd.Series(y_train))
    model = get_model(name, params=params, scale_pos_weight=spw)
    model.fit(X_train, y_train)
    return model


def build_optuna_objective(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    use_smote: bool = False,
    smote_sampling_strategy: float = 0.1,
) -> Callable[[optuna.Trial], float]:
    """
    Build an Optuna objective optimizing mean average precision (PR-AUC)
    across StratifiedKFold CV — PR-AUC, not accuracy/ROC-AUC, is the right
    metric under 0.17% positive-class imbalance.

    Args:
        use_smote: if True, SMOTE is applied inside each CV fold via an
            imblearn Pipeline (fit only on that fold's training portion,
            never touching the validation portion — avoiding leakage), and
            the model's own class_weight/scale_pos_weight correction is
            disabled to avoid double-correcting for imbalance. If False
            (default), the model's built-in imbalance handling is used
            instead, and no resampling happens.
    """
    scale_pos_weight = 1.0 if use_smote else get_scale_pos_weight(pd.Series(y))
    class_weight = None if use_smote else "balanced"
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        if name == "logistic_regression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "liblinear",
            }
        elif name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        elif name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        elif name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "num_leaves": trial.suggest_int("num_leaves", 15, 128),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        elif name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            }
        else:
            raise ValueError(f"Unknown model name '{name}'.")

        model = get_model(
            name, params=params, scale_pos_weight=scale_pos_weight, class_weight=class_weight
        )

        if use_smote:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            estimator = ImbPipeline(
                [
                    ("smote", SMOTE(sampling_strategy=smote_sampling_strategy, random_state=RANDOM_STATE)),
                    ("clf", model),
                ]
            )
        else:
            estimator = model

        scores = cross_val_score(estimator, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        return float(scores.mean())

    return objective


def run_optuna_study(
    name: str,
    X,
    y,
    n_trials: int = 50,
    n_splits: int = 5,
    direction: str = "maximize",
    use_smote: bool = False,
    smote_sampling_strategy: float = 0.1,
) -> optuna.Study:
    """Run an Optuna study for a given model family and return the completed study."""
    objective = build_optuna_objective(
        name, X, y, n_splits=n_splits, use_smote=use_smote,
        smote_sampling_strategy=smote_sampling_strategy,
    )
    study = optuna.create_study(
        direction=direction, sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def save_model(model: Any, path: Path | None = None) -> None:
    """
    Persist a trained model. CatBoost models use their native .cbm format
    (faster load, smaller file); everything else uses joblib.
    """
    path = path or (MODELS_DIR / "model.cbm")
    if isinstance(model, CatBoostClassifier):
        model.save_model(str(path))
    else:
        joblib.dump(model, path)
    print(f"✅ Model saved to {path}")


def load_model(path: Path | None = None, model_type: str = "catboost") -> Any:
    """Load a trained model. model_type must match how it was saved."""
    path = path or (MODELS_DIR / "model.cbm")
    if model_type == "catboost":
        model = CatBoostClassifier()
        model.load_model(str(path))
        return model
    return joblib.load(path)


def save_training_summary(
    model_family: str,
    strategy: str,
    best_params: dict,
    cv_pr_auc: float,
    model: Any,
    path: Path | None = None,
) -> Path:
    """
    Persist metadata about the champion model choice: which family won,
    which imbalance strategy it used, its tuned hyperparameters, its CV
    PR-AUC, and how it was serialized ("catboost" native format vs
    "joblib"). src.predict.FraudPredictor reads this to know how to load
    the model artifact without hardcoding an assumption about which
    family was chosen.
    """
    path = path or (MODELS_DIR / "training_summary.json")
    saved_as = "catboost" if isinstance(model, CatBoostClassifier) else "joblib"
    summary = {
        "model_family": model_family,
        "strategy": strategy,
        "best_params": best_params,
        "cv_pr_auc": cv_pr_auc,
        "saved_as": saved_as,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Training summary saved to {path}")
    return path
