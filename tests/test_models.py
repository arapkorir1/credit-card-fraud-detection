"""
Unit tests for src.models.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import MODEL_REGISTRY, get_model, train_model


@pytest.fixture
def tiny_classification_data():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(200, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.choice([0, 1], size=200, p=[0.9, 0.1]))
    return X, y


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_get_model_returns_correct_type(name):
    model = get_model(name, scale_pos_weight=9.0)
    assert model is not None


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_train_model_fits_and_predicts(name, tiny_classification_data):
    X, y = tiny_classification_data
    model = train_model(name, X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)


def test_get_model_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_model("not_a_real_model")