"""
Shared pytest fixtures for the test suite.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    """A tiny synthetic dataframe mimicking the real schema, for fast unit tests."""
    return pd.DataFrame(
        {
            **{f"V{i}": [0.1, -0.2, 0.3, -0.4] for i in range(1, 29)},
            "Time": [0, 1000, 2000, 3000],
            "Amount": [10.0, 250.5, 5.0, 999.99],
            "Class": [0, 0, 0, 1],
        }
    )