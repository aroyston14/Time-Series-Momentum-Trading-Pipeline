# Shared fixtures for the tests: small fixed price datasets and a sample config whose outputs all
# go into a temporary folder

from pathlib import Path

import numpy as np
import pytest

from momentum_ml.config import load_config
from momentum_ml.sample_data import generate_sample_universe

ROOT = Path(__file__).resolve().parents[1]


# One synthetic ticker with around 300 business days of clean OHLCV data
@pytest.fixture
def prices():
    return generate_sample_universe(
        "2020-01-01", "2021-03-31", seed=1, tickers=["SPY"], inject_defects=False
    )["SPY"]


# Three synthetic tickers, including SPY so the market features can be built
@pytest.fixture
def universe():
    return generate_sample_universe(
        "2018-01-01", "2021-12-31", seed=3, tickers=["SPY", "QQQ", "IWM"], inject_defects=False
    )


# Loads the sample config with every output folder redirected into tmp_path, so tests never
# overwrite the real outputs. Any extra keyword arguments are merged over the config.
def make_config(tmp_path, **overrides):
    redirected_paths = {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "sample_dir": str(ROOT / "data" / "sample"),
        },
        "outputs": {
            "models_dir": str(tmp_path / "models"),
            "tables_dir": str(tmp_path / "tables"),
            "figures_dir": str(tmp_path / "figures"),
            "report_path": str(tmp_path / "report.md"),
        },
    }
    from momentum_ml.config import deep_merge

    return load_config(ROOT / "configs" / "sample.yaml", deep_merge(redirected_paths, overrides))


# The sample config, isolated in a temporary folder
@pytest.fixture
def config(tmp_path):
    return make_config(tmp_path)


# A very small config for the end-to-end tests (3 tickers and tiny grids) so they run quickly
@pytest.fixture
def fast_config(tmp_path):
    return make_config(
        tmp_path,
        data={"tickers": ["SPY", "QQQ", "DIA"], "start": "2016-01-01", "end": "2020-12-31"},
        walk_forward={"min_train_days": 500, "test_days": 250},
        tuning={"cv_splits": 2, "n_jobs": 1},
        models={
            "logreg_l1": {"grid": {"C": [0.1]}},
            "logreg_l2": {"grid": {"C": [0.1, 1.0]}},
            "random_forest": {
                "grid": {
                    "n_estimators": [30],
                    "max_depth": [3],
                    "min_samples_leaf": [50],
                    "max_features": ["sqrt"],
                }
            },
            "xgboost": {
                "grid": {
                    "learning_rate": [0.1],
                    "max_depth": [2],
                    "n_estimators": [30],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                }
            },
        },
        evaluation={"permutation_importance": {"n_repeats": 2, "max_rows": 1000}},
    )


# Checks two feature frames are equal, treating NaNs in the same place as equal
def assert_frame_close(first, second):
    np.testing.assert_allclose(
        first.to_numpy(dtype=float),
        second.to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
