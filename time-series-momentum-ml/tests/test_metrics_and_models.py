# Tests for the classification metrics, building the models, and reproducibility with a fixed seed

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_ml import build_features, train
from momentum_ml.config import ConfigError, load_config
from momentum_ml.metrics import classification_metrics
from momentum_ml.models import build_model_specs, fit_with_params, predict_proba_up


def test_classification_metrics_values():
    outcomes = np.array([1, 0, 1, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.4, 0.6, 0.7])
    metrics = classification_metrics(outcomes, probabilities)
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (2, 1, 1, 1)
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["brier"] == pytest.approx(np.mean((probabilities - outcomes) ** 2))
    assert 0.5 < metrics["roc_auc"] <= 1


def test_roc_auc_nan_with_single_class():
    metrics = classification_metrics(np.ones(4), np.full(4, 0.6))
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == 1.0


def test_majority_baseline_reports_prior_probability(universe, config):
    from momentum_ml.features import build_feature_panel, feature_columns

    panel = build_feature_panel(universe, config)
    features = feature_columns(panel)
    spec = build_model_specs(config)["majority"]
    model = fit_with_params(spec, {}, panel[features], panel["target"])

    # The baseline should always predict the training base rate
    prob_up = predict_proba_up(model, panel[features].head(3))
    np.testing.assert_allclose(prob_up, panel["target"].mean())


def test_scaler_only_in_logistic_pipelines(config):
    specs = build_model_specs(config)
    for name, spec in specs.items():
        has_scaler = (
            hasattr(spec.estimator, "named_steps") and "scaler" in spec.estimator.named_steps
        )
        assert has_scaler == name.startswith("logreg")


def test_config_validation_errors(tmp_path):
    main_config = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(f"extends: {main_config}\nsplit: {{train_frac: 0.8, val_frac: 0.3}}\n")
    with pytest.raises(ConfigError, match="sum to < 1"):
        load_config(bad_config)


# Two full training runs with the same seed should give identical predictions
def test_reproducible_training_with_fixed_seed(fast_config, tmp_path):
    fast_config["walk_forward"]["enabled"] = False
    build_features.run(fast_config)
    tables_dir = tmp_path / "tables"

    first_metadata = train.run(fast_config)
    first_test = pd.read_csv(tables_dir / "predictions_test.csv")
    first_validation = pd.read_csv(tables_dir / "predictions_validation.csv")

    second_metadata = train.run(fast_config)
    second_test = pd.read_csv(tables_dir / "predictions_test.csv")
    second_validation = pd.read_csv(tables_dir / "predictions_validation.csv")

    pd.testing.assert_frame_equal(first_test, second_test)
    pd.testing.assert_frame_equal(first_validation, second_validation)
    assert first_metadata["selected_model"] == second_metadata["selected_model"]

    first_params = {repr(info["best_params"]) for info in first_metadata["models"].values()}
    second_params = {repr(info["best_params"]) for info in second_metadata["models"].values()}
    assert first_params == second_params
