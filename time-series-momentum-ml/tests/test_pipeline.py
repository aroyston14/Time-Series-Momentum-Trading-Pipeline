# End to end test of the whole pipeline on the offline sample data, using small grids

import json

import pandas as pd

from momentum_ml import build_features, evaluate, train
from momentum_ml.data import acquire_data


def test_end_to_end_sample_pipeline(fast_config, tmp_path):
    acquire_data(fast_config)
    build_features.run(fast_config)
    metadata = train.run(fast_config)
    summary = evaluate.run(fast_config)

    # Every expected table, the model metadata and the main figures should have been written
    tables_dir = tmp_path / "tables"
    expected_tables = (
        "splits.csv",
        "predictions_validation.csv",
        "predictions_test.csv",
        "predictions_walk_forward.csv",
        "classification_metrics.csv",
        "classification_metrics.json",
        "backtest_metrics_test.csv",
        "backtest_metrics.json",
        "sensitivity_test_selected.csv",
        "model_selection_validation.csv",
    )
    for name in expected_tables:
        assert (tables_dir / name).exists(), name
    assert (tmp_path / "models" / "model_metadata.json").exists()

    figure_names = {path.name for path in (tmp_path / "figures").glob("*.png")}
    expected_figures = (
        "test_cumulative.png",
        "test_drawdown.png",
        "test_rolling_vol.png",
        "test_roc_curves.png",
        "test_calibration.png",
        "test_confusion_matrices.png",
        "importance_permutation.png",
    )
    for name in expected_figures:
        assert name in figure_names, name

    # The report should contain all 13 sections as top level headings
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    headings = [
        "Research question",
        "Data",
        "Feature definitions",
        "Target definition",
        "Train, validation and test methodology",
        "How I prevented look-ahead bias",
        "Models and hyperparameters used",
        "Classification results",
        "Backtest results",
        "Benchmark comparison",
        "Feature importance",
        "Some assumptions and limitations of the project",
        "Conclusions",
    ]
    for heading in headings:
        assert f"\n# {heading}:\n" in report, heading

    # Selection never looks at the test data, so the selected model has to be the best
    # non-baseline model on validation
    selection_table = pd.read_csv(tables_dir / "model_selection_validation.csv")
    candidates = selection_table[selection_table["model"] != "majority"]
    assert candidates.loc[candidates["log_loss"].idxmin(), "model"] == metadata["selected_model"]

    backtest_table = pd.read_csv(tables_dir / "backtest_metrics_test.csv")
    expected_strategies = {"buy_and_hold", "always_long", "ma_rule_50d", "majority"}
    assert expected_strategies <= set(backtest_table["strategy"])

    saved_metadata = json.loads(
        (tmp_path / "models" / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert saved_metadata["seed"] == fast_config["project"]["seed"]
    assert summary["report_path"].exists()
