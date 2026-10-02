# Command line entry point that works out the out-of-sample metrics, feature importance,
# backtests and figures, then writes the Markdown report.
#
# Run it after momentum_ml.train:
#     python -m momentum_ml.evaluate --config configs/config.yaml

# Standard library modules for the command line arguments, reading the metadata and logging
import argparse
import json
import logging

# joblib loads the fitted models that train.py saved
import joblib
import numpy as np
import pandas as pd

from momentum_ml import plots
from momentum_ml.backtest import (
    backtest_positions,
    benchmark_signals,
    buy_and_hold_returns,
    performance_metrics,
    run_strategy,
    signals_from_probs,
    trade_returns,
)
from momentum_ml.build_features import load_panel
from momentum_ml.config import load_config, set_seeds, setup_logging
from momentum_ml.features import FWD_RET_COL, TARGET_COL
from momentum_ml.importance import all_importances
from momentum_ml.metrics import classification_metrics
from momentum_ml.splits import chronological_split
from momentum_ml.train import _safe, output_dirs

logger = logging.getLogger("momentum_ml.evaluate")

BENCH_BH = "buy_and_hold"


# Reads the saved predictions for one split, returning None if that split was not produced
def _read_predictions(tables_dir, split_name):
    path = tables_dir / f"predictions_{split_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


# Classification metrics for every (split, model) pair
def classification_table(predictions):
    rows = []
    for split_name, split_predictions in predictions.items():
        for model, model_predictions in split_predictions.groupby("model", sort=False):
            rows.append(
                {
                    "split": split_name,
                    "model": model,
                    **classification_metrics(
                        model_predictions[TARGET_COL].to_numpy(),
                        model_predictions["prob_up"].to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(rows)


# Turns one model's probabilities into trading signals using the backtest settings in the config
def _signals(prob_up, backtest_config):
    return signals_from_probs(
        prob_up,
        float(backtest_config["threshold"]),
        bool(backtest_config.get("long_short", False)),
        float(backtest_config.get("short_threshold", 0.5)),
    )


# Backtests every model, plus the benchmarks, over the rows of one prediction split
def backtest_split(predictions, panel, config):
    backtest_config = config["backtest"]
    ma_column = f"dist_ma_{int(backtest_config['ma_rule_window'])}"
    metrics_rows = []
    daily_returns = {}

    # The benchmarks need the same (date, ticker) rows as the models, plus the MA distance column
    # from the panel. Any single model's rows will do for this.
    first_model = predictions["model"].iloc[0]
    benchmark_frame = predictions[predictions["model"] == first_model][
        ["date", "ticker", FWD_RET_COL]
    ].merge(
        panel[["date", "ticker", ma_column]],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    for model, model_predictions in predictions.groupby("model", sort=False):
        model_predictions = model_predictions.sort_values(
            ["date", "ticker"], kind="mergesort"
        ).reset_index(drop=True)
        signals = _signals(model_predictions["prob_up"], backtest_config)
        portfolio, metrics = run_strategy(model_predictions, signals, backtest_config)
        metrics_rows.append({"strategy": model, "kind": "model", **metrics})
        daily_returns[model] = portfolio["net_ret"]

    benchmark_frame = benchmark_frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(
        drop=True
    )
    for name, signals in benchmark_signals(benchmark_frame, backtest_config).items():
        portfolio, metrics = run_strategy(benchmark_frame, signals, backtest_config)
        metrics_rows.append({"strategy": name, "kind": "benchmark", **metrics})
        daily_returns[name] = portfolio["net_ret"]

    # Buy and hold is handled separately because its weights drift instead of being rebalanced
    buy_and_hold = buy_and_hold_returns(benchmark_frame, float(backtest_config["cost_bps"]))
    metrics = performance_metrics(
        buy_and_hold,
        float(backtest_config.get("risk_free_rate", 0.0)),
        int(backtest_config.get("periods_per_year", 252)),
    )
    metrics["n_trades"] = benchmark_frame["ticker"].nunique()
    metrics["avg_trade_return"] = float("nan")
    metrics_rows.append({"strategy": BENCH_BH, "kind": "benchmark", **metrics})
    daily_returns[BENCH_BH] = buy_and_hold["net_ret"]
    return pd.DataFrame(metrics_rows), daily_returns


# Flags degenerate models and measures how far each test strategy really differs from the passive
# benchmarks. A model is constant when every test probability is the same (for example an L1
# logistic regression whose coefficients were all shrunk to zero). A model like that holds the
# same position every day, so its backtest is really just a benchmark rather than a signal.
def model_diagnostics(predictions, importances, test_returns, config, const_tol=1e-9):
    backtest_config = config["backtest"]
    threshold = float(backtest_config["threshold"])
    test_predictions = predictions["test"]
    always_long = test_returns.get("always_long")
    buy_and_hold = test_returns.get(BENCH_BH)

    diagnostics = {}
    for model, model_predictions in test_predictions.groupby("model", sort=False):
        prob_up = model_predictions["prob_up"].to_numpy(dtype=float)
        signals = _signals(prob_up, backtest_config)
        info = {
            "prob_min": float(prob_up.min()),
            "prob_max": float(prob_up.max()),
            "prob_std": float(prob_up.std()),
            "constant_prediction": bool(prob_up.max() - prob_up.min() <= const_tol),
            "long_fraction": float((signals > 0).mean()),
            # A constant output sitting exactly on the threshold means only the ">=" rule decides
            # the position, so the tiniest change to the threshold would flip it
            "on_threshold_tie": bool(np.all(np.abs(prob_up - threshold) <= const_tol)),
        }

        # Counting the non-zero coefficients for the logistic regressions
        coefficients = importances.get(f"coef_{model}")
        if coefficients is not None:
            info["n_coefficients"] = int(len(coefficients))
            info["n_nonzero_coefficients"] = int((coefficients["coef_raw_units"].abs() > 0).sum())

        model_returns = test_returns.get(model)
        if model_returns is not None and always_long is not None:
            info["identical_to_always_long"] = bool(
                np.allclose(
                    model_returns.to_numpy(),
                    always_long.reindex(model_returns.index).to_numpy(),
                    atol=1e-12,
                )
            )

        # Daily active return against buy and hold. The t-stat ignores autocorrelation, so it is
        # only a rough guide.
        if model_returns is not None and buy_and_hold is not None:
            active_returns = (model_returns - buy_and_hold.reindex(model_returns.index)).dropna()
            active_std = float(active_returns.std(ddof=1))
            info["active_vs_bh_mean_daily"] = float(active_returns.mean())
            if active_std > 0:
                info["active_vs_bh_tstat"] = float(
                    active_returns.mean() / active_std * np.sqrt(len(active_returns))
                )
            else:
                info["active_vs_bh_tstat"] = float("nan")

        # The majority baseline is constant by design, so it is never flagged
        info["degenerate"] = bool(
            model != "majority"
            and (
                info["constant_prediction"]
                or info.get("n_nonzero_coefficients", 1) == 0
                or info.get("identical_to_always_long", False)
            )
        )
        diagnostics[model] = info
    return diagnostics


# The selected model's test backtest under different costs and execution lags.
# This is only reported for transparency, none of these settings are chosen from it.
def sensitivity_table(predictions, model, config):
    backtest_config = config["backtest"]
    model_predictions = (
        predictions[predictions["model"] == model]
        .sort_values(["date", "ticker"], kind="mergesort")
        .reset_index(drop=True)
    )
    signals = _signals(model_predictions["prob_up"], backtest_config)
    reported_metrics = (
        "annualised_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "total_cost",
    )

    rows = []
    for lag in (0, 1):
        for cost in (0.0, 2.0, 5.0, 10.0, 20.0):
            _, metrics = run_strategy(
                model_predictions, signals, backtest_config, execution_lag=lag, cost_bps=cost
            )
            row = {"execution_lag": lag, "cost_bps": cost}
            for key in reported_metrics:
                row[key] = metrics[key]
            rows.append(row)
    return pd.DataFrame(rows)


# Test performance of the selected model on each ticker, compared with just holding that ticker
def per_ticker_table(predictions, model, config):
    backtest_config = config["backtest"]
    model_predictions = predictions[predictions["model"] == model].copy()
    model_predictions["signal"] = _signals(model_predictions["prob_up"], backtest_config)
    model_predictions["hold"] = 1.0
    reported_metrics = (
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "n_trades",
    )

    rows = []
    for ticker, ticker_predictions in model_predictions.groupby("ticker"):
        for label, signal_column in (("model", "signal"), ("hold", "hold")):
            per_ticker = backtest_positions(
                ticker_predictions[["date", "ticker", FWD_RET_COL, signal_column]],
                signal_column,
                float(backtest_config["cost_bps"]),
                int(backtest_config.get("execution_lag", 0)),
            )
            portfolio = per_ticker.set_index("date")[
                ["net_ret", "gross_ret", "cost", "turnover", "position"]
            ].rename(columns={"position": "exposure"})
            metrics = performance_metrics(
                portfolio,
                float(backtest_config.get("risk_free_rate", 0.0)),
                trades=trade_returns(per_ticker),
            )
            row = {"ticker": ticker, "strategy": model if label == "model" else "hold_ticker"}
            for key in reported_metrics:
                row[key] = metrics[key]
            rows.append(row)
    return pd.DataFrame(rows)


# Works out all of the evaluation outputs, writes the report and returns a summary dict
def run(config):
    set_seeds(int(config["project"]["seed"]))
    dirs = output_dirs(config)
    metadata_path = dirs["models"] / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found, run `python -m momentum_ml.train` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = metadata["selected_model"]
    features = metadata["feature_columns"]
    panel = load_panel(config)
    splits, _ = chronological_split(panel, config)

    predictions = {}
    for split_name in ("validation", "test", "walk_forward"):
        split_predictions = _read_predictions(dirs["tables"], split_name)
        if split_predictions is not None:
            predictions[split_name] = split_predictions
    if "test" not in predictions:
        raise FileNotFoundError("predictions_test.csv missing, run training first")

    # Classification metrics
    classification = classification_table(predictions)
    classification.to_csv(dirs["tables"] / "classification_metrics.csv", index=False)
    (dirs["tables"] / "classification_metrics.json").write_text(
        classification.to_json(orient="records", indent=2), encoding="utf-8"
    )

    # Feature importance, using the models fitted on train and scored on the validation data
    validation = splits["validation"]
    train_fitted_models = {}
    for name in metadata["models"]:
        model_path = dirs["models"] / "train_fit" / f"{_safe(name)}.joblib"
        if model_path.exists():
            train_fitted_models[name] = joblib.load(model_path)
    importances = all_importances(
        train_fitted_models, validation[features], validation[TARGET_COL], config
    )
    for key, importance_table in importances.items():
        importance_table.to_csv(dirs["tables"] / f"importance_{_safe(key)}.csv", index=False)

    # Backtests for each split
    backtest_tables = {}
    backtest_returns = {}
    for split_name in ("test", "walk_forward", "validation"):
        if split_name in predictions:
            metrics_table, daily_returns = backtest_split(predictions[split_name], panel, config)
            metrics_table.insert(0, "split", split_name)
            metrics_table.to_csv(dirs["tables"] / f"backtest_metrics_{split_name}.csv", index=False)
            pd.DataFrame(daily_returns).to_csv(
                dirs["tables"] / f"backtest_daily_returns_{split_name}.csv", index_label="date"
            )
            backtest_tables[split_name] = metrics_table
            backtest_returns[split_name] = daily_returns
    pd.concat(backtest_tables.values()).to_json(
        dirs["tables"] / "backtest_metrics.json", orient="records", indent=2
    )

    # Checking whether any model (in particular the selected one) is degenerate
    diagnostics = model_diagnostics(predictions, importances, backtest_returns["test"], config)
    (dirs["tables"] / "model_diagnostics_test.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    if diagnostics[selected]["degenerate"]:
        logger.warning(
            "Selected model %s is degenerate on test (constant P(up)=%.4f, long %.0f%% of rows); "
            "its backtest replicates a passive benchmark rather than a signal",
            selected,
            diagnostics[selected]["prob_min"],
            100 * diagnostics[selected]["long_fraction"],
        )

    sensitivity = sensitivity_table(predictions["test"], selected, config)
    sensitivity.to_csv(dirs["tables"] / "sensitivity_test_selected.csv", index=False)
    per_ticker = per_ticker_table(predictions["test"], selected, config)
    per_ticker.to_csv(dirs["tables"] / "per_ticker_test_selected.csv", index=False)

    # Figures
    figures_dir = dirs["figures"]
    ma_name = f"ma_rule_{int(config['backtest']['ma_rule_window'])}d"
    highlight = [selected, BENCH_BH, ma_name, "majority"]
    figures = {}

    # Always-long is left off the equity plots because it sits on top of buy and hold
    for split_name in ("test", "walk_forward"):
        if split_name in backtest_returns:
            plotted_returns = {
                name: returns
                for name, returns in backtest_returns[split_name].items()
                if name != "always_long"
            }
            figures[split_name] = plots.plot_equity_panels(
                plotted_returns,
                highlight,
                figures_dir / split_name,
                split_name.replace("_", "-") + " period",
                int(config["backtest"].get("rolling_vol_window", 63)),
            )

    for split_name in ("validation", "test"):
        confusion_matrices = {}
        curves = {}
        for model, model_predictions in predictions[split_name].groupby("model", sort=False):
            scores = classification[
                (classification["split"] == split_name) & (classification["model"] == model)
            ].iloc[0]
            confusion_matrices[model] = np.array(
                [[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]], dtype=int
            )
            curves[model] = (
                model_predictions[TARGET_COL].to_numpy(),
                model_predictions["prob_up"].to_numpy(),
            )
        figures.setdefault(split_name, [])
        figures[split_name].append(
            plots.plot_confusion_matrices(
                confusion_matrices,
                figures_dir / f"{split_name}_confusion_matrices.png",
                f"Confusion matrices — {split_name} (threshold 0.5)",
            )
        )
        figures[split_name].append(
            plots.plot_roc_curves(
                curves,
                selected,
                figures_dir / f"{split_name}_roc_curves.png",
                f"ROC curves — {split_name}",
            )
        )
        figures[split_name].append(
            plots.plot_calibration(
                {name: curve for name, curve in curves.items() if name != "majority"},
                figures_dir / f"{split_name}_calibration.png",
                f"Calibration — {split_name}",
            )
        )

    # One panel per type of native importance
    native_panels = {}
    for key, importance_table in importances.items():
        if key.startswith("coef_logreg_l2"):
            native_panels["LR-L2 coefficient (per 1 SD)"] = (importance_table, "coef_per_sd")
        elif key.startswith("impurity_"):
            native_panels["RF impurity"] = (importance_table, "importance")
        elif key.startswith("gain_"):
            native_panels["XGBoost gain (normalised)"] = (importance_table, "importance")
    if native_panels:
        figures["importance"] = [
            plots.plot_importance(
                native_panels,
                figures_dir / "importance_native.png",
                "Native feature importance (models fitted on training data)",
            )
        ]

    permutation_panels = {}
    for key, importance_table in importances.items():
        if key.startswith("permutation_"):
            permutation_panels[key.removeprefix("permutation_")] = (
                importance_table,
                "importance_mean",
            )
    if permutation_panels:
        scoring = config["evaluation"]["permutation_importance"].get("scoring", "roc_auc")
        figures.setdefault("importance", []).append(
            plots.plot_importance(
                permutation_panels,
                figures_dir / "importance_permutation.png",
                f"Permutation importance on validation set (Δ {scoring})",
            )
        )

    from momentum_ml.report import write_report

    summary = {
        "meta": metadata,
        "classification": classification,
        "backtests": backtest_tables,
        "sensitivity": sensitivity,
        "diagnostics": diagnostics,
        "per_ticker": per_ticker,
        "importances": importances,
        "figures": figures,
        "panel": panel,
        "splits": splits,
    }
    report_path = write_report(config, summary)
    logger.info("Report written to %s", report_path)
    summary["report_path"] = report_path
    return summary


# Entry point for python -m momentum_ml.evaluate
def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained models and write the report.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config["project"].get("log_level", "INFO"))
    run(config)


if __name__ == "__main__":
    main()
