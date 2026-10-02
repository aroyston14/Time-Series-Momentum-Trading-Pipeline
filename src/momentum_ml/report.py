# Builds the Markdown research report from the evaluation outputs

import os
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_ml.config import all_tickers, resolve_path
from momentum_ml.features import describe_features


# Turns a DataFrame into a GitHub-flavoured Markdown table, without needing an extra library.
# Columns listed in pct_cols are shown as percentages and NaN values are shown as a dash.
def md_table(df, floatfmt=".4f", pct_cols=()):
    if df.empty:
        return "_(empty)_"

    # Formats a single cell depending on its type
    def format_cell(column, value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "–"
            if column in pct_cols:
                return f"{value * 100:.2f}%"
            return format(value, floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "✓" if value else ""
        return str(value)

    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for i in range(len(df)):
        cells = [format_cell(column, df[column].iloc[i]) for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Path of a figure relative to the report, with forward slashes so the links work on any OS
def _rel(path, report_path):
    return os.path.relpath(path, report_path.parent).replace(os.sep, "/")


# Returns the last date in the cached raw data for the given tickers, which is the most recent price
# the run actually used. The feature panel stops a day earlier, because the final day has no
# next-day outcome. Returns None if the raw files cannot be read.
def _last_data_date(config, tickers):
    from momentum_ml.data import load_raw, raw_path

    raw_dir = resolve_path(config, config["data"]["raw_dir"])
    try:
        last_dates = [load_raw(raw_path(raw_dir, ticker)).index.max() for ticker in tickers]
    except Exception:
        return None
    return max(last_dates).date() if last_dates else None


# Returns the first row where the column equals the value, or None if there is no such row
def _row(df, column, value):
    subset = df[df[column] == value]
    return subset.iloc[0] if len(subset) else None


# Puts together the 13 section research report from the evaluation outputs and saves it.
# Every section is added line by line to report_lines, which is joined up and written at the end.
def write_report(config, summary):
    output_path = resolve_path(config, config["outputs"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = summary["meta"]
    classification = summary["classification"]
    backtests = summary["backtests"]
    selected = metadata["selected_model"]
    panel = summary["panel"]
    backtest_config = config["backtest"]
    target_config = config["target"]
    is_sample = config["data"]["source"] == "sample"
    ma_name = f"ma_rule_{int(backtest_config['ma_rule_window'])}d"
    report_lines = []
    add = report_lines.append

    # Title, plus a note when the run used the synthetic sample data
    add("## Time-Series Momentum ML — Research Report\n")
    if is_sample:
        add(
            "Note: this run used the synthetic sample data from the offline generator in "
            "`momentum_ml.sample_data`, not market prices. It shows the pipeline works end-to-end, "
            "but none of the numbers below say anything about real markets.\n"
        )

    # The research question
    horizon = int(target_config["horizon"])
    when = "tomorrow" if horizon == 1 else f"in {horizon} trading days"
    add("# Research question:\n")
    add(
        "A research pipeline aimed at determining whether features from lagged prices and volumes "
        f"can predict if an asset's adjusted close will be higher {when} than today, before checking "
        "if this predictive power can survive as an out-of-sample trading signal.\n"
    )

    # The data used
    add("# Data:\n")
    requested_tickers = all_tickers(config)
    add(
        f"* Source: {'synthetic generator (offline sample)' if is_sample else 'Yahoo Finance via `yfinance` (auto_adjust=False)'}"
    )
    add(
        f"* Tickers used: {', '.join(sorted(panel['ticker'].unique()))} (requested: {', '.join(requested_tickers)}; market ticker: {config['data']['market_ticker']})"
    )
    # The end of the time frame is the last date actually in the data, rather than the config value,
    # since Yahoo treats the configured end date as exclusive
    last_date = _last_data_date(config, sorted(panel["ticker"].unique()))
    end_text = last_date or config["data"].get("end") or "latest"
    add(f"* TimeFrame used: {config['data']['start']} → {end_text}")
    add(
        f"* Feature Panel: {len(panel):,} rows, {panel['date'].nunique():,} dates, {panel['date'].min().date()} → {panel['date'].max().date()}"
    )
    add(f"* Up-day base rate (whole panel): {panel['target'].mean():.2%}")
    add(
        "* Fields: open/high/low are unadjusted session prices; `close` is split-adjusted only; "
        "`adj_close` is split- and dividend-adjusted and drives every return, feature and target; "
        "volume is split-adjusted share volume."
    )
    add(
        "* Cleaning: duplicate dates dropped (keep last), rows with missing/non-positive adjusted "
        "close dropped, inconsistent high<low and negative volume set to missing, calendar gaps "
        "logged but never filled. Details per ticker: `download_manifest.json` in the raw data folder."
    )
    add(
        "* Limitations: Yahoo data are vendor-adjusted and can be revised retroactively; the "
        "universe consists of instruments that exist today (survivorship); ETF closes are not "
        "perfectly synchronous with the underlying (notably EFA, whose holdings trade in "
        "earlier time zones).\n"
    )

    # A definition of every feature
    add("# Feature definitions:\n")
    definitions = describe_features(config)
    add(
        md_table(
            pd.DataFrame({"feature": list(definitions), "definition": list(definitions.values())})
        )
    )
    add("")

    # How the target is defined
    add("# Target definition:\n")
    add(
        f"`fwd_ret_t = AdjClose_(t+{target_config['horizon']}) / AdjClose_t − 1`; "
        f"`target_t = 1 if fwd_ret_t > {target_config['threshold']} else 0`. Rows whose outcome is not yet "
        "observable (the final horizon days) are dropped. The forward return is stored only for "
        "evaluation and backtesting and is excluded from the model inputs.\n"
    )

    # The train, validation and test methodology
    add("# Train, validation and test methodology:\n")
    add(md_table(pd.DataFrame(metadata["splits"])))
    add("")
    add(
        f"* Splits are by date across all tickers (no shuffling); the last {metadata['purge_days']} "
        "date(s) of train and of validation are purged because their labels depend on prices in "
        "the next segment.\n"
        f"* Hyperparameter tuning: grid search on the training split only, with "
        f"`TimeSeriesSplit` ({config['tuning']['cv_splits']} folds, grouped by date, gap = purge) "
        f"scored by `{config['tuning']['scoring']}`.\n"
        f"* Model selection: the non-baseline model with the best validation "
        f"`{metadata['selection_metric']}` (fitted on train). Test data were not used.\n"
        f"* Final test predictions: every model refitted on {'train + validation' if metadata['refit_on_train_val'] else 'train'} "
        "with its selected hyperparameters, then scored once on the test split.\n"
        "* Walk-forward (expanding window): hyperparameters fixed from the tuning step; each "
        "model is refitted on all dates before each block and predicts the block. "
        f"Scope: `{config.get('walk_forward', {}).get('scope', 'full')}`. This is a robustness view and "
        "was not used for selection.\n"
    )
    if metadata.get("walk_forward_folds"):
        add(md_table(pd.DataFrame(metadata["walk_forward_folds"])))
        add("")

    # The controls against look-ahead bias
    add("# How I prevented look-ahead bias:\n")
    add(
        "* Every rolling window ends at `t` (trailing, never centred); only positive shifts are used.\n"
        f"* Market (SPY) features are additionally lagged by {config['features']['market_features'].get('lag', 1)} day(s).\n"
        "* No imputation or forward-filling of prices; warm-up rows are dropped.\n"
        "* StandardScaler lives inside the logistic-regression Pipeline, so it is fitted on each "
        "training fold only.\n"
        "* Boundary purging between splits and between CV folds removes overlapping labels.\n"
        "* Unit tests recompute features on data truncated at `t` and on data with every price "
        "after `t` perturbed; row `t` must be identical (`tests/test_features.py`).\n"
        "* The backtest position for row `t` uses only the probability for row `t` and earns "
        "`fwd_ret_t`, realised after the decision.\n"
    )

    # The models and their chosen hyperparameters
    add("# Models and hyperparameters used:\n")
    rows = []
    for name, model_info in metadata["models"].items():

        # The grid size is the number of hyperparameter combinations that were tried
        grid = model_info["grid"]
        grid_size = int(np.prod([len(values) for values in grid.values()])) if grid else 0

        # The "clf__" pipeline prefix is dropped to make the parameter names easier to read
        chosen_params = ", ".join(
            f"{param.replace('clf__', '')}={value}"
            for param, value in model_info["best_params"].items()
        )
        cv_score = model_info["cv_score"] if model_info["cv_score"] is not None else float("nan")
        rows.append(
            {
                "model": name,
                "estimator": model_info["estimator"],
                "grid size": grid_size,
                "selected hyperparameters": chosen_params or "–",
                f"CV {model_info['cv_scoring']}": cv_score,
            }
        )
    add(md_table(pd.DataFrame(rows)))
    add("")
    add(
        "Logistic regressions use `class_weight='balanced'` and standardised inputs; the random "
        "forest uses `class_weight='balanced_subsample'`; tree models see unscaled inputs. "
        f"Seed: {metadata['seed']}."
    )
    notes = [
        model_info["note"] for model_info in metadata["models"].values() if model_info.get("note")
    ]
    for note in notes:
        add(f"\nNote: {note}")
    add("")

    # The classification results for each split
    add("# Classification results:\n")
    shown_columns = [
        "model",
        "n",
        "base_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
        "brier",
        "acc_pvalue_vs_majority",
    ]
    for split in ("validation", "test", "walk_forward"):
        subset = classification[classification["split"] == split]
        if subset.empty:
            continue
        subset = subset[shown_columns].copy()
        subset.insert(1, "selected", subset["model"] == selected)
        title = {
            "validation": "Validation (models fitted on train) - used for selection",
            "test": "Test (untouched until now; models refitted on train+validation)",
            "walk_forward": "Walk-forward out-of-sample (concatenated blocks)",
        }[split]
        add(f"## {title}:\n")
        add(md_table(subset))
        add("")
    add(
        f"Selected model: {selected} (validation {metadata['selection_metric']} = {metadata['selected_validation_score']:.5f}; "
        f"better than the majority baseline on validation: {'yes' if metadata['selected_beats_majority_on_validation'] else 'no'}).\n"
    )
    add(
        "`acc_pvalue_vs_majority` is a one-sided binomial test treating rows as independent; same-day "
        "rows across correlated ETFs are not independent, so these p-values are optimistic, and no "
        "correction for the number of models has been applied.\n"
    )
    figures = summary["figures"]
    for split in ("test", "validation"):
        for figure_path in figures.get(split, []):
            if any(keyword in figure_path.name for keyword in ("roc", "calibration", "confusion")):
                add(f"![{figure_path.stem}]({_rel(figure_path, output_path)})")
    add("")

    # The backtest results and the benchmark comparison
    columns = [
        "strategy",
        "kind",
        "cumulative_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "avg_trade_return",
        "n_trades",
        "annual_turnover",
        "avg_exposure",
        "total_cost",
    ]
    percent_columns = (
        "cumulative_return",
        "annualised_return",
        "annualised_volatility",
        "max_drawdown",
        "hit_rate",
        "avg_trade_return",
        "avg_exposure",
        "total_cost",
    )
    add("# Backtest results:\n")
    add(
        f"Rules: long when P(up) ≥ {backtest_config['threshold']}, otherwise cash"
        f"{' (short below ' + str(backtest_config['short_threshold']) + ')' if backtest_config.get('long_short') else ''}; "
        f"{backtest_config['cost_bps']} bp per one-way trade; execution lag {backtest_config.get('execution_lag', 0)} "
        "(0 = position taken at the close of the signal day, an idealised market-on-close "
        "assumption); equal-weight sleeves per ticker; cash earns 0; Sharpe uses "
        f"rf = {backtest_config.get('risk_free_rate', 0.0):.2%}. All figures are net of costs.\n"
    )
    test_backtest = backtests["test"]
    add("## Test period:\n")
    add(md_table(test_backtest[columns], ".3f", percent_columns))
    add("")
    for figure_path in figures.get("test", []):
        if any(
            keyword in figure_path.name for keyword in ("cumulative", "drawdown", "rolling_vol")
        ):
            add(f"![{figure_path.stem}]({_rel(figure_path, output_path)})")
    add("")
    add(f"## Sensitivity of the selected model ({selected}) on the test period:\n")
    add(
        "This is only reported for transparency, none of these settings were chosen using the test data.\n"
    )
    add(
        md_table(summary["sensitivity"], ".3f", ("annualised_return", "max_drawdown", "total_cost"))
    )
    add("")
    add("## Per-ticker test results (selected model vs. holding the ticker):\n")
    add(
        md_table(
            summary["per_ticker"],
            ".3f",
            ("annualised_return", "annualised_volatility", "max_drawdown", "hit_rate"),
        )
    )
    add("")
    if "walk_forward" in backtests:
        add("## Walk-forward period:\n")
        add(md_table(backtests["walk_forward"][columns], ".3f", percent_columns))
        add("")
        for figure_path in figures.get("walk_forward", []):
            add(f"![{figure_path.stem}]({_rel(figure_path, output_path)})")
        add("")

    add("# Benchmark comparison:\n")
    # Comparing the selected model against each benchmark on the test period
    selected_row = _row(test_backtest, "strategy", selected)
    bh_row = _row(test_backtest, "strategy", "buy_and_hold")
    ma_row = _row(test_backtest, "strategy", ma_name)
    majority_row = _row(test_backtest, "strategy", "majority")
    lines = []
    for label, row in (
        ("buy-and-hold", bh_row),
        ("always-long", _row(test_backtest, "strategy", "always_long")),
        (f"{ma_name} rule", ma_row),
        ("majority-class model", majority_row),
    ):
        if row is None or selected_row is None:
            continue
        lines.append(
            f"* vs {label}: Sharpe {selected_row['sharpe']:.2f} vs {row['sharpe']:.2f}; annualised return "
            f"{selected_row['annualised_return']:.2%} vs {row['annualised_return']:.2%}; max drawdown "
            f"{selected_row['max_drawdown']:.2%} vs {row['max_drawdown']:.2%}."
        )
    add("\n".join(lines))
    add("")
    if majority_row is not None and metadata["models"].get("majority"):
        add(
            "The majority-class model predicts the training set's more frequent class every day; when "
            "that class is 'up' its strategy is identical to always-long.\n"
        )
    # If the selected model is degenerate, the comparisons above are meaningless, so a warning
    # is added pointing to the conclusions
    diagnostics = summary.get("diagnostics", {})
    selected_diag = diagnostics.get(selected, {})
    if selected_diag.get("degenerate"):
        add(
            f"Warning: the selected model ({selected}) outputs a constant P(up) = "
            f"{selected_diag['prob_min']:.4f} on every test row and is long "
            f"{selected_diag['long_fraction']:.0%} of the time, so the comparisons above set a passive "
            "benchmark against itself rather than testing a signal. See the conclusions below.\n"
        )

    # Feature importance
    add("# Feature importance:\n")
    importances = summary["importances"]
    add(
        "Native importances come from models fitted on the training split; permutation importance "
        f"is the drop in validation `{config['evaluation']['permutation_importance'].get('scoring', 'roc_auc')}` "
        "when a feature is shuffled (mean ± std over repeats).\n"
    )
    for key in sorted(importances):
        if key.startswith("permutation_"):
            continue
        subset = importances[key].head(8)
        add(f"{key} (top 8):\n")
        add(md_table(subset))
        add("")
    selected_permutation = importances.get(f"permutation_{selected}")
    if selected_permutation is not None:
        add(f"Permutation importance for {selected} (top 10):\n")
        add(md_table(selected_permutation.head(10), ".5f"))
        add("")
        # Counting the features whose effect is bigger than twice the noise between repeats
        is_significant = (
            selected_permutation["importance_mean"] > 2 * selected_permutation["importance_std"]
        )
        significant = selected_permutation[is_significant]
        add(
            f"{len(significant)} of {len(selected_permutation)} features have a permutation effect larger than twice its "
            "repeat-to-repeat standard deviation for the selected model. "
            "Importance measures describe what a model uses, not causal effects, and they are "
            "unstable when signal is weak and features are highly correlated (e.g. the 5-day return, "
            "5-day MA distance and 5/20 MA spread overlap heavily).\n"
        )
    for figure_path in figures.get("importance", []):
        add(f"![{figure_path.stem}]({_rel(figure_path, output_path)})")
    add("")

    # The assumptions and limitations of the study
    add("# Some assumptions and limitations of the project:\n")
    add(
        "* Survivorship bias: the universe is chosen today from instruments that still exist and "
        "are liquid; individual stocks added via config inherit this bias more severely.\n"
        "* Transaction costs: a flat bp cost ignores spreads that widen in stress, market impact, "
        "borrow costs for shorts, taxes and the price difference between the signal close and an "
        "achievable fill. Daily strategies are very cost-sensitive (see the sensitivity table).\n"
        "* Execution timing: lag 0 assumes trading at the same close that generated the "
        "signal, which is not strictly achievable; lag 1 is shown as a conservative alternative.\n"
        "* Non-stationarity and regime change: relationships between momentum features and "
        "next-day returns drift over time (rate regimes, volatility regimes, market structure); a "
        "single test window may be unrepresentative.\n"
        "* Multiple testing: several model families × hyperparameter grids × feature choices "
        "were tried. Even with a clean test set, the best of many strategies is biased upward; the "
        "binomial p-values and Sharpe t-stats are uncorrected and assume independence.\n"
        "* Prediction accuracy ≠ economic value: a model can beat 50% accuracy by predicting "
        "'up' in a rising market, or have AUC > 0.5 while its errors fall on large-move days. "
        "Only cost-adjusted, risk-adjusted returns relative to simple benchmarks measure value.\n"
        "* Cross-sectional dependence: the ETFs are highly correlated, so the effective sample "
        "is much smaller than the row count.\n"
        "* Data quality: vendor adjustments and revisions; no point-in-time guarantees.\n"
    )

    # The conclusions
    add("# Conclusions:\n")
    selected_test_scores = _row(
        classification[classification["split"] == "test"], "model", selected
    )
    sensitivity = summary["sensitivity"]
    # The sensitivity row with a one day execution lag at the configured cost
    lag1_rows = sensitivity[
        (sensitivity["execution_lag"] == 1)
        & (sensitivity["cost_bps"] == float(backtest_config["cost_bps"]))
    ]
    conclusions = []
    if selected_test_scores is not None:
        conclusions.append(
            f"* On the unseen test period the selected model ({selected}) achieved accuracy "
            f"{selected_test_scores['accuracy']:.3f} against an up-day base rate of {selected_test_scores['base_rate']:.3f}, ROC-AUC "
            f"{selected_test_scores['roc_auc']:.3f} and log loss {selected_test_scores['log_loss']:.4f}."
        )
    # When the selected model is degenerate, the conclusions explain what happened and why it was
    # still chosen, rather than presenting its backtest as if it were a real signal
    degenerate = bool(selected_diag.get("degenerate"))
    if degenerate:
        num_nonzero = selected_diag.get("n_nonzero_coefficients")
        num_coefficients = selected_diag.get("n_coefficients")
        zeroed_text = (
            f"its regularisation shrank {num_coefficients - num_nonzero} of {num_coefficients} coefficients to zero, so it "
            if num_coefficients is not None and num_nonzero == 0
            else "it "
        )
        tie_text = (
            f" That value sits exactly on the trading threshold ({float(backtest_config['threshold']):.2f}), so "
            "the position is decided purely by the `P ≥ threshold` tie-break; a threshold an "
            "infinitesimal amount higher would have kept it in cash every day instead."
            if selected_diag.get("on_threshold_tie")
            else ""
        )
        conclusions.append(
            f"* The selected model is degenerate and is not a trading signal. {selected} was chosen "
            f"by validation log loss, but {zeroed_text}outputs the same probability "
            f"(P(up) = {selected_diag['prob_min']:.4f}) for every row and ignores the features entirely. "
            f"It is long {selected_diag['long_fraction']:.0%} of the time"
            + (
                ", and its daily returns are identical to the always-long benchmark."
                if selected_diag.get("identical_to_always_long")
                else "."
            )
            + tie_text
        )
        is_coin_flip = abs(selected_diag["prob_min"] - 0.5) <= 1e-9
        conclusions.append(
            "* Why the selection procedure picked it: "
            + (
                "a constant 0.5 forecast scores log loss ln 2 ≈ 0.6931 by construction"
                if is_coin_flip
                else "a constant forecast still earns a reasonable log loss"
            )
            + f" (validation: {metadata['selected_validation_score']:.4f}), and every model that "
            "actually used the features scored worse on validation, so the features made "
            "probabilistic forecasts worse than "
            + ("a coin flip" if is_coin_flip else "ignoring them")
            + ". Choosing the 'least bad' model then rewards the model that switched itself off. "
            f'The correct reading of this selection result is "no model", not a win for {selected}'
            + (
                f"; its test ROC-AUC of {selected_test_scores['roc_auc']:.3f} says the same."
                if selected_test_scores is not None
                else "."
            )
        )
    # Comparing against the passive benchmarks. Outperformance is only claimed if the model is not
    # degenerate, beats both benchmarks with and without the execution lag, and the difference in
    # daily returns has a t-stat above 2.
    if selected_row is not None and bh_row is not None:
        always_long_row = _row(test_backtest, "strategy", "always_long")
        passive = [row for row in (bh_row, always_long_row) if row is not None]
        best_passive = max(row["sharpe"] for row in passive)
        lag1_sharpe = float(lag1_rows["sharpe"].iloc[0]) if len(lag1_rows) else float("nan")
        active_tstat = selected_diag.get("active_vs_bh_tstat", float("nan"))
        conclusions.append(
            f"* After {backtest_config['cost_bps']} bp costs its net Sharpe was {selected_row['sharpe']:.2f} versus "
            f"{bh_row['sharpe']:.2f} for buy-and-hold"
            + (
                f" and {always_long_row['sharpe']:.2f} for always-long"
                if always_long_row is not None
                else ""
            )
            + (f" ({lag1_sharpe:.2f} with a one-day execution lag)." if len(lag1_rows) else ".")
        )
        beats_both_lags = selected_row["sharpe"] > best_passive and (
            not len(lag1_rows) or lag1_sharpe > best_passive
        )
        if degenerate:
            conclusions.append(
                "* That Sharpe is the market's, not the model's: the small gap to buy-and-hold "
                "comes only from always-long rebalancing to equal weights daily while buy-and-hold "
                "lets weights drift"
                + (
                    f" (daily active-return t-stat vs buy-and-hold {active_tstat:.2f})."
                    if np.isfinite(active_tstat)
                    else "."
                )
                + " It is not evidence that the model beats buy-and-hold."
            )
        elif beats_both_lags and np.isfinite(active_tstat) and active_tstat > 2 and not is_sample:
            conclusions.append(
                f"* The selected model beat both passive benchmarks under both timing assumptions, "
                f"with a daily active-return t-stat of {active_tstat:.2f} versus buy-and-hold. Given the "
                "number of configurations tried, a single test window and the limitations above, "
                "this is at most weak, provisional evidence; it should be re-tested on a later, "
                "genuinely new period before being taken seriously."
            )
        elif beats_both_lags:
            conclusions.append(
                "* Its Sharpe was nominally above the passive benchmarks, but the difference is "
                "not statistically distinguishable from zero"
                + (
                    f" (daily active-return t-stat vs buy-and-hold {active_tstat:.2f})"
                    if np.isfinite(active_tstat)
                    else ""
                )
                + ", so this is not evidence of a profitable strategy"
                + (" (and the data are synthetic)." if is_sample else ".")
            )
        else:
            conclusions.append(
                "* This does not constitute evidence of a profitable strategy: the result does "
                "not beat simple passive benchmarks after costs on unseen data"
                + (" (and the data are synthetic)." if is_sample else ".")
            )
        # If the selected model is degenerate, the best model that actually uses the features is
        # the real test of the research question, so its results are reported as well
        if degenerate:
            validation_scores = classification[classification["split"] == "validation"]
            live_models = validation_scores[
                (validation_scores["model"] != "majority")
                & validation_scores["model"].map(
                    lambda model_name: not diagnostics.get(model_name, {}).get("degenerate", False)
                )
            ]
            if len(live_models):
                best_live = live_models.sort_values("log_loss").iloc[0]
                best_live_backtest = _row(test_backtest, "strategy", best_live["model"])
                best_live_scores = _row(
                    classification[classification["split"] == "test"], "model", best_live["model"]
                )
                if best_live_backtest is not None and best_live_scores is not None:
                    best_live_tstat = diagnostics[best_live["model"]].get(
                        "active_vs_bh_tstat", float("nan")
                    )
                    beat_benchmarks = (
                        best_live_backtest["sharpe"] > best_passive
                        and np.isfinite(best_live_tstat)
                        and best_live_tstat > 2
                    )
                    verdict = "beat" if beat_benchmarks else "did not beat"
                    conclusions.append(
                        f"* The best model that actually uses the features ({best_live['model']}, "
                        f"validation log loss {best_live['log_loss']:.4f}) is the honest test of the "
                        f"research question. On the test period it had ROC-AUC "
                        f"{best_live_scores['roc_auc']:.3f}, net Sharpe {best_live_backtest['sharpe']:.2f} vs "
                        f"{best_passive:.2f} for the best passive benchmark, and was long "
                        f"{diagnostics[best_live['model']]['long_fraction']:.0%} of the time, and it {verdict} "
                        "the passive benchmarks."
                    )
    if not metadata["selected_beats_majority_on_validation"]:
        conclusions.append(
            "* No model beat the majority-class baseline on validation log loss. The features "
            "carry little or no usable probabilistic information about next-day direction in this "
            "sample, which is consistent with the weak-form efficiency of liquid index ETFs."
        )
    conclusions.append(
        "* Any apparent edge should be treated as a hypothesis for further testing (new periods, "
        "more assets, realistic execution), not as a trading recommendation."
    )
    add("\n".join(conclusions))
    add("")
    add("---\n")
    add(
        f"_Generated by `momentum_ml.evaluate` from config `{Path(config.get('_config_path', '')).name}`; "
        f"seed {metadata['seed']}; package versions: {metadata['versions']}._\n"
    )
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path
