## Time-Series Momentum ML — Research Report

Note: this run used the synthetic sample data from the offline generator in `momentum_ml.sample_data`, not market prices. It shows the pipeline works end-to-end, but none of the numbers below say anything about real markets.

# Research question:

A research pipeline aimed at determining whether features from lagged prices and volumes can predict if an asset's adjusted close will be higher tomorrow than today, before checking if this predictive power can survive as an out-of-sample trading signal.

# Data:

* Source: synthetic generator (offline sample)
* Tickers used: DIA, EFA, IWM, QQQ, SPY (requested: SPY, QQQ, IWM, DIA, EFA; market ticker: SPY)
* TimeFrame used: 2014-01-01 → 2024-12-31
* Feature Panel: 14,020 rows, 2,809 dates, 2014-03-26 → 2024-12-30
* Up-day base rate (whole panel): 52.78%
* Fields: open/high/low are unadjusted session prices; `close` is split-adjusted only; `adj_close` is split- and dividend-adjusted and drives every return, feature and target; volume is split-adjusted share volume.
* Cleaning: duplicate dates dropped (keep last), rows with missing/non-positive adjusted close dropped, inconsistent high<low and negative volume set to missing, calendar gaps logged but never filled. Details per ticker: `download_manifest.json` in the raw data folder.
* Limitations: Yahoo data are vendor-adjusted and can be revised retroactively; the universe consists of instruments that exist today (survivorship); ETF closes are not perfectly synchronous with the underlying (notably EFA, whose holdings trade in earlier time zones).

# Feature definitions:

| feature | definition |
|---|---|
| ret_1d | AdjClose_t / AdjClose_(t-1) - 1 |
| ret_2d | AdjClose_t / AdjClose_(t-2) - 1 |
| ret_5d | AdjClose_t / AdjClose_(t-5) - 1 |
| ret_10d | AdjClose_t / AdjClose_(t-10) - 1 |
| ret_20d | AdjClose_t / AdjClose_(t-20) - 1 |
| ret_60d | AdjClose_t / AdjClose_(t-60) - 1 |
| vol_5d | std of daily adj-close returns over days t-4..t (annualised x sqrt(252)) |
| vol_10d | std of daily adj-close returns over days t-9..t (annualised x sqrt(252)) |
| vol_20d | std of daily adj-close returns over days t-19..t (annualised x sqrt(252)) |
| vol_60d | std of daily adj-close returns over days t-59..t (annualised x sqrt(252)) |
| dist_ma_5 | AdjClose_t / SMA5_t - 1 (SMA over days t-4..t) |
| dist_ma_20 | AdjClose_t / SMA20_t - 1 (SMA over days t-19..t) |
| dist_ma_50 | AdjClose_t / SMA50_t - 1 (SMA over days t-49..t) |
| ma_spread_5_20 | SMA5_t / SMA20_t - 1 |
| hl_range | (High_t - Low_t) / Close_t  (same-day unadjusted; equals adjusted ratio) |
| oc_return | Close_t / Open_t - 1 (intraday return) |
| volume_pct_change | Volume_t / Volume_(t-1) - 1 |
| volume_rel_ma20 | Volume_t / mean(Volume over days t-19..t) |
| mdd_20d | max drawdown of AdjClose within days t-19..t (<= 0) |
| mdd_60d | max drawdown of AdjClose within days t-59..t (<= 0) |
| mkt_ret_1d | SPY 1-day return, lagged 1 day(s) |
| mkt_ret_5d | SPY 5-day return, lagged 1 day(s) |
| mkt_ret_20d | SPY 20-day return, lagged 1 day(s) |
| mkt_vol_20d | SPY 20-day volatility (annualised x sqrt(252)), lagged 1 day(s) |

# Target definition:

`fwd_ret_t = AdjClose_(t+1) / AdjClose_t − 1`; `target_t = 1 if fwd_ret_t > 0.0 else 0`. Rows whose outcome is not yet observable (the final horizon days) are dropped. The forward return is stored only for evaluation and backtesting and is excluded from the model inputs.

# Train, validation and test methodology:

| name | start | end | n_dates | n_rows |
|---|---|---|---|---|
| train | 2014-03-26 | 2020-09-07 | 1684 | 8395 |
| validation | 2020-09-09 | 2022-11-01 | 560 | 2800 |
| test | 2022-11-03 | 2024-12-30 | 563 | 2815 |

* Splits are by date across all tickers (no shuffling); the last 1 date(s) of train and of validation are purged because their labels depend on prices in the next segment.
* Hyperparameter tuning: grid search on the training split only, with `TimeSeriesSplit` (3 folds, grouped by date, gap = purge) scored by `neg_log_loss`.
* Model selection: the non-baseline model with the best validation `log_loss` (fitted on train). Test data were not used.
* Final test predictions: every model refitted on train + validation with its selected hyperparameters, then scored once on the test split.
* Walk-forward (expanding window): hyperparameters fixed from the tuning step; each model is refitted on all dates before each block and predicts the block. Scope: `full`. This is a robustness view and was not used for selection.

| fold | train_start | train_end | test_start | test_end | train_rows | test_rows |
|---|---|---|---|---|---|---|
| 0 | 2014-03-26 | 2017-02-14 | 2017-02-16 | 2019-01-22 | 3750 | 2520 |
| 1 | 2014-03-26 | 2019-01-21 | 2019-01-23 | 2020-12-28 | 6270 | 2520 |
| 2 | 2014-03-26 | 2020-12-25 | 2020-12-29 | 2022-12-02 | 8790 | 2520 |
| 3 | 2014-03-26 | 2022-12-01 | 2022-12-05 | 2024-11-07 | 11310 | 2520 |
| 4 | 2014-03-26 | 2024-11-06 | 2024-11-08 | 2024-12-30 | 13830 | 185 |

# How I prevented look-ahead bias:

* Every rolling window ends at `t` (trailing, never centred); only positive shifts are used.
* Market (SPY) features are additionally lagged by 1 day(s).
* No imputation or forward-filling of prices; warm-up rows are dropped.
* StandardScaler lives inside the logistic-regression Pipeline, so it is fitted on each training fold only.
* Boundary purging between splits and between CV folds removes overlapping labels.
* Unit tests recompute features on data truncated at `t` and on data with every price after `t` perturbed; row `t` must be identical (`tests/test_features.py`).
* The backtest position for row `t` uses only the probability for row `t` and earns `fwd_ret_t`, realised after the decision.

# Models and hyperparameters used:

| model | estimator | grid size | selected hyperparameters | CV neg_log_loss |
|---|---|---|---|---|
| majority | DummyClassifier | 0 | – | – |
| logreg_none | Pipeline | 0 | – | – |
| logreg_l1 | Pipeline | 3 | C=0.01 | -0.6931 |
| logreg_l2 | Pipeline | 3 | C=0.01 | -0.6953 |
| random_forest | RandomForestClassifier | 2 | max_depth=3, max_features=sqrt, min_samples_leaf=100, n_estimators=100 | -0.6932 |
| xgboost | XGBClassifier | 8 | colsample_bytree=0.8, learning_rate=0.05, max_depth=2, n_estimators=100, reg_alpha=1.0, reg_lambda=10.0, subsample=0.8 | -0.6957 |

Logistic regressions use `class_weight='balanced'` and standardised inputs; the random forest uses `class_weight='balanced_subsample'`; tree models see unscaled inputs. Seed: 7.

# Classification results:

## Validation (models fitted on train) - used for selection:

| model | selected | n | base_rate | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | log_loss | brier | acc_pvalue_vs_majority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority |  | 2800 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.6667 | 0.5000 | 0.6959 | 0.2514 | 0.5075 |
| logreg_none |  | 2800 | 0.5000 | 0.4800 | 0.4800 | 0.4766 | 0.4079 | 0.4396 | 0.4882 | 0.6979 | 0.2523 | 0.9836 |
| logreg_l1 |  | 2800 | 0.5000 | 0.5271 | 0.5271 | 0.5314 | 0.4600 | 0.4931 | 0.5343 | 0.6924 | 0.2496 | 0.0022 |
| logreg_l2 |  | 2800 | 0.5000 | 0.5018 | 0.5018 | 0.5022 | 0.4079 | 0.4501 | 0.4986 | 0.6954 | 0.2511 | 0.4325 |
| random_forest | ✓ | 2800 | 0.5000 | 0.5196 | 0.5196 | 0.5194 | 0.5271 | 0.5232 | 0.5220 | 0.6923 | 0.2496 | 0.0197 |
| xgboost |  | 2800 | 0.5000 | 0.5171 | 0.5171 | 0.5101 | 0.8671 | 0.6423 | 0.5137 | 0.6961 | 0.2514 | 0.0363 |

## Test (untouched until now; models refitted on train+validation):

| model | selected | n | base_rate | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | log_loss | brier | acc_pvalue_vs_majority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority |  | 2815 | 0.5290 | 0.5290 | 0.5000 | 0.5290 | 1.0000 | 0.6919 | 0.5000 | 0.6915 | 0.2492 | 0.5077 |
| logreg_none |  | 2815 | 0.5290 | 0.4934 | 0.4891 | 0.5195 | 0.5635 | 0.5406 | 0.4906 | 0.6964 | 0.2516 | 0.9999 |
| logreg_l1 |  | 2815 | 0.5290 | 0.5087 | 0.5076 | 0.5363 | 0.5265 | 0.5313 | 0.5047 | 0.6934 | 0.2501 | 0.9850 |
| logreg_l2 |  | 2815 | 0.5290 | 0.4924 | 0.4880 | 0.5185 | 0.5635 | 0.5401 | 0.4930 | 0.6953 | 0.2511 | 1.0000 |
| random_forest | ✓ | 2815 | 0.5290 | 0.5172 | 0.5050 | 0.5325 | 0.7159 | 0.6107 | 0.5099 | 0.6935 | 0.2501 | 0.8970 |
| xgboost |  | 2815 | 0.5290 | 0.5329 | 0.5105 | 0.5349 | 0.8959 | 0.6698 | 0.4978 | 0.6932 | 0.2500 | 0.3460 |

## Walk-forward out-of-sample (concatenated blocks):

| model | selected | n | base_rate | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | log_loss | brier | acc_pvalue_vs_majority |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority |  | 10265 | 0.5249 | 0.5249 | 0.5000 | 0.5249 | 1.0000 | 0.6884 | 0.5026 | 0.6922 | 0.2495 | 0.5040 |
| logreg_none |  | 10265 | 0.5249 | 0.4963 | 0.4952 | 0.5203 | 0.5178 | 0.5191 | 0.4968 | 0.6967 | 0.2518 | 1.0000 |
| logreg_l1 |  | 10265 | 0.5249 | 0.5222 | 0.5168 | 0.5387 | 0.6242 | 0.5783 | 0.5154 | 0.6930 | 0.2499 | 0.7134 |
| logreg_l2 |  | 10265 | 0.5249 | 0.5008 | 0.4996 | 0.5245 | 0.5243 | 0.5244 | 0.5000 | 0.6949 | 0.2509 | 1.0000 |
| random_forest | ✓ | 10265 | 0.5249 | 0.5111 | 0.5054 | 0.5293 | 0.6186 | 0.5705 | 0.5109 | 0.6931 | 0.2500 | 0.9976 |
| xgboost |  | 10265 | 0.5249 | 0.5150 | 0.4970 | 0.5231 | 0.8586 | 0.6501 | 0.4959 | 0.6963 | 0.2515 | 0.9786 |

Selected model: random_forest (validation log_loss = 0.69226; better than the majority baseline on validation: yes).

`acc_pvalue_vs_majority` is a one-sided binomial test treating rows as independent; same-day rows across correlated ETFs are not independent, so these p-values are optimistic, and no correction for the number of models has been applied.

![test_confusion_matrices](figures/test_confusion_matrices.png)
![test_roc_curves](figures/test_roc_curves.png)
![test_calibration](figures/test_calibration.png)
![validation_confusion_matrices](figures/validation_confusion_matrices.png)
![validation_roc_curves](figures/validation_roc_curves.png)
![validation_calibration](figures/validation_calibration.png)

# Backtest results:

Rules: long when P(up) ≥ 0.5, otherwise cash; 5.0 bp per one-way trade; execution lag 0 (0 = position taken at the close of the signal day, an idealised market-on-close assumption); equal-weight sleeves per ticker; cash earns 0; Sharpe uses rf = 0.00%. All figures are net of costs.

## Test period:

| strategy | kind | cumulative_return | annualised_return | annualised_volatility | sharpe | max_drawdown | calmar | hit_rate | avg_trade_return | n_trades | annual_turnover | avg_exposure | total_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | model | 26.22% | 10.98% | 11.99% | 0.929 | -10.81% | 1.016 | 52.22% | 25.72% | 5 | 0.448 | 100.00% | 0.05% |
| logreg_none | model | -4.53% | -2.05% | 8.72% | -0.194 | -14.80% | -0.139 | 47.60% | -0.04% | 447 | 79.673 | 57.37% | 8.90% |
| logreg_l1 | model | 4.23% | 1.87% | 7.90% | 0.274 | -9.83% | 0.190 | 51.53% | 0.06% | 367 | 65.260 | 51.94% | 7.29% |
| logreg_l2 | model | -3.75% | -1.70% | 8.67% | -0.154 | -13.72% | -0.124 | 46.29% | -0.04% | 433 | 77.077 | 57.48% | 8.61% |
| random_forest | model | 8.37% | 3.67% | 9.56% | 0.424 | -11.01% | 0.333 | 51.01% | 0.17% | 274 | 48.610 | 71.12% | 5.43% |
| xgboost | model | 21.94% | 9.28% | 11.25% | 0.846 | -10.96% | 0.847 | 52.04% | 0.55% | 195 | 34.465 | 88.60% | 3.85% |
| always_long | benchmark | 26.22% | 10.98% | 11.99% | 0.929 | -10.81% | 1.016 | 52.22% | 25.72% | 5 | 0.448 | 100.00% | 0.05% |
| ma_rule_50d | benchmark | 6.58% | 2.89% | 8.58% | 0.375 | -9.20% | 0.315 | 50.60% | 0.30% | 118 | 20.769 | 63.62% | 2.32% |
| buy_and_hold | benchmark | 25.72% | 10.79% | 11.96% | 0.916 | -10.94% | 0.986 | 51.69% | – | 5 | 0.448 | 100.00% | 0.05% |

![test_cumulative](figures/test_cumulative.png)
![test_drawdown](figures/test_drawdown.png)
![test_rolling_vol](figures/test_rolling_vol.png)

## Sensitivity of the selected model (random_forest) on the test period:

This is only reported for transparency, none of these settings were chosen using the test data.

| execution_lag | cost_bps | annualised_return | sharpe | max_drawdown | annual_turnover | total_cost |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 6.21% | 0.678 | -10.27% | 48.610 | 0.00% |
| 0 | 2.000 | 5.19% | 0.577 | -10.53% | 48.610 | 2.17% |
| 0 | 5.000 | 3.67% | 0.424 | -11.01% | 48.610 | 5.43% |
| 0 | 10.000 | 1.18% | 0.170 | -13.44% | 48.610 | 10.86% |
| 0 | 20.000 | -3.63% | -0.338 | -18.63% | 48.610 | 21.72% |
| 1 | 0.000 | 4.98% | 0.554 | -12.03% | 48.610 | 0.00% |
| 1 | 2.000 | 3.97% | 0.453 | -12.67% | 48.610 | 2.17% |
| 1 | 5.000 | 2.46% | 0.301 | -13.62% | 48.610 | 5.43% |
| 1 | 10.000 | 0.00% | 0.048 | -15.70% | 48.610 | 10.86% |
| 1 | 20.000 | -4.74% | -0.456 | -21.28% | 48.610 | 21.72% |

## Per-ticker test results (selected model vs. holding the ticker):

| ticker | strategy | annualised_return | annualised_volatility | sharpe | max_drawdown | hit_rate | n_trades |
|---|---|---|---|---|---|---|---|
| DIA | random_forest | 2.75% | 10.48% | 0.312 | -12.12% | 53.77% | 53 |
| DIA | hold_ticker | 9.51% | 12.17% | 0.808 | -12.21% | 54.17% | 1 |
| EFA | random_forest | 4.60% | 12.81% | 0.415 | -16.96% | 52.33% | 57 |
| EFA | hold_ticker | 9.75% | 14.47% | 0.715 | -12.94% | 51.51% | 1 |
| IWM | random_forest | -6.81% | 14.42% | -0.417 | -25.01% | 51.17% | 59 |
| IWM | hold_ticker | 5.19% | 17.89% | 0.372 | -14.29% | 52.22% | 1 |
| QQQ | random_forest | 14.46% | 12.84% | 1.116 | -11.23% | 53.02% | 57 |
| QQQ | hold_ticker | 17.36% | 15.60% | 1.104 | -17.89% | 51.87% | 1 |
| SPY | random_forest | 3.00% | 9.25% | 0.366 | -12.96% | 54.09% | 48 |
| SPY | hold_ticker | 11.70% | 10.79% | 1.080 | -10.11% | 54.71% | 1 |

## Walk-forward period:

| strategy | kind | cumulative_return | annualised_return | annualised_volatility | sharpe | max_drawdown | calmar | hit_rate | avg_trade_return | n_trades | annual_turnover | avg_exposure | total_cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| majority | model | 135.30% | 11.07% | 11.53% | 0.968 | -19.36% | 0.572 | 52.61% | 139.22% | 5 | 0.123 | 100.00% | 0.05% |
| logreg_none | model | 14.53% | 1.68% | 7.78% | 0.253 | -14.19% | 0.118 | 50.54% | 0.05% | 1481 | 72.617 | 52.24% | 29.58% |
| logreg_l1 | model | 94.28% | 8.49% | 8.53% | 0.998 | -10.34% | 0.821 | 52.54% | 0.55% | 717 | 35.081 | 60.82% | 14.29% |
| logreg_l2 | model | 21.96% | 2.47% | 7.81% | 0.351 | -13.33% | 0.185 | 49.51% | 0.08% | 1470 | 72.053 | 52.47% | 29.35% |
| random_forest | model | 72.98% | 6.96% | 8.35% | 0.848 | -12.58% | 0.553 | 51.53% | 0.23% | 1249 | 61.202 | 61.34% | 24.93% |
| xgboost | model | 79.16% | 7.42% | 10.37% | 0.742 | -15.77% | 0.471 | 51.83% | 0.42% | 756 | 36.996 | 86.15% | 15.07% |
| always_long | benchmark | 135.30% | 11.07% | 11.53% | 0.968 | -19.36% | 0.572 | 52.61% | 139.22% | 5 | 0.123 | 100.00% | 0.05% |
| ma_rule_50d | benchmark | 66.71% | 6.47% | 8.42% | 0.787 | -11.62% | 0.557 | 51.87% | 0.78% | 368 | 17.970 | 61.46% | 7.32% |
| buy_and_hold | benchmark | 139.22% | 11.30% | 11.38% | 0.998 | -19.19% | 0.589 | 51.92% | – | 5 | 0.123 | 100.00% | 0.05% |

![walk_forward_cumulative](figures/walk_forward_cumulative.png)
![walk_forward_drawdown](figures/walk_forward_drawdown.png)
![walk_forward_rolling_vol](figures/walk_forward_rolling_vol.png)

# Benchmark comparison:

* vs buy-and-hold: Sharpe 0.42 vs 0.92; annualised return 3.67% vs 10.79%; max drawdown -11.01% vs -10.94%.
* vs always-long: Sharpe 0.42 vs 0.93; annualised return 3.67% vs 10.98%; max drawdown -11.01% vs -10.81%.
* vs ma_rule_50d rule: Sharpe 0.42 vs 0.38; annualised return 3.67% vs 2.89%; max drawdown -11.01% vs -9.20%.
* vs majority-class model: Sharpe 0.42 vs 0.93; annualised return 3.67% vs 10.98%; max drawdown -11.01% vs -10.81%.

The majority-class model predicts the training set's more frequent class every day; when that class is 'up' its strategy is identical to always-long.

# Feature importance:

Native importances come from models fitted on the training split; permutation importance is the drop in validation `roc_auc` when a feature is shuffled (mean ± std over repeats).

coef_logreg_l1 (top 8):

| feature | coef_per_sd | abs_coef_per_sd | coef_raw_units | feature_sd |
|---|---|---|---|---|
| ret_10d | 0.0176 | 0.0176 | 0.5781 | 0.0304 |
| ret_1d | -0.0119 | 0.0119 | -1.2594 | 0.0094 |
| mkt_ret_20d | 0.0112 | 0.0112 | 0.3440 | 0.0327 |
| mdd_20d | 0.0055 | 0.0055 | 0.2293 | 0.0240 |
| ma_spread_5_20 | 0.0012 | 0.0012 | 0.0641 | 0.0191 |
| ret_20d | 0.0000 | 0.0000 | 0.0000 | 0.0447 |
| ret_2d | 0.0000 | 0.0000 | 0.0000 | 0.0132 |
| ret_5d | 0.0000 | 0.0000 | 0.0000 | 0.0210 |

coef_logreg_l2 (top 8):

| feature | coef_per_sd | abs_coef_per_sd | coef_raw_units | feature_sd |
|---|---|---|---|---|
| ret_10d | 0.1430 | 0.1430 | 4.6965 | 0.0304 |
| mdd_20d | 0.1072 | 0.1072 | 4.4717 | 0.0240 |
| ma_spread_5_20 | -0.0732 | 0.0732 | -3.8450 | 0.0191 |
| dist_ma_20 | -0.0701 | 0.0701 | -2.8778 | 0.0244 |
| dist_ma_50 | 0.0677 | 0.0677 | 1.6042 | 0.0422 |
| vol_10d | 0.0626 | 0.0626 | 1.0923 | 0.0573 |
| mkt_ret_1d | -0.0608 | 0.0608 | -8.4874 | 0.0072 |
| ret_1d | -0.0594 | 0.0594 | -6.2930 | 0.0094 |

coef_logreg_none (top 8):

| feature | coef_per_sd | abs_coef_per_sd | coef_raw_units | feature_sd |
|---|---|---|---|---|
| ret_10d | 0.2585 | 0.2585 | 8.4883 | 0.0304 |
| dist_ma_20 | -0.1914 | 0.1914 | -7.8576 | 0.0244 |
| mdd_20d | 0.1782 | 0.1782 | 7.4359 | 0.0240 |
| ma_spread_5_20 | -0.1676 | 0.1676 | -8.7958 | 0.0191 |
| dist_ma_50 | 0.1535 | 0.1535 | 3.6379 | 0.0422 |
| ret_1d | -0.1012 | 0.1012 | -10.7149 | 0.0094 |
| mkt_ret_1d | -0.0777 | 0.0777 | -10.8432 | 0.0072 |
| ret_60d | -0.0748 | 0.0748 | -0.8685 | 0.0861 |

gain_xgboost (top 8):

| feature | gain | split_count | importance |
|---|---|---|---|
| ma_spread_5_20 | 10.1479 | 4.0000 | 0.0564 |
| mkt_ret_5d | 9.9676 | 37.0000 | 0.0554 |
| ret_10d | 9.8302 | 12.0000 | 0.0546 |
| ret_1d | 9.1692 | 16.0000 | 0.0509 |
| ret_2d | 8.7207 | 8.0000 | 0.0484 |
| dist_ma_50 | 8.6904 | 9.0000 | 0.0483 |
| oc_return | 8.5344 | 5.0000 | 0.0474 |
| mkt_vol_20d | 8.2413 | 38.0000 | 0.0458 |

impurity_random_forest (top 8):

| feature | importance |
|---|---|
| mkt_ret_5d | 0.1047 |
| ret_10d | 0.0750 |
| mkt_ret_20d | 0.0703 |
| ret_1d | 0.0671 |
| mkt_vol_20d | 0.0588 |
| ret_2d | 0.0479 |
| ma_spread_5_20 | 0.0465 |
| dist_ma_50 | 0.0456 |

Permutation importance for random_forest (top 10):

| feature | importance_mean | importance_std |
|---|---|---|
| mkt_ret_5d | 0.01941 | 0.00113 |
| ret_10d | 0.00416 | 0.00110 |
| ma_spread_5_20 | 0.00262 | 0.00057 |
| oc_return | 0.00202 | 0.00138 |
| mkt_ret_20d | 0.00153 | 0.00122 |
| vol_10d | 0.00150 | 0.00093 |
| mdd_20d | 0.00147 | 0.00093 |
| ret_1d | 0.00139 | 0.00181 |
| mkt_vol_20d | 0.00138 | 0.00066 |
| ret_5d | 0.00130 | 0.00052 |

6 of 24 features have a permutation effect larger than twice its repeat-to-repeat standard deviation for the selected model. Importance measures describe what a model uses, not causal effects, and they are unstable when signal is weak and features are highly correlated (e.g. the 5-day return, 5-day MA distance and 5/20 MA spread overlap heavily).

![importance_native](figures/importance_native.png)
![importance_permutation](figures/importance_permutation.png)

# Some assumptions and limitations of the project:

* Survivorship bias: the universe is chosen today from instruments that still exist and are liquid; individual stocks added via config inherit this bias more severely.
* Transaction costs: a flat bp cost ignores spreads that widen in stress, market impact, borrow costs for shorts, taxes and the price difference between the signal close and an achievable fill. Daily strategies are very cost-sensitive (see the sensitivity table).
* Execution timing: lag 0 assumes trading at the same close that generated the signal, which is not strictly achievable; lag 1 is shown as a conservative alternative.
* Non-stationarity and regime change: relationships between momentum features and next-day returns drift over time (rate regimes, volatility regimes, market structure); a single test window may be unrepresentative.
* Multiple testing: several model families × hyperparameter grids × feature choices were tried. Even with a clean test set, the best of many strategies is biased upward; the binomial p-values and Sharpe t-stats are uncorrected and assume independence.
* Prediction accuracy ≠ economic value: a model can beat 50% accuracy by predicting 'up' in a rising market, or have AUC > 0.5 while its errors fall on large-move days. Only cost-adjusted, risk-adjusted returns relative to simple benchmarks measure value.
* Cross-sectional dependence: the ETFs are highly correlated, so the effective sample is much smaller than the row count.
* Data quality: vendor adjustments and revisions; no point-in-time guarantees.

# Conclusions:

* On the unseen test period the selected model (random_forest) achieved accuracy 0.517 against an up-day base rate of 0.529, ROC-AUC 0.510 and log loss 0.6935.
* After 5.0 bp costs its net Sharpe was 0.42 versus 0.92 for buy-and-hold and 0.93 for always-long (0.30 with a one-day execution lag).
* This does not constitute evidence of a profitable strategy: the result does not beat simple passive benchmarks after costs on unseen data (and the data are synthetic).
* Any apparent edge should be treated as a hypothesis for further testing (new periods, more assets, realistic execution), not as a trading recommendation.

---

_Generated by `momentum_ml.evaluate` from config `sample.yaml`; seed 7; package versions: {'python': '3.13.15', 'pandas': '3.0.6', 'numpy': '2.5.3', 'scikit-learn': '1.9.1', 'xgboost_available': True, 'xgboost': '3.4.1'}._
