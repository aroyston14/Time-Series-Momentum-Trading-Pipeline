# time-series-momentum-ml

A research pipeline aimed at determining whether features from lagged prices and volumes can predict if an asset's adjusted close will be higher tomorrow than today, before checking if this predictive power can survive as an out-of-sample trading signal.


Result:
No model beat a buy-and-hold benchmark in out-of-sample testing after costs.

# How to Install:

```bash
git clone https://github.com/aroyston14/Time-Series-Momentum-Trading-Pipeline.git time-series-momentum-ml
cd time-series-momentum-ml
python -m venv .venv
.venv\Scripts\activate # or Mac: source .venv/bin/activate         
pip install --upgrade pip
pip install -e ".[dev]"              
```

# Running the synthetic demo dataset in data/sample/

```bash
python -m momentum_ml.run_pipeline --config configs/sample.yaml
```
The sample results can be found in reports/sample.

# Using the real data from Yahoo Finance (via yfinance): SPY QQQ IWM DIA EFA from 2010-01-01

```bash
python -m momentum_ml.download_data                       
python -m momentum_ml.build_features
python -m momentum_ml.train
python -m momentum_ml.evaluate                           
# or all four steps in 1:
python -m momentum_ml.run_pipeline
```

Setting the download options (can also be set in `configs/config.yaml`):

```bash
python -m momentum_ml.download_data --tickers SPY QQQ AAPL MSFT --start 2012-01-01 --end 2025-01-01 \
    --output-dir data/raw --force
```

# List of relevant commands:

`python -m momentum_ml.download_data` - downloads sample
`python -m momentum_ml.build_features` - builds the feature and  target panel in 'data/processed/features.csv'
`python -m momentum_ml.train` - tunes ( using TimeSeriesSplit), selects on validation, refits, predicts test, walks-forward
`python -m momentum_ml.evaluate` - produces metrics, importance, backtests, figures and the markdown report
`python -m momentum_ml.run_pipeline` - carries out all of the above
`python -m momentum_ml.sample_data` - regenerates the sample data

# File Structure

```
time-series-momentum-ml/
├── pyproject.toml            # dependencies, pytest, ruff, black config
├── configs/
│   ├── config.yaml           # main (real-data) configuration — fully commented
│   └── sample.yaml           # offline synthetic-data config (extends config.yaml)
├── data/
│   ├── raw/                  # cached downloads + download_manifest.json
│   ├── processed/            # features.csv + features_schema.json
│   └── sample/               # committed synthetic CSVs for the offline demo
├── models/                   # fitted models (train_fit/, final_fit/) + model_metadata.json
├── notebooks/                # exploration only — the pipeline lives in src/
├── reports/
│   ├── figures/              # PNG figures
│   ├── tables/               # CSV/JSON metrics, predictions, importances
│   └── research_report.md
├── src/momentum_ml/
│   ├── config.py             # YAML loading, extends/merge, validation, seeds
│   ├── data.py               # yfinance download, cleaning, caching, manifest
│   ├── sample_data.py        # synthetic data generator
│   ├── features.py           # features + target (no look-ahead)
│   ├── splits.py             # chronological split, purged date-grouped CV, walk-forward
│   ├── models.py             # model zoo, grids, tuning
│   ├── metrics.py            # classification metrics
│   ├── importance.py         # LR coefficients, RF impurity, XGB gain, permutation
│   ├── backtest.py           # positions, costs, benchmarks, performance metrics
│   ├── plots.py / report.py  # figures and the Markdown report
│   └── download_data.py, build_features.py, train.py, evaluate.py, run_pipeline.py  # CLIs
└── tests/                    # pytest suite
```

# Price data fields
`open`, `high`, `low` - session prices, not adjusted for dividends
`close` - official close, split-adjusted only (Yahoo convention) 
`adj_close` - close adjusted for splits **and** dividends/distributions
`volume` - shares traded (split-adjusted by Yahoo)

The same-day ratios such as `(high − low) / close` or `close / open − 1` use unadjusted fields from
the same day, so the adjustment factor cancels.

# How I prevented look-ahead bias

1. Every rolling window is trailing (`rolling(w)` ending at t); no centred windows, no negative shifts.
2. Only the target looks forward, and it is excluded from model inputs by construction
   (`features.feature_columns`).
3. No forward-filling or imputation of prices; rows with missing inputs are dropped.
4. Market features are lagged (default 1 day; `lag: 0` is also look-ahead-free for US-listed assets
   sharing SPY's close, but 1 is conservative).
5. Splits are by date, so every ticker's row for a date lands in the same split; the last
   `horizon` dates of train and validation are purged because their labels use prices from the
   next segment. CV folds use the same date grouping with a gap.
6. `StandardScaler` sits inside the logistic-regression `Pipeline`, so it is re-fitted on each fold.
7. The tests (`tests/test_features.py`) recompute features on data truncated at t, and on data with
   every price after t randomly perturbed, and assert row t is unchanged — for single assets and for
   the full panel including market features.

# Models used:

Majority-class baseline (`DummyClassifier(strategy="prior")`)
Logistic regression (unregularised, L1, L2; `class_weight="balanced"`, scaled in a Pipeline, `C` tuned)
random forest (`class_weight="balanced_subsample"`, trees/depth/min leaf/max features tuned)
XGBoost (learning rate/depth/estimators/subsample/colsample and L1/L2 leaf-weight penalties `reg_alpha`/`reg_lambda` tuned)

The tree models do not scale inputs.

Seeds come from `project.seed`; `n_jobs=1` to ensure determinism.

# Backtesting conventions:

* Long if P(up) ≥ `backtest.threshold` (0.50), else cash. Optional long/short mode.
* `execution_lag: 0` — position taken at the close of t (market-on-close; idealised because the
  close is also an input). `execution_lag: 1` trades a day later. Both are reported.
* Position `pos_t` earns `fwd_ret_t`, realised after the decision.
* Cost = |Δposition| × `cost_bps` (5 bp one-way default); the first entry is charged.
* Equal-weight sleeves per ticker, rebalanced daily; cash earns 0; Sharpe uses `risk_free_rate`.
* Benchmarks: buy-and-hold (equal initial weights, drifting), always-long (daily-rebalanced),
  `ma_rule_50d` (long when price > 50-day SMA), and the majority-class model.
* Metrics: cumulative/annualised return, volatility, Sharpe (+ rough t-stat), max drawdown, Calmar
  (≥ 1 year only), hit rate, average trade return, turnover, number of trades, total cost.

# Outputs:

`data/raw/*.csv`, `download_manifest.json` - cleaned raw data and download log
`data/processed/features.csv`, `features_schema.json` - feature panel and definitions 
`models/model_metadata.json`, `models/*_fit/*.joblib` - hyperparameters, CV scores, versions, fitted models
`reports/tables/predictions_{validation,test,walk_forward}.csv` - out-of-sample probabilities
`reports/tables/classification_metrics.{csv,json}` - accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, log loss, Brier, confusion counts
`reports/tables/backtest_metrics_*.csv`, `backtest_metrics.json` - strategy and benchmark performance 
`reports/tables/importance_*.csv` - LR coefficients, RF impurity, XGB gain, permutation importance 
`reports/tables/sensitivity_test_selected.csv` - cost × execution-lag sensitivity 
`reports/figures/*.png` - cumulative returns, drawdowns, rolling vol, confusion matrices, ROC, calibration, importance
`reports/research_report.md` - the research report 

# Testing:

```bash
pytest                      # 49 tests, ~15 s
ruff check src tests
black --check src tests
```

Tests cover target construction, return/rolling calculations, absence of future data, chronological
split ordering and purging, CV and walk-forward folds, transaction costs, execution lag, max drawdown,
benchmarks, missing-value handling, failed downloads, config validation, reproducibility with a fixed
seed, and an end-to-end pipeline run.

# Some assumptions and limitations of the project:

* Yahoo's adjusted close is taken as the total-return series; vendor data can be revised.
* The universe is survivorship-biased (chosen today). EFA's close is not synchronous with its
  foreign holdings.
* Flat bp costs ignore spread widening, impact and taxes; daily strategies are very cost-sensitive.
* Many configurations are tried; uncorrected p-values and Sharpe t-stats overstate significance.
* Accuracy above 50 % is not economic value — compare cost-adjusted Sharpe to the benchmarks.
* This is further discussed in the limitations section of the report

# Use of generative AI in this project:

This was the first professional research project I have conducted of this size and I used generative AI to guide me regarding proper programming etiquette and file structure, along with help with debugging and some data formatting issues. Furthermore, I used an LLM to produce the report based on the results of each model, having given it the information and limitations I wanted to include.