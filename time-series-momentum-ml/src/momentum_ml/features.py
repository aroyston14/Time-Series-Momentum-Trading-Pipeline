# Building the features and the target.
#
# Timing convention: row t of the feature panel describes trading date t, and every feature on
# that row only uses data dated on or before t. Rolling windows always end at t (they are never
# centred) and shift is only ever used with positive lags. The target on row t is the forward
# adjusted close return from t to t + horizon. It is the only column that looks ahead, and it is
# never used as a model input.
#
# tests/test_features.py checks for leakage by recomputing the features on data cut off at t,
# and on data where every price after t has been changed. Row t must come out the same.

import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

TARGET_COL = "target"
FWD_RET_COL = "fwd_ret"
ID_COLS = ["date", "ticker"]


# Simple return over the last `window` days at each date, P_t / P_(t-window) - 1
def simple_returns(prices, window):
    if window < 1:
        raise ValueError(f"return window must be >= 1, got {window}")
    return prices / prices.shift(window) - 1.0


# Sample standard deviation of the daily returns over the last `window` days, including day t
def rolling_volatility(prices, window, annualise=True, periods=252):
    if window < 2:
        raise ValueError(f"volatility window must be >= 2, got {window}")
    daily_returns = prices.pct_change(fill_method=None)
    volatility = daily_returns.rolling(window, min_periods=window).std(ddof=1)
    return volatility * np.sqrt(periods) if annualise else volatility


# How far the price is above or below its trailing simple moving average, as a fraction
def ma_distance(prices, window):
    return prices / prices.rolling(window, min_periods=window).mean() - 1.0


# The fast trailing moving average relative to the slow one, minus one
def ma_spread(prices, fast, slow):
    if fast >= slow:
        raise ValueError(f"ma_spread fast window ({fast}) must be < slow window ({slow})")
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    return fast_ma / slow_ma - 1.0


# Maximum drawdown of a price or equity path, as a fraction that is zero or negative (e.g. -0.25).
# Returns nan if the path is empty or contains a NaN.
def max_drawdown(values):
    path = np.asarray(values, dtype=float)
    if path.size == 0 or np.isnan(path).any():
        return float("nan")
    running_peaks = np.maximum.accumulate(path)
    return float(np.min(path / running_peaks - 1.0))


# Maximum drawdown within the last `window` prices at each date.
# sliding_window_view lets every window be worked out at once rather than looping over dates.
def rolling_max_drawdown(prices, window):
    values = prices.to_numpy(dtype=float)
    result = np.full(values.shape, np.nan)
    if len(values) >= window:
        windows = sliding_window_view(values, window)
        running_peaks = np.maximum.accumulate(windows, axis=1)
        drawdowns = np.min(windows / running_peaks - 1.0, axis=1)
        drawdowns[np.isnan(windows).any(axis=1)] = np.nan
        result[window - 1 :] = drawdowns
    return pd.Series(result, index=prices.index)


# Works out all of the single-asset features for one ticker's cleaned OHLCV data.
# df must be sorted by date, and feature_config is the "features" section of the config.
# The rows at the start where the windows have not filled up yet contain NaN.
def compute_asset_features(df, feature_config):
    if not df.index.is_monotonic_increasing:
        raise ValueError("Input frame must be sorted by date before feature construction")
    prices = df["adj_close"].astype(float)
    annualise = bool(feature_config.get("annualise_volatility", True))

    # The order features are added in here sets the column order of the panel
    features = {}
    for window in feature_config["return_windows"]:
        features[f"ret_{window}d"] = simple_returns(prices, int(window))
    for window in feature_config["volatility_windows"]:
        features[f"vol_{window}d"] = rolling_volatility(prices, int(window), annualise=annualise)
    for window in feature_config["ma_distance_windows"]:
        features[f"dist_ma_{window}"] = ma_distance(prices, int(window))
    fast, slow = (int(window) for window in feature_config["ma_spread"])
    features[f"ma_spread_{fast}_{slow}"] = ma_spread(prices, fast, slow)

    # Intraday ratios use the unadjusted fields from the same day, so the adjustment cancels out
    features["hl_range"] = (df["high"] - df["low"]) / df["close"]
    features["oc_return"] = df["close"] / df["open"] - 1.0

    # Zero volume days are treated as missing to avoid dividing by zero
    volume = df["volume"].astype(float).where(df["volume"] > 0)
    features["volume_pct_change"] = volume.pct_change(fill_method=None)
    volume_window = int(feature_config.get("volume_ma_window", 20))
    volume_ma = volume.rolling(volume_window, min_periods=volume_window).mean()
    features[f"volume_rel_ma{volume_window}"] = volume / volume_ma

    for window in feature_config["drawdown_windows"]:
        features[f"mdd_{window}d"] = rolling_max_drawdown(prices, int(window))

    feature_frame = pd.DataFrame(features, index=df.index)
    return feature_frame.replace([np.inf, -np.inf], np.nan)


# Market (SPY) return and volatility features, shifted forward by the configured lag.
# A positive lag means row t holds SPY information from t - lag. The columns start with "mkt_".
def compute_market_features(market_df, market_config, feature_config):
    prices = market_df["adj_close"].astype(float)
    annualise = bool(feature_config.get("annualise_volatility", True))
    lag = int(market_config.get("lag", 1))
    if lag < 0:
        raise ValueError("market_features.lag must be >= 0 (a negative lag would leak future data)")

    features = {}
    for window in market_config.get("return_windows", [1, 5, 20]):
        features[f"mkt_ret_{window}d"] = simple_returns(prices, int(window))
    for window in market_config.get("volatility_windows", [20]):
        features[f"mkt_vol_{window}d"] = rolling_volatility(
            prices, int(window), annualise=annualise
        )
    return pd.DataFrame(features, index=market_df.index).shift(lag)


# The forward return and the binary up/down target.
# fwd_ret_t = P_(t+h) / P_t - 1, and target_t = 1 if fwd_ret_t > threshold, otherwise 0.
# The last `horizon` rows have no known outcome yet, so they are NaN in both.
def compute_target(prices, horizon=1, threshold=0.0):
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    forward_return = prices.shift(-horizon) / prices - 1.0
    target = (forward_return > threshold).astype(float).where(forward_return.notna())
    return forward_return, target


# The model input columns, which is everything apart from the identifiers, target and forward return
def feature_columns(panel):
    excluded = set(ID_COLS) | {TARGET_COL, FWD_RET_COL}
    return [column for column in panel.columns if column not in excluded]


# Builds the long (date, ticker) panel of features, forward returns and targets.
# Any row missing a feature or target (window warm-up, the last `horizon` days, missing volume,
# no market data that day) is dropped and the counts are logged. Nothing is imputed or
# forward-filled.
def build_feature_panel(raw_data, config):
    feature_config = config["features"]
    target_config = config["target"]
    market_config = feature_config.get("market_features", {}) or {}
    market_ticker = str(config["data"].get("market_ticker", "SPY")).upper()
    from momentum_ml.config import all_tickers

    tickers = [ticker for ticker in all_tickers(config) if ticker in raw_data]

    market_features = None
    if market_config.get("enabled", False):
        if market_ticker not in raw_data:
            raise ValueError(
                f"market_features.enabled is true but market ticker {market_ticker} has no "
                "data; disable market features or fix the download"
            )
        market_features = compute_market_features(
            raw_data[market_ticker], market_config, feature_config
        )

    ticker_panels = []
    for ticker in tickers:
        df = raw_data[ticker].sort_index()
        features = compute_asset_features(df, feature_config)

        # The asset's own calendar is used, so any days SPY is missing become NaN
        if market_features is not None:
            features = features.join(market_features, how="left")

        forward_return, target = compute_target(
            df["adj_close"], int(target_config["horizon"]), float(target_config["threshold"])
        )
        features[FWD_RET_COL] = forward_return
        features[TARGET_COL] = target

        rows_before = len(features)
        features = features.dropna()
        logger.info(
            "%s: %d feature rows kept, %d dropped (warm-up / NaN / no outcome)",
            ticker,
            len(features),
            rows_before - len(features),
        )
        features.insert(0, "ticker", ticker)
        ticker_panels.append(features.reset_index())

    if not ticker_panels:
        raise ValueError("No tickers available to build features")

    panel = pd.concat(ticker_panels, ignore_index=True)
    panel[TARGET_COL] = panel[TARGET_COL].astype(int)
    panel = panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    logger.info(
        "Feature panel: %d rows, %d tickers, %d features, %s -> %s, up-day rate %.3f",
        len(panel),
        panel["ticker"].nunique(),
        len(feature_columns(panel)),
        panel["date"].min().date(),
        panel["date"].max().date(),
        panel[TARGET_COL].mean(),
    )
    return panel


# A readable definition of every feature the config produces, which is used in the report
def describe_features(config):
    feature_config = config["features"]
    market_config = feature_config.get("market_features", {}) or {}
    annualised_note = (
        " (annualised x sqrt(252))" if feature_config.get("annualise_volatility", True) else ""
    )

    definitions = {}
    for window in feature_config["return_windows"]:
        definitions[f"ret_{window}d"] = f"AdjClose_t / AdjClose_(t-{window}) - 1"
    for window in feature_config["volatility_windows"]:
        definitions[f"vol_{window}d"] = (
            f"std of daily adj-close returns over days t-{int(window) - 1}..t{annualised_note}"
        )
    for window in feature_config["ma_distance_windows"]:
        definitions[f"dist_ma_{window}"] = (
            f"AdjClose_t / SMA{window}_t - 1 (SMA over days t-{int(window) - 1}..t)"
        )
    fast, slow = feature_config["ma_spread"]
    definitions[f"ma_spread_{fast}_{slow}"] = f"SMA{fast}_t / SMA{slow}_t - 1"
    definitions["hl_range"] = (
        "(High_t - Low_t) / Close_t  (same-day unadjusted; equals adjusted ratio)"
    )
    definitions["oc_return"] = "Close_t / Open_t - 1 (intraday return)"
    definitions["volume_pct_change"] = "Volume_t / Volume_(t-1) - 1"
    volume_window = feature_config.get("volume_ma_window", 20)
    definitions[f"volume_rel_ma{volume_window}"] = (
        f"Volume_t / mean(Volume over days t-{int(volume_window) - 1}..t)"
    )
    for window in feature_config["drawdown_windows"]:
        definitions[f"mdd_{window}d"] = (
            f"max drawdown of AdjClose within days t-{int(window) - 1}..t (<= 0)"
        )

    if market_config.get("enabled", False):
        lag = int(market_config.get("lag", 1))
        market_ticker = config["data"].get("market_ticker", "SPY")
        for window in market_config.get("return_windows", [1, 5, 20]):
            definitions[f"mkt_ret_{window}d"] = (
                f"{market_ticker} {window}-day return, lagged {lag} day(s)"
            )
        for window in market_config.get("volatility_windows", [20]):
            definitions[f"mkt_vol_{window}d"] = (
                f"{market_ticker} {window}-day volatility{annualised_note}, lagged {lag} day(s)"
            )
    return definitions
