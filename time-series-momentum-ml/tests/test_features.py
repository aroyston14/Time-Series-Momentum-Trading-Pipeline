# Tests for the return and rolling feature calculations, and that no future data is used

import numpy as np
import pandas as pd
import pytest

from conftest import assert_frame_close
from momentum_ml.features import (
    FWD_RET_COL,
    TARGET_COL,
    build_feature_panel,
    compute_asset_features,
    compute_market_features,
    feature_columns,
    ma_distance,
    ma_spread,
    max_drawdown,
    rolling_max_drawdown,
    rolling_volatility,
    simple_returns,
)


def test_simple_returns():
    prices = pd.Series([100.0, 110.0, 99.0, 108.9])
    np.testing.assert_allclose(simple_returns(prices, 1).iloc[1:], [0.1, -0.1, 0.1])
    assert simple_returns(prices, 2).iloc[2] == pytest.approx(99 / 100 - 1)
    assert simple_returns(prices, 2).iloc[:2].isna().all()


def test_rolling_volatility_matches_numpy():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 50)))
    volatility = rolling_volatility(prices, 10, annualise=False)

    # The value on day 30 should be the std of the returns for days 21 to 30 inclusive
    daily_returns = prices.pct_change().to_numpy()
    expected = np.std(daily_returns[30 - 9 : 31], ddof=1)
    assert volatility.iloc[30] == pytest.approx(expected)
    assert volatility.iloc[:10].isna().all() and not np.isnan(volatility.iloc[10])
    assert rolling_volatility(prices, 10, annualise=True).iloc[30] == pytest.approx(
        expected * np.sqrt(252)
    )


def test_moving_average_features():
    prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert ma_distance(prices, 3).iloc[5] == pytest.approx(6 / 5 - 1)
    assert ma_spread(prices, 2, 4).iloc[5] == pytest.approx(5.5 / 4.5 - 1)
    with pytest.raises(ValueError):
        ma_spread(prices, 4, 2)


def test_max_drawdown_known_values():
    assert max_drawdown([100, 120, 90, 130, 65]) == pytest.approx(65 / 130 - 1)
    assert max_drawdown([1, 2, 3]) == 0.0
    assert max_drawdown([100, 50, 75]) == pytest.approx(-0.5)
    assert np.isnan(max_drawdown([]))


def test_rolling_max_drawdown_matches_naive():
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 120)))
    vectorised = rolling_max_drawdown(prices, 20)

    # Working out each window one at a time should give the same answer
    looped = [np.nan] * 19
    for i in range(19, len(prices)):
        looped.append(max_drawdown(prices.iloc[i - 19 : i + 1]))
    np.testing.assert_allclose(vectorised, looped, equal_nan=True)


def test_intraday_and_volume_features(prices, config):
    features = compute_asset_features(prices, config["features"])
    i = 100
    row = prices.iloc[i]
    assert features["hl_range"].iloc[i] == pytest.approx((row["high"] - row["low"]) / row["close"])
    assert features["oc_return"].iloc[i] == pytest.approx(row["close"] / row["open"] - 1)
    assert features["volume_pct_change"].iloc[i] == pytest.approx(
        row["volume"] / prices["volume"].iloc[i - 1] - 1
    )
    assert features["volume_rel_ma20"].iloc[i] == pytest.approx(
        row["volume"] / prices["volume"].iloc[i - 19 : i + 1].mean()
    )
    assert features["ret_5d"].iloc[i] == pytest.approx(
        prices["adj_close"].iloc[i] / prices["adj_close"].iloc[i - 5] - 1
    )


# The tests below check for look-ahead bias


@pytest.mark.parametrize("row_index", [70, 120, 200, 250])
def test_features_identical_when_future_is_truncated(prices, config, row_index):
    full = compute_asset_features(prices, config["features"])
    truncated = compute_asset_features(prices.iloc[: row_index + 1], config["features"])
    assert_frame_close(full.iloc[[row_index]], truncated.iloc[[row_index]])


def test_features_unchanged_when_future_prices_are_perturbed(prices, config):
    cutoff = 150

    # Randomly scaling every price after the cutoff by between 0.5 and 1.5
    shocked = prices.copy()
    rng = np.random.default_rng(99)
    shocked.iloc[cutoff + 1 :] = shocked.iloc[cutoff + 1 :] * rng.uniform(
        0.5, 1.5, size=shocked.iloc[cutoff + 1 :].shape
    )
    original_features = compute_asset_features(prices, config["features"])
    shocked_features = compute_asset_features(shocked, config["features"])
    assert_frame_close(original_features.iloc[: cutoff + 1], shocked_features.iloc[: cutoff + 1])

    # The change should show up after the cutoff, otherwise the test would not prove anything
    original_after = original_features.iloc[cutoff + 1 :].fillna(0).to_numpy()
    shocked_after = shocked_features.iloc[cutoff + 1 :].fillna(0).to_numpy()
    assert not np.allclose(original_after, shocked_after)


def test_market_features_are_lagged(universe, config):
    spy = universe["SPY"]
    market_config = dict(config["features"]["market_features"], lag=1)
    market_features = compute_market_features(spy, market_config, config["features"])

    # The value on row t should equal SPY's 1-day return on t-1
    spy_returns = spy["adj_close"].pct_change()
    assert market_features["mkt_ret_1d"].iloc[50] == pytest.approx(spy_returns.iloc[49])
    with pytest.raises(ValueError):
        compute_market_features(spy, dict(market_config, lag=-1), config["features"])


# End to end check that the panel features up to a date are the same when every observation after
# that date is removed
def test_panel_rows_do_not_use_future_observations(universe, config):
    cutoff = pd.Timestamp("2020-06-30")
    full_panel = build_feature_panel(universe, config)
    truncated_data = {ticker: df.loc[:cutoff] for ticker, df in universe.items()}
    truncated_panel = build_feature_panel(truncated_data, config)

    features = feature_columns(full_panel)
    before_cutoff = full_panel[full_panel["date"] < cutoff].set_index(["date", "ticker"])[features]
    truncated = truncated_panel.set_index(["date", "ticker"])[features].reindex(before_cutoff.index)
    assert_frame_close(before_cutoff, truncated)


def test_panel_structure(universe, config):
    panel = build_feature_panel(universe, config)
    features = feature_columns(panel)
    assert TARGET_COL not in features and FWD_RET_COL not in features and "date" not in features
    assert not panel[features + [TARGET_COL, FWD_RET_COL]].isna().any().any()
    assert set(panel[TARGET_COL].unique()) <= {0, 1}
    assert not panel.duplicated(["date", "ticker"]).any()
    assert panel["date"].is_monotonic_increasing
