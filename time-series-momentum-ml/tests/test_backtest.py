# Tests for the transaction costs, execution timing, benchmarks and performance metrics

import numpy as np
import pandas as pd
import pytest

from momentum_ml.backtest import (
    backtest_positions,
    benchmark_signals,
    buy_and_hold_returns,
    performance_metrics,
    portfolio_returns,
    signals_from_probs,
    trade_returns,
)


# Builds a one ticker backtest frame from a list of signals and forward returns
def _frame(signals, forward_returns, ticker="A"):
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(signals)),
            "ticker": ticker,
            "fwd_ret": forward_returns,
            "signal": signals,
        }
    )


def test_transaction_costs_hand_example():
    frame = _frame([1, 1, 0, 1, 0], [0.01, 0.02, -0.03, 0.01, 0.05])
    result = backtest_positions(frame, cost_bps=5.0, execution_lag=0)
    np.testing.assert_allclose(result["turnover"], [1, 0, 1, 1, 1])
    np.testing.assert_allclose(result["cost"], [0.0005, 0, 0.0005, 0.0005, 0.0005])
    np.testing.assert_allclose(result["gross_ret"], [0.01, 0.02, 0.0, 0.01, 0.0])
    np.testing.assert_allclose(result["net_ret"], [0.0095, 0.02, -0.0005, 0.0095, -0.0005])


def test_zero_cost_equals_gross():
    frame = _frame([1, 0, 1], [0.01, 0.02, 0.03])
    result = backtest_positions(frame, cost_bps=0.0)
    np.testing.assert_allclose(result["net_ret"], result["gross_ret"])


def test_long_short_flip_costs_two_units():
    signals = signals_from_probs(
        [0.6, 0.4, 0.6], threshold=0.5, long_short=True, short_threshold=0.5
    )
    np.testing.assert_array_equal(signals, [1, -1, 1])

    # Going from long to short is a change of two units of position
    result = backtest_positions(_frame(signals, [0.01, -0.02, 0.01]), cost_bps=10.0)
    np.testing.assert_allclose(result["turnover"], [1, 2, 2])
    np.testing.assert_allclose(result["gross_ret"], [0.01, 0.02, 0.01])


def test_long_only_default_never_shorts():
    signals = signals_from_probs([0.1, 0.49, 0.5, 0.9])
    np.testing.assert_array_equal(signals, [0, 0, 1, 1])


def test_execution_lag_delays_position():
    frame = _frame([1, 0, 0, 1], [0.01, 0.02, 0.03, 0.04])
    result = backtest_positions(frame, cost_bps=0.0, execution_lag=1)
    np.testing.assert_allclose(result["position"], [0, 1, 0, 0])
    np.testing.assert_allclose(result["gross_ret"], [0, 0.02, 0, 0])


# Changing future signals must not change the profit and loss of earlier days
def test_position_uses_only_its_own_row_signal():
    forward_returns = [0.01, -0.02, 0.03, 0.04]
    original = backtest_positions(_frame([1, 0, 1, 1], forward_returns), cost_bps=5)
    changed_future = backtest_positions(_frame([1, 0, 0, 0], forward_returns), cost_bps=5)
    pd.testing.assert_frame_equal(original.iloc[:2], changed_future.iloc[:2])


def test_costs_are_per_ticker_and_portfolio_is_equal_weight():
    frame = pd.concat([_frame([1, 1], [0.02, 0.0], "A"), _frame([0, 1], [0.04, 0.02], "B")])
    per_ticker = backtest_positions(frame, cost_bps=10.0)
    portfolio = portfolio_returns(per_ticker)

    # Day 1: A enters (10bp cost) and earns 2% while B is in cash.
    # Day 2: A holds and earns 0%, while B enters and earns 2%.
    np.testing.assert_allclose(portfolio["net_ret"], [(0.02 - 0.001) / 2, (0.02 - 0.001) / 2])
    np.testing.assert_allclose(portfolio["turnover"], [0.5, 0.5])


def test_trade_returns_include_entry_and_exit_costs():
    per_ticker = backtest_positions(_frame([1, 1, 0, 0], [0.10, 0.10, 0.5, 0.0]), cost_bps=100.0)
    trades = trade_returns(per_ticker)
    assert len(trades) == 1
    assert trades.iloc[0] == pytest.approx((1 + 0.10 - 0.01) * 1.10 * (1 - 0.01) - 1)


def test_buy_and_hold_lets_weights_drift():
    frame = pd.concat([_frame([1, 1], [1.0, 0.0], "A"), _frame([1, 1], [0.0, 0.5], "B")])

    # Day 1: A doubles, so the value is (2 + 1) / 2 = 1.5.
    # Day 2: B rises 50%, so the value is (2 + 1.5) / 2 = 1.75.
    no_costs = buy_and_hold_returns(frame, cost_bps=0.0)
    np.testing.assert_allclose(no_costs["net_ret"], [0.5, 1.75 / 1.5 - 1])

    # Only the first day should be charged
    with_costs = buy_and_hold_returns(frame, cost_bps=5.0)
    assert with_costs["net_ret"].iloc[0] == pytest.approx(0.5 - 0.0005)
    assert with_costs["cost"].sum() == pytest.approx(0.0005)


def test_benchmark_signals():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "ticker": "A",
            "fwd_ret": 0.0,
            "dist_ma_50": [0.01, -0.02, 0.0],
        }
    )
    signals = benchmark_signals(frame, {"ma_rule_window": 50})
    np.testing.assert_array_equal(signals["always_long"], [1, 1, 1])
    np.testing.assert_array_equal(signals["ma_rule_50d"], [1, 0, 0])

    # There is no dist_ma_20 column, so a 20 day rule should raise an error
    with pytest.raises(ValueError):
        benchmark_signals(frame, {"ma_rule_window": 20})


def test_performance_metrics_known_series():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0] * 63)
    portfolio = pd.DataFrame({"net_ret": returns, "turnover": 0.0, "cost": 0.0, "exposure": 1.0})
    metrics = performance_metrics(portfolio, risk_free_rate=0.0)

    equity = np.cumprod(1 + returns)
    running_peaks = np.maximum.accumulate(np.r_[1, equity])[1:]
    assert metrics["cumulative_return"] == pytest.approx(equity.iloc[-1] - 1)
    assert metrics["annualised_return"] == pytest.approx(
        equity.iloc[-1] ** (252 / len(returns)) - 1
    )
    assert metrics["sharpe"] == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))
    assert metrics["max_drawdown"] == pytest.approx(np.min(equity / running_peaks - 1))
    assert metrics["hit_rate"] == pytest.approx(0.5)

    # A positive risk-free rate should lower the Sharpe ratio
    metrics_with_rf = performance_metrics(portfolio, risk_free_rate=0.05)
    assert metrics_with_rf["sharpe"] < metrics["sharpe"]


def test_model_diagnostics_flags_constant_model_as_degenerate():
    from momentum_ml.evaluate import model_diagnostics

    rng = np.random.default_rng(0)
    num_days = 50
    dates = pd.bdate_range("2024-01-01", periods=num_days)
    forward_returns = rng.normal(0, 0.01, num_days)

    # "zeroed" always predicts exactly 0.5, "live" varies, and "majority" is the baseline
    model_probabilities = {
        "zeroed": np.full(num_days, 0.5),
        "live": rng.uniform(0.3, 0.7, num_days),
        "majority": np.full(num_days, 0.54),
    }
    prediction_frames = []
    for model, probabilities in model_probabilities.items():
        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": "A",
                    "model": model,
                    "prob_up": probabilities,
                    "fwd_ret": forward_returns,
                }
            )
        )
    predictions = {"test": pd.concat(prediction_frames, ignore_index=True)}
    importances = {
        "coef_zeroed": pd.DataFrame({"feature": ["f1", "f2"], "coef_raw_units": [0.0, 0.0]}),
        "coef_live": pd.DataFrame({"feature": ["f1", "f2"], "coef_raw_units": [0.3, 0.0]}),
    }

    always_long_returns = pd.Series(forward_returns, index=dates)
    live_probabilities = predictions["test"].query("model == 'live'")["prob_up"]
    live_returns = pd.Series(np.where(live_probabilities >= 0.5, forward_returns, 0.0), index=dates)
    test_returns = {
        "zeroed": always_long_returns,
        "live": live_returns,
        "majority": always_long_returns,
        "always_long": always_long_returns,
        "buy_and_hold": always_long_returns,
    }
    config = {"backtest": {"threshold": 0.5}}
    diagnostics = model_diagnostics(predictions, importances, test_returns, config)

    zeroed = diagnostics["zeroed"]
    assert zeroed["degenerate"] and zeroed["constant_prediction"] and zeroed["on_threshold_tie"]
    assert zeroed["n_nonzero_coefficients"] == 0 and zeroed["identical_to_always_long"]
    assert zeroed["long_fraction"] == 1.0
    assert not diagnostics["live"]["degenerate"]
    assert diagnostics["live"]["n_nonzero_coefficients"] == 1

    # The majority baseline is constant by design and should never be flagged
    assert not diagnostics["majority"]["degenerate"]
