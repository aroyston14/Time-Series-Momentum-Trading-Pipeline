# Tests for how the target is built

import numpy as np
import pandas as pd
import pytest

from momentum_ml.features import compute_target


def test_next_day_target_hand_example():
    prices = pd.Series([100.0, 101.0, 100.0, 100.0, 102.0])
    forward_return, target = compute_target(prices, horizon=1, threshold=0.0)
    np.testing.assert_allclose(forward_return.iloc[:4], [0.01, 100 / 101 - 1, 0.0, 0.02])

    # A flat day (zero return) is not an up day, since the target needs fwd_ret > threshold
    assert target.iloc[:4].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert np.isnan(forward_return.iloc[-1]) and np.isnan(target.iloc[-1])


def test_target_aligned_to_decision_date():
    prices = pd.Series([10.0, 11.0, 9.0], index=pd.date_range("2024-01-01", periods=3))
    forward_return, _ = compute_target(prices)

    # Row t holds the return from t to t+1
    assert forward_return.loc["2024-01-01"] == pytest.approx(0.1)
    assert forward_return.loc["2024-01-02"] == pytest.approx(9 / 11 - 1)


def test_multi_day_horizon_and_threshold():
    prices = pd.Series([100.0, 100.5, 101.0, 100.0])
    forward_return, target = compute_target(prices, horizon=2, threshold=0.005)
    np.testing.assert_allclose(forward_return.iloc[:2], [0.01, 100 / 100.5 - 1])
    assert target.iloc[:2].tolist() == [1.0, 0.0]
    assert target.iloc[2:].isna().all()


def test_invalid_horizon():
    with pytest.raises(ValueError):
        compute_target(pd.Series([1.0, 2.0]), horizon=0)
