# Tests for the data cleaning, missing values, and handling downloads that fail

import json

import numpy as np
import pandas as pd
import pytest

from momentum_ml import data as data_module
from momentum_ml.data import DataError, acquire_data, clean_ohlcv
from momentum_ml.features import build_feature_panel, feature_columns
from momentum_ml.sample_data import generate_sample_universe


# A small Yahoo-style dataset with a duplicate date, a missing adj close, a high below the low
# on 2024-01-04, a negative volume and a 10 day gap
def _raw():
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-15"]
    )
    return pd.DataFrame(
        {
            "Open": [10, 11, 11, 12, 13, 14],
            "High": [11, 12, 12.5, 11, 14, 15],
            "Low": [9, 10, 10, 12.5, 12, 13],
            "Close": [10.5, 11.5, 11.6, 12.0, 13.5, 14.5],
            "Adj Close": [10.4, 11.4, 11.5, np.nan, 13.4, 14.4],
            "Volume": [100, 200, 210, 300, -5, 400],
        },
        index=dates,
    )


def test_clean_ohlcv_handles_defects():
    df, report = clean_ohlcv(_raw(), "X")
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]

    # The last of the duplicate rows is the one kept
    assert report.duplicate_dates_dropped == 1
    assert df.loc["2024-01-03", "adj_close"] == 11.5

    assert report.missing_adj_close_dropped == 1 and pd.Timestamp("2024-01-04") not in df.index
    assert report.negative_volume_nulled == 1 and np.isnan(df.loc["2024-01-05", "volume"])
    assert report.calendar_gaps_over_5_days == 1
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates


def test_clean_ohlcv_inconsistent_high_low_nulled():
    raw = _raw()
    raw.loc["2024-01-04", "Adj Close"] = 12.0
    df, report = clean_ohlcv(raw, "X")
    assert report.inconsistent_high_low_nulled == 1
    assert df.loc["2024-01-04", ["high", "low"]].isna().all()


def test_clean_ohlcv_multiindex_and_missing_adj_close():
    raw = _raw().drop(columns=["Adj Close"])
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["X"]])
    df, report = clean_ohlcv(raw, "X")
    assert (df["adj_close"] == df["close"]).all()
    assert any("fell back to close" in note for note in report.notes)


def test_missing_values_drop_rows_without_imputation(universe, config):
    edited_universe = {ticker: df.copy() for ticker, df in universe.items()}
    qqq = edited_universe["QQQ"]
    qqq.iloc[300, qqq.columns.get_loc("volume")] = np.nan
    panel = build_feature_panel(edited_universe, config)
    qqq_panel = panel[panel["ticker"] == "QQQ"]

    # The day with the missing volume and the day after it (which needs it for the volume
    # percentage change) should be left out rather than filled in
    missing_day = qqq.index[300]
    next_day = qqq.index[301]
    assert missing_day not in set(qqq_panel["date"]) and next_day not in set(qqq_panel["date"])
    assert not panel[feature_columns(panel)].isna().any().any()


def test_failed_ticker_is_skipped_and_logged(config, monkeypatch):
    config["data"].update(source="yahoo", tickers=["SPY", "QQQ", "BAD"], max_retries=1)
    good_data = generate_sample_universe(tickers=["SPY", "QQQ"])

    # Stand-in for the Yahoo download that fails for the "BAD" ticker
    def fake_download(ticker, *args, **kwargs):
        if ticker == "BAD":
            raise DataError("BAD: Yahoo returned no rows")
        return good_data[ticker]

    monkeypatch.setattr(data_module, "download_yahoo", fake_download)
    price_data = acquire_data(config, force=True)
    assert set(price_data) == {"SPY", "QQQ"}

    raw_dir = data_module.resolve_path(config, config["data"]["raw_dir"])
    manifest = json.loads((raw_dir / "download_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tickers"]["BAD"]["status"] == "failed"
    assert manifest["tickers"]["SPY"]["status"] == "ok"

    # The cached files should be reused on the next call, so any download attempt fails the test
    monkeypatch.setattr(
        data_module, "download_yahoo", lambda *args, **kwargs: pytest.fail("should use cache")
    )
    config["data"]["tickers"] = ["SPY", "QQQ"]
    cached_data = acquire_data(config)
    pd.testing.assert_frame_equal(price_data["SPY"], cached_data["SPY"], check_freq=False)


def test_all_tickers_failing_raises(config, monkeypatch):
    config["data"].update(source="yahoo", tickers=["BAD1", "BAD2"], max_retries=1)
    config["features"]["market_features"]["enabled"] = False

    def failing_download(*args, **kwargs):
        raise DataError("network down")

    monkeypatch.setattr(data_module, "download_yahoo", failing_download)
    with pytest.raises(DataError, match="No modelling tickers"):
        acquire_data(config, force=True)
