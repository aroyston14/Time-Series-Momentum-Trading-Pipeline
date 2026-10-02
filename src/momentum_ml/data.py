# Downloading, cleaning and caching the raw daily price data.
#
# Price fields (one row per trading date, indexed by the exchange date):
#   open      - first traded price of the session, not adjusted for splits or dividends
#   high      - highest traded price of the session, unadjusted
#   low       - lowest traded price of the session, unadjusted
#   close     - official closing price, adjusted for splits only (Yahoo convention)
#   adj_close - closing price adjusted for splits and dividends. Ratios of adj_close give total
#               returns, so it is used for every return, momentum, volatility, moving average,
#               drawdown and target calculation
#   volume    - shares traded in the session (Yahoo split-adjusts volume)
#
# Intraday ratios (high/low range, open to close return) use the unadjusted fields from the same
# day, so the adjustment factor cancels out.

# Standard library modules for the manifest file, logging, retry delays and the cleaning report
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from momentum_ml.config import all_tickers, resolve_path

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
MANIFEST_NAME = "download_manifest.json"


# Raised when there is a data problem the run cannot recover from, e.g. no usable tickers
class DataError(RuntimeError):
    pass


# Records what clean_ohlcv changed for one ticker, so it can be saved to the download manifest
@dataclass
class CleaningReport:
    ticker: str
    rows_in: int = 0
    rows_out: int = 0
    duplicate_dates_dropped: int = 0
    missing_adj_close_dropped: int = 0
    nonpositive_price_dropped: int = 0
    inconsistent_high_low_nulled: int = 0
    negative_volume_nulled: int = 0
    calendar_gaps_over_5_days: int = 0
    first_date: str | None = None
    last_date: str | None = None
    notes: list[str] = field(default_factory=list)


# Maps the yfinance or CSV column names onto the names in PRICE_COLUMNS
def _normalise_columns(df):

    # Newer versions of yfinance return (Price, Ticker) columns even for a single ticker,
    # so only the first level is kept
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    new_names = {}
    for column in df.columns:
        new_names[column] = str(column).strip().lower().replace(" ", "_")
    return df.rename(columns=new_names)


# Cleans one ticker's daily OHLCV data without making up any values.
# Every step is logged and counted in the returned CleaningReport:
#   1. Normalise the column names and the date index (timezone-naive, midnight, sorted).
#   2. Drop duplicate dates, keeping the last one since later vendor rows are corrections.
#   3. Drop rows with a missing or non-positive adj_close. The target and all the momentum
#      features depend on it, and forward-filling prices would create fake zero returns.
#   4. Set inconsistent high < low pairs and negative volumes to NaN rather than dropping the
#      row, so its other fields stay usable. Features that depend on them are dropped later.
#   5. Count calendar gaps longer than five days (holidays never go beyond four). Gaps are not
#      filled, returns simply span the gap, which is how the asset could actually be traded.
# Returns the cleaned data (with exactly the PRICE_COLUMNS) and the report.
def clean_ohlcv(raw, ticker):
    report = CleaningReport(ticker=ticker, rows_in=len(raw))
    df = _normalise_columns(raw)

    # Standardising the date index
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()]
    df.index = df.index.normalize()
    df.index.name = "date"

    # If there is no adjusted close the plain close is used instead, which means returns will
    # not include dividends, so a warning is logged
    if "adj_close" not in df.columns:
        if "close" not in df.columns:
            raise DataError(
                f"{ticker}: neither 'adj_close' nor 'close' present; columns={list(df.columns)}"
            )
        report.notes.append(
            "adj_close missing from source; fell back to close (no dividend adjustment)"
        )
        logger.warning(
            "%s: adj_close missing, using close instead (returns exclude dividends)", ticker
        )
        df["adj_close"] = df["close"]

    # Any other missing column is added as NaN so the rest of the pipeline still works
    for column in PRICE_COLUMNS:
        if column not in df.columns:
            report.notes.append(f"column '{column}' missing; filled with NaN")
            logger.warning("%s: column %s missing; dependent features will be NaN", ticker, column)
            df[column] = np.nan
    df = df[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()

    # Step 2, duplicate dates
    is_duplicate = df.index.duplicated(keep="last")
    report.duplicate_dates_dropped = int(is_duplicate.sum())
    df = df[~is_duplicate]

    # Step 3, missing or non-positive prices
    is_missing = df["adj_close"].isna()
    report.missing_adj_close_dropped = int(is_missing.sum())
    df = df[~is_missing]

    price_columns = ["open", "high", "low", "close", "adj_close"]
    is_nonpositive = (df["adj_close"] <= 0) | (df[price_columns[:-1]] <= 0).any(axis=1)
    report.nonpositive_price_dropped = int(is_nonpositive.sum())
    df = df[~is_nonpositive]

    # Step 4, impossible high/low pairs and negative volumes
    bad_high_low = df["high"] < df["low"]
    report.inconsistent_high_low_nulled = int(bad_high_low.sum())
    df.loc[bad_high_low, ["high", "low"]] = np.nan

    negative_volume = df["volume"] < 0
    report.negative_volume_nulled = int(negative_volume.sum())
    df.loc[negative_volume, "volume"] = np.nan

    # Step 5, counting long calendar gaps
    if len(df) > 1:
        gap_days = df.index.to_series().diff().dt.days
        report.calendar_gaps_over_5_days = int((gap_days > 5).sum())

    report.rows_out = len(df)
    if len(df):
        report.first_date = df.index[0].strftime("%Y-%m-%d")
        report.last_date = df.index[-1].strftime("%Y-%m-%d")

    # Only log the counts that are non-zero
    counted_fields = (
        "duplicate_dates_dropped",
        "missing_adj_close_dropped",
        "nonpositive_price_dropped",
        "inconsistent_high_low_nulled",
        "negative_volume_nulled",
        "calendar_gaps_over_5_days",
    )
    for name in counted_fields:
        if getattr(report, name):
            logger.info("%s: %s = %d", ticker, name, getattr(report, name))
    return df, report


# Downloads one ticker's unadjusted OHLC, Adj Close and Volume from Yahoo using yfinance.
# auto_adjust=False is set so that both Close and Adj Close are returned, and Yahoo treats the
# end date as exclusive. Failed attempts are retried with a growing wait, and a DataError is
# raised if every attempt fails or returns no rows.
def download_yahoo(ticker, start, end, max_retries=3, retry_wait_seconds=2.0):

    # yfinance is imported here so the offline sample run works without it installed
    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - depends on environment
        raise DataError(
            "yfinance is not installed. Run `pip install yfinance` or use data.source: sample"
        ) from error

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if df is not None and len(df) > 0:
                logger.info("%s: downloaded %d rows (attempt %d)", ticker, len(df), attempt)
                return df
            last_error = DataError(f"{ticker}: Yahoo returned no rows")

        # Network errors, bad JSON and rate limits all end up here and are retried
        except Exception as error:
            last_error = error

        logger.warning(
            "%s: download attempt %d/%d failed: %s", ticker, attempt, max_retries, last_error
        )
        if attempt < max_retries:
            time.sleep(retry_wait_seconds * attempt)
    raise DataError(f"{ticker}: all {max_retries} download attempts failed: {last_error}")


# Returns where a ticker's cached raw CSV is stored
def raw_path(raw_dir, ticker):
    return raw_dir / f"{ticker.upper()}.csv"


# Saves the cleaned OHLCV data to a CSV with a date column
def save_raw(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date", float_format="%.6f")


# Reads back a cached raw CSV that was written by save_raw
def load_raw(path):
    if not path.exists():
        raise DataError(
            f"Raw data file not found: {path}. Run `python -m momentum_ml.download_data` first."
        )
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")

    missing_columns = [column for column in PRICE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DataError(f"{path} is missing columns {missing_columns}")
    return df[PRICE_COLUMNS]


# Downloads (or loads from the cache or the sample data) and cleans every ticker in the config.
# A ticker that fails is logged and skipped, and the run carries on as long as at least one
# modelling ticker is left. The market ticker (SPY by default) is always fetched when market
# features are switched on. Every decision is written to a JSON manifest next to the raw files.
# Returns a dict of ticker -> cleaned OHLCV data for the tickers that worked.
def acquire_data(config, force=None):
    data_config = config["data"]
    raw_dir = resolve_path(config, data_config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    force = data_config.get("force_download", False) if force is None else force
    end = data_config.get("end") or date.today().isoformat()

    # The market ticker is needed for the market features even if it is not being modelled
    tickers = all_tickers(config)
    market_ticker = str(data_config.get("market_ticker", "SPY")).upper()
    needed_tickers = list(tickers)
    market_features_on = config["features"].get("market_features", {}).get("enabled", False)
    if market_features_on and market_ticker not in needed_tickers:
        needed_tickers.append(market_ticker)

    manifest = {
        "source": data_config["source"],
        "start": data_config["start"],
        "end": end,
        "requested": needed_tickers,
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "tickers": {},
    }
    price_data = {}
    for ticker in needed_tickers:
        path = raw_path(raw_dir, ticker)
        entry = {"path": str(path)}
        try:

            # Use the cached file if there is one, unless a fresh download has been asked for
            if path.exists() and not force:
                df = load_raw(path)
                entry["action"] = "loaded_from_cache"
                logger.info(
                    "%s: using cached raw file %s (%d rows); set force_download to refresh",
                    ticker,
                    path,
                    len(df),
                )
                report = CleaningReport(ticker=ticker, rows_in=len(df), rows_out=len(df))
            else:
                if data_config["source"] == "sample":
                    from momentum_ml.sample_data import load_sample_ticker

                    raw = load_sample_ticker(config, ticker)
                    entry["action"] = "copied_from_sample"
                else:
                    raw = download_yahoo(
                        ticker,
                        data_config["start"],
                        end,
                        int(data_config.get("max_retries", 3)),
                        float(data_config.get("retry_wait_seconds", 2.0)),
                    )
                    entry["action"] = "downloaded"

                # Clean the data, trim it to the requested window and check there is enough
                df, report = clean_ohlcv(raw, ticker)
                df = df.loc[pd.Timestamp(data_config["start"]) : pd.Timestamp(end)]
                report.rows_out = len(df)
                min_rows = data_config.get("min_rows", 300)
                if len(df) < int(min_rows):
                    raise DataError(
                        f"{ticker}: only {len(df)} clean rows (< data.min_rows={min_rows})"
                    )
                save_raw(df, path)

            entry.update({"status": "ok", "cleaning": asdict(report), "rows": len(df)})
            price_data[ticker] = df
        except Exception as error:
            entry.update({"status": "failed", "error": str(error)})
            logger.error("%s: skipped (%s)", ticker, error)
        manifest["tickers"][ticker] = entry

    (raw_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    # The run can only continue if at least one of the modelling tickers loaded
    usable_tickers = [ticker for ticker in tickers if ticker in price_data]
    if not usable_tickers:
        raise DataError(
            "No modelling tickers could be loaded; see the log and the download manifest"
        )
    failed_tickers = [ticker for ticker in needed_tickers if ticker not in price_data]
    if failed_tickers:
        logger.warning("Continuing without failed tickers: %s", failed_tickers)
    return price_data


# Loads the cached raw files for the configured tickers, fetching any that are missing
def load_all_raw(config):
    return acquire_data(config, force=False)
