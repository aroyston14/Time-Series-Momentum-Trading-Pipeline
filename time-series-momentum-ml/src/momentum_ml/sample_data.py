# Generates a fixed synthetic dataset so the pipeline can run offline without an API key.
#
# It produces Yahoo-shaped daily OHLCV files for a small ETF-like universe with:
#   - a common market factor with volatility clustering (GARCH(1,1) style) and a small drift
#   - a weak negative day-to-day autocorrelation in the market factor (short-term reversal)
#   - a beta and some idiosyncratic noise for each asset
#   - unadjusted OHLC worked out from a dividend-adjusted series with quarterly payouts, so close
#     and adj_close differ in the same way they do on Yahoo
#   - volume that rises with the size of the daily move
#
# A few deliberate defects (a duplicate row, a missing adj close, a multi-day gap) are added so
# that the cleaning code gets tested. None of this is market data, so any "edge" found on it
# comes from the generator.
#
# Regenerate the files with: python -m momentum_ml.sample_data

# Standard library modules for the command line arguments and logging
import argparse
import logging

import numpy as np
import pandas as pd

from momentum_ml.config import load_config, resolve_path, setup_logging

logger = logging.getLogger(__name__)

# ticker -> (beta to the market, idiosyncratic daily vol, start price, annual dividend yield)
SAMPLE_UNIVERSE = {
    "SPY": (1.00, 0.0000, 180.0, 0.018),
    "QQQ": (1.15, 0.0060, 85.0, 0.006),
    "IWM": (1.20, 0.0080, 110.0, 0.012),
    "DIA": (0.95, 0.0040, 165.0, 0.020),
    "EFA": (0.85, 0.0070, 65.0, 0.028),
}


# Generates synthetic OHLCV data (indexed by date) for each ticker.
# Tickers that are not in SAMPLE_UNIVERSE are given a random beta and volatility, so extra
# symbols can be added to the sample config.
def generate_sample_universe(
    start="2014-01-01", end="2024-12-31", seed=2024, tickers=None, inject_defects=True
):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, name="date")
    num_days = len(dates)

    # The market factor follows a GARCH(1,1) variance process with weak AR(1) mean reversion
    omega, alpha, garch_beta = 1.5e-6, 0.09, 0.89
    variance = np.empty(num_days)
    market_returns = np.empty(num_days)
    variance[0] = omega / (1 - alpha - garch_beta)
    shocks = rng.standard_t(df=5, size=num_days) / np.sqrt(5 / 3)
    previous_return = 0.0
    for i in range(num_days):
        if i > 0:
            variance[i] = omega + alpha * market_returns[i - 1] ** 2 + garch_beta * variance[i - 1]
        market_returns[i] = 0.0003 - 0.04 * previous_return + np.sqrt(variance[i]) * shocks[i]
        previous_return = market_returns[i]

    sample_data = {}
    for ticker in tickers or list(SAMPLE_UNIVERSE):
        default_params = (
            float(rng.uniform(0.7, 1.4)),
            float(rng.uniform(0.008, 0.02)),
            50.0,
            0.01,
        )
        beta, idio_vol, start_price, dividend_yield = SAMPLE_UNIVERSE.get(ticker, default_params)
        returns = beta * market_returns + idio_vol * rng.standard_normal(num_days)
        adj_close = start_price * np.exp(np.cumsum(np.log1p(returns)))

        # Quarterly dividends. The unadjusted close is the adjusted close divided by the
        # cumulative adjustment factor from all of the future ex-dividend dates, i.e.
        # adj_close[t] = close[t] * product over future ex-dates of (1 - dividend_yield / 4)
        dividend_days = np.zeros(num_days, dtype=bool)
        dividend_days[63::63] = True
        adjustment_factor = np.ones(num_days)
        cumulative_factor = 1.0
        for i in range(num_days - 1, -1, -1):
            adjustment_factor[i] = cumulative_factor
            if dividend_days[i]:
                cumulative_factor *= 1 - dividend_yield / 4
        close = adj_close / adjustment_factor

        # The open is the previous close plus an overnight move, and the high and low are spread
        # around the open and close in proportion to how volatile the day was
        overnight_move = 0.25 * returns + 0.002 * rng.standard_normal(num_days)
        open_price = np.r_[close[0], close[:-1]] * np.exp(overnight_move)
        intraday_vol = np.abs(returns) + 0.004 + 0.002 * np.abs(rng.standard_normal(num_days))
        high = np.maximum(open_price, close) * (
            1 + 0.5 * intraday_vol * rng.uniform(0.2, 1.0, num_days)
        )
        low = np.minimum(open_price, close) * (
            1 - 0.5 * intraday_vol * rng.uniform(0.2, 1.0, num_days)
        )
        base_volume = rng.uniform(5e6, 6e7)
        volume = base_volume * np.exp(0.3 * rng.standard_normal(num_days) + 25 * np.abs(returns))

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": adj_close,
                "volume": np.round(volume),
            },
            index=dates,
        )

        # IWM gets a duplicate date, a missing adj close and a multi-day gap (like an exchange
        # outage), and EFA gets a missing volume
        if inject_defects and ticker == "IWM":
            df = pd.concat([df, df.iloc[[100]]]).sort_index()
            df.iloc[200, df.columns.get_loc("adj_close")] = np.nan
            df = df.drop(df.index[400:404])
        if inject_defects and ticker == "EFA":
            df.iloc[300, df.columns.get_loc("volume")] = np.nan
        sample_data[ticker] = df
    return sample_data


# Returns the folder holding the committed sample CSVs
def sample_dir(config):
    return resolve_path(config, config["data"].get("sample_dir", "data/sample"))


# Returns the raw (uncleaned) sample data for a ticker, generating the file if it is missing
def load_sample_ticker(config, ticker):
    path = sample_dir(config) / f"{ticker.upper()}.csv"
    if not path.exists():
        logger.info("Sample file %s not found; generating synthetic data", path)
        write_sample_files(config, [ticker])
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


# Generates and saves the sample CSVs for the given tickers (the whole sample universe by default)
def write_sample_files(config, tickers=None):
    folder = sample_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    sample_data = generate_sample_universe(tickers=tickers)
    paths = []
    for ticker, df in sample_data.items():
        path = folder / f"{ticker}.csv"
        df.to_csv(path, index_label="date", float_format="%.6f")
        paths.append(path)
        logger.info("Wrote synthetic sample %s (%d rows)", path, len(df))
    return paths


# Entry point for python -m momentum_ml.sample_data, which regenerates the committed sample files
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministic SYNTHETIC sample data so the pipeline runs offline without "
        "an API key."
    )
    parser.add_argument("--config", default="configs/sample.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config["project"].get("log_level", "INFO"))
    write_sample_files(config)


if __name__ == "__main__":
    main()
