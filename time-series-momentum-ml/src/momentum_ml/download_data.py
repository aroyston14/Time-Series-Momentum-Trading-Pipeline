# Command line entry point that downloads (or loads the sample) daily OHLCV data and caches it
# in the data.raw_dir folder.
#
# Examples:
#     python -m momentum_ml.download_data
#     python -m momentum_ml.download_data --tickers SPY QQQ AAPL --start 2012-01-01 --end 2025-01-01
#     python -m momentum_ml.download_data --config configs/sample.yaml
#     python -m momentum_ml.download_data --force          (ignore the cache and re-download)

# Standard library modules for the command line arguments and logging
import argparse
import logging

from momentum_ml.config import load_config, set_seeds, setup_logging
from momentum_ml.data import acquire_data

logger = logging.getLogger("momentum_ml.download_data")


# Turns the command line flags into a nested dict that can be merged over the config.
# Only flags the user actually passed are included.
def build_overrides(args):
    data_overrides = {}
    if args.tickers:
        data_overrides["tickers"] = [ticker.upper() for ticker in args.tickers]
        data_overrides["extra_tickers"] = []
    if args.start:
        data_overrides["start"] = args.start
    if args.end:
        data_overrides["end"] = args.end
    if args.output_dir:
        data_overrides["raw_dir"] = args.output_dir
    if args.source:
        data_overrides["source"] = args.source
    return {"data": data_overrides} if data_overrides else {}


# Entry point for python -m momentum_ml.download_data
def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and cache daily OHLCV data.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--tickers", nargs="+", help="override data.tickers (and clear extra_tickers)"
    )
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD (exclusive on Yahoo)")
    parser.add_argument("--output-dir", help="override data.raw_dir")
    parser.add_argument("--source", choices=["yahoo", "sample"])
    parser.add_argument(
        "--force", action="store_true", help="re-download even if cached files exist"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, build_overrides(args))
    setup_logging(config["project"].get("log_level", "INFO"))
    set_seeds(int(config["project"]["seed"]))
    price_data = acquire_data(config, force=args.force or None)

    # Log a one line summary of what was loaded for each ticker
    for ticker, df in price_data.items():
        logger.info(
            "%-6s %5d rows  %s -> %s", ticker, len(df), df.index[0].date(), df.index[-1].date()
        )


if __name__ == "__main__":
    main()
