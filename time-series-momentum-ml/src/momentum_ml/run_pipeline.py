# Command line entry point that runs every stage in order: download, features, train, evaluate.
#
# Examples:
#     python -m momentum_ml.run_pipeline --config configs/sample.yaml   (offline, synthetic data)
#     python -m momentum_ml.run_pipeline                                (real data via yfinance)

# Standard library modules for the command line arguments, logging and timing the run
import argparse
import logging
import time

# Each stage of the pipeline lives in its own module
from momentum_ml import build_features, evaluate, train
from momentum_ml.config import load_config, set_seeds, setup_logging
from momentum_ml.data import acquire_data

logger = logging.getLogger("momentum_ml.run_pipeline")


# Entry point for python -m momentum_ml.run_pipeline
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the full research pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config["project"].get("log_level", "INFO"))
    set_seeds(int(config["project"]["seed"]))
    start_time = time.time()

    # Passing None rather than False lets the config's force_download setting decide
    logger.info("Step 1/4: data")
    acquire_data(config, force=args.force_download or None)

    logger.info("Step 2/4: features")
    build_features.run(config)

    logger.info("Step 3/4: train")
    metadata = train.run(config)

    logger.info("Step 4/4: evaluate")
    summary = evaluate.run(config)

    logger.info(
        "Done in %.1fs. Selected model: %s. Report: %s",
        time.time() - start_time,
        metadata["selected_model"],
        summary["report_path"],
    )


if __name__ == "__main__":
    main()
