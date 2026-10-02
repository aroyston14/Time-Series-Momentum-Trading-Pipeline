# Command line entry point that builds the (date, ticker) feature panel from the cached raw data.
#
# Example:
#     python -m momentum_ml.build_features --config configs/config.yaml

# Standard library modules for the command line arguments, saving the schema and logging
import argparse
import json
import logging

import pandas as pd

from momentum_ml.config import load_config, resolve_path, set_seeds, setup_logging
from momentum_ml.data import load_all_raw
from momentum_ml.features import build_feature_panel, describe_features, feature_columns

logger = logging.getLogger("momentum_ml.build_features")
PANEL_NAME = "features.csv"


# Returns where the processed feature panel is saved
def panel_path(config):
    return resolve_path(config, config["data"]["processed_dir"]) / PANEL_NAME


# Loads the raw data, builds the feature panel and saves it alongside a JSON schema describing it
def run(config):
    raw_data = load_all_raw(config)
    panel = build_feature_panel(raw_data, config)

    output_path = panel_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False, float_format="%.8g")

    # The schema records what is in the panel and how each feature was defined
    schema = {
        "rows": len(panel),
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "start": str(panel["date"].min().date()),
        "end": str(panel["date"].max().date()),
        "feature_columns": feature_columns(panel),
        "definitions": describe_features(config),
        "target": config["target"],
    }
    (output_path.parent / "features_schema.json").write_text(
        json.dumps(schema, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Saved feature panel to %s", output_path)
    return panel


# Loads the saved feature panel, building it first if it has not been made yet
def load_panel(config):
    path = panel_path(config)
    if not path.exists():
        logger.info("Feature panel %s not found; building it", path)
        return run(config)

    panel = pd.read_csv(path, parse_dates=["date"])
    return panel.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


# Entry point for python -m momentum_ml.build_features
def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the feature panel from cached raw data.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config["project"].get("log_level", "INFO"))
    set_seeds(int(config["project"]["seed"]))
    run(config)


if __name__ == "__main__":
    main()
