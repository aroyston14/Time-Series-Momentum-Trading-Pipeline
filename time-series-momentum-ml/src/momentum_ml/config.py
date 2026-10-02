# Loading, validating and resolving paths for the YAML config files.
# A config can start with "extends: other.yaml" (relative to itself), in which case it is merged
# on top of that parent file, so variants only need to list the keys they change.

# Standard library modules for copying nested dicts, logging, seeding and file paths
import copy
import logging
import random
from pathlib import Path

# numpy is seeded alongside python's random module, and yaml is used to read the config files
import numpy as np
import yaml

DEFAULT_CONFIG = Path("configs/config.yaml")
logger = logging.getLogger(__name__)


# Raised whenever a config file is missing or has an invalid value
class ConfigError(ValueError):
    pass


# Merges the override dict into a copy of the base dict. Nested dicts are merged key by key,
# while any other value in the override simply replaces the base value.
def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Reads one YAML file, following its "extends" chain upwards.
# visited_paths keeps track of the files already read so that a circular chain raises an error
# instead of recursing forever.
def _read_yaml(path, visited_paths=None):
    path = path.resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    visited = visited_paths or set()
    if path in visited:
        raise ConfigError(f"Circular 'extends' chain involving {path}")
    visited.add(path)

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")

    # If this file extends another one, load the parent first and merge this file on top of it
    parent_file = data.pop("extends", None)
    if parent_file:
        parent_config = _read_yaml(path.parent / parent_file, visited)
        data = deep_merge(parent_config, data)
    return data


# Works out the project root, which is the folder containing the configs/ directory.
# If the config is stored somewhere else, the current working directory is used instead.
def project_root_for(config_path):
    config_path = config_path.resolve()
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return Path.cwd()


# Loads, merges and validates a config. Defaults to configs/config.yaml when no path is given.
# overrides is an optional nested dict merged on top, which the CLIs and tests use.
# The returned dict also contains the absolute project root under "_root".
def load_config(path=None, overrides=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    config = _read_yaml(config_path)
    if overrides:
        config = deep_merge(config, overrides)
    config["_root"] = str(project_root_for(config_path))
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


# Checks that every required section is present and that values are in a sensible range,
# raising a ConfigError with a clear message if anything is wrong.
def validate_config(config):
    required_sections = (
        "project",
        "data",
        "features",
        "target",
        "split",
        "tuning",
        "models",
        "backtest",
        "outputs",
    )
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required config section '{section}'")

    data_config = config["data"]
    if data_config.get("source") not in {"yahoo", "sample"}:
        raise ConfigError(
            f"data.source must be 'yahoo' or 'sample', got {data_config.get('source')!r}"
        )
    if not all_tickers(config):
        raise ConfigError("data.tickers (plus extra_tickers) must contain at least one ticker")

    if int(config["target"].get("horizon", 1)) < 1:
        raise ConfigError("target.horizon must be >= 1")

    # The fractions only matter when explicit split dates have not been given.
    # They must leave some room for a test period at the end.
    split_config = config["split"]
    if not (split_config.get("train_end") and split_config.get("val_end")):
        train_frac = float(split_config.get("train_frac", 0))
        val_frac = float(split_config.get("val_frac", 0))
        if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
            raise ConfigError(
                f"split.train_frac ({train_frac}) and split.val_frac ({val_frac}) must be > 0 "
                "and sum to < 1 so that a test period remains"
            )

    backtest_config = config["backtest"]
    if not 0.0 <= float(backtest_config["threshold"]) <= 1.0:
        raise ConfigError("backtest.threshold must be in [0, 1]")
    if float(backtest_config["cost_bps"]) < 0:
        raise ConfigError("backtest.cost_bps must be non-negative")
    if int(backtest_config.get("execution_lag", 0)) < 0:
        raise ConfigError("backtest.execution_lag must be >= 0")

    # The moving average benchmark reuses one of the MA distance features, so its window must exist
    ma_windows = [int(window) for window in config["features"]["ma_distance_windows"]]
    if int(backtest_config["ma_rule_window"]) not in ma_windows:
        raise ConfigError(
            f"backtest.ma_rule_window={backtest_config['ma_rule_window']} must be one of "
            f"features.ma_distance_windows={config['features']['ma_distance_windows']}"
        )


# Returns the modelling universe, tickers followed by extra_tickers, with duplicates removed.
# A dict is used rather than a set so that the original order is kept.
def all_tickers(config):
    unique_tickers = {}
    ticker_list = list(config["data"].get("tickers") or []) + list(
        config["data"].get("extra_tickers") or []
    )
    for ticker in ticker_list:
        unique_tickers[str(ticker).upper().strip()] = None
    return list(unique_tickers)


# Turns a path from the config into an absolute path, relative to the project root
def resolve_path(config, relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else Path(config["_root"]) / path


# Seeds python's and numpy's global random number generators.
# The models themselves are given random_state explicitly.
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


# Sets up logging once with the same format for every module
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
