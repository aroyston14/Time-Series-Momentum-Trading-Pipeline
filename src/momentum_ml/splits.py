# Chronological splitting of the (date, ticker) panel.
#
# Every split works on unique dates, so all of the tickers' rows for a given date end up in the
# same split and nothing leaks across assets. Rows are never shuffled.
#
# Purging: with a `horizon` day target, the labels on the last `horizon` dates of a segment
# depend on prices inside the next segment. Those dates are dropped ("purged") from the earlier
# segment so that no training label depends on validation or test prices.

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# Summary of the date range covered by one split
@dataclass(frozen=True)
class SplitInfo:
    name: str
    start: str
    end: str
    n_dates: int
    n_rows: int

    # Returns the fields as a plain dict, for saving to CSV or JSON
    def as_dict(self):
        return self.__dict__.copy()


# Builds the SplitInfo for one segment, using blank dates if the segment is empty
def _split_info(name, df):
    if df.empty:
        return SplitInfo(name, "", "", 0, 0)
    return SplitInfo(
        name,
        str(df["date"].min().date()),
        str(df["date"].max().date()),
        int(df["date"].nunique()),
        len(df),
    )


# Number of dates to purge at each boundary, which defaults to the target horizon
def purge_days(config):
    purge_setting = config["split"].get("purge_days")
    return int(config["target"]["horizon"]) if purge_setting is None else int(purge_setting)


# Splits the panel into train, validation and test sets by date.
# Explicit split.train_end and split.val_end dates (both inclusive) are used if they are given,
# otherwise train_frac and val_frac of the unique dates are used and the rest becomes the test set.
# Returns a dict of the three DataFrames and a list of SplitInfo summaries.
def chronological_split(panel, config):
    split_config = config["split"]
    dates = np.sort(panel["date"].unique())
    if len(dates) < 10:
        raise ValueError(f"Too few dates ({len(dates)}) to split chronologically")

    if split_config.get("train_end") and split_config.get("val_end"):
        train_end = pd.Timestamp(split_config["train_end"])
        val_end = pd.Timestamp(split_config["val_end"])
        if not train_end < val_end:
            raise ValueError(
                f"split.train_end ({train_end.date()}) must be before split.val_end "
                f"({val_end.date()})"
            )
        train_dates = dates[dates <= train_end]
        val_dates = dates[(dates > train_end) & (dates <= val_end)]
        test_dates = dates[dates > val_end]
    else:
        num_dates = len(dates)
        num_train = int(num_dates * float(split_config["train_frac"]))
        num_val = int(num_dates * float(split_config["val_frac"]))
        train_dates = dates[:num_train]
        val_dates = dates[num_train : num_train + num_val]
        test_dates = dates[num_train + num_val :]

    # Purging the end of the train and validation segments, since their labels use prices
    # from the following segment
    purge = purge_days(config)
    if purge > 0:
        train_dates = train_dates[:-purge] if len(train_dates) > purge else train_dates[:0]
        val_dates = val_dates[:-purge] if len(val_dates) > purge else val_dates[:0]

    for name, segment_dates in (
        ("train", train_dates),
        ("validation", val_dates),
        ("test", test_dates),
    ):
        if len(segment_dates) == 0:
            raise ValueError(f"The {name} split is empty; adjust split dates/fractions")

    splits = {
        "train": panel[panel["date"].isin(train_dates)].copy(),
        "validation": panel[panel["date"].isin(val_dates)].copy(),
        "test": panel[panel["date"].isin(test_dates)].copy(),
    }
    split_infos = [_split_info(name, df) for name, df in splits.items()]
    for info in split_infos:
        logger.info(
            "Split %-10s %s -> %s  (%d dates, %d rows)",
            info.name,
            info.start,
            info.end,
            info.n_dates,
            info.n_rows,
        )
    assert_chronological(list(splits.values()))
    return splits, split_infos


# Raises an AssertionError unless every segment finishes strictly before the next one starts
def assert_chronological(segments):
    for earlier, later in zip(segments, segments[1:], strict=False):
        if len(earlier) and len(later) and not earlier["date"].max() < later["date"].min():
            raise AssertionError(
                f"Segments overlap or are out of order: {earlier['date'].max()} >= "
                f"{later['date'].min()}"
            )


# TimeSeriesSplit run over the unique dates, then mapped back to row positions so it can be
# passed straight to GridSearchCV as the cv argument. `gap` dates are left out between each
# training fold and its validation fold to purge overlapping labels.
# dates should be the date column of the training data, in row order.
def date_grouped_ts_cv(dates, n_splits, gap=1):
    row_dates = pd.to_datetime(dates).to_numpy()
    unique_dates = np.sort(np.unique(row_dates))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    folds = []
    for train_index, val_index in splitter.split(unique_dates):
        train_rows = np.flatnonzero(np.isin(row_dates, unique_dates[train_index]))
        val_rows = np.flatnonzero(np.isin(row_dates, unique_dates[val_index]))
        folds.append((train_rows, val_rows))
    return folds


# One expanding window fold, which trains on every date up to train_end and tests on the next block
@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_dates: np.ndarray
    test_dates: np.ndarray


# Generates the expanding window walk-forward folds over the unique dates.
# Fold k trains on dates [0, min_train_days + k * test_days - purge) and tests on the next
# test_days dates. The last block may be shorter than the others.
def walk_forward_folds(dates, min_train_days, test_days, purge=1):
    unique_dates = np.sort(np.unique(pd.to_datetime(np.asarray(dates))))
    if len(unique_dates) <= min_train_days:
        raise ValueError(
            f"walk-forward needs more than min_train_days={min_train_days} dates, "
            f"have {len(unique_dates)}"
        )

    fold_number = 0
    test_start = min_train_days
    while test_start < len(unique_dates):
        train_dates = unique_dates[: max(test_start - purge, 1)]
        test_dates = unique_dates[test_start : test_start + test_days]
        yield WalkForwardFold(
            fold_number,
            str(pd.Timestamp(train_dates[0]).date()),
            str(pd.Timestamp(train_dates[-1]).date()),
            str(pd.Timestamp(test_dates[0]).date()),
            str(pd.Timestamp(test_dates[-1]).date()),
            train_dates,
            test_dates,
        )
        fold_number += 1
        test_start += test_days
