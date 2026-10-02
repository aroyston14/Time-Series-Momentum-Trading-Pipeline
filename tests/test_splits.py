# Tests for the chronological split ordering, purging, CV folds and walk-forward folds

import numpy as np
import pandas as pd
import pytest

from momentum_ml.features import build_feature_panel
from momentum_ml.splits import (
    assert_chronological,
    chronological_split,
    date_grouped_ts_cv,
    walk_forward_folds,
)


@pytest.fixture
def panel(universe, config):
    return build_feature_panel(universe, config)


def test_fraction_split_is_chronological_and_purged(panel, config):
    splits, split_infos = chronological_split(panel, config)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    assert (
        train["date"].max()
        < validation["date"].min()
        < validation["date"].max()
        < test["date"].min()
    )

    # Exactly one date is purged between train and validation, and between validation and test
    dates = np.sort(panel["date"].unique())
    train_end_index = np.searchsorted(dates, train["date"].max())
    val_start_index = np.searchsorted(dates, validation["date"].min())
    val_end_index = np.searchsorted(dates, validation["date"].max())
    test_start_index = np.searchsorted(dates, test["date"].min())
    assert val_start_index - train_end_index == 2
    assert test_start_index - val_end_index == 2

    # Every ticker's row for a date ends up in the same split
    for split in (train, validation, test):
        assert set(split["date"]).isdisjoint(set(panel["date"]) - set(split["date"]))
    assert [info.name for info in split_infos] == ["train", "validation", "test"]
    assert all(info.start <= info.end for info in split_infos)


def test_explicit_date_split(panel, config):
    config["split"].update(train_end="2019-12-31", val_end="2020-12-31", purge_days=0)
    splits, _ = chronological_split(panel, config)
    assert splits["train"]["date"].max() <= pd.Timestamp("2019-12-31")
    assert splits["validation"]["date"].min() > pd.Timestamp("2019-12-31")
    assert splits["test"]["date"].min() > pd.Timestamp("2020-12-31")

    # The train end date has to come before the validation end date
    config["split"].update(train_end="2021-01-01", val_end="2020-01-01")
    with pytest.raises(ValueError):
        chronological_split(panel, config)


def test_split_does_not_shuffle(panel, config):
    splits, _ = chronological_split(panel, config)
    for split in splits.values():
        assert split["date"].is_monotonic_increasing


def test_assert_chronological_detects_overlap():
    earlier = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"])})
    later = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"])})
    with pytest.raises(AssertionError):
        assert_chronological([earlier, later])


def test_date_grouped_time_series_cv(panel):
    folds = date_grouped_ts_cv(panel["date"], n_splits=3, gap=1)
    assert len(folds) == 3

    previous_val_end = None
    for train_rows, val_rows in folds:
        train_dates = panel["date"].iloc[train_rows]
        val_dates = panel["date"].iloc[val_rows]
        assert train_dates.max() < val_dates.min()

        # A whole date is skipped between the folds (the gap) and no date is split across folds
        assert set(train_dates).isdisjoint(set(val_dates))
        between = (panel["date"] > train_dates.max()) & (panel["date"] < val_dates.min())
        assert len(np.unique(panel["date"][between])) == 1

        if previous_val_end is not None:
            assert val_dates.min() > previous_val_end
        previous_val_end = val_dates.max()


def test_walk_forward_expanding_windows():
    dates = pd.bdate_range("2020-01-01", periods=100)
    folds = list(walk_forward_folds(dates, min_train_days=40, test_days=25, purge=1))
    assert len(folds) == 3

    # The training window expands each fold, and the first one is purged by one day
    train_sizes = [len(fold.train_dates) for fold in folds]
    assert train_sizes == sorted(train_sizes) and train_sizes[0] == 39

    # Every fold trains on dates before its test block and starts from the first date
    for fold in folds:
        assert fold.train_dates.max() < fold.test_dates.min()
        assert fold.train_dates[0] == np.datetime64(dates[0])

    # The test blocks do not overlap
    all_test_dates = np.concatenate([fold.test_dates for fold in folds])
    assert len(all_test_dates) == len(np.unique(all_test_dates)) == 60
