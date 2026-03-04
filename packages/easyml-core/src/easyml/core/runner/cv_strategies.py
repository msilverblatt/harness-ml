"""Cross-validation strategy bridge for PipelineRunner.

Maps BacktestConfig cv_strategy strings to fold generation logic
and returns (train_seasons, test_season) tuples for use by
PipelineRunner.backtest().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from easyml.core.runner.schema import BacktestConfig


def generate_cv_folds(
    df: pd.DataFrame,
    bt_config: BacktestConfig,
) -> list[tuple[list[int], int]]:
    """Generate (train_seasons, test_season) tuples from CV strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with 'season' column.
    bt_config : BacktestConfig
        Backtest configuration.

    Returns
    -------
    list of (train_seasons, test_season) tuples
    """
    strategy = bt_config.cv_strategy

    # Determine available seasons
    if bt_config.seasons:
        all_seasons = sorted(bt_config.seasons)
    else:
        all_seasons = sorted(df["season"].unique().tolist())

    if strategy == "leave_one_season_out":
        return _loso_folds(all_seasons, bt_config.min_train_folds)
    elif strategy == "expanding_window":
        return _expanding_window_folds(all_seasons, bt_config.min_train_folds)
    elif strategy == "sliding_window":
        return _sliding_window_folds(all_seasons, bt_config.window_size)
    elif strategy == "purged_kfold":
        return _purged_kfold_folds(all_seasons, bt_config.n_folds, bt_config.purge_gap)
    else:
        raise ValueError(f"Unknown cv_strategy: {strategy!r}")


def _loso_folds(
    seasons: list[int],
    min_train_folds: int,
) -> list[tuple[list[int], int]]:
    """Leave-one-season-out: test each season using all prior seasons."""
    folds = []
    for test_season in seasons:
        train_seasons = [s for s in seasons if s < test_season]
        if len(train_seasons) < min_train_folds:
            continue
        folds.append((train_seasons, test_season))
    return folds


def _expanding_window_folds(
    seasons: list[int],
    min_train_folds: int,
) -> list[tuple[list[int], int]]:
    """Expanding window: each season tested using all prior seasons.

    Identical to LOSO -- min_train_folds controls the minimum number
    of training seasons required.
    """
    return _loso_folds(seasons, min_train_folds)


def _sliding_window_folds(
    seasons: list[int],
    window_size: int | None,
) -> list[tuple[list[int], int]]:
    """Sliding window: train on last window_size prior seasons."""
    if window_size is None:
        raise ValueError(
            "sliding_window strategy requires window_size to be set "
            "in BacktestConfig"
        )
    folds = []
    for test_season in seasons:
        prior = [s for s in seasons if s < test_season]
        if not prior:
            continue
        train_seasons = prior[-window_size:]
        folds.append((train_seasons, test_season))
    return folds


def _purged_kfold_folds(
    seasons: list[int],
    n_folds: int | None,
    purge_gap: int,
) -> list[tuple[list[int], int]]:
    """Purged k-fold: split seasons into groups, purge nearby seasons.

    For each fold group, test on those seasons and train on all others
    except those within purge_gap of any test season.  Emits one
    (train_seasons, test_season) tuple per test season in each group.
    """
    if n_folds is None:
        raise ValueError(
            "purged_kfold strategy requires n_folds to be set "
            "in BacktestConfig"
        )

    season_arr = np.array(seasons)
    groups = np.array_split(season_arr, n_folds)

    folds = []
    for group in groups:
        test_set = set(group.tolist())

        # Build embargo set: seasons within purge_gap of any test season
        embargo_set: set[int] = set()
        for ts in test_set:
            for offset in range(1, purge_gap + 1):
                embargo_set.add(ts - offset)
                embargo_set.add(ts + offset)
        embargo_set -= test_set

        # Training = all seasons not in test and not embargoed
        all_set = set(seasons)
        train_seasons = sorted(all_set - test_set - embargo_set)

        for test_season in sorted(test_set):
            folds.append((train_seasons, test_season))

    return folds
