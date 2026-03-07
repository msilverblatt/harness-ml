"""Cross-validation strategy bridge for PipelineRunner.

Maps BacktestConfig cv_strategy strings to fold generation logic
and returns (train_folds, test_fold) tuples for use by
PipelineRunner.backtest().
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harnessml.core.runner.schema import BacktestConfig


def generate_cv_folds(
    df: pd.DataFrame,
    bt_config: BacktestConfig,
) -> list[tuple[list[int], int]]:
    """Generate (train_folds, test_fold) tuples from CV strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with the fold column (specified by bt_config.fold_column).
    bt_config : BacktestConfig
        Backtest configuration.

    Returns
    -------
    list of (train_fold_values, test_fold_value) tuples
    """
    strategy = bt_config.cv_strategy
    fold_col = bt_config.fold_column

    if fold_col not in df.columns:
        raise ValueError(
            f"Fold column {fold_col!r} not found in data. "
            f"Set backtest.fold_column to an existing column."
        )

    # All fold values available in df (used as training pool)
    all_df_folds = sorted(df[fold_col].unique().tolist())

    # Test folds: specified in config or all df folds
    if bt_config.fold_values:
        test_folds = sorted(bt_config.fold_values)
    else:
        test_folds = all_df_folds

    if strategy == "leave_one_out":
        return _loso_folds(test_folds, bt_config.min_train_folds, all_df_folds)
    elif strategy == "expanding_window":
        return _expanding_window_folds(test_folds, bt_config.min_train_folds, all_df_folds)
    elif strategy == "sliding_window":
        return _sliding_window_folds(test_folds, bt_config.window_size, all_df_folds)
    elif strategy == "purged_kfold":
        return _purged_kfold_folds(test_folds, bt_config.n_folds, bt_config.purge_gap)
    else:
        raise ValueError(f"Unknown cv_strategy: {strategy!r}")


def _loso_folds(
    folds: list[int],
    min_train_folds: int,
    training_pool: list[int] | None = None,
) -> list[tuple[list[int], int]]:
    """Leave-one-out: test each fold using all other folds for training.

    Parameters
    ----------
    folds : list[int]
        Fold values to use as test folds.
    min_train_folds : int
        Minimum number of training folds required.
    training_pool : list[int] | None
        All available fold values for training (can include values
        outside the test fold list). Defaults to ``folds`` if not set.
    """
    pool = sorted(training_pool or folds)
    result = []
    for test_fold in folds:
        train_folds = [f for f in pool if f != test_fold]
        if len(train_folds) < min_train_folds:
            continue
        result.append((train_folds, test_fold))
    return result


def _expanding_window_folds(
    folds: list[int],
    min_train_folds: int,
    training_pool: list[int] | None = None,
) -> list[tuple[list[int], int]]:
    """Expanding window: each fold tested using all prior folds only.

    Unlike LOSO, this is temporal — only folds with values less than
    the test fold are used for training. min_train_folds controls the
    minimum number of training folds required.
    """
    pool = sorted(training_pool or folds)
    result = []
    for test_fold in folds:
        train_folds = [f for f in pool if f < test_fold]
        if len(train_folds) < min_train_folds:
            continue
        result.append((train_folds, test_fold))
    return result


def _sliding_window_folds(
    folds: list[int],
    window_size: int | None,
    training_pool: list[int] | None = None,
) -> list[tuple[list[int], int]]:
    """Sliding window: train on last window_size prior folds."""
    if window_size is None:
        raise ValueError(
            "sliding_window strategy requires window_size to be set "
            "in BacktestConfig"
        )
    pool = sorted(training_pool or folds)
    result = []
    for test_fold in folds:
        prior = [f for f in pool if f < test_fold]
        if not prior:
            continue
        train_folds = prior[-window_size:]
        result.append((train_folds, test_fold))
    return result


def _purged_kfold_folds(
    folds: list[int],
    n_folds: int | None,
    purge_gap: int,
) -> list[tuple[list[int], int]]:
    """Purged k-fold: split fold values into groups, purge nearby values.

    For each fold group, test on those values and train on all others
    except those within purge_gap of any test value. Emits one
    (train_folds, test_fold) tuple per test value in each group.
    """
    if n_folds is None:
        raise ValueError(
            "purged_kfold strategy requires n_folds to be set "
            "in BacktestConfig"
        )

    fold_arr = np.array(folds)
    groups = np.array_split(fold_arr, n_folds)

    result = []
    for group in groups:
        test_set = set(group.tolist())

        # Build embargo set: values within purge_gap of any test value
        embargo_set: set[int] = set()
        for tv in test_set:
            for offset in range(1, purge_gap + 1):
                embargo_set.add(tv - offset)
                embargo_set.add(tv + offset)
        embargo_set -= test_set

        # Training = all folds not in test and not embargoed
        all_set = set(folds)
        train_folds = sorted(all_set - test_set - embargo_set)

        for test_fold in sorted(test_set):
            result.append((train_folds, test_fold))

    return result
