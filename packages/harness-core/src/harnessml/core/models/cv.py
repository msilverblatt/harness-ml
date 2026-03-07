"""Temporal cross-validation strategies.

All strategies return lists of ``Fold`` objects and enforce temporal ordering
(training data always precedes test data in time).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from harnessml.core.schemas.contracts import Fold

# ---------------------------------------------------------------------------
# Protocol for all CV strategies
# ---------------------------------------------------------------------------

class CVStrategy(Protocol):
    """Minimal interface shared by all CV splitters."""

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        ...


# ---------------------------------------------------------------------------
# Leave-One-Season-Out
# ---------------------------------------------------------------------------

class LeaveOneSeasonOut:
    """For each unique fold_id, use all *prior* fold_ids as training.

    Parameters
    ----------
    min_train_folds : int
        Minimum number of distinct prior fold_ids required to include a
        test fold.  Fold_ids with fewer prior folds are skipped.
    """

    def __init__(self, min_train_folds: int = 1) -> None:
        self.min_train_folds = min_train_folds

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        fold_ids = np.asarray(fold_ids)
        unique_ids = np.sort(np.unique(fold_ids))
        folds: list[Fold] = []

        for test_id in unique_ids:
            prior_ids = unique_ids[unique_ids < test_id]
            if len(prior_ids) < self.min_train_folds:
                continue

            train_mask = np.isin(fold_ids, prior_ids)
            test_mask = fold_ids == test_id

            folds.append(Fold(
                fold_id=int(test_id),
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            ))

        return folds


# ---------------------------------------------------------------------------
# Expanding Window
# ---------------------------------------------------------------------------

class ExpandingWindow:
    """Like LOSO but ``min_train_size`` counts *rows*, not folds.

    Parameters
    ----------
    min_train_size : int
        Minimum number of training *rows* required to include a test fold.
    """

    def __init__(self, min_train_size: int = 1) -> None:
        self.min_train_size = min_train_size

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        fold_ids = np.asarray(fold_ids)
        unique_ids = np.sort(np.unique(fold_ids))
        folds: list[Fold] = []

        for test_id in unique_ids:
            train_mask = fold_ids < test_id
            test_mask = fold_ids == test_id

            if train_mask.sum() < self.min_train_size:
                continue

            folds.append(Fold(
                fold_id=int(test_id),
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            ))

        return folds


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------

class SlidingWindow:
    """Use only the last ``window_size`` prior fold_ids for training.

    Parameters
    ----------
    window_size : int
        Number of prior fold_ids to include in the training set.
    """

    def __init__(self, window_size: int = 2) -> None:
        self.window_size = window_size

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        fold_ids = np.asarray(fold_ids)
        unique_ids = np.sort(np.unique(fold_ids))
        folds: list[Fold] = []

        for test_id in unique_ids:
            prior_ids = unique_ids[unique_ids < test_id]
            if len(prior_ids) == 0:
                continue

            window_ids = prior_ids[-self.window_size:]
            train_mask = np.isin(fold_ids, window_ids)
            test_mask = fold_ids == test_id

            folds.append(Fold(
                fold_id=int(test_id),
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            ))

        return folds


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------

class PurgedKFold:
    """Standard k-fold with temporal embargo purging.

    Fold_ids within ``embargo_size`` of any test fold_id are removed from
    training to prevent temporal leakage.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    embargo_size : int
        Fold_ids within this distance of the test fold are purged from
        training.
    """

    def __init__(self, n_splits: int = 3, embargo_size: int = 1) -> None:
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        fold_ids = np.asarray(fold_ids)
        unique_ids = np.sort(np.unique(fold_ids))

        # Split unique fold_ids into n_splits groups
        splits = np.array_split(unique_ids, self.n_splits)
        folds: list[Fold] = []

        for split_idx, test_ids in enumerate(splits):
            test_id_set = set(test_ids.tolist())

            # Build embargo set: fold_ids within embargo_size of any test id
            embargo_set: set[int] = set()
            for tid in test_ids:
                for offset in range(1, self.embargo_size + 1):
                    embargo_set.add(int(tid - offset))
                    embargo_set.add(int(tid + offset))
            # Remove test ids themselves from embargo (they belong to test)
            embargo_set -= test_id_set

            # Training = all fold_ids not in test and not embargoed
            train_ids = set(unique_ids.tolist()) - test_id_set - embargo_set
            train_mask = np.isin(fold_ids, list(train_ids))
            test_mask = np.isin(fold_ids, list(test_id_set))

            folds.append(Fold(
                fold_id=int(test_ids[0]),
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
            ))

        return folds


# ---------------------------------------------------------------------------
# Nested CV (outer strategy + calibration carve-out)
# ---------------------------------------------------------------------------

class NestedCV:
    """Wraps an outer CV strategy and carves out a calibration set.

    The calibration set is taken from the most recent fold_id(s) in the
    outer training set, sized to approximately ``inner_calibration_fraction``
    of the original training rows.

    Parameters
    ----------
    outer : CVStrategy
        Any CV strategy instance.
    inner_calibration_fraction : float
        Fraction of outer training data to reserve for calibration
        (taken from the most recent fold).
    """

    def __init__(
        self,
        outer: CVStrategy,
        inner_calibration_fraction: float = 0.25,
    ) -> None:
        self.outer = outer
        self.inner_calibration_fraction = inner_calibration_fraction

    def split(self, X: np.ndarray | None, /, *, fold_ids: np.ndarray) -> list[Fold]:
        fold_ids = np.asarray(fold_ids)
        outer_folds = self.outer.split(X, fold_ids=fold_ids)
        nested_folds: list[Fold] = []

        for fold in outer_folds:
            train_idx = fold.train_idx
            train_fold_ids = fold_ids[train_idx]

            # Identify the most recent fold_id in training
            unique_train_ids = np.sort(np.unique(train_fold_ids))

            # Carve calibration from most recent fold(s) until we hit the fraction
            target_cal_size = int(len(train_idx) * self.inner_calibration_fraction)
            cal_indices: list[int] = []
            remaining_train_indices: list[int] = []

            # Walk backwards through fold_ids, moving entire folds to calibration
            for fid in reversed(unique_train_ids):
                fid_mask = train_fold_ids == fid
                fid_indices = train_idx[fid_mask]

                if len(cal_indices) + len(fid_indices) <= target_cal_size:
                    cal_indices.extend(fid_indices.tolist())
                elif len(cal_indices) == 0:
                    # Even the most recent fold is larger than target — take a
                    # random subsample from it
                    rng = np.random.default_rng(fold.fold_id)
                    chosen = rng.choice(fid_indices, size=target_cal_size, replace=False)
                    cal_indices.extend(chosen.tolist())
                    leftover = set(fid_indices.tolist()) - set(chosen.tolist())
                    remaining_train_indices.extend(sorted(leftover))
                    break
                else:
                    # Adding this fold would exceed target — stop here
                    remaining_train_indices.extend(fid_indices.tolist())
                    break

            # All fold_ids not consumed by calibration go to training
            consumed = set(cal_indices)
            for idx in train_idx:
                if idx not in consumed and idx not in remaining_train_indices:
                    remaining_train_indices.append(idx)

            remaining_train_indices.sort()

            nested_folds.append(Fold(
                fold_id=fold.fold_id,
                train_idx=np.array(sorted(remaining_train_indices), dtype=np.intp),
                test_idx=fold.test_idx,
                calibration_idx=np.array(sorted(cal_indices), dtype=np.intp),
            ))

        return nested_folds
