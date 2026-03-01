"""Tests for temporal cross-validation strategies."""
import numpy as np
import pytest

from easyml.schemas.core import Fold
from easyml.models.cv import (
    LeaveOneSeasonOut,
    ExpandingWindow,
    SlidingWindow,
    PurgedKFold,
    NestedCV,
)


def test_loso_basic():
    fold_ids = np.array([2015, 2015, 2016, 2016, 2017, 2017])
    loso = LeaveOneSeasonOut(min_train_folds=1)
    folds = loso.split(None, fold_ids=fold_ids)
    assert len(folds) == 2  # 2016, 2017 (2015 has 0 prior folds so it's skipped)
    # 2016 uses [2015] as training (1 fold), 2017 uses [2015, 2016] (2 folds)
    for fold in folds:
        train_seasons = set(fold_ids[fold.train_idx])
        test_season = set(fold_ids[fold.test_idx])
        assert max(train_seasons) < min(test_season)


def test_loso_min_train_folds():
    fold_ids = np.array([2015, 2016, 2017])
    loso = LeaveOneSeasonOut(min_train_folds=2)
    folds = loso.split(None, fold_ids=fold_ids)
    assert len(folds) == 1  # only 2017 has 2+ training folds


def test_expanding_window():
    fold_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    ew = ExpandingWindow(min_train_size=1)
    folds = ew.split(None, fold_ids=fold_ids)
    # fold 2 has train from [1], fold 3 from [1,2], fold 4 from [1,2,3]
    assert len(folds) == 3


def test_sliding_window():
    fold_ids = np.array([1, 2, 3, 4, 5])
    sw = SlidingWindow(window_size=2)
    folds = sw.split(None, fold_ids=fold_ids)
    fold_5 = [f for f in folds if f.fold_id == 5][0]
    train_ids = set(fold_ids[fold_5.train_idx])
    assert train_ids == {3, 4}


def test_purged_kfold():
    fold_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    pkf = PurgedKFold(n_splits=3, embargo_size=1)
    folds = pkf.split(None, fold_ids=fold_ids)
    assert len(folds) == 3
    # embargo: folds adjacent to test fold are excluded from training
    for fold in folds:
        train_ids = set(fold_ids[fold.train_idx])
        test_ids = set(fold_ids[fold.test_idx])
        for test_id in test_ids:
            # embargo_size=1 means fold_id +/- 1 of test should not be in train
            assert (test_id - 1) not in train_ids or (test_id - 1) in test_ids
            assert (test_id + 1) not in train_ids or (test_id + 1) in test_ids


def test_nested_cv():
    fold_ids = np.array([2015] * 20 + [2016] * 20 + [2017] * 20)
    nested = NestedCV(
        outer=LeaveOneSeasonOut(min_train_folds=1),
        inner_calibration_fraction=0.25,
    )
    folds = nested.split(None, fold_ids=fold_ids)
    for fold in folds:
        assert fold.calibration_idx is not None
        assert len(fold.calibration_idx) > 0
        # calibration indices should be a subset of the original outer training set
        # train_idx here is the remaining training after carving out calibration
        assert len(set(fold.train_idx) & set(fold.calibration_idx)) == 0


def test_temporal_ordering_enforced():
    fold_ids = np.array([2015, 2016, 2017, 2018])
    for strategy in [LeaveOneSeasonOut(), ExpandingWindow(), SlidingWindow(window_size=2)]:
        folds = strategy.split(None, fold_ids=fold_ids)
        for fold in folds:
            max_train = max(fold_ids[fold.train_idx])
            min_test = min(fold_ids[fold.test_idx])
            assert max_train < min_test


def test_loso_fold_ids_are_correct():
    """Verify each LOSO fold has the expected fold_id."""
    fold_ids = np.array([10, 10, 20, 20, 30, 30])
    loso = LeaveOneSeasonOut(min_train_folds=1)
    folds = loso.split(None, fold_ids=fold_ids)
    assert [f.fold_id for f in folds] == [20, 30]


def test_expanding_window_grows():
    """Training set should grow with each fold."""
    fold_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    ew = ExpandingWindow(min_train_size=1)
    folds = ew.split(None, fold_ids=fold_ids)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)  # monotonically increasing


def test_sliding_window_fixed_size():
    """Training window should not exceed window_size folds."""
    fold_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    sw = SlidingWindow(window_size=2)
    folds = sw.split(None, fold_ids=fold_ids)
    for fold in folds:
        train_unique = np.unique(fold_ids[fold.train_idx])
        assert len(train_unique) <= 2


def test_nested_cv_calibration_from_recent_fold():
    """Calibration data should come from the most recent training fold."""
    fold_ids = np.array([2015] * 20 + [2016] * 20 + [2017] * 20)
    nested = NestedCV(
        outer=LeaveOneSeasonOut(min_train_folds=1),
        inner_calibration_fraction=0.25,
    )
    folds = nested.split(None, fold_ids=fold_ids)
    # For fold testing 2017: training is [2015, 2016], calibration from 2016
    fold_2017 = [f for f in folds if f.fold_id == 2017][0]
    cal_fold_ids = set(fold_ids[fold_2017.calibration_idx])
    # Calibration should be from the most recent fold (2016)
    assert 2016 in cal_fold_ids
