import numpy as np
import pandas as pd
import pytest

from harness.ml.config.project import CVConfig
from harness.ml.runners.cross_validation import generate_folds


def make_df(n=100):
    rng = np.random.RandomState(0)
    return pd.DataFrame({"x": rng.randn(n)})


def make_df_with_col(n=100, col="fold", n_groups=10):
    df = make_df(n)
    df[col] = [i % n_groups for i in range(n)]
    return df


class TestKFold:
    def test_produces_n_folds(self):
        df = make_df(100)
        cfg = CVConfig(strategy="kfold", n_folds=5)
        folds = generate_folds(df, cfg)
        assert len(folds) == 5

    def test_no_train_test_overlap(self):
        df = make_df(100)
        cfg = CVConfig(strategy="kfold", n_folds=5)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_all_indices_covered(self):
        df = make_df(100)
        cfg = CVConfig(strategy="kfold", n_folds=5)
        all_test = np.concatenate([t for _, t in generate_folds(df, cfg)])
        assert sorted(all_test) == list(range(100))


class TestStratifiedKFold:
    def test_requires_target(self):
        df = make_df(100)
        cfg = CVConfig(strategy="stratified_kfold", n_folds=5)
        with pytest.raises(ValueError, match="requires target"):
            generate_folds(df, cfg)

    def test_produces_n_folds(self):
        df = make_df(100)
        target = pd.Series([i % 2 for i in range(100)])
        cfg = CVConfig(strategy="stratified_kfold", n_folds=5)
        folds = generate_folds(df, cfg, target=target)
        assert len(folds) == 5

    def test_similar_class_distribution(self):
        n = 200
        df = make_df(n)
        target = pd.Series([0] * 150 + [1] * 50)
        cfg = CVConfig(strategy="stratified_kfold", n_folds=5)
        for train_idx, test_idx in generate_folds(df, cfg, target=target):
            test_rate = target.iloc[test_idx].mean()
            # Global positive rate is 0.25; each fold should be close
            assert abs(test_rate - 0.25) < 0.05

    def test_no_train_test_overlap(self):
        df = make_df(100)
        target = pd.Series([i % 2 for i in range(100)])
        cfg = CVConfig(strategy="stratified_kfold", n_folds=5)
        for train_idx, test_idx in generate_folds(df, cfg, target=target):
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestGroupKFold:
    def test_requires_fold_column(self):
        df = make_df(100)
        cfg = CVConfig(strategy="group_kfold", n_folds=5)
        with pytest.raises(ValueError, match="requires fold_column"):
            generate_folds(df, cfg)

    def test_produces_n_folds(self):
        df = make_df_with_col(100, "group", n_groups=10)
        cfg = CVConfig(strategy="group_kfold", n_folds=5, fold_column="group")
        folds = generate_folds(df, cfg)
        assert len(folds) == 5

    def test_groups_dont_split(self):
        df = make_df_with_col(100, "group", n_groups=10)
        cfg = CVConfig(strategy="group_kfold", n_folds=5, fold_column="group")
        for train_idx, test_idx in generate_folds(df, cfg):
            train_groups = set(df["group"].iloc[train_idx])
            test_groups = set(df["group"].iloc[test_idx])
            assert train_groups.isdisjoint(test_groups)


class TestLeaveOneOut:
    def test_requires_fold_column(self):
        df = make_df(50)
        cfg = CVConfig(strategy="leave_one_out")
        with pytest.raises(ValueError, match="requires fold_column"):
            generate_folds(df, cfg)

    def test_one_fold_per_unique_value(self):
        df = make_df_with_col(100, "season", n_groups=5)
        cfg = CVConfig(strategy="leave_one_out", fold_column="season", min_train_folds=1)
        folds = generate_folds(df, cfg)
        assert len(folds) == 5

    def test_min_train_folds_respected(self):
        # With min_train_folds=3 and 5 groups, first 2 groups are skipped (only 4 or 3 in train)
        # group 0: train has groups 1-4 (4 rows each), always >= 3 samples but need >= 3 folds
        # Actually min_train_folds=3 means train_idx length >= 3
        df = make_df_with_col(50, "grp", n_groups=5)
        cfg = CVConfig(strategy="leave_one_out", fold_column="grp", min_train_folds=40)
        folds = generate_folds(df, cfg)
        for train_idx, _ in folds:
            assert len(train_idx) >= 40

    def test_no_train_test_overlap(self):
        df = make_df_with_col(100, "wk", n_groups=5)
        cfg = CVConfig(strategy="leave_one_out", fold_column="wk", min_train_folds=1)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestExpandingWindow:
    def test_requires_fold_column(self):
        df = make_df(50)
        cfg = CVConfig(strategy="expanding_window")
        with pytest.raises(ValueError, match="requires fold_column"):
            generate_folds(df, cfg)

    def test_train_grows(self):
        df = make_df_with_col(100, "period", n_groups=10)
        cfg = CVConfig(strategy="expanding_window", fold_column="period", min_train_folds=2)
        folds = generate_folds(df, cfg)
        train_sizes = [len(t) for t, _ in folds]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_test_is_next_period(self):
        df = make_df_with_col(100, "period", n_groups=10)
        sorted_vals = sorted(df["period"].unique())
        cfg = CVConfig(strategy="expanding_window", fold_column="period", min_train_folds=2)
        folds = generate_folds(df, cfg)
        for i, (train_idx, test_idx) in enumerate(folds):
            expected_test_val = sorted_vals[i + 2]
            assert set(df["period"].iloc[test_idx]) == {expected_test_val}

    def test_no_train_test_overlap(self):
        df = make_df_with_col(100, "period", n_groups=10)
        cfg = CVConfig(strategy="expanding_window", fold_column="period", min_train_folds=2)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestSlidingWindow:
    def test_requires_fold_column(self):
        df = make_df(50)
        cfg = CVConfig(strategy="sliding_window")
        with pytest.raises(ValueError, match="requires fold_column"):
            generate_folds(df, cfg)

    def test_fixed_train_size(self):
        df = make_df_with_col(100, "period", n_groups=10)
        cfg = CVConfig(strategy="sliding_window", fold_column="period", min_train_folds=3)
        folds = generate_folds(df, cfg)
        # Each fold should have the same number of training rows (3 periods * rows_per_period)
        train_sizes = [len(t) for t, _ in folds]
        assert len(set(train_sizes)) == 1

    def test_no_train_test_overlap(self):
        df = make_df_with_col(100, "period", n_groups=10)
        cfg = CVConfig(strategy="sliding_window", fold_column="period", min_train_folds=2)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_window_slides(self):
        df = make_df_with_col(100, "period", n_groups=10)
        sorted_vals = sorted(df["period"].unique())
        cfg = CVConfig(strategy="sliding_window", fold_column="period", min_train_folds=2)
        folds = generate_folds(df, cfg)
        for i, (train_idx, _) in enumerate(folds):
            expected_train_vals = set(sorted_vals[i:i + 2])
            actual_train_vals = set(df["period"].iloc[train_idx])
            assert actual_train_vals == expected_train_vals


class TestPurgedKFold:
    def test_produces_n_folds(self):
        df = make_df(100)
        cfg = CVConfig(strategy="purged_kfold", n_folds=5)
        folds = generate_folds(df, cfg)
        assert len(folds) == 5

    def test_gap_between_train_and_test(self):
        df = make_df(200)
        cfg = CVConfig(strategy="purged_kfold", n_folds=5)
        n = len(df)
        gap = max(1, n // (cfg.n_folds * 10))
        for train_idx, test_idx in generate_folds(df, cfg):
            test_min, test_max = test_idx.min(), test_idx.max()
            # No training index should be within gap distance of test boundary
            near_boundary = train_idx[
                (train_idx >= test_min - gap) & (train_idx <= test_max + gap)
            ]
            assert len(near_boundary) == 0

    def test_no_train_test_overlap(self):
        df = make_df(100)
        cfg = CVConfig(strategy="purged_kfold", n_folds=5)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(train_idx, test_idx)) == 0


class TestBootstrap:
    def test_produces_n_folds(self):
        df = make_df(100)
        cfg = CVConfig(strategy="bootstrap", n_folds=5)
        folds = generate_folds(df, cfg)
        assert len(folds) == 5

    def test_oob_samples_exist(self):
        df = make_df(100)
        cfg = CVConfig(strategy="bootstrap", n_folds=5)
        for _, test_idx in generate_folds(df, cfg):
            assert len(test_idx) > 0

    def test_train_has_repeated_indices(self):
        df = make_df(100)
        cfg = CVConfig(strategy="bootstrap", n_folds=3)
        for train_idx, _ in generate_folds(df, cfg):
            # With replacement, should have duplicates
            assert len(train_idx) == len(df)
            assert len(np.unique(train_idx)) < len(train_idx)

    def test_oob_not_in_train(self):
        df = make_df(100)
        cfg = CVConfig(strategy="bootstrap", n_folds=3)
        for train_idx, test_idx in generate_folds(df, cfg):
            assert len(np.intersect1d(np.unique(train_idx), test_idx)) == 0


class TestUnknownStrategy:
    def test_raises_value_error(self):
        df = make_df(50)
        cfg = CVConfig(strategy="not_a_real_strategy")
        with pytest.raises(ValueError, match="Unknown CV strategy"):
            generate_folds(df, cfg)
