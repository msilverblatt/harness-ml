import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from harness.ml.config.project import CVConfig


def generate_folds(
    df: pd.DataFrame,
    config: CVConfig,
    target: pd.Series | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index splits."""
    n = len(df)
    strategy = config.strategy

    if strategy == "kfold":
        kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
        return list(kf.split(df))

    elif strategy == "stratified_kfold":
        if target is None:
            raise ValueError("stratified_kfold requires target")
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=42)
        return list(skf.split(df, target))

    elif strategy == "group_kfold":
        if config.fold_column is None:
            raise ValueError("group_kfold requires fold_column")
        gkf = GroupKFold(n_splits=config.n_folds)
        groups = df[config.fold_column]
        return list(gkf.split(df, groups=groups))

    elif strategy == "leave_one_out":
        if config.fold_column is None:
            raise ValueError("leave_one_out requires fold_column")
        fold_values = config.fold_values or sorted(df[config.fold_column].unique())
        folds = []
        for val in fold_values:
            test_mask = df[config.fold_column] == val
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) >= config.min_train_folds:
                folds.append((train_idx, test_idx))
        return folds

    elif strategy == "expanding_window":
        if config.fold_column is None:
            raise ValueError("expanding_window requires fold_column")
        fold_values = config.fold_values or sorted(df[config.fold_column].unique())
        folds = []
        for i in range(config.min_train_folds, len(fold_values)):
            train_vals = fold_values[:i]
            test_val = fold_values[i]
            train_idx = np.where(df[config.fold_column].isin(train_vals))[0]
            test_idx = np.where(df[config.fold_column] == test_val)[0]
            folds.append((train_idx, test_idx))
        return folds

    elif strategy == "sliding_window":
        if config.fold_column is None:
            raise ValueError("sliding_window requires fold_column")
        fold_values = config.fold_values or sorted(df[config.fold_column].unique())
        window_size = config.min_train_folds
        folds = []
        for i in range(window_size, len(fold_values)):
            train_vals = fold_values[i - window_size:i]
            test_val = fold_values[i]
            train_idx = np.where(df[config.fold_column].isin(train_vals))[0]
            test_idx = np.where(df[config.fold_column] == test_val)[0]
            folds.append((train_idx, test_idx))
        return folds

    elif strategy == "purged_kfold":
        kf = KFold(n_splits=config.n_folds, shuffle=False)
        folds = []
        gap = max(1, n // (config.n_folds * 10))  # ~10% of fold size
        for train_idx, test_idx in kf.split(df):
            test_min, test_max = test_idx.min(), test_idx.max()
            purge_mask = (train_idx >= test_min - gap) & (train_idx <= test_max + gap)
            purged_train = train_idx[~purge_mask]
            folds.append((purged_train, test_idx))
        return folds

    elif strategy == "bootstrap":
        rng = np.random.RandomState(42)
        folds = []
        for _ in range(config.n_folds):
            train_idx = rng.choice(n, size=n, replace=True)
            test_idx = np.setdiff1d(np.arange(n), np.unique(train_idx))
            if len(test_idx) > 0:
                folds.append((train_idx, test_idx))
        return folds

    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")
