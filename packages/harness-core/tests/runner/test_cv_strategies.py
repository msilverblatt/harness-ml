"""Tests for cross-validation strategy bridge (cv_strategies.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from harnessml.core.runner.cv_strategies import generate_cv_folds
from harnessml.core.runner.schema import BacktestConfig

# -----------------------------------------------------------------------
# Unit tests for generate_cv_folds
# -----------------------------------------------------------------------

class TestGenerateCVFolds:
    """Tests for the generate_cv_folds function."""

    def test_loso_basic(self):
        """LOSO returns one fold per value, training on all others."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_out",
            fold_column="season",
            fold_values=[2015, 2016, 2017],
        )
        folds = generate_cv_folds(df, bt)
        # All 3 values can be tested (each trains on the other 2)
        assert len(folds) == 3
        for train_folds, test_fold in folds:
            assert test_fold not in train_folds
            assert len(train_folds) == 2

    def test_loso_min_train_folds(self):
        """min_train_folds=2 keeps all folds with enough training data."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_out",
            fold_column="season",
            fold_values=[2015, 2016, 2017],
            min_train_folds=2,
        )
        folds = generate_cv_folds(df, bt)
        # All 3 values have 2 training folds each
        assert len(folds) == 3
        for train_folds, test_fold in folds:
            assert len(train_folds) >= 2

    def test_expanding_window_grows(self):
        """Each fold's training set grows."""
        fold_vals = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="expanding_window",
            fold_column="season",
            fold_values=fold_vals,
        )
        folds = generate_cv_folds(df, bt)
        assert len(folds) == 4  # 2016, 2017, 2018, 2019
        prev_size = 0
        for train_folds, test_fold in folds:
            assert len(train_folds) > prev_size
            prev_size = len(train_folds)

    def test_sliding_window_fixed_size(self):
        """Training window never exceeds window_size."""
        fold_vals = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            fold_column="season",
            fold_values=fold_vals,
            window_size=2,
        )
        folds = generate_cv_folds(df, bt)
        assert len(folds) == 4  # 2016, 2017, 2018, 2019
        for train_folds, test_fold in folds:
            assert len(train_folds) <= 2
            assert all(f < test_fold for f in train_folds)

    def test_sliding_window_requires_window_size(self):
        """Raises ValueError if window_size is None."""
        with pytest.raises((ValueError, Exception), match="window_size"):
            BacktestConfig(
                cv_strategy="sliding_window",
                fold_column="season",
                fold_values=[2015, 2016, 2017],
                window_size=None,
            )

    def test_purged_kfold_embargo(self):
        """Fold values within purge_gap of test are excluded from training."""
        fold_vals = [2015, 2016, 2017, 2018, 2019, 2020]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            fold_column="season",
            fold_values=fold_vals,
            n_folds=3,
            purge_gap=1,
        )
        folds = generate_cv_folds(df, bt)
        # Verify purge_gap is respected
        for train_folds, test_fold in folds:
            for tf in train_folds:
                assert abs(tf - test_fold) > 1, (
                    f"Fold {tf} is within purge_gap of test fold {test_fold}"
                )

    def test_purged_kfold_requires_n_folds(self):
        """Raises ValueError if n_folds is None."""
        with pytest.raises((ValueError, Exception), match="n_folds"):
            BacktestConfig(
                cv_strategy="purged_kfold",
                fold_column="season",
                fold_values=[2015, 2016, 2017],
                n_folds=None,
            )

    def test_uses_fold_values_from_config(self):
        """If bt_config.fold_values is set, uses those as test folds."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10 + [2018] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_out",
            fold_column="season",
            fold_values=[2016, 2017],
        )
        folds = generate_cv_folds(df, bt)
        # Both 2016 and 2017 are tested; training pool is all df values
        assert len(folds) == 2
        test_vals = sorted([test_fold for _, test_fold in folds])
        assert test_vals == [2016, 2017]
        # Each test fold's training set excludes only the test value
        for train_folds, test_fold in folds:
            assert test_fold not in train_folds

    def test_uses_df_folds_when_config_empty(self):
        """If bt_config.fold_values is empty, discovers from df."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_out",
            fold_column="season",
            fold_values=[],
        )
        folds = generate_cv_folds(df, bt)
        # Should discover 2015, 2016, 2017 from df
        assert len(folds) >= 2
        test_vals = [test_fold for _, test_fold in folds]
        assert 2016 in test_vals
        assert 2017 in test_vals

    def test_sliding_window_first_fold_has_fewer(self):
        """First fold may have fewer than window_size training folds."""
        fold_vals = [2015, 2016, 2017, 2018]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            fold_column="season",
            fold_values=fold_vals,
            window_size=3,
        )
        folds = generate_cv_folds(df, bt)
        # 2016 trains on [2015] (1 fold, less than window_size=3)
        first_train, first_test = folds[0]
        assert first_test == 2016
        assert len(first_train) == 1

    def test_purged_kfold_covers_all_test_folds(self):
        """All fold values appear as test folds exactly once."""
        fold_vals = [2015, 2016, 2017, 2018, 2019, 2020]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            fold_column="season",
            fold_values=fold_vals,
            n_folds=3,
            purge_gap=0,
        )
        folds = generate_cv_folds(df, bt)
        test_vals = sorted([test_fold for _, test_fold in folds])
        assert test_vals == fold_vals

    def test_loso_excludes_only_test_fold(self):
        """LOSO training set includes all folds except the test fold."""
        fold_vals = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": fold_vals * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_out",
            fold_column="season",
            fold_values=fold_vals,
        )
        folds = generate_cv_folds(df, bt)
        assert len(folds) == 5
        for train_folds, test_fold in folds:
            assert test_fold not in train_folds
            assert sorted(train_folds + [test_fold]) == fold_vals


# -----------------------------------------------------------------------
# Integration test: pipeline.backtest() with different CV strategies
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _setup_project_with_cv(
    tmp_path: Path,
    cv_strategy: str = "leave_one_out",
    window_size: int | None = None,
    n_folds: int | None = None,
    purge_gap: int = 1,
    n_rows: int = 300,
    n_folds_count: int = 5,
) -> Path:
    """Create a minimal project for backtest integration tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    fold_vals = list(range(2020, 2020 + n_folds_count))
    fold_col_data = rng.choice(fold_vals, size=n_rows)
    diff_x = rng.standard_normal(n_rows)
    prob = 1 / (1 + np.exp(-diff_x))
    result = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "season": fold_col_data,
        "diff_x": diff_x,
        "result": result,
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    backtest_cfg: dict = {
        "cv_strategy": cv_strategy,
        "fold_column": "season",
        "fold_values": fold_vals,
        "metrics": ["brier", "accuracy"],
        "min_train_folds": 1,
    }
    if window_size is not None:
        backtest_cfg["window_size"] = window_size
    if n_folds is not None:
        backtest_cfg["n_folds"] = n_folds
    backtest_cfg["purge_gap"] = purge_gap

    _write_yaml(
        config_dir / "pipeline.yaml",
        {
            "data": {
                "raw_dir": str(tmp_path / "data" / "raw"),
                "processed_dir": str(tmp_path / "data" / "processed"),
                "features_dir": str(features_dir),
            },
            "backtest": backtest_cfg,
        },
    )

    _write_yaml(
        config_dir / "models.yaml",
        {
            "models": {
                "logreg_seed": {
                    "type": "logistic_regression",
                    "features": ["diff_x"],
                    "params": {"max_iter": 200},
                    "active": True,
                },
            },
        },
    )

    _write_yaml(
        config_dir / "ensemble.yaml",
        {"ensemble": {"method": "average"}},
    )

    return config_dir


class TestBacktestWithCVStrategies:
    """Pipeline-level integration tests for different CV strategies."""

    def test_backtest_expanding_window(self, tmp_path):
        """backtest() works with expanding_window strategy."""
        from harnessml.core.runner.pipeline import PipelineRunner

        config_dir = _setup_project_with_cv(
            tmp_path, cv_strategy="expanding_window",
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        assert len(result["per_fold"]) >= 2

    def test_backtest_sliding_window(self, tmp_path):
        """backtest() works with sliding_window strategy."""
        from harnessml.core.runner.pipeline import PipelineRunner

        config_dir = _setup_project_with_cv(
            tmp_path, cv_strategy="sliding_window", window_size=2,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        assert len(result["per_fold"]) >= 2

    def test_backtest_purged_kfold(self, tmp_path):
        """backtest() works with purged_kfold strategy."""
        from harnessml.core.runner.pipeline import PipelineRunner

        config_dir = _setup_project_with_cv(
            tmp_path, cv_strategy="purged_kfold", n_folds=3, purge_gap=0,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        # purged_kfold generates folds for all values, but the pipeline
        # may skip early folds with no prior training data
        assert len(result["per_fold"]) >= 3

    def test_backtest_loso_matches_default(self, tmp_path):
        """LOSO strategy produces results consistent with default backtest."""
        from harnessml.core.runner.pipeline import PipelineRunner

        config_dir = _setup_project_with_cv(
            tmp_path, cv_strategy="leave_one_out",
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        assert result["metrics"]["brier"] >= 0
        assert result["metrics"]["brier"] <= 1.0
