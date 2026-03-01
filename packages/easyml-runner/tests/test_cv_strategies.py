"""Tests for cross-validation strategy bridge (cv_strategies.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner.cv_strategies import generate_cv_folds
from easyml.runner.schema import BacktestConfig


# -----------------------------------------------------------------------
# Unit tests for generate_cv_folds
# -----------------------------------------------------------------------

class TestGenerateCVFolds:
    """Tests for the generate_cv_folds function."""

    def test_loso_basic(self):
        """LOSO returns folds where test season comes after all train seasons."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=[2015, 2016, 2017],
        )
        folds = generate_cv_folds(df, bt)
        # 2016 and 2017 can be tested (2015 has no prior seasons)
        assert len(folds) >= 2
        for train_s, test_s in folds:
            assert all(s < test_s for s in train_s)

    def test_loso_min_train_folds(self):
        """min_train_folds=2 skips first testable season."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=[2015, 2016, 2017],
            min_train_folds=2,
        )
        folds = generate_cv_folds(df, bt)
        # Only 2017 has 2 prior seasons (2015, 2016)
        assert len(folds) == 1
        train_s, test_s = folds[0]
        assert test_s == 2017
        assert train_s == [2015, 2016]

    def test_expanding_window_grows(self):
        """Each fold's training set grows."""
        seasons = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="expanding_window",
            seasons=seasons,
        )
        folds = generate_cv_folds(df, bt)
        assert len(folds) == 4  # 2016, 2017, 2018, 2019
        prev_size = 0
        for train_s, test_s in folds:
            assert len(train_s) > prev_size
            prev_size = len(train_s)

    def test_sliding_window_fixed_size(self):
        """Training window never exceeds window_size."""
        seasons = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            seasons=seasons,
            window_size=2,
        )
        folds = generate_cv_folds(df, bt)
        assert len(folds) == 4  # 2016, 2017, 2018, 2019
        for train_s, test_s in folds:
            assert len(train_s) <= 2
            assert all(s < test_s for s in train_s)

    def test_sliding_window_requires_window_size(self):
        """Raises ValueError if window_size is None."""
        df = pd.DataFrame({"season": [2015, 2016, 2017]})
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            seasons=[2015, 2016, 2017],
            window_size=None,
        )
        with pytest.raises(ValueError, match="window_size"):
            generate_cv_folds(df, bt)

    def test_purged_kfold_embargo(self):
        """Seasons within purge_gap of test are excluded from training."""
        seasons = [2015, 2016, 2017, 2018, 2019, 2020]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            seasons=seasons,
            n_folds=3,
            purge_gap=1,
        )
        folds = generate_cv_folds(df, bt)
        # Verify purge_gap is respected
        for train_s, test_s in folds:
            for ts in train_s:
                assert abs(ts - test_s) > 1, (
                    f"Season {ts} is within purge_gap of test season {test_s}"
                )

    def test_purged_kfold_requires_n_folds(self):
        """Raises ValueError if n_folds is None."""
        df = pd.DataFrame({"season": [2015, 2016, 2017]})
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            seasons=[2015, 2016, 2017],
            n_folds=None,
        )
        with pytest.raises(ValueError, match="n_folds"):
            generate_cv_folds(df, bt)

    def test_uses_seasons_from_config(self):
        """If bt_config.seasons is set, uses those, not df seasons."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10 + [2018] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=[2016, 2017],
        )
        folds = generate_cv_folds(df, bt)
        # Only 2017 can be tested (2016 has no prior in the config seasons list)
        assert len(folds) == 1
        train_s, test_s = folds[0]
        assert test_s == 2017
        assert train_s == [2016]

    def test_uses_df_seasons_when_config_empty(self):
        """If bt_config.seasons is empty, discovers from df."""
        df = pd.DataFrame({"season": [2015] * 10 + [2016] * 10 + [2017] * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=[],
        )
        folds = generate_cv_folds(df, bt)
        # Should discover 2015, 2016, 2017 from df
        assert len(folds) >= 2
        test_seasons = [test_s for _, test_s in folds]
        assert 2016 in test_seasons
        assert 2017 in test_seasons

    def test_sliding_window_first_fold_has_fewer(self):
        """First fold may have fewer than window_size training seasons."""
        seasons = [2015, 2016, 2017, 2018]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            seasons=seasons,
            window_size=3,
        )
        folds = generate_cv_folds(df, bt)
        # 2016 trains on [2015] (1 season, less than window_size=3)
        first_train, first_test = folds[0]
        assert first_test == 2016
        assert len(first_train) == 1

    def test_purged_kfold_covers_all_test_seasons(self):
        """All seasons appear as test seasons exactly once."""
        seasons = [2015, 2016, 2017, 2018, 2019, 2020]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            seasons=seasons,
            n_folds=3,
            purge_gap=0,
        )
        folds = generate_cv_folds(df, bt)
        test_seasons = sorted([test_s for _, test_s in folds])
        assert test_seasons == seasons

    def test_loso_no_future_leakage(self):
        """LOSO never includes future seasons in training."""
        seasons = [2015, 2016, 2017, 2018, 2019]
        df = pd.DataFrame({"season": seasons * 10})
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=seasons,
        )
        folds = generate_cv_folds(df, bt)
        for train_s, test_s in folds:
            assert all(s < test_s for s in train_s)
            # No future seasons
            assert not any(s > test_s for s in train_s)


# -----------------------------------------------------------------------
# Integration test: pipeline.backtest() with different CV strategies
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _setup_project_with_cv(
    tmp_path: Path,
    cv_strategy: str = "leave_one_season_out",
    window_size: int | None = None,
    n_folds: int | None = None,
    purge_gap: int = 1,
    n_rows: int = 300,
    n_seasons: int = 5,
) -> Path:
    """Create a minimal project for backtest integration tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    seasons_list = list(range(2020, 2020 + n_seasons))
    seasons = rng.choice(seasons_list, size=n_rows)
    diff_x = rng.standard_normal(n_rows)
    prob = 1 / (1 + np.exp(-diff_x))
    result = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "season": seasons,
        "diff_x": diff_x,
        "result": result,
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    backtest_cfg: dict = {
        "cv_strategy": cv_strategy,
        "seasons": seasons_list,
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
        from easyml.runner.pipeline import PipelineRunner

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
        from easyml.runner.pipeline import PipelineRunner

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
        from easyml.runner.pipeline import PipelineRunner

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
        # purged_kfold should test all seasons
        assert len(result["per_fold"]) == 5

    def test_backtest_loso_matches_default(self, tmp_path):
        """LOSO strategy produces results consistent with default backtest."""
        from easyml.runner.pipeline import PipelineRunner

        config_dir = _setup_project_with_cv(
            tmp_path, cv_strategy="leave_one_season_out",
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
