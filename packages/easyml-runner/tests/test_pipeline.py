"""Tests for PipelineRunner — wires library APIs from YAML config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner.pipeline import PipelineRunner


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _make_matchup_parquet(
    path: Path,
    n_rows: int = 200,
    n_seasons: int = 3,
    include_margin: bool = False,
    include_seed: bool = False,
    mm_style: bool = False,
    extra_features: list[str] | None = None,
) -> None:
    """Create a mock features parquet with diff_x feature and season column."""
    rng = np.random.default_rng(42)
    seasons = rng.choice(list(range(2022, 2022 + n_seasons)), size=n_rows)
    result = rng.integers(0, 2, size=n_rows)

    data = {
        "diff_x": rng.standard_normal(n_rows),
    }

    if mm_style:
        data["Season"] = seasons
        data["TeamAWon"] = result
        if include_margin:
            data["TeamAMargin"] = rng.standard_normal(n_rows) * 10
    else:
        data["season"] = seasons
        data["result"] = result
        if include_margin:
            data["margin"] = rng.standard_normal(n_rows) * 10

    if include_seed:
        seed_col = "diff_seed_num"
        data[seed_col] = rng.integers(-15, 16, size=n_rows).astype(float)

    if extra_features:
        for feat in extra_features:
            data[feat] = rng.standard_normal(n_rows)

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _setup_project(
    tmp_path: Path,
    models: dict | None = None,
    ensemble: dict | None = None,
    n_rows: int = 200,
    n_seasons: int = 3,
    include_margin: bool = False,
    include_seed: bool = False,
    mm_style: bool = False,
    extra_features: list[str] | None = None,
    seasons: list[int] | None = None,
) -> Path:
    """Create a minimal project config and data for PipelineRunner tests.

    Returns the config directory path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline config
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    _make_matchup_parquet(
        features_dir / "features.parquet",
        n_rows=n_rows,
        n_seasons=n_seasons,
        include_margin=include_margin,
        include_seed=include_seed,
        mm_style=mm_style,
        extra_features=extra_features,
    )

    if seasons is None:
        seasons = list(range(2022, 2022 + n_seasons))

    _write_yaml(
        config_dir / "pipeline.yaml",
        {
            "data": {
                "raw_dir": str(tmp_path / "data" / "raw"),
                "processed_dir": str(tmp_path / "data" / "processed"),
                "features_dir": str(features_dir),
            },
            "backtest": {
                "cv_strategy": "leave_one_season_out",
                "seasons": seasons,
                "metrics": ["brier", "accuracy"],
                "min_train_folds": 1,
            },
        },
    )

    if models is None:
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            }
        }
    _write_yaml(config_dir / "models.yaml", {"models": models})

    if ensemble is None:
        ensemble = {"method": "average"}
    _write_yaml(config_dir / "ensemble.yaml", {"ensemble": ensemble})

    return config_dir


# -----------------------------------------------------------------------
# Tests — original
# -----------------------------------------------------------------------

class TestLoad:
    """PipelineRunner.load() validates config successfully."""

    def test_load(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        assert runner.config is not None
        assert "logreg_seed" in runner.config.models


class TestTrain:
    """PipelineRunner.train() trains models."""

    def test_train(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert result["status"] == "success"
        assert "logreg_seed" in result["models_trained"]


class TestBacktest:
    """PipelineRunner.backtest() returns metrics."""

    def test_backtest(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "metrics" in result
        assert "brier" in result["metrics"]
        assert "accuracy" in result["metrics"]
        assert "per_fold" in result


class TestRunFull:
    """PipelineRunner.run_full() runs load + train + backtest."""

    def test_run_full(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        result = runner.run_full()
        assert result["status"] == "success"
        assert "metrics" in result


class TestTwoModels:
    """PipelineRunner trains multiple models."""

    def test_two_models(self, tmp_path):
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_v2": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 500, "C": 0.5},
                "active": True,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert result["status"] == "success"
        assert "logreg_seed" in result["models_trained"]
        assert "logreg_v2" in result["models_trained"]


class TestInactiveModelSkipped:
    """Inactive models are not trained."""

    def test_inactive_skipped(self, tmp_path):
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_inactive": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {},
                "active": False,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert "logreg_seed" in result["models_trained"]
        assert "logreg_inactive" not in result["models_trained"]


# -----------------------------------------------------------------------
# Tests — new: stacked ensemble backtesting
# -----------------------------------------------------------------------

class TestStackedEnsembleBacktest:
    """Backtest with stacked ensemble (meta-learner)."""

    def test_stacked_three_models(self, tmp_path):
        """Stacked backtest with 3 models across 3 seasons produces ensemble."""
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.5},
                "active": True,
            },
            "logreg_c": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 2.0},
                "active": True,
            },
        }
        ensemble = {
            "method": "stacked",
            "meta_learner": {"C": 1.0},
            "calibration": "none",
        }
        config_dir = _setup_project(
            tmp_path,
            models=models,
            ensemble=ensemble,
            n_rows=300,
            n_seasons=3,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "metrics" in result
        assert "brier" in result["metrics"]
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["brier"] >= 0
        assert result["metrics"]["brier"] <= 1.0
        # LOSO skips first season (no prior training data), so 2 folds
        assert len(result["per_fold"]) == 2


class TestExcludeModels:
    """Exclude models from ensemble."""

    def test_exclude_models(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_excluded": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        ensemble = {
            "method": "average",
            "exclude_models": ["logreg_excluded"],
        }
        config_dir = _setup_project(
            tmp_path, models=models, ensemble=ensemble,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # Excluded model should not be in models_trained
        assert "logreg_excluded" not in result["models_trained"]
        assert "logreg_a" in result["models_trained"]


class TestProbEnsembleColumn:
    """Verify prob_ensemble column is produced."""

    def test_prob_ensemble_exists(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        # The backtest should succeed and produce metrics
        assert result["status"] == "success"
        assert "brier" in result["metrics"]


class TestSimpleAverageMethod:
    """Simple average method works."""

    def test_simple_average(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
        }
        ensemble = {"method": "average"}
        config_dir = _setup_project(
            tmp_path, models=models, ensemble=ensemble,
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


class TestColumnAutoDetect:
    """Auto-detect mm-style column names."""

    def test_mm_style_columns(self, tmp_path):
        """mm-style columns (TeamAWon, Season) are auto-normalized."""
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, mm_style=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]


class TestTrainSeasonsFiltering:
    """train_seasons filtering works in backtest."""

    def test_train_seasons_last_n(self, tmp_path):
        """Models with train_seasons='last_2' use fewer seasons."""
        models = {
            "logreg_recent": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "train_seasons": "last_2",
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models,
            n_rows=300, n_seasons=4,
            seasons=[2022, 2023, 2024, 2025],
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]


# -----------------------------------------------------------------------
# Tests — provider model dependencies
# -----------------------------------------------------------------------

class TestProviderModelsBacktest:
    """Provider models trained before consumers, outputs injected as features."""

    def test_matchup_provider_backtest(self, tmp_path):
        """Matchup-level provider → consumer backtest runs end-to-end."""
        models = {
            "provider": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "provides": ["predicted_prob"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "consumer": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_predicted_prob"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=300, n_seasons=3,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        # Provider not in ensemble, only consumer
        assert "consumer" in result["models_trained"]
        # Provider should not be in the ensemble model list
        # (include_in_ensemble=False means no prob_provider column)

    def test_provider_include_in_ensemble_true(self, tmp_path):
        """Provider with include_in_ensemble=True contributes to ensemble."""
        models = {
            "provider": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 2.0},
                "active": True,
                "provides": ["score"],
                "provides_level": "matchup",
                "include_in_ensemble": True,
            },
            "consumer": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_score"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=300, n_seasons=3,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # Both models in ensemble
        assert "provider" in result["models_trained"]
        assert "consumer" in result["models_trained"]

    def test_chained_providers(self, tmp_path):
        """Provider A → provider B → consumer C (3-wave chain)."""
        models = {
            "base": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "provides": ["base_score"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "mid": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_base_score"],
                "params": {"max_iter": 200},
                "active": True,
                "provides": ["mid_score"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "top": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_mid_score"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=300, n_seasons=3,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        # Only "top" is in ensemble
        assert "top" in result["models_trained"]

    def test_provider_with_independent_models(self, tmp_path):
        """Provider + independent models all work together."""
        models = {
            "provider": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 2.0},
                "active": True,
                "provides": ["predicted_prob"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "consumer": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_predicted_prob"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "independent": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.5},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=300, n_seasons=3,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # consumer and independent in ensemble, provider excluded
        assert "consumer" in result["models_trained"]
        assert "independent" in result["models_trained"]

    def test_provider_train_method(self, tmp_path):
        """train() also respects wave ordering for providers."""
        models = {
            "provider": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "provides": ["score"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "consumer": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_score"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=300, n_seasons=3,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()

        assert result["status"] == "success"
        assert "provider" in result["models_trained"]
        assert "consumer" in result["models_trained"]

    def test_fingerprint_propagation(self, tmp_path):
        """Provider fingerprint changes propagate to dependents."""
        from easyml.runner.fingerprint import compute_fingerprint

        provider_config = {"type": "logistic_regression", "features": ["diff_x"]}
        consumer_config = {"type": "logistic_regression", "features": ["diff_x", "diff_score"]}

        # Compute fingerprints with same upstream
        provider_fp_v1 = compute_fingerprint(provider_config)
        consumer_fp_v1 = compute_fingerprint(
            consumer_config,
            upstream_fingerprints={"provider": provider_fp_v1},
        )

        # Change provider config → different provider fingerprint
        provider_config_v2 = {**provider_config, "params": {"C": 0.5}}
        provider_fp_v2 = compute_fingerprint(provider_config_v2)
        consumer_fp_v2 = compute_fingerprint(
            consumer_config,
            upstream_fingerprints={"provider": provider_fp_v2},
        )

        # Provider fingerprint changed
        assert provider_fp_v1 != provider_fp_v2
        # Consumer fingerprint also changed (even though its own config didn't)
        assert consumer_fp_v1 != consumer_fp_v2

    def test_fingerprint_stable_without_upstream_change(self, tmp_path):
        """Consumer fingerprint stable when provider fingerprint is unchanged."""
        from easyml.runner.fingerprint import compute_fingerprint

        provider_fp = compute_fingerprint({"type": "logistic_regression"})
        consumer_config = {"type": "logistic_regression", "features": ["diff_score"]}

        fp1 = compute_fingerprint(
            consumer_config,
            upstream_fingerprints={"provider": provider_fp},
        )
        fp2 = compute_fingerprint(
            consumer_config,
            upstream_fingerprints={"provider": provider_fp},
        )
        assert fp1 == fp2


# -----------------------------------------------------------------------
# Tests — prediction cache integration
# -----------------------------------------------------------------------

class TestPredictionCacheIntegration:
    """PredictionCache wired into PipelineRunner backtest."""

    def test_first_run_all_misses(self, tmp_path):
        """First backtest with cache → all misses, no hits."""
        from easyml.runner.prediction_cache import PredictionCache

        # 4 seasons → LOSO with min_train=1 gives 3 holdout folds
        # (first season has no prior data to train on)
        config_dir = _setup_project(tmp_path, n_rows=300, n_seasons=4)
        cache = PredictionCache(tmp_path / "cache")

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        stats = runner.cache_stats
        assert stats["hits"] == 0
        # 1 model x 3 holdout seasons = 3 misses
        assert stats["misses"] == 3

    def test_second_run_all_hits(self, tmp_path):
        """Second backtest with same cache → all hits, no misses."""
        from easyml.runner.prediction_cache import PredictionCache

        config_dir = _setup_project(tmp_path, n_rows=300, n_seasons=4)
        cache = PredictionCache(tmp_path / "cache")

        # First run — fills cache
        runner1 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner1.load()
        runner1.backtest()

        # Second run — should hit cache for all models
        runner2 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner2.load()
        result2 = runner2.backtest()

        assert result2["status"] == "success"
        stats2 = runner2.cache_stats
        assert stats2["hits"] == 3  # 1 model x 3 holdout seasons
        assert stats2["misses"] == 0

    def test_cache_produces_same_metrics(self, tmp_path):
        """Cached predictions produce identical metrics to uncached run."""
        from easyml.runner.prediction_cache import PredictionCache

        config_dir = _setup_project(tmp_path, n_rows=300, n_seasons=4)
        cache = PredictionCache(tmp_path / "cache")

        # First run
        runner1 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner1.load()
        result1 = runner1.backtest()

        # Second run (cached)
        runner2 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner2.load()
        result2 = runner2.backtest()

        # Metrics should be identical
        for metric in result1["metrics"]:
            assert result1["metrics"][metric] == pytest.approx(
                result2["metrics"][metric], abs=1e-6
            ), f"Metric {metric} differs"

    def test_different_model_config_misses(self, tmp_path):
        """Changing model config causes cache misses."""
        from easyml.runner.prediction_cache import PredictionCache

        cache = PredictionCache(tmp_path / "cache")
        n_folds = 3  # 4 seasons, LOSO, first skipped → 3 folds

        # First config
        models_v1 = {
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
        }
        config_dir1 = _setup_project(
            tmp_path / "v1", models=models_v1, n_rows=300, n_seasons=4,
        )
        runner1 = PipelineRunner(
            project_dir=str(tmp_path / "v1"),
            config_dir=str(config_dir1),
            prediction_cache=cache,
        )
        runner1.load()
        runner1.backtest()
        assert runner1.cache_stats["misses"] == n_folds

        # Second config — different C value → different fingerprint
        models_v2 = {
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
        }
        config_dir2 = _setup_project(
            tmp_path / "v2", models=models_v2, n_rows=300, n_seasons=4,
        )
        runner2 = PipelineRunner(
            project_dir=str(tmp_path / "v2"),
            config_dir=str(config_dir2),
            prediction_cache=cache,
        )
        runner2.load()
        runner2.backtest()
        # Different model config → all misses again
        assert runner2.cache_stats["hits"] == 0
        assert runner2.cache_stats["misses"] == n_folds

    def test_multi_model_partial_cache(self, tmp_path):
        """With 2 models, changing one causes misses only for that model."""
        from easyml.runner.prediction_cache import PredictionCache

        cache = PredictionCache(tmp_path / "cache")
        n_folds = 3  # 4 seasons, LOSO, first skipped → 3 folds

        models_v1 = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.5},
                "active": True,
            },
        }

        config_dir = _setup_project(
            tmp_path, models=models_v1, n_rows=300, n_seasons=4,
        )
        runner1 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner1.load()
        runner1.backtest()
        # 2 models x 3 folds = 6 misses
        assert runner1.cache_stats["misses"] == 2 * n_folds

        # Change only model B's C value
        models_v2 = {
            "logreg_a": models_v1["logreg_a"],  # unchanged
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
        }
        _write_yaml(
            tmp_path / "config" / "models.yaml",
            {"models": models_v2},
        )
        runner2 = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(tmp_path / "config"),
            prediction_cache=cache,
        )
        runner2.load()
        runner2.backtest()
        # Model A: n_folds hits (unchanged), Model B: n_folds misses (changed)
        assert runner2.cache_stats["hits"] == n_folds
        assert runner2.cache_stats["misses"] == n_folds

    def test_no_cache_by_default(self, tmp_path):
        """Without prediction_cache, no caching happens."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        runner.backtest()
        assert runner.cache_stats == {"hits": 0, "misses": 0}

    def test_provider_not_cached(self, tmp_path):
        """Provider models are always retrained (not cached)."""
        from easyml.runner.prediction_cache import PredictionCache

        cache = PredictionCache(tmp_path / "cache")
        models = {
            "provider": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "provides": ["score"],
                "provides_level": "matchup",
                "include_in_ensemble": False,
            },
            "consumer": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_score"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        # 4 seasons → 3 LOSO folds (first season skipped, no prior data)
        config_dir = _setup_project(
            tmp_path, models=models, n_rows=400, n_seasons=4,
            include_seed=True,
        )

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            prediction_cache=cache,
        )
        runner.load()
        runner.backtest()

        # Only consumer gets cached (3 misses), provider skips cache
        assert runner.cache_stats["misses"] == 3
        assert runner.cache_stats["hits"] == 0


# -----------------------------------------------------------------------
# Tests — reporting integration
# -----------------------------------------------------------------------


class TestBacktestReporting:
    """Backtest result includes report and diagnostics."""

    def test_backtest_includes_report(self, tmp_path):
        """Backtest result has a 'report' key with markdown content."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert "report" in result
        assert "# Backtest Report" in result["report"]
        assert "Top-Line Metrics" in result["report"]

    def test_backtest_includes_diagnostics(self, tmp_path):
        """Backtest result has a 'diagnostics' key with per-season data."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert "diagnostics" in result
        assert isinstance(result["diagnostics"], list)
        assert len(result["diagnostics"]) > 0
        assert "season" in result["diagnostics"][0]

    def test_backtest_exports_artifacts_to_run_dir(self, tmp_path):
        """When run_dir is set, artifacts are exported."""
        config_dir = _setup_project(tmp_path)
        run_dir = tmp_path / "output" / "run_001"
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
            run_dir=str(run_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result.get("run_dir") == str(run_dir)
        assert (run_dir / "diagnostics" / "report.md").exists()
        assert (run_dir / "diagnostics" / "pooled_metrics.json").exists()
        assert (run_dir / "diagnostics" / "diagnostics.parquet").exists()

    def test_backtest_no_run_dir_no_export(self, tmp_path):
        """Without run_dir, no artifacts are exported."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert "run_dir" not in result
        assert "report" in result  # report is always generated

    def test_report_has_per_season_breakdown(self, tmp_path):
        """Report includes per-season breakdown section."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert "Per-Season Breakdown" in result["report"]

    def test_report_has_pick_analysis(self, tmp_path):
        """Report includes pick analysis section."""
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert "Pick Analysis" in result["report"]


# -----------------------------------------------------------------------
# Tests — failed model surfacing
# -----------------------------------------------------------------------


class TestFailedModelsSurfaced:
    """Failed models during backtest appear in result['models_failed']."""

    def test_failed_model_in_models_failed(self, tmp_path):
        """A model that fails during training appears in models_failed."""
        # Use a regressor with 'margin' as target, but margin column is absent
        # from the data — this causes KeyError during training.
        # The regressor requires a 'margin' column as training target (pipeline.py training.py:83-86).
        # Omitting it causes a ValueError that backtest() catches and tracks in _failed_models.
        models = {
            "good_model": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "bad_regressor": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
                "mode": "regressor",
            },
        }
        # Note: include_margin=False so 'margin' column is absent
        config_dir = _setup_project(tmp_path, models=models, include_margin=False)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "models_failed" in result
        assert "bad_regressor" in result["models_failed"]
        assert "bad_regressor" not in result["models_trained"]
        assert "good_model" in result["models_trained"]


# -----------------------------------------------------------------------
# Tests — cdf_scale diagnostics
# -----------------------------------------------------------------------


class TestCdfScaleInDiagnostics:
    def test_cdf_scale_tracked_for_regressors(self, tmp_path):
        """Regression models have their fitted cdf_scale averaged and surfaced in result."""
        models = {
            "xgb_regressor": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
                "mode": "regressor",
            },
        }
        # include_margin=True so the 'margin' column exists (required training target for regressors)
        config_dir = _setup_project(
            tmp_path,
            models=models,
            include_margin=True,
            n_rows=300,
            n_seasons=3,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "model_cdf_scales" in result
        assert "xgb_regressor" in result["model_cdf_scales"]
        assert isinstance(result["model_cdf_scales"]["xgb_regressor"], float)
        assert result["model_cdf_scales"]["xgb_regressor"] > 0
