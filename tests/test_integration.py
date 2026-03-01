"""End-to-end integration tests exercising all 7 easyml packages together.

Tests the full pipeline: config -> features -> models -> ensemble -> metrics ->
experiments -> guardrails, using lightweight mock data and sklearn-only models.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

# -- Package 1: easyml-schemas --
from easyml.schemas import (
    ArtifactDecl,
    EnsembleConfig,
    ExperimentResult,
    GuardrailViolation,
    ModelConfig,
    PipelineConfig,
    StageConfig,
    accuracy,
    brier_score,
    calibration_table,
    ece,
    log_loss,
    model_audit,
    model_correlations,
)

# -- Package 2: easyml-config --
from easyml.config import deep_merge, load_config_file, resolve_config

# -- Package 3: easyml-features --
from easyml.features import FeatureBuilder, FeatureRegistry, PairwiseFeatureBuilder

# -- Package 4: easyml-models --
from easyml.models import (
    LeaveOneSeasonOut,
    ModelRegistry,
    StackedEnsemble,
    TrainOrchestrator,
)

# -- Package 5: easyml-data --
from easyml.data import (
    SourceRegistry,
    StageGuard,
    generate_dvc_yaml,
)

# -- Package 6: easyml-experiments --
from easyml.experiments import ExperimentManager, ExperimentError

# -- Package 7: easyml-guardrails --
from easyml.guardrails import (
    AuditLogger,
    Guardrail,
    GuardrailError,
    NamingConventionGuardrail,
    FeatureLeakageGuardrail,
    TemporalOrderingGuardrail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_data(n_per_season: int = 50, seasons: tuple[int, ...] = (2022, 2023)) -> pd.DataFrame:
    """Generate mock team-season data with realistic column names."""
    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            rows.append({
                "entity_id": f"team_{i}",
                "period_id": season,
                "wins": rng.integers(5, 30),
                "losses": rng.integers(2, 20),
                "points_per_game": rng.uniform(55, 90),
                "opp_points_per_game": rng.uniform(55, 90),
                "seed_num": rng.integers(1, 17),
            })
    return pd.DataFrame(rows)


def _make_matchup_data(
    raw_df: pd.DataFrame,
    n_per_season: int = 30,
) -> pd.DataFrame:
    """Generate mock matchup data (binary outcomes) from entity data."""
    rng = np.random.default_rng(123)
    rows = []
    for season in raw_df["period_id"].unique():
        teams = raw_df[raw_df["period_id"] == season]["entity_id"].unique()
        for _ in range(n_per_season):
            a, b = rng.choice(teams, size=2, replace=False)
            rows.append({
                "entity_a_id": a,
                "entity_b_id": b,
                "period_id": season,
                "outcome": int(rng.random() > 0.5),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: Full pipeline integration
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Integration test proving all 7 packages compose together."""

    def test_full_pipeline(self, tmp_path):
        # ---------------------------------------------------------------
        # Step 1: Config (easyml-config)
        # ---------------------------------------------------------------
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        pipeline_yaml = {
            "backtest": {"seasons": [2022, 2023], "min_train_seasons": 1},
        }
        models_yaml = {
            "models": {
                "logreg_seed": {
                    "type": "logistic_regression",
                    "mode": "classifier",
                    "features": ["diff_win_rate", "diff_scoring_margin", "diff_seed_num"],
                    "params": {"C": 1.0, "max_iter": 200},
                    "active": True,
                },
                "elasticnet_basic": {
                    "type": "elastic_net",
                    "mode": "classifier",
                    "features": ["diff_win_rate", "diff_scoring_margin"],
                    "params": {"C": 0.5, "l1_ratio": 0.5, "max_iter": 200},
                    "active": True,
                },
            },
        }
        ensemble_yaml = {
            "ensemble": {
                "method": "stacked",
                "meta_learner_type": "logistic",
                "meta_learner_params": {"C": 1.0},
                "temperature": 1.0,
            },
        }

        (config_dir / "pipeline.yaml").write_text(yaml.dump(pipeline_yaml))
        (config_dir / "models.yaml").write_text(yaml.dump(models_yaml))
        (config_dir / "ensemble.yaml").write_text(yaml.dump(ensemble_yaml))

        config = resolve_config(
            config_dir=config_dir,
            file_map={
                "pipeline": "pipeline.yaml",
                "models": "models.yaml",
                "ensemble": "ensemble.yaml",
            },
        )
        assert "models" in config
        assert "ensemble" in config
        assert config["ensemble"]["method"] == "stacked"

        # Overlay merge
        overlay = {"ensemble": {"temperature": 1.5}}
        config_with_overlay = deep_merge(config, overlay)
        assert config_with_overlay["ensemble"]["temperature"] == 1.5
        # Original unchanged
        assert config["ensemble"]["temperature"] == 1.0

        # ---------------------------------------------------------------
        # Step 2: Features (easyml-features)
        # ---------------------------------------------------------------
        registry = FeatureRegistry()

        @registry.register(
            name="win_rate",
            category="resume",
            level="team",
            output_columns=["win_rate"],
        )
        def compute_win_rate(df, cfg):
            result = df[["entity_id", "period_id"]].copy()
            result["win_rate"] = df["wins"] / (df["wins"] + df["losses"])
            return result

        @registry.register(
            name="scoring_margin",
            category="efficiency",
            level="team",
            output_columns=["scoring_margin"],
        )
        def compute_scoring_margin(df, cfg):
            result = df[["entity_id", "period_id"]].copy()
            result["scoring_margin"] = df["points_per_game"] - df["opp_points_per_game"]
            return result

        @registry.register(
            name="seed_num",
            category="seeding",
            level="team",
            output_columns=["seed_num"],
        )
        def compute_seed(df, cfg):
            return df[["entity_id", "period_id", "seed_num"]].copy()

        assert len(registry) == 3
        assert "win_rate" in registry

        raw_data = _make_raw_data()
        cache_dir = tmp_path / "feature_cache"
        manifest_path = tmp_path / "manifest.json"

        builder = FeatureBuilder(
            registry=registry,
            cache_dir=cache_dir,
            manifest_path=manifest_path,
        )
        features_df = builder.build_all(raw_data, config)

        assert "win_rate" in features_df.columns
        assert "scoring_margin" in features_df.columns
        assert "seed_num" in features_df.columns
        assert len(features_df) == 100  # 50 per season x 2 seasons

        # Incremental caching: rebuild should use cached parquets
        features_df2 = builder.build_all(raw_data, config)
        assert len(features_df2) == 100

        # Pairwise features
        matchup_df = _make_matchup_data(raw_data)
        pairwise = PairwiseFeatureBuilder(methods=["diff"])
        matchup_features = pairwise.build(
            entity_df=features_df,
            matchups=matchup_df,
            feature_columns=["win_rate", "scoring_margin", "seed_num"],
        )
        assert "diff_win_rate" in matchup_features.columns
        assert "diff_scoring_margin" in matchup_features.columns

        # ---------------------------------------------------------------
        # Step 3: Model training (easyml-models)
        # ---------------------------------------------------------------
        model_registry = ModelRegistry.with_defaults()
        assert "logistic_regression" in model_registry
        assert "elastic_net" in model_registry

        # Build feature matrix from matchup features
        feature_cols = ["diff_win_rate", "diff_scoring_margin", "diff_seed_num"]
        X = matchup_features[feature_cols].values.astype(np.float64)
        y = matchup_features["outcome"].values.astype(np.float64)

        # Handle NaN in features (from matchup merge misses)
        nan_mask = np.isnan(X).any(axis=1)
        X = X[~nan_mask]
        y = y[~nan_mask]
        seasons = matchup_features["period_id"].values[~nan_mask]

        model_dir = tmp_path / "models"
        orchestrator = TrainOrchestrator(
            model_registry=model_registry,
            model_configs=config["models"],
            output_dir=model_dir,
            failure_policy="raise",
            use_fingerprint=False,  # skip fingerprint for test
        )

        trained_models = orchestrator.train_all(
            X=X, y=y, feature_columns=feature_cols,
        )
        assert len(trained_models) == 2
        assert "logreg_seed" in trained_models
        assert "elasticnet_basic" in trained_models

        # ---------------------------------------------------------------
        # Step 4: Predictions + Ensemble (easyml-models)
        # ---------------------------------------------------------------
        # The orchestrator subsets features during training using the model's
        # feature list mapped against feature_columns.  We replicate that
        # same subsetting here for prediction.
        predictions = {}
        for model_name, model in trained_models.items():
            model_cfg = config["models"][model_name]
            model_features = model_cfg["features"]
            feat_indices = [
                feature_cols.index(f)
                for f in model_features
                if f in feature_cols
            ]
            X_sub = X[:, feat_indices] if feat_indices else X
            predictions[model_name] = model.predict_proba(X_sub)

        # Verify predictions are in valid range
        for name, preds in predictions.items():
            assert np.all(preds >= 0) and np.all(preds <= 1), f"{name} predictions out of range"

        # Stacked ensemble with LOSO CV
        cv = LeaveOneSeasonOut(min_train_folds=1)
        ensemble = StackedEnsemble(
            method="stacked",
            meta_learner_type="logistic",
            meta_learner_params={"C": 1.0},
        )
        ensemble.fit(predictions, y, cv=cv, fold_ids=seasons)

        ensemble_preds = ensemble.predict(predictions)
        assert len(ensemble_preds) == len(y)
        assert np.all(ensemble_preds >= 0) and np.all(ensemble_preds <= 1)

        # Coefficients
        coefs = ensemble.coefficients()
        assert len(coefs) == 2

        # ---------------------------------------------------------------
        # Step 5: Metrics (easyml-schemas)
        # ---------------------------------------------------------------
        brier = brier_score(y, ensemble_preds)
        acc = accuracy(y, ensemble_preds)
        ll = log_loss(y, ensemble_preds)
        cal_ece = ece(y, ensemble_preds)

        # With random data, metrics should be reasonable but not great
        assert 0.0 <= brier <= 0.5
        assert 0.0 <= acc <= 1.0
        assert ll > 0.0
        assert 0.0 <= cal_ece <= 1.0

        # Calibration table
        cal_table = calibration_table(y, ensemble_preds, n_bins=5)
        assert isinstance(cal_table, list)
        assert all("mean_predicted" in row for row in cal_table)

        # Model correlations
        corrs = model_correlations(predictions)
        assert len(corrs) == 1  # 2 models -> 1 pair
        pair_key = list(corrs.keys())[0]
        assert "|" in pair_key

        # Per-model audit
        audit = model_audit(predictions, y, metrics=["brier", "accuracy"])
        assert "logreg_seed" in audit
        assert "brier" in audit["logreg_seed"]
        assert "accuracy" in audit["logreg_seed"]

        # ---------------------------------------------------------------
        # Step 6: Experiments (easyml-experiments)
        # ---------------------------------------------------------------
        exp_dir = tmp_path / "experiments"
        log_path = tmp_path / "EXPERIMENT_LOG.md"

        mgr = ExperimentManager(
            experiments_dir=exp_dir,
            naming_pattern=r"^exp-\d{3}-.+$",
            log_path=log_path,
        )

        # Create experiment
        exp_path = mgr.create("exp-001-test-integration")
        assert exp_path.exists()
        assert (exp_path / "overlay.yaml").exists()

        # Invalid naming raises
        with pytest.raises(ExperimentError):
            mgr.create("bad_name")

        # Change detection
        overlay_dict = {"models": {"logreg_seed": {"params": {"C": 2.0}}}}
        report = mgr.detect_changes(config, overlay_dict)
        assert "logreg_seed" in report.changed_models

        # Log the experiment
        mgr.log(
            experiment_id="exp-001-test-integration",
            hypothesis="Test that integration works",
            changes="Changed logreg C from 1.0 to 2.0",
            verdict="keep",
            notes=f"Brier: {brier:.4f}, Accuracy: {acc:.4f}",
        )
        assert log_path.exists()
        assert "exp-001-test-integration" in log_path.read_text()

        # ---------------------------------------------------------------
        # Step 7: Guardrails (easyml-guardrails)
        # ---------------------------------------------------------------

        # Naming convention guardrail
        naming_guard = NamingConventionGuardrail(pattern=r"^exp-\d{3}-.+$")
        naming_guard.check({"experiment_id": "exp-001-test"})  # should pass
        with pytest.raises(GuardrailError):
            naming_guard.check({"experiment_id": "bad-name-123"})

        # Feature leakage guardrail (non-overridable)
        leakage_guard = FeatureLeakageGuardrail(denylist=["kp_adj_o", "post_tournament_data"])
        leakage_guard.check({"model_features": ["win_rate", "seed_num"]})  # pass
        with pytest.raises(GuardrailError):
            leakage_guard.check({"model_features": ["win_rate", "kp_adj_o"]})
        # Non-overridable: human_override should NOT suppress
        with pytest.raises(GuardrailError):
            leakage_guard.check(
                {"model_features": ["kp_adj_o"]},
                human_override=True,
            )

        # Temporal ordering guardrail (non-overridable)
        temporal_guard = TemporalOrderingGuardrail()
        temporal_guard.check({"training_seasons": [2020, 2021], "test_season": 2022})
        with pytest.raises(GuardrailError):
            temporal_guard.check({"training_seasons": [2021, 2023], "test_season": 2022})

        # Audit logger
        audit_log = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=audit_log)
        logger.log_invocation(
            tool="train_models",
            args={"gender": "M"},
            guardrails_passed=True,
            result_status="success",
            duration_s=12.5,
        )
        entries = logger.query(tool="train_models")
        assert len(entries) == 1
        assert entries[0]["result_status"] == "success"

    def test_ensemble_average_method(self, tmp_path):
        """Test that averaging ensemble works as an alternative to stacking."""
        rng = np.random.default_rng(99)
        n = 60
        y = rng.integers(0, 2, size=n).astype(float)
        preds = {
            "model_a": rng.uniform(0.2, 0.8, size=n),
            "model_b": rng.uniform(0.2, 0.8, size=n),
        }

        ensemble = StackedEnsemble(method="average")
        ensemble.fit(preds, y)
        result = ensemble.predict(preds)

        expected = (preds["model_a"] + preds["model_b"]) / 2
        np.testing.assert_allclose(result, expected)

    def test_config_variant_resolution(self, tmp_path):
        """Test that config variant resolution loads variant-specific files."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Base config
        (config_dir / "pipeline.yaml").write_text(
            yaml.dump({"gender": "M", "seasons": [2022, 2023]})
        )
        # Women's variant
        (config_dir / "pipeline_w.yaml").write_text(
            yaml.dump({"gender": "W", "seasons": [2022, 2023, 2024]})
        )

        base_cfg = load_config_file(config_dir, "pipeline.yaml")
        assert base_cfg["gender"] == "M"

        variant_cfg = load_config_file(config_dir, "pipeline.yaml", variant="w")
        assert variant_cfg["gender"] == "W"
        assert len(variant_cfg["seasons"]) == 3

    def test_schema_validation(self):
        """Test that Pydantic schemas enforce correct types."""
        mc = ModelConfig(
            name="test_model",
            type="logistic_regression",
            mode="classifier",
            features=["win_rate"],
            params={"C": 1.0},
        )
        assert mc.active is True
        assert mc.n_seeds == 1

        ec = EnsembleConfig(
            method="stacked",
            meta_learner_type="logistic",
        )
        assert ec.temperature == 1.0
        assert ec.clip_floor == 0.0

        # GuardrailViolation serialization
        gv = GuardrailViolation(
            blocked=True,
            rule="test_rule",
            message="Test message",
            source="test",
        )
        d = gv.to_dict()
        assert d["blocked"] is True
        assert d["rule"] == "test_rule"
        assert "override_hint" not in d  # None values excluded

    def test_cv_loso_produces_temporal_folds(self):
        """LOSO CV should produce folds where training < test in time."""
        cv = LeaveOneSeasonOut(min_train_folds=1)
        fold_ids = np.array([2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023])
        folds = cv.split(None, fold_ids=fold_ids)

        assert len(folds) == 3  # 2021, 2022, 2023 (2020 has no prior)
        for fold in folds:
            train_seasons = set(fold_ids[fold.train_idx])
            test_season = fold_ids[fold.test_idx[0]]
            assert all(s < test_season for s in train_seasons)


# ---------------------------------------------------------------------------
# Test: DVC config generation (Task 8.2)
# ---------------------------------------------------------------------------

class TestDVCGeneration:
    """Tests DVC YAML generation from typed artifact declarations."""

    def test_dvc_generation_from_artifacts(self):
        """Validate DVC config generation from StageConfig + ArtifactDecl."""
        stages = {
            "ingest": StageConfig(
                script="pipelines/ingest.py",
                consumes=[],
                produces=[
                    ArtifactDecl(name="raw_data", type="data", path="data/processed/"),
                ],
            ),
            "featurize": StageConfig(
                script="pipelines/featurize.py",
                consumes=["raw_data"],
                produces=[
                    ArtifactDecl(name="features", type="features", path="data/features/"),
                ],
            ),
            "train": StageConfig(
                script="pipelines/train.py",
                consumes=["features"],
                produces=[
                    ArtifactDecl(name="models", type="model", path="models/"),
                ],
            ),
        }

        dvc_yaml = generate_dvc_yaml(stages)

        # Top-level structure
        assert "stages" in dvc_yaml

        # All three stages present
        assert "ingest" in dvc_yaml["stages"]
        assert "featurize" in dvc_yaml["stages"]
        assert "train" in dvc_yaml["stages"]

        # Ingest has no consumed deps (only its script)
        ingest_deps = dvc_yaml["stages"]["ingest"]["deps"]
        assert "pipelines/ingest.py" in ingest_deps
        assert len(ingest_deps) == 1  # just the script

        # Featurize depends on ingest output
        feat_deps = dvc_yaml["stages"]["featurize"]["deps"]
        assert "pipelines/featurize.py" in feat_deps
        assert "data/processed/" in feat_deps

        # Train depends on featurize output
        train_deps = dvc_yaml["stages"]["train"]["deps"]
        assert "data/features/" in train_deps

        # Outputs match artifact paths
        assert dvc_yaml["stages"]["ingest"]["outs"] == ["data/processed/"]
        assert dvc_yaml["stages"]["featurize"]["outs"] == ["data/features/"]
        assert dvc_yaml["stages"]["train"]["outs"] == ["models/"]

        # Commands use uv run
        for stage_name, stage_def in dvc_yaml["stages"].items():
            assert stage_def["cmd"].startswith("uv run python ")

    def test_dvc_missing_artifact_raises(self):
        """Consuming a non-existent artifact should raise KeyError."""
        stages = {
            "featurize": StageConfig(
                script="pipelines/featurize.py",
                consumes=["nonexistent_data"],
                produces=[
                    ArtifactDecl(name="features", type="features", path="data/features/"),
                ],
            ),
        }

        with pytest.raises(KeyError, match="nonexistent_data"):
            generate_dvc_yaml(stages)

    def test_dvc_multi_output_stage(self):
        """A stage can produce multiple artifacts."""
        stages = {
            "ingest": StageConfig(
                script="pipelines/ingest.py",
                consumes=[],
                produces=[
                    ArtifactDecl(name="games", type="data", path="data/games.parquet"),
                    ArtifactDecl(name="teams", type="data", path="data/teams.parquet"),
                ],
            ),
            "featurize": StageConfig(
                script="pipelines/featurize.py",
                consumes=["games", "teams"],
                produces=[
                    ArtifactDecl(name="features", type="features", path="data/features/"),
                ],
            ),
        }

        dvc_yaml = generate_dvc_yaml(stages)

        assert len(dvc_yaml["stages"]["ingest"]["outs"]) == 2
        feat_deps = dvc_yaml["stages"]["featurize"]["deps"]
        assert "data/games.parquet" in feat_deps
        assert "data/teams.parquet" in feat_deps


# ---------------------------------------------------------------------------
# Test: Source registry integration
# ---------------------------------------------------------------------------

class TestSourceRegistryIntegration:
    """Validate SourceRegistry works with real decorator patterns."""

    def test_register_and_run_source(self, tmp_path):
        """Register a source, run it, verify output."""
        source_reg = SourceRegistry()

        @source_reg.register(
            name="mock_kaggle",
            category="external",
            outputs=[str(tmp_path / "output.csv")],
            temporal_safety="pre_tournament",
        )
        def fetch_kaggle(output_dir, config):
            pd.DataFrame({"team": ["Duke", "UNC"], "wins": [25, 22]}).to_csv(
                output_dir / "output.csv", index=False,
            )

        assert "mock_kaggle" in source_reg
        meta = source_reg.get_metadata("mock_kaggle")
        assert meta.temporal_safety == "pre_tournament"

        source_reg.run("mock_kaggle", tmp_path, {})
        assert (tmp_path / "output.csv").exists()


# ---------------------------------------------------------------------------
# Test: Experiment do-not-retry integration
# ---------------------------------------------------------------------------

class TestExperimentDoNotRetry:
    """Verify DNR patterns block experiments across packages."""

    def test_do_not_retry_persistence(self, tmp_path):
        """DNR patterns should persist to disk and reload."""
        exp_dir = tmp_path / "experiments"
        dnr_path = tmp_path / "do_not_retry.json"
        log_path = tmp_path / "EXPERIMENT_LOG.md"

        mgr = ExperimentManager(
            experiments_dir=exp_dir,
            log_path=log_path,
            do_not_retry_path=dnr_path,
        )

        mgr.add_do_not_retry(
            pattern="temperature scaling",
            reference="EXP-002",
            reason="T>1.0 hurts monotonically",
        )
        assert dnr_path.exists()

        # Reload from disk
        mgr2 = ExperimentManager(
            experiments_dir=exp_dir,
            log_path=log_path,
            do_not_retry_path=dnr_path,
        )
        with pytest.raises(ExperimentError):
            mgr2.check_do_not_retry("Try temperature scaling T=1.5")
