import pytest
from harnessml.core.schemas.contracts import (
    FeatureMeta, TemporalFilter, ModelConfig, EnsembleConfig,
    StageConfig, ArtifactDecl, PipelineConfig, RunManifest,
    ExperimentResult, GuardrailViolation, Fold, SourceMeta,
)

def test_feature_meta_basic():
    fm = FeatureMeta(
        name="scoring_margin",
        category="offense",
        level="entity",
        output_columns=["scoring_margin", "scoring_margin_std"],
    )
    assert fm.name == "scoring_margin"
    assert fm.nan_strategy == "median"  # default

def test_feature_meta_temporal_filter():
    fm = FeatureMeta(
        name="test",
        category="test",
        level="entity",
        output_columns=["x"],
        temporal_filter=TemporalFilter(exclude_event_types=["tournament"]),
    )
    assert fm.temporal_filter.exclude_event_types == ["tournament"]

def test_feature_meta_tainted_columns():
    fm = FeatureMeta(
        name="test",
        category="test",
        level="entity",
        output_columns=["x"],
        tainted_columns=["kp_adj_o", "kp_sos"],
    )
    assert "kp_adj_o" in fm.tainted_columns

def test_model_config():
    mc = ModelConfig(
        name="xgb_core",
        type="xgboost",
        mode="classifier",
        features=["diff_prior", "diff_adj_em"],
        params={"max_depth": 3, "learning_rate": 0.05},
    )
    assert mc.type == "xgboost"
    assert mc.mode == "classifier"

def test_model_config_defaults():
    mc = ModelConfig(
        name="test",
        type="xgboost",
        mode="classifier",
        features=["x"],
        params={},
    )
    assert mc.active is True  # default
    assert mc.pre_calibrate is None  # default
    assert mc.n_seeds == 1  # default

def test_ensemble_config():
    ec = EnsembleConfig(
        method="stacked",
        meta_learner_params={"C": 2.5},
    )
    assert ec.method == "stacked"

def test_artifact_decl():
    ad = ArtifactDecl(name="base_features", type="features", path="data/features/")
    assert ad.type == "features"

def test_stage_config():
    sc = StageConfig(
        script="pipelines/featurize.py",
        consumes=["raw_data"],
        produces=[ArtifactDecl(name="features", type="features", path="data/features/")],
    )
    assert sc.consumes == ["raw_data"]

def test_run_manifest():
    rm = RunManifest(
        run_id="20260228_143021",
        created_at="2026-02-28T14:30:21Z",
        labels=["current"],
        stage="train",
    )
    assert "current" in rm.labels

def test_experiment_result():
    er = ExperimentResult(
        experiment_id="exp-055-test",
        baseline_metrics={"brier": 0.1752},
        result_metrics={"brier": 0.1748},
        delta={"brier": -0.0004},
        verdict="keep",
        models_trained=["xgb_core"],
    )
    assert er.delta["brier"] == -0.0004

def test_guardrail_violation():
    gv = GuardrailViolation(
        blocked=True,
        rule="sanity_check",
        message="Sanity check failed",
        source="scripts/sanity_check.py",
        override_hint="Re-call with human_override=true",
    )
    assert gv.blocked is True
    assert gv.to_dict()["rule"] == "sanity_check"

def test_guardrail_violation_non_overridable():
    gv = GuardrailViolation(
        blocked=True,
        rule="feature_leakage",
        message="Leaky column detected",
        source="guardrails",
        overridable=False,
    )
    assert gv.overridable is False

def test_fold():
    import numpy as np
    f = Fold(
        fold_id=2018,
        train_idx=np.array([0, 1, 2]),
        test_idx=np.array([3, 4]),
        calibration_idx=None,
    )
    assert f.fold_id == 2018
    assert len(f.train_idx) == 3


def test_source_meta():
    sm = SourceMeta(
        name="kenpom_archive",
        category="external",
        outputs=["data/external/kenpom/"],
        temporal_safety="pre_event",
        leakage_notes="Archive endpoint returns ratings as of Selection Sunday",
    )
    assert sm.temporal_safety == "pre_event"


def test_source_meta_defaults():
    sm = SourceMeta(
        name="test",
        category="internal",
        outputs=["data/test/"],
        temporal_safety="unknown",
    )
    assert sm.leakage_notes == ""
    assert sm.freshness_check is None
