"""Tests for EnsembleDef sub-config decomposition."""
from harnessml.core.runner.schema import (
    CalibrationConfig,
    EnsembleDef,
    LogitAdjustment,
    PostProcessingConfig,
)


class TestCalibrationConfig:
    def test_defaults(self):
        c = CalibrationConfig()
        assert c.method == "spline"
        assert c.spline_prob_max == 0.985
        assert c.spline_n_bins == 20

    def test_custom(self):
        c = CalibrationConfig(method="isotonic", spline_prob_max=0.95)
        assert c.method == "isotonic"
        assert c.spline_prob_max == 0.95


class TestPostProcessingConfig:
    def test_defaults(self):
        pp = PostProcessingConfig()
        assert pp.temperature == 1.0
        assert pp.clip_floor == 0.0
        assert pp.prior_compression == 0.0
        assert pp.prior_feature is None
        assert pp.logit_adjustments == []


class TestEnsembleDefFlatKwargs:
    """Flat kwargs should route to sub-configs."""

    def test_calibration_string(self):
        e = EnsembleDef(method="stacked", calibration="isotonic")
        assert e.calibration == "isotonic"
        assert e.calibration_config.method == "isotonic"

    def test_spline_params(self):
        e = EnsembleDef(method="stacked", spline_prob_max=0.9, spline_n_bins=15)
        assert e.spline_prob_max == 0.9
        assert e.spline_n_bins == 15
        assert e.calibration_config.spline_prob_max == 0.9
        assert e.calibration_config.spline_n_bins == 15

    def test_post_processing_params(self):
        e = EnsembleDef(method="average", temperature=0.8, clip_floor=0.02)
        assert e.temperature == 0.8
        assert e.clip_floor == 0.02
        assert e.post_processing.temperature == 0.8

    def test_prior_compression(self):
        e = EnsembleDef(method="stacked", prior_compression=0.1, prior_compression_threshold=5)
        assert e.prior_compression == 0.1
        assert e.prior_compression_threshold == 5

    def test_logit_adjustments(self):
        adj = [{"columns": ["a", "b"], "strength": 0.2}]
        e = EnsembleDef(method="stacked", logit_adjustments=adj)
        assert len(e.logit_adjustments) == 1
        assert e.logit_adjustments[0].strength == 0.2

    def test_defaults(self):
        e = EnsembleDef(method="stacked")
        assert e.calibration == "spline"
        assert e.temperature == 1.0
        assert e.clip_floor == 0.0

    def test_nested_calibration_config(self):
        e = EnsembleDef(
            method="stacked",
            calibration_config={"method": "platt", "spline_n_bins": 10},
        )
        assert e.calibration == "platt"
        assert e.spline_n_bins == 10


class TestEnsembleDefPropertySetters:
    def test_set_calibration(self):
        e = EnsembleDef(method="stacked")
        e.calibration = "isotonic"
        assert e.calibration_config.method == "isotonic"

    def test_set_temperature(self):
        e = EnsembleDef(method="stacked")
        e.temperature = 0.5
        assert e.post_processing.temperature == 0.5
