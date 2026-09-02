import numpy as np
import pytest

from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.runners.postprocessing import apply_postprocessing


def make_predictions(seed=0, n=100):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.05, 0.95, n)


class TestNoOp:
    def test_no_op_config_returns_same_predictions(self):
        preds = make_predictions()
        config = EnsembleConfig(temperature=1.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        np.testing.assert_allclose(result, preds)

    def test_does_not_modify_input_array(self):
        preds = make_predictions()
        original = preds.copy()
        config = EnsembleConfig(temperature=1.0, clip_floor=None)
        apply_postprocessing(preds, config)
        np.testing.assert_array_equal(preds, original)


class TestTemperatureScaling:
    def test_temperature_gt_1_pushes_toward_0_5(self):
        """High temperature should move predictions closer to 0.5."""
        preds = np.array([0.1, 0.3, 0.7, 0.9])
        config = EnsembleConfig(temperature=2.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        # All results should be closer to 0.5 than originals
        assert np.all(np.abs(result - 0.5) < np.abs(preds - 0.5))

    def test_temperature_lt_1_makes_predictions_more_extreme(self):
        """Low temperature should push predictions further from 0.5."""
        preds = np.array([0.3, 0.4, 0.6, 0.7])
        config = EnsembleConfig(temperature=0.5, clip_floor=None)
        result = apply_postprocessing(preds, config)
        assert np.all(np.abs(result - 0.5) > np.abs(preds - 0.5))

    def test_temperature_2_matches_formula_on_known_input(self):
        """Verify temperature=2.0 on a specific input matches the formula."""
        p = 0.8
        preds = np.array([p])
        config = EnsembleConfig(temperature=2.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        # Manual calculation
        logit = np.log(p / (1 - p))
        scaled_logit = logit / 2.0
        expected = 1.0 / (1.0 + np.exp(-scaled_logit))
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_temperature_1_is_identity(self):
        preds = make_predictions()
        config = EnsembleConfig(temperature=1.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        np.testing.assert_allclose(result, preds)


class TestClipFloor:
    def test_clip_floor_enforces_minimum(self):
        preds = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
        floor = 0.05
        config = EnsembleConfig(temperature=1.0, clip_floor=floor)
        result = apply_postprocessing(preds, config)
        assert np.all(result >= floor)
        assert np.all(result <= 1.0 - floor)

    def test_clip_floor_enforces_maximum(self):
        preds = np.array([0.95, 0.98, 1.0])
        floor = 0.1
        config = EnsembleConfig(temperature=1.0, clip_floor=floor)
        result = apply_postprocessing(preds, config)
        assert np.all(result <= 1.0 - floor)

    def test_clip_floor_none_does_not_clip(self):
        preds = np.array([0.01, 0.99])
        config = EnsembleConfig(temperature=1.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        np.testing.assert_allclose(result, preds)


class TestAlwaysInUnitInterval:
    def test_output_always_in_0_1(self):
        rng = np.random.RandomState(1)
        preds = rng.uniform(0, 1, 500)
        configs = [
            EnsembleConfig(temperature=0.1, clip_floor=None),
            EnsembleConfig(temperature=5.0, clip_floor=None),
            EnsembleConfig(temperature=1.0, clip_floor=0.05),
            EnsembleConfig(temperature=2.0, clip_floor=0.1),
        ]
        for config in configs:
            result = apply_postprocessing(preds, config)
            assert np.all(result >= 0.0), f"Below 0 for {config}"
            assert np.all(result <= 1.0), f"Above 1 for {config}"

    def test_edge_case_near_0_and_1(self):
        preds = np.array([1e-10, 1.0 - 1e-10])
        config = EnsembleConfig(temperature=1.0, clip_floor=None)
        result = apply_postprocessing(preds, config)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
