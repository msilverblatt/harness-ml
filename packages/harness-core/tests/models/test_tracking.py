"""Tests for TrackingCallback."""

from harnessml.core.models.tracking import TrackingCallback


class MockTracker(TrackingCallback):
    def __init__(self):
        self.events = []

    def on_model_trained(self, model_name, metrics, duration_s):
        self.events.append(("trained", model_name))

    def on_backtest_complete(self, metrics):
        self.events.append(("backtest", metrics))

    def on_experiment_logged(self, experiment_id, verdict, metrics):
        self.events.append(("experiment", experiment_id, verdict))


def test_tracking_callback():
    tracker = MockTracker()
    tracker.on_model_trained("xgb_core", {"brier": 0.18}, 5.2)
    tracker.on_backtest_complete({"brier": 0.175})
    assert len(tracker.events) == 2


def test_tracking_callback_experiment_logged():
    tracker = MockTracker()
    tracker.on_experiment_logged("exp-001", "keep", {"brier": 0.17})
    assert tracker.events[0] == ("experiment", "exp-001", "keep")


def test_default_no_ops():
    """Base class methods should be callable without error (no-ops)."""

    class MinimalTracker(TrackingCallback):
        pass

    tracker = MinimalTracker()
    # These should not raise
    tracker.on_model_trained("model", {}, 1.0)
    tracker.on_backtest_complete({})
    tracker.on_experiment_logged("exp", "revert", {})
