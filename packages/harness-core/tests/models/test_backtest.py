"""Tests for BacktestRunner."""
import numpy as np
import pytest

from harnessml.core.models.backtest import BacktestRunner, BacktestResult


def test_backtest_runner():
    per_fold = {
        1: {"preds": {"m1": np.array([0.9, 0.1]), "m2": np.array([0.8, 0.2])}, "y": np.array([1, 0])},
        2: {"preds": {"m1": np.array([0.7, 0.3]), "m2": np.array([0.6, 0.4])}, "y": np.array([1, 0])},
    }

    runner = BacktestRunner(metrics=["brier", "accuracy"])
    result = runner.run(per_fold)

    assert isinstance(result, BacktestResult)
    assert "brier" in result.pooled_metrics
    assert "accuracy" in result.pooled_metrics
    assert len(result.per_fold_metrics) == 2


def test_backtest_perfect_predictions():
    per_fold = {
        1: {"preds": {"m1": np.array([1.0, 0.0])}, "y": np.array([1, 0])},
    }

    runner = BacktestRunner(metrics=["brier", "accuracy"])
    result = runner.run(per_fold)

    assert result.pooled_metrics["brier"] == pytest.approx(0.0)
    assert result.pooled_metrics["accuracy"] == pytest.approx(1.0)


def test_backtest_coin_flip():
    per_fold = {
        1: {"preds": {"m1": np.array([0.5, 0.5, 0.5, 0.5])}, "y": np.array([1, 0, 1, 0])},
    }

    runner = BacktestRunner(metrics=["brier"])
    result = runner.run(per_fold)
    assert result.pooled_metrics["brier"] == pytest.approx(0.25)


def test_backtest_averages_multiple_models():
    """When multiple models are present, predictions should be averaged."""
    per_fold = {
        1: {
            "preds": {
                "m1": np.array([1.0, 0.0]),
                "m2": np.array([0.5, 0.5]),
            },
            "y": np.array([1, 0]),
        },
    }
    runner = BacktestRunner(metrics=["brier"])
    result = runner.run(per_fold)
    # Averaged preds: [0.75, 0.25], y: [1, 0]
    # Brier: ((0.75-1)^2 + (0.25-0)^2) / 2 = (0.0625 + 0.0625) / 2 = 0.0625
    assert result.pooled_metrics["brier"] == pytest.approx(0.0625)


def test_backtest_all_metrics():
    per_fold = {
        1: {"preds": {"m1": np.array([0.9, 0.1, 0.7, 0.3])}, "y": np.array([1, 0, 1, 0])},
    }

    runner = BacktestRunner(metrics=["brier", "accuracy", "log_loss", "ece"])
    result = runner.run(per_fold)

    assert "brier" in result.pooled_metrics
    assert "accuracy" in result.pooled_metrics
    assert "log_loss" in result.pooled_metrics
    assert "ece" in result.pooled_metrics

    # All metrics should be finite
    for v in result.pooled_metrics.values():
        assert np.isfinite(v)


def test_backtest_per_fold_metrics():
    per_fold = {
        1: {"preds": {"m1": np.array([1.0, 0.0])}, "y": np.array([1, 0])},
        2: {"preds": {"m1": np.array([0.5, 0.5])}, "y": np.array([1, 0])},
    }

    runner = BacktestRunner(metrics=["brier", "accuracy"])
    result = runner.run(per_fold)

    assert result.per_fold_metrics[1]["brier"] == pytest.approx(0.0)
    assert result.per_fold_metrics[1]["accuracy"] == pytest.approx(1.0)
    assert result.per_fold_metrics[2]["brier"] == pytest.approx(0.25)
    assert result.per_fold_metrics[2]["accuracy"] == pytest.approx(0.5)


def test_backtest_pooled_arrays():
    per_fold = {
        1: {"preds": {"m1": np.array([0.9, 0.1])}, "y": np.array([1, 0])},
        2: {"preds": {"m1": np.array([0.7, 0.3])}, "y": np.array([1, 0])},
    }

    runner = BacktestRunner(metrics=["brier"])
    result = runner.run(per_fold)

    assert len(result.pooled_y_true) == 4
    assert len(result.pooled_y_pred) == 4


def test_backtest_supports_extended_metrics():
    """BacktestRunner should accept auc_roc, f1, precision, recall."""
    runner = BacktestRunner(metrics=["brier", "accuracy", "auc_roc", "f1", "precision", "recall"])

    result = runner.run({
        1: {"preds": {"m1": np.array([0.9, 0.1, 0.8, 0.3])}, "y": np.array([1, 0, 1, 0])},
    })

    assert "auc_roc" in result.pooled_metrics
    assert "f1" in result.pooled_metrics
    assert "precision" in result.pooled_metrics
    assert "recall" in result.pooled_metrics
    assert 0.0 <= result.pooled_metrics["auc_roc"] <= 1.0


def test_backtest_auc_alias():
    """'auc' should be accepted as alias for 'auc_roc'."""
    runner = BacktestRunner(metrics=["auc"])
    result = runner.run({
        1: {"preds": {"m1": np.array([0.9, 0.1, 0.8, 0.3])}, "y": np.array([1, 0, 1, 0])},
    })
    assert "auc" in result.pooled_metrics


def test_backtest_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        BacktestRunner(metrics=["nonexistent"])
