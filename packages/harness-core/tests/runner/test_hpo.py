"""Tests for harnessml.core.runner.hpo utilities."""
import pytest


def test_create_pruner_median():
    from harnessml.core.runner.hpo import create_pruner

    pruner = create_pruner("median")
    import optuna

    assert isinstance(pruner, optuna.pruners.MedianPruner)


def test_create_pruner_hyperband():
    from harnessml.core.runner.hpo import create_pruner

    pruner = create_pruner("hyperband")
    import optuna

    assert isinstance(pruner, optuna.pruners.HyperbandPruner)


def test_create_pruner_threshold():
    from harnessml.core.runner.hpo import create_pruner

    pruner = create_pruner("threshold", upper=0.9)
    import optuna

    assert isinstance(pruner, optuna.pruners.ThresholdPruner)


def test_create_pruner_none():
    from harnessml.core.runner.hpo import create_pruner

    pruner = create_pruner("none")
    import optuna

    assert isinstance(pruner, optuna.pruners.NopPruner)


def test_create_pruner_with_kwargs():
    from harnessml.core.runner.hpo import create_pruner

    pruner = create_pruner("median", n_startup_trials=10, n_warmup_steps=5)
    assert pruner is not None


def test_create_pruner_unknown():
    from harnessml.core.runner.hpo import create_pruner

    with pytest.raises(ValueError, match="Unknown pruner strategy"):
        create_pruner("nonexistent")


def test_multi_objective_study():
    from harnessml.core.runner.hpo import create_multi_objective_study

    study = create_multi_objective_study(
        objectives=["brier", "accuracy"],
        directions=["minimize", "maximize"],
    )

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x, 1 - x

    study.optimize(objective, n_trials=10)
    assert len(study.trials) == 10


def test_multi_objective_study_default_directions():
    from harnessml.core.runner.hpo import create_multi_objective_study

    study = create_multi_objective_study(objectives=["a", "b", "c"])

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x, x * 2, x * 3

    study.optimize(objective, n_trials=5)
    assert len(study.trials) == 5


def test_multi_objective_study_direction_length_mismatch():
    from harnessml.core.runner.hpo import create_multi_objective_study

    with pytest.raises(ValueError, match="Length mismatch"):
        create_multi_objective_study(
            objectives=["a", "b"],
            directions=["minimize"],
        )


def test_analyze_importance():
    import optuna
    from harnessml.core.runner.hpo import analyze_importance

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study()

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x + y * 0.01

    study.optimize(objective, n_trials=30)
    importance = analyze_importance(study)
    assert "x" in importance
    assert "y" in importance


def test_get_pareto_front():
    from harnessml.core.runner.hpo import (
        create_multi_objective_study,
        get_pareto_front,
    )

    study = create_multi_objective_study(
        objectives=["a", "b"],
        directions=["minimize", "minimize"],
    )

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x, 1 - x

    study.optimize(objective, n_trials=20)
    front = get_pareto_front(study)
    assert len(front) > 0
    assert "number" in front[0]
    assert "values" in front[0]
    assert "params" in front[0]
