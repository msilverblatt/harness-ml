"""Hyperparameter optimization utilities wrapping Optuna."""
from __future__ import annotations

from typing import Any


def create_pruner(strategy: str = "median", **kwargs) -> Any:
    """Create an Optuna pruner by strategy name.

    Parameters
    ----------
    strategy : str
        One of "median", "hyperband", "threshold", "none".
    **kwargs
        Forwarded to the pruner constructor.

    Returns
    -------
    optuna.pruners.BasePruner
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for HPO. Install with: pip install optuna")

    pruners = {
        "median": optuna.pruners.MedianPruner,
        "hyperband": optuna.pruners.HyperbandPruner,
        "threshold": optuna.pruners.ThresholdPruner,
        "none": optuna.pruners.NopPruner,
    }
    if strategy not in pruners:
        raise ValueError(
            f"Unknown pruner strategy: {strategy!r}. "
            f"Must be one of: {sorted(pruners)}"
        )
    return pruners[strategy](**kwargs)


def create_multi_objective_study(
    objectives: list[str],
    directions: list[str] | None = None,
    pruner_strategy: str = "median",
    study_name: str | None = None,
    **kwargs,
) -> Any:
    """Create a multi-objective Optuna study.

    Parameters
    ----------
    objectives : list[str]
        Names of the objectives (used for documentation; length determines
        the number of directions).
    directions : list[str] | None
        List of "minimize" or "maximize" per objective.  Defaults to
        ``["minimize"] * len(objectives)``.
    pruner_strategy : str
        Pruner strategy name passed to :func:`create_pruner`.
    study_name : str | None
        Optional Optuna study name.
    **kwargs
        Forwarded to ``optuna.create_study``.

    Returns
    -------
    optuna.Study
    """
    import optuna  # noqa: F811

    if directions is None:
        directions = ["minimize"] * len(objectives)
    if len(directions) != len(objectives):
        raise ValueError(
            f"Length mismatch: {len(objectives)} objectives but "
            f"{len(directions)} directions."
        )
    pruner = create_pruner(pruner_strategy)
    study = optuna.create_study(
        directions=directions,
        pruner=pruner,
        study_name=study_name,
        **kwargs,
    )
    return study


def analyze_importance(study: Any) -> dict[str, float]:
    """Analyze hyperparameter importance from a completed study.

    Parameters
    ----------
    study : optuna.Study
        A study with completed trials.

    Returns
    -------
    dict[str, float]
        Parameter name to importance score mapping.
    """
    import optuna  # noqa: F811

    return dict(optuna.importance.get_param_importances(study))


def get_pareto_front(study: Any) -> list[dict]:
    """Extract Pareto-optimal trials from a multi-objective study.

    Parameters
    ----------
    study : optuna.Study
        A multi-objective study.

    Returns
    -------
    list[dict]
        Each dict has keys ``number``, ``values``, ``params``.
    """
    trials = study.best_trials
    return [
        {"number": t.number, "values": t.values, "params": t.params}
        for t in trials
    ]
