"""Bayesian exploration engine for automated experiment search.

Lets the agent define a typed search space (continuous, integer, categorical,
subset axes covering hyperparams, features, models, and ensemble settings) and
runs Optuna-driven trials against PipelineRunner.backtest(), sharing a
PredictionCache across trials so unchanged models are never retrained.

Typical usage via MCP::

    manage_experiments(
        action="explore",
        search_space={
            "axes": [
                {"key": "models.xgb.params.learning_rate",
                 "type": "continuous", "low": 0.01, "high": 0.3},
                {"key": "models.xgb.params.max_depth",
                 "type": "integer", "low": 3, "high": 10},
            ],
            "budget": 20,
        },
    )
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator

from easyml.runner.sweep import set_nested_key

logger = logging.getLogger(__name__)

# Metrics where lower is better — used for Optuna direction
_LOWER_IS_BETTER = {"brier", "brier_score", "ece", "log_loss"}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class AxisDef(BaseModel):
    """A single dimension in the exploration search space."""

    key: str
    type: Literal["continuous", "integer", "categorical", "subset"]

    # continuous / integer
    low: float | None = None
    high: float | None = None
    log: bool = False

    # categorical
    values: list[Any] | None = None

    # subset
    candidates: list[str] | None = None
    min_size: int = 1

    @model_validator(mode="after")
    def _validate_axis(self) -> AxisDef:
        if self.type in ("continuous", "integer"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Axis '{self.key}' ({self.type}) requires 'low' and 'high'."
                )
        elif self.type == "categorical":
            if not self.values or len(self.values) < 2:
                raise ValueError(
                    f"Axis '{self.key}' (categorical) requires 'values' with at least 2 items."
                )
        elif self.type == "subset":
            if not self.candidates or len(self.candidates) < 1:
                raise ValueError(
                    f"Axis '{self.key}' (subset) requires 'candidates' with at least 1 item."
                )
        return self


class ExplorationSpace(BaseModel):
    """Full search space definition for an exploration run."""

    axes: list[AxisDef]
    budget: int = 20
    primary_metric: str = "brier"
    baseline: bool = True
    description: str = ""


# ---------------------------------------------------------------------------
# Subset key routing
# ---------------------------------------------------------------------------

# Semantic subset keys that get special overlay translation
_SUBSET_ROUTES: dict[str, str] = {
    "models.active": "models.{item}.active",
    "features.include": "data.feature_defs.{item}.enabled",
}


def _is_category_subset(key: str) -> bool:
    """Check if this is a features.categories subset axis."""
    return key == "features.categories"


def _apply_subset_to_overlay(
    overlay: dict,
    axis: AxisDef,
    active_items: list[str],
) -> None:
    """Apply subset selections to the overlay dict."""
    if axis.key in _SUBSET_ROUTES:
        template = _SUBSET_ROUTES[axis.key]
        for candidate in axis.candidates:
            path = template.format(item=candidate)
            set_nested_key(overlay, path, candidate in active_items)
    elif _is_category_subset(axis.key):
        # Category-level toggling: enable/disable all features in each category
        # This is handled at the overlay level by setting a category flag
        for candidate in axis.candidates:
            set_nested_key(
                overlay,
                f"data.feature_categories.{candidate}.enabled",
                candidate in active_items,
            )
    else:
        # Generic subset — store the active list at the dot-path
        set_nested_key(overlay, axis.key, active_items)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _build_overlay(
    trial: Any,  # optuna.trial.Trial
    axes: list[AxisDef],
) -> dict:
    """Suggest values from Optuna and build a config overlay dict."""
    overlay: dict = {}

    for axis in axes:
        if axis.type == "continuous":
            val = trial.suggest_float(
                axis.key, axis.low, axis.high, log=axis.log,
            )
            set_nested_key(overlay, axis.key, val)

        elif axis.type == "integer":
            val = trial.suggest_int(axis.key, int(axis.low), int(axis.high))
            set_nested_key(overlay, axis.key, val)

        elif axis.type == "categorical":
            val = trial.suggest_categorical(axis.key, axis.values)
            set_nested_key(overlay, axis.key, val)

        elif axis.type == "subset":
            # Suggest a boolean per candidate, enforce min_size
            active: list[str] = []
            for candidate in axis.candidates:
                include = trial.suggest_categorical(
                    f"{axis.key}__{candidate}", [True, False],
                )
                if include:
                    active.append(candidate)

            # If below min_size, prune this trial
            if len(active) < axis.min_size:
                import optuna
                raise optuna.TrialPruned(
                    f"Subset '{axis.key}' has {len(active)} items, "
                    f"min_size is {axis.min_size}"
                )

            _apply_subset_to_overlay(overlay, axis, active)

    return overlay


def _make_objective(
    project_dir: Path,
    config_dir: Path,
    space: ExplorationSpace,
    pred_cache: Any,  # PredictionCache
    trials_dir: Path,
    trial_results: list[dict],
) -> Any:  # Callable[[optuna.Trial], float]
    """Build the Optuna objective function."""

    def objective(trial: Any) -> float:
        from easyml.runner.pipeline import PipelineRunner

        overlay = _build_overlay(trial, space.axes)

        # Save trial overlay
        trial_dir = trials_dir / f"trial-{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "overlay.yaml").write_text(
            yaml.dump(overlay, default_flow_style=False, sort_keys=False),
        )

        try:
            runner = PipelineRunner(
                project_dir=project_dir,
                config_dir=str(config_dir),
                overlay=overlay,
                prediction_cache=pred_cache,
            )
            runner.load()
            bt_result = runner.backtest()
        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            (trial_dir / "error.txt").write_text(str(exc))
            import optuna
            raise optuna.TrialPruned(f"Backtest failed: {exc}")

        metrics = bt_result.get("metrics", {})

        # Save trial results
        trial_data = {
            "trial": trial.number,
            "overlay": overlay,
            "metrics": metrics,
            "cache_stats": getattr(runner, "cache_stats", {}),
        }
        (trial_dir / "results.json").write_text(
            json.dumps(trial_data, indent=2, default=str),
        )
        trial_results.append(trial_data)

        # Store all metrics as user attributes for the report
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("overlay", overlay)

        # Return the primary metric value
        primary = space.primary_metric
        # Try exact match, then with _score suffix
        value = metrics.get(primary) or metrics.get(f"{primary}_score")
        if value is None:
            import optuna
            raise optuna.TrialPruned(
                f"Primary metric '{primary}' not found in results. "
                f"Available: {sorted(metrics.keys())}"
            )
        return float(value)

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_exploration(
    project_dir: Path,
    search_space: dict | ExplorationSpace,
    config_dir: Path | None = None,
) -> dict:
    """Run a Bayesian exploration over the search space.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    search_space : dict | ExplorationSpace
        Search space definition (parsed from MCP JSON or already validated).
    config_dir : Path | None
        Config directory override. Defaults to ``project_dir / "config"``.

    Returns
    -------
    dict
        Keys: ``exploration_id``, ``n_trials``, ``best``, ``all_trials``,
        ``baseline_metrics``, ``parameter_importance``, ``report``.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for exploration. "
            "Install with: pip install 'easyml-runner[explore]'"
        )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    project_dir = Path(project_dir)
    if config_dir is None:
        config_dir = project_dir / "config"
    config_dir = Path(config_dir)

    # Parse search space
    if isinstance(search_space, dict):
        space = ExplorationSpace(**search_space)
    else:
        space = search_space

    # Create exploration directory
    expl_dir = _next_exploration_dir(project_dir)
    exploration_id = expl_dir.name
    expl_dir.mkdir(parents=True, exist_ok=True)

    # Save the search space
    (expl_dir / "space.yaml").write_text(
        yaml.dump(
            space.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
        ),
    )

    # Shared prediction cache
    from easyml.runner.prediction_cache import PredictionCache
    cache_dir = expl_dir / ".cache"
    pred_cache = PredictionCache(cache_dir)

    # Run baseline if requested
    baseline_metrics: dict = {}
    if space.baseline:
        logger.info("Running baseline backtest...")
        baseline_metrics = _run_baseline(
            project_dir, config_dir, expl_dir,
        )

    # Create Optuna study
    direction = (
        "minimize"
        if space.primary_metric in _LOWER_IS_BETTER
        or f"{space.primary_metric}_score" in _LOWER_IS_BETTER
        else "maximize"
    )

    study = optuna.create_study(
        study_name=exploration_id,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run trials
    trials_dir = expl_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_results: list[dict] = []

    objective = _make_objective(
        project_dir, config_dir, space, pred_cache,
        trials_dir, trial_results,
    )

    study.optimize(
        objective,
        n_trials=space.budget,
        catch=(Exception,),
    )

    # Save study
    _save_study_json(expl_dir / "study.json", study)

    # Save best overlay
    best_overlay = {}
    if study.best_trial:
        best_overlay = study.best_trial.user_attrs.get("overlay", {})
        (expl_dir / "best_overlay.yaml").write_text(
            yaml.dump(best_overlay, default_flow_style=False, sort_keys=False),
        )

    # Compute parameter importance
    param_importance: dict[str, float] = {}
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 4:
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            logger.warning("Could not compute parameter importance")

    # Build report
    report = format_exploration_report(
        exploration_id=exploration_id,
        space=space,
        study=study,
        baseline_metrics=baseline_metrics,
        param_importance=param_importance,
        trial_results=trial_results,
    )

    (expl_dir / "report.md").write_text(report)

    # Collect cache stats
    total_hits = sum(
        r.get("cache_stats", {}).get("hits", 0) for r in trial_results
    )
    total_misses = sum(
        r.get("cache_stats", {}).get("misses", 0) for r in trial_results
    )

    return {
        "exploration_id": exploration_id,
        "n_trials": len(completed),
        "n_pruned": len(study.trials) - len(completed),
        "best": {
            "trial": study.best_trial.number if study.best_trial else None,
            "value": study.best_value if study.best_trial else None,
            "overlay": best_overlay,
            "metrics": study.best_trial.user_attrs if study.best_trial else {},
        },
        "baseline_metrics": baseline_metrics,
        "parameter_importance": param_importance,
        "cache_stats": {"hits": total_hits, "misses": total_misses},
        "report": report,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_exploration_dir(project_dir: Path) -> Path:
    """Auto-number the next exploration directory."""
    experiments_dir = project_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^expl-(\d{3})$")
    max_num = 0
    for entry in experiments_dir.iterdir():
        m = pattern.match(entry.name)
        if m:
            max_num = max(max_num, int(m.group(1)))

    return experiments_dir / f"expl-{max_num + 1:03d}"


def _run_baseline(
    project_dir: Path,
    config_dir: Path,
    expl_dir: Path,
) -> dict:
    """Run a baseline backtest (no overlay) and save results."""
    from easyml.runner.pipeline import PipelineRunner

    baseline_dir = expl_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    try:
        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        metrics = result.get("metrics", {})

        (baseline_dir / "results.json").write_text(
            json.dumps({"metrics": metrics}, indent=2, default=str),
        )
        return metrics
    except Exception as exc:
        logger.warning("Baseline backtest failed: %s", exc)
        (baseline_dir / "error.txt").write_text(str(exc))
        return {}


def _save_study_json(path: Path, study: Any) -> None:
    """Serialize the Optuna study to JSON for later analysis."""
    import optuna

    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "user_attrs": {
                k: v for k, v in t.user_attrs.items()
                if k != "overlay"  # overlays saved separately per trial
            },
        })

    data = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_value": study.best_value if study.best_trial else None,
        "trials": trials_data,
    }
    path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_exploration_report(
    exploration_id: str,
    space: ExplorationSpace,
    study: Any,  # optuna.Study
    baseline_metrics: dict,
    param_importance: dict[str, float],
    trial_results: list[dict],
) -> str:
    """Build the full markdown exploration report."""
    import optuna

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = len(study.trials) - len(completed)

    lines: list[str] = [f"## Exploration Report: `{exploration_id}`\n"]

    if space.description:
        lines.append(f"_{space.description}_\n")

    # --- Summary ---
    if study.best_trial:
        best = study.best_trial
        best_val = study.best_value
        delta_str = ""
        if baseline_metrics:
            base_val = (
                baseline_metrics.get(space.primary_metric)
                or baseline_metrics.get(f"{space.primary_metric}_score")
            )
            if base_val is not None:
                delta = best_val - base_val
                delta_str = f", baseline={base_val:.4f}, delta={delta:+.4f}"

        lines.append(
            f"**Best trial**: #{best.number} of {len(study.trials)} "
            f"({space.primary_metric}={best_val:.4f}{delta_str})"
        )
    else:
        lines.append("**No completed trials.**")

    # Cache stats
    total_hits = sum(
        r.get("cache_stats", {}).get("hits", 0) for r in trial_results
    )
    total_total = total_hits + sum(
        r.get("cache_stats", {}).get("misses", 0) for r in trial_results
    )
    if total_total > 0:
        pct = total_hits / total_total * 100
        lines.append(f"**Cache reuse**: {pct:.0f}% of model-season predictions were cache hits")

    lines.append(f"**Budget used**: {len(completed)}/{space.budget} trials completed, {pruned} pruned\n")

    # --- Best config table ---
    if study.best_trial:
        lines.append("### Best Configuration\n")
        lines.append("| Axis | Value |")
        lines.append("|------|-------|")
        for axis in space.axes:
            if axis.type == "subset":
                # Reconstruct active items from params
                active = [
                    c for c in axis.candidates
                    if study.best_trial.params.get(f"{axis.key}__{c}")
                ]
                lines.append(f"| {axis.key} | {', '.join(active)} |")
            else:
                val = study.best_trial.params.get(axis.key, "?")
                if isinstance(val, float):
                    lines.append(f"| {axis.key} | {val:.6g} |")
                else:
                    lines.append(f"| {axis.key} | {val} |")
        lines.append("")

    # --- All trials table ---
    if completed:
        lines.append(f"### All Trials (ranked by {space.primary_metric})\n")

        # Collect all metric keys across completed trials
        all_metric_keys: set[str] = set()
        for t in completed:
            for k in t.user_attrs:
                if k != "overlay" and isinstance(t.user_attrs[k], (int, float)):
                    all_metric_keys.add(k)
        metric_cols = sorted(all_metric_keys)

        # Build short axis labels
        axis_labels = []
        for axis in space.axes:
            short = axis.key.split(".")[-1]
            axis_labels.append(short)

        header_parts = ["#"] + metric_cols + axis_labels
        lines.append("| " + " | ".join(header_parts) + " |")
        lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

        # Sort by primary metric
        reverse = space.primary_metric not in _LOWER_IS_BETTER
        ranked = sorted(
            completed,
            key=lambda t: t.value if t.value is not None else float("inf"),
            reverse=reverse,
        )

        for t in ranked:
            row = [str(t.number)]

            # Metrics
            for mk in metric_cols:
                val = t.user_attrs.get(mk)
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                elif val is not None:
                    row.append(str(val))
                else:
                    row.append("")

            # Axis values
            for axis in space.axes:
                if axis.type == "subset":
                    active = [
                        c for c in axis.candidates
                        if t.params.get(f"{axis.key}__{c}")
                    ]
                    row.append(",".join(active) if active else "(none)")
                else:
                    val = t.params.get(axis.key, "")
                    if isinstance(val, float):
                        row.append(f"{val:.4g}")
                    else:
                        row.append(str(val))

            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # --- Parameter importance ---
    if param_importance:
        lines.append("### Parameter Importance\n")
        lines.append("| Axis | Importance |")
        lines.append("|------|-----------|")
        for param, imp in sorted(
            param_importance.items(), key=lambda x: x[1], reverse=True,
        ):
            lines.append(f"| {param} | {imp:.2f} |")
        lines.append("")

    # --- Baseline comparison ---
    if baseline_metrics and study.best_trial:
        lines.append("### Baseline Comparison\n")
        lines.append("| Metric | Baseline | Best Trial | Delta |")
        lines.append("|--------|----------|------------|-------|")
        best_attrs = study.best_trial.user_attrs
        for mk in sorted(baseline_metrics.keys()):
            base_v = baseline_metrics[mk]
            best_v = best_attrs.get(mk)
            if isinstance(base_v, (int, float)) and isinstance(best_v, (int, float)):
                delta = best_v - base_v
                lines.append(
                    f"| {mk} | {base_v:.4f} | {best_v:.4f} | {delta:+.4f} |"
                )
            elif isinstance(base_v, (int, float)):
                lines.append(f"| {mk} | {base_v:.4f} | N/A | N/A |")
        lines.append("")

    return "\n".join(lines)
