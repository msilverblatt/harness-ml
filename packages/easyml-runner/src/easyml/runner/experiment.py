"""Experiment change detection and smart execution."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChangeSet:
    """Extended change detection result for the runner."""

    changed_models: list[str] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    removed_models: list[str] = field(default_factory=list)
    ensemble_changed: bool = False
    feature_config_changed: bool = False

    @property
    def ensemble_only(self) -> bool:
        """True when only ensemble config changed, no model changes."""
        return (
            not self.changed_models
            and not self.new_models
            and not self.removed_models
            and self.ensemble_changed
            and not self.feature_config_changed
        )

    @property
    def total_changes(self) -> int:
        return (
            len(self.changed_models)
            + len(self.new_models)
            + len(self.removed_models)
            + int(self.ensemble_changed)
            + int(self.feature_config_changed)
        )


@dataclass
class ExperimentResult:
    """Complete experiment result with baseline comparison."""

    experiment_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    change_set: ChangeSet = field(default_factory=ChangeSet)
    models_trained: list[str] = field(default_factory=list)


def detect_experiment_changes(
    production_config: dict,
    overlay: dict,
) -> ChangeSet:
    """Detect what an experiment overlay changes vs production.

    Uses easyml.experiments.ExperimentManager.detect_changes() for
    model and ensemble comparison, then extends with feature_config check.

    Parameters
    ----------
    production_config : dict
        The current production configuration dict.
    overlay : dict
        The experiment overlay dict (same shape, only overrides).

    Returns
    -------
    ChangeSet
        Summary of what the overlay changes.
    """
    try:
        from easyml.experiments.manager import ExperimentManager
    except ImportError:
        logger.warning(
            "easyml-experiments not installed; falling back to basic change detection"
        )
        return _fallback_detect_changes(production_config, overlay)

    manager = ExperimentManager(experiments_dir=".")
    report = manager.detect_changes(production_config, overlay)

    change_set = ChangeSet(
        changed_models=list(report.changed_models),
        new_models=list(report.new_models),
        removed_models=list(report.removed_models),
        ensemble_changed=len(report.ensemble_changes) > 0,
    )

    # Check feature_config changes
    prod_fc = production_config.get("feature_config", {})
    overlay_fc = overlay.get("feature_config", {})
    if overlay_fc and overlay_fc != prod_fc:
        change_set.feature_config_changed = True

    return change_set


def _fallback_detect_changes(
    production_config: dict,
    overlay: dict,
) -> ChangeSet:
    """Basic change detection when easyml-experiments is not available."""
    change_set = ChangeSet()

    prod_models = production_config.get("models", {})
    overlay_models = overlay.get("models", {})

    for name in overlay_models:
        if name in prod_models:
            if overlay_models[name] != prod_models[name]:
                change_set.changed_models.append(name)
        else:
            change_set.new_models.append(name)

    if "models" in overlay:
        for name in prod_models:
            if name not in overlay_models:
                change_set.removed_models.append(name)

    prod_ensemble = production_config.get("ensemble", {})
    overlay_ensemble = overlay.get("ensemble", {})
    if overlay_ensemble:
        for key in overlay_ensemble:
            if key not in prod_ensemble or overlay_ensemble[key] != prod_ensemble[key]:
                change_set.ensemble_changed = True
                break

    prod_fc = production_config.get("feature_config", {})
    overlay_fc = overlay.get("feature_config", {})
    if overlay_fc and overlay_fc != prod_fc:
        change_set.feature_config_changed = True

    return change_set


def load_baseline_metrics(run_dir: Path) -> dict[str, float]:
    """Load pooled metrics from a previous run directory.

    Looks for run_dir/diagnostics/pooled_metrics.json.
    Returns empty dict if not found.

    Parameters
    ----------
    run_dir : Path
        Path to a run output directory.

    Returns
    -------
    dict[str, float]
        Metric name -> value, or empty dict if file is missing.
    """
    metrics_path = run_dir / "diagnostics" / "pooled_metrics.json"
    if not metrics_path.exists():
        return {}

    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load baseline metrics from %s: %s", metrics_path, exc)
        return {}


def compute_deltas(
    experiment_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute metric differences (experiment - baseline).

    For brier/ece/logloss: negative = improvement.
    For accuracy: positive = improvement.

    Parameters
    ----------
    experiment_metrics : dict[str, float]
        Metrics from the experiment run.
    baseline_metrics : dict[str, float]
        Metrics from the baseline run.

    Returns
    -------
    dict[str, float]
        Delta for each metric present in both dicts.
    """
    deltas = {}
    for key in experiment_metrics:
        if key in baseline_metrics:
            try:
                deltas[key] = experiment_metrics[key] - baseline_metrics[key]
            except (TypeError, ValueError):
                # Skip non-numeric values (e.g. n_samples as int)
                pass
    return deltas


def format_change_summary(change_set: ChangeSet) -> str:
    """Format change set as human-readable summary string.

    Parameters
    ----------
    change_set : ChangeSet
        The detected changes.

    Returns
    -------
    str
        Multi-line summary of changes.
    """
    if change_set.total_changes == 0:
        return "No changes detected."

    lines = [f"Detected {change_set.total_changes} change(s):"]

    if change_set.changed_models:
        lines.append(f"  Changed models: {', '.join(change_set.changed_models)}")
    if change_set.new_models:
        lines.append(f"  New models: {', '.join(change_set.new_models)}")
    if change_set.removed_models:
        lines.append(f"  Removed models: {', '.join(change_set.removed_models)}")
    if change_set.ensemble_changed:
        lines.append("  Ensemble config changed")
    if change_set.feature_config_changed:
        lines.append("  Feature config changed")

    return "\n".join(lines)


def format_delta_table(
    experiment_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    deltas: dict[str, float],
) -> str:
    """Format metric comparison as a markdown table.

    Parameters
    ----------
    experiment_metrics : dict[str, float]
        Metrics from the experiment run.
    baseline_metrics : dict[str, float]
        Metrics from the baseline run.
    deltas : dict[str, float]
        Pre-computed deltas (experiment - baseline).

    Returns
    -------
    str
        Markdown-formatted comparison table.
    """
    lines = [
        "| Metric | Baseline | Experiment | Delta |",
        "|--------|----------|------------|-------|",
    ]

    for key in sorted(deltas.keys()):
        base_val = baseline_metrics.get(key)
        exp_val = experiment_metrics.get(key)
        delta_val = deltas[key]

        # Format values
        if isinstance(base_val, float):
            base_str = f"{base_val:.4f}"
        else:
            base_str = str(base_val)

        if isinstance(exp_val, float):
            exp_str = f"{exp_val:.4f}"
        else:
            exp_str = str(exp_val)

        if isinstance(delta_val, float):
            delta_str = f"{delta_val:+.4f}"
        else:
            delta_str = str(delta_val)

        lines.append(f"| {key} | {base_str} | {exp_str} | {delta_str} |")

    return "\n".join(lines)
