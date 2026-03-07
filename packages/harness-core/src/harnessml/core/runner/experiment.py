"""Experiment change detection, lifecycle management, and smart execution."""
from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from harnessml.core.config.merge import resolve_feature_mutations

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

    Uses harnessml.experiments.ExperimentManager.detect_changes() for
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
        from harnessml.core.runner.experiment_manager import ExperimentManager
    except ImportError:
        logger.warning(
            "harnessml-experiments not installed; falling back to basic change detection"
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
    """Basic change detection when harnessml-experiments is not available."""
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


# -----------------------------------------------------------------------
# Auto-numbering
# -----------------------------------------------------------------------

_EXP_NUMBER_RE = re.compile(r"^(?:\w+-)?exp-(\d{3})")


def auto_next_id(
    experiments_dir: Path,
    prefix: str = "exp",
) -> str:
    """Scan *experiments_dir* and return the next zero-padded experiment ID.

    Recognises patterns like ``exp-001-description``,
    ``w-exp-005-thing``, or bare ``exp-012``.

    Returns
    -------
    str
        e.g. ``"exp-003"`` if the highest existing number is 2.
    """
    experiments_dir = Path(experiments_dir)
    max_num = 0

    if experiments_dir.exists():
        for entry in experiments_dir.iterdir():
            if not entry.is_dir():
                continue
            m = _EXP_NUMBER_RE.match(entry.name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    return f"{prefix}-{max_num + 1:03d}"


# -----------------------------------------------------------------------
# Auto-logging
# -----------------------------------------------------------------------

_LOG_HEADER = (
    "| ID | Date | Hypothesis | Changes | Accuracy | Brier | ECE | LogLoss | Verdict | Conclusion | Notes |\n"
    "|-----|------|------------|---------|----------|-------|-----|---------|---------|------------|-------|\n"
)


def auto_log_result(
    log_path: Path,
    experiment_id: str,
    hypothesis: str,
    changes: str,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    verdict: str,
    conclusion: str = "",
    notes: str = "",
) -> None:
    """Append a row to an ``EXPERIMENT_LOG.md`` markdown table.

    Creates the file with headers if it does not exist.

    Parameters
    ----------
    log_path : Path
        Path to the experiment log markdown file.
    metrics : dict
        Experiment metrics (looks for accuracy, brier_score, ece, log_loss).
    baseline_metrics : dict
        Baseline metrics for delta computation.
    verdict : str
        Experiment outcome: "improved", "regressed", "neutral", etc.
    """
    log_path = Path(log_path)

    if not log_path.exists():
        log_path.write_text(f"# Experiment Log\n\n{_LOG_HEADER}")

    def _fmt(key: str) -> str:
        val = metrics.get(key)
        base = baseline_metrics.get(key)
        if val is None:
            return "-"
        val_str = f"{val:.4f}"
        if base is not None:
            delta = val - base
            val_str += f" ({delta:+.4f})"
        return val_str

    date_str = datetime.date.today().isoformat()
    row = (
        f"| {experiment_id} | {date_str} | {hypothesis} | {changes} "
        f"| {_fmt('accuracy')} | {_fmt('brier_score')} | {_fmt('ece')} "
        f"| {_fmt('log_loss')} | {verdict} | {conclusion} | {notes} |\n"
    )

    with open(log_path, "a") as f:
        f.write(row)


# -----------------------------------------------------------------------
# Frozen config
# -----------------------------------------------------------------------

def save_frozen_config(
    experiment_dir: Path,
    resolved_config: dict,
    production_run_id: str | None = None,
    features_hash: str = "",
    cache_stats: dict | None = None,
) -> Path:
    """Save full reproduction context for an experiment.

    Writes ``frozen_config.json`` with the resolved config, production
    run ID, features hash, and cache hit/miss stats.

    Returns
    -------
    Path
        Path to the written frozen config file.
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    frozen = {
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "production_run_id": production_run_id,
        "features_hash": features_hash,
        "cache_stats": cache_stats or {},
        "resolved_config": resolved_config,
    }

    path = experiment_dir / "frozen_config.json"
    path.write_text(json.dumps(frozen, indent=2, default=str))
    return path


# -----------------------------------------------------------------------
# Promote with safety checks
# -----------------------------------------------------------------------

# Metrics where lower is better
_LOWER_IS_BETTER = {"brier", "brier_score", "ece", "log_loss"}


def promote_experiment(
    experiment_id: str,
    experiments_dir: Path,
    config_dir: Path,
    primary_metric: str = "brier_score",
    max_improvement: float = 0.05,
) -> dict[str, Any]:
    """Promote an experiment's config changes to production.

    Safety checks:
    1. Experiment must have results.
    2. Primary metric must improve over baseline.
    3. Suspiciously large improvements (possible leakage) are flagged.

    Parameters
    ----------
    primary_metric : str
        Metric to evaluate for improvement.
    max_improvement : float
        Improvement larger than this triggers a leakage warning and
        blocks promotion.

    Returns
    -------
    dict
        ``{"promoted": bool, "reason": str, ...}``
    """
    experiments_dir = Path(experiments_dir)
    config_dir = Path(config_dir)
    exp_dir = experiments_dir / experiment_id

    # Load experiment results
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        return {"promoted": False, "reason": f"No results.json found in {exp_dir}"}

    results = json.loads(results_path.read_text())
    exp_metrics = results.get("metrics", {})
    baseline_metrics = results.get("baseline_metrics", {})

    if primary_metric not in exp_metrics:
        return {
            "promoted": False,
            "reason": f"Primary metric '{primary_metric}' not found in experiment results",
        }

    exp_val = exp_metrics[primary_metric]
    base_val = baseline_metrics.get(primary_metric)

    if base_val is None:
        return {
            "promoted": False,
            "reason": f"No baseline value for '{primary_metric}' to compare against",
        }

    # Compute improvement (positive = better)
    if primary_metric in _LOWER_IS_BETTER:
        improvement = base_val - exp_val  # lower is better → positive improvement
    else:
        improvement = exp_val - base_val  # higher is better → positive improvement

    if improvement <= 0:
        return {
            "promoted": False,
            "reason": f"No improvement on {primary_metric}: "
            f"baseline={base_val:.4f}, experiment={exp_val:.4f}",
        }

    if improvement > max_improvement:
        return {
            "promoted": False,
            "reason": "suspicious_improvement",
            "warning": (
                f"Improvement of {improvement:.4f} on {primary_metric} exceeds "
                f"threshold of {max_improvement:.4f}. Possible leakage — "
                f"review before promoting."
            ),
        }

    # Apply overlay to production config
    overlay_path = exp_dir / "overlay.yaml"
    if not overlay_path.exists():
        return {"promoted": False, "reason": f"No overlay.yaml found in {exp_dir}"}

    overlay = yaml.safe_load(overlay_path.read_text()) or {}
    changes_applied = _apply_overlay_to_config(overlay, config_dir)

    return {
        "promoted": True,
        "experiment_id": experiment_id,
        "improvement": {primary_metric: improvement},
        "changes": changes_applied,
    }


def _apply_overlay_to_config(overlay: dict, config_dir: Path) -> list[str]:
    """Apply overlay changes to production config YAML files.

    Returns list of change descriptions.
    """
    changes: list[str] = []
    skip_keys = {"description", "hypothesis", "sweep"}

    # Model changes → models.yaml
    if "models" in overlay:
        models_path = config_dir / "models.yaml"
        if models_path.exists():
            existing = yaml.safe_load(models_path.read_text()) or {}
        else:
            existing = {}

        existing_models = existing.get("models", {})
        for model_name, model_config in overlay["models"].items():
            if model_name in existing_models:
                existing_models[model_name].update(model_config)
                changes.append(f"Updated model: {model_name}")
            else:
                existing_models[model_name] = model_config
                changes.append(f"Added model: {model_name}")
        existing["models"] = existing_models
        resolve_feature_mutations(existing)
        models_path.write_text(
            yaml.dump(existing, default_flow_style=False, sort_keys=False)
        )

    # Ensemble changes → ensemble.yaml
    if "ensemble" in overlay:
        ensemble_path = config_dir / "ensemble.yaml"
        if ensemble_path.exists():
            existing = yaml.safe_load(ensemble_path.read_text()) or {}
        else:
            existing = {}

        existing_ensemble = existing.get("ensemble", {})
        existing_ensemble.update(overlay["ensemble"])
        existing["ensemble"] = existing_ensemble
        ensemble_path.write_text(
            yaml.dump(existing, default_flow_style=False, sort_keys=False)
        )
        changes.append("Updated ensemble config")

    # Log other keys that were in overlay but not applied
    for key in overlay:
        if key not in ("models", "ensemble") and key not in skip_keys:
            changes.append(f"Skipped overlay key: {key}")

    return changes


# -----------------------------------------------------------------------
# Sweep execution
# -----------------------------------------------------------------------

def run_sweep(
    overlay_path: Path,
    config_dir: Path,
    project_dir: Path,
    experiments_dir: Path,
    experiment_id: str,
    variant: str | None = None,
    primary_metric: str = "brier_score",
) -> dict[str, Any]:
    """Run a sweep experiment.

    1. Load overlay from *overlay_path*.
    2. ``expand_sweep(overlay)`` → list of concrete overlays.
    3. For each variant:
       a. Create sub-experiment dir ``experiments/{id}-v{NN}/``.
       b. Write the variant overlay.
       c. Run ``PipelineRunner.backtest()`` with a shared prediction
          cache (unchanged models are trained once, then cached).
       d. Save results + frozen config.
    4. Rank variants by *primary_metric*.
    5. Return summary with best variant highlighted.

    Parameters
    ----------
    overlay_path : Path
        Path to the base overlay YAML.
    config_dir : Path
        Project config directory.
    project_dir : Path
        Root project directory.
    experiments_dir : Path
        Root experiments directory.
    experiment_id : str
        Experiment identifier.
    variant : str | None
        Gender variant (e.g. ``"w"``).
    primary_metric : str
        Metric to rank variants by (default ``"brier_score"``).

    Returns
    -------
    dict
        Keys: ``is_sweep``, ``experiment_id``, ``n_variants``,
        ``results`` (ranked), ``best``, ``total_cache_stats``.
    """
    from harnessml.core.runner.pipeline import PipelineRunner
    from harnessml.core.runner.prediction_cache import PredictionCache
    from harnessml.core.runner.sweep import expand_sweep

    overlay = yaml.safe_load(overlay_path.read_text()) or {}
    variants = expand_sweep(overlay)

    if len(variants) <= 1:
        return {"is_sweep": False, "variants": variants}

    # Shared prediction cache across variants — unchanged models
    # only need to train once for each holdout fold.
    cache_dir = experiments_dir / experiment_id / ".cache"
    pred_cache = PredictionCache(cache_dir)

    results: list[dict] = []
    for i, variant_overlay in enumerate(variants):
        variant_id = f"{experiment_id}-v{i:02d}"
        variant_dir = experiments_dir / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Write variant overlay
        variant_overlay_path = variant_dir / "overlay.yaml"
        variant_overlay_path.write_text(
            yaml.dump(variant_overlay, default_flow_style=False, sort_keys=False)
        )

        try:
            runner = PipelineRunner(
                project_dir=project_dir,
                config_dir=str(config_dir),
                variant=variant,
                overlay=variant_overlay,
                prediction_cache=pred_cache,
            )
            runner.load()
            bt_result = runner.backtest()

            results_data = {
                "variant_id": variant_id,
                "description": variant_overlay.get("description", ""),
                "metrics": bt_result.get("metrics", {}),
                "cache_stats": runner.cache_stats,
            }
            (variant_dir / "results.json").write_text(
                json.dumps(results_data, indent=2, default=str)
            )

            save_frozen_config(
                variant_dir,
                resolved_config=variant_overlay,
                cache_stats=runner.cache_stats,
            )

            results.append(results_data)

        except Exception as exc:
            logger.exception("Sweep variant %s failed", variant_id)
            results.append({
                "variant_id": variant_id,
                "description": variant_overlay.get("description", ""),
                "error": str(exc),
            })

    ranked = _rank_sweep_results(results, primary_metric)

    return {
        "is_sweep": True,
        "experiment_id": experiment_id,
        "n_variants": len(variants),
        "results": ranked,
        "best": ranked[0] if ranked else None,
        "total_cache_stats": {
            "hits": sum(
                r.get("cache_stats", {}).get("hits", 0) for r in results
            ),
            "misses": sum(
                r.get("cache_stats", {}).get("misses", 0) for r in results
            ),
        },
    }


def _rank_sweep_results(
    results: list[dict],
    primary_metric: str,
) -> list[dict]:
    """Rank sweep results by *primary_metric*, best first."""
    scored = [
        r for r in results
        if "error" not in r
        and primary_metric in r.get("metrics", {})
    ]

    if not scored:
        return results

    # Lower is better for brier/ece/log_loss, higher for accuracy
    reverse = primary_metric not in _LOWER_IS_BETTER
    scored.sort(
        key=lambda r: r["metrics"][primary_metric],
        reverse=reverse,
    )

    for i, r in enumerate(scored):
        r["rank"] = i + 1

    # Append errored variants at the end
    errored = [r for r in results if "error" in r]
    return scored + errored


def format_sweep_summary(sweep_result: dict) -> str:
    """Format sweep results as markdown summary.

    Parameters
    ----------
    sweep_result : dict
        Output from ``run_sweep()``.

    Returns
    -------
    str
        Markdown-formatted summary.
    """
    if not sweep_result.get("is_sweep"):
        return "Not a sweep experiment."

    lines = [
        f"## Sweep Results: {sweep_result['experiment_id']}",
        f"Variants: {sweep_result['n_variants']}",
        "",
    ]

    results = sweep_result.get("results", [])
    if results:
        lines.append("| Rank | Variant | Description | Brier | Accuracy |")
        lines.append("|------|---------|-------------|-------|----------|")
        for r in results:
            if "error" in r:
                lines.append(
                    f"| - | {r['variant_id']} | {r['description']} "
                    f"| ERROR | {r['error'][:40]} |"
                )
                continue
            metrics = r.get("metrics", {})
            rank = r.get("rank", "-")
            brier = metrics.get("brier_score")
            acc = metrics.get("accuracy")
            brier_str = f"{brier:.4f}" if brier is not None else "-"
            acc_str = f"{acc:.4f}" if acc is not None else "-"
            lines.append(
                f"| {rank} | {r['variant_id']} | {r['description']} "
                f"| {brier_str} | {acc_str} |"
            )
        lines.append("")

    best = sweep_result.get("best")
    if best:
        lines.append(f"**Best variant:** {best['variant_id']}")
        lines.append(f"**Description:** {best.get('description', '')}")

    stats = sweep_result.get("total_cache_stats", {})
    if stats:
        total = stats.get("hits", 0) + stats.get("misses", 0)
        if total > 0:
            lines.append(
                f"\nCache: {stats['hits']} hits, {stats['misses']} misses "
                f"({stats['hits'] / total:.0%} hit rate)"
            )

    return "\n".join(lines)
