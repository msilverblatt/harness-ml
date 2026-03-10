"""Pipeline, backtest, run, diagnostics, and exploration operations."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from harnessml.core.runner.config_writer._helpers import (
    _LOWER_IS_BETTER,
    _get_config_dir,
    _load_yaml,
    _save_yaml,
)

logger = logging.getLogger(__name__)


def configure_backtest(
    project_dir: Path,
    *,
    cv_strategy: str | None = None,
    fold_values: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    fold_column: str | None = None,
    n_folds: int | None = None,
    window_size: int | None = None,
    group_column: str | None = None,
    eval_filter: str | None = None,
) -> str:
    """Update backtest section of pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "backtest" not in data:
        data["backtest"] = {}

    bt = data["backtest"]

    if cv_strategy is not None:
        from harnessml.core.runner.schema import _CV_STRATEGY_ALIASES
        resolved_strategy = _CV_STRATEGY_ALIASES.get(cv_strategy, cv_strategy)

        # Validate required companion parameters
        effective_n_folds = n_folds or bt.get("n_folds")
        effective_window = window_size or bt.get("window_size")
        effective_group = group_column or bt.get("group_column")

        if resolved_strategy in ("stratified_kfold", "purged_kfold") and not effective_n_folds:
            return f"**Error**: `{resolved_strategy}` strategy requires `n_folds`. Pass n_folds=5 (or similar)."
        if resolved_strategy == "sliding_window" and not effective_window:
            return "**Error**: `sliding_window` strategy requires `window_size`. Pass window_size=3 (or similar)."
        if resolved_strategy == "group_kfold" and not effective_group:
            return "**Error**: `group_kfold` strategy requires `group_column`. Pass group_column='col_name'."

        bt["cv_strategy"] = resolved_strategy
    if fold_values is not None:
        bt["fold_values"] = fold_values
    if metrics is not None:
        bt["metrics"] = metrics
    if min_train_folds is not None:
        bt["min_train_folds"] = min_train_folds
    if fold_column is not None:
        bt["fold_column"] = fold_column
    if n_folds is not None:
        bt["n_folds"] = n_folds
    if window_size is not None:
        bt["window_size"] = window_size
    if group_column is not None:
        bt["group_column"] = group_column
    if eval_filter is not None:
        bt["eval_filter"] = eval_filter

    _save_yaml(pipeline_path, data)

    fold_col_str = f"\n- Fold column: {bt['fold_column']}" if "fold_column" in bt else ""
    return (
        f"**Updated backtest config**\n"
        f"- CV strategy: {bt.get('cv_strategy', 'N/A')}\n"
        f"- Fold values: {bt.get('fold_values', [])}\n"
        f"- Metrics: {bt.get('metrics', [])}"
        f"{fold_col_str}"
    )


def show_config(project_dir: Path) -> str:
    """Show the resolved project configuration."""
    config_dir = _get_config_dir(Path(project_dir))
    from harnessml.core.runner.validation.validator import validate_project

    result = validate_project(str(config_dir))

    if not result.valid:
        return f"**Config validation failed**:\n{result.format()}"

    config = result.config
    lines = ["## Project Configuration\n"]

    # Models
    lines.append(f"### Models ({len(config.models)})\n")
    for name, m in sorted(config.models.items()):
        status = "active" if m.active else "inactive"
        lines.append(f"- **{name}**: {m.type} ({status}, {len(m.features)} features)")

    # Ensemble
    lines.append("\n### Ensemble\n")
    lines.append(f"- Method: {config.ensemble.method}")
    lines.append(f"- Temperature: {config.ensemble.temperature}")

    # Backtest
    lines.append("\n### Backtest\n")
    lines.append(f"- Strategy: {config.backtest.cv_strategy}")
    lines.append(f"- Fold values: {config.backtest.fold_values}")
    lines.append(f"- Metrics: {config.backtest.metrics}")

    return "\n".join(lines)


def check_guardrails(project_dir: Path) -> str:
    """Run configured guardrails and return a status report.

    Checks:
    - Feature leakage: model features vs denylist
    - Naming conventions: experiment IDs vs pattern
    - Model configuration: active models have features

    Returns markdown report with pass/fail per guardrail.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    models_data = _load_yaml(config_dir / "models.yaml")
    sources_data = _load_yaml(config_dir / "sources.yaml")

    guardrail_config = sources_data.get("guardrails", {})
    models = models_data.get("models", {})

    results = []

    # 1. Feature leakage check
    denylist = set(guardrail_config.get("feature_leakage_denylist", []))
    if denylist:
        violations = []
        for model_name, model_def in models.items():
            model_features = set(model_def.get("features", []))
            found = denylist & model_features
            if found:
                violations.append((model_name, sorted(found)))

        if violations:
            details = "; ".join(
                f"`{m}` uses {cols}" for m, cols in violations
            )
            results.append(("Feature Leakage", "FAIL", details))
        else:
            results.append(("Feature Leakage", "PASS", f"No denied columns found (denylist: {len(denylist)} entries)"))
    else:
        results.append(("Feature Leakage", "SKIP", "No denylist configured"))

    # 1b. Auto-detect leaky features (name patterns + correlation)
    try:
        from harnessml.core.guardrails.inventory import detect_leaky_columns
        from harnessml.core.runner.data.utils import get_feature_columns, get_features_df, load_data_config

        config = load_data_config(project_dir)
        df = get_features_df(project_dir, config)
        feature_cols = get_feature_columns(df, config)
        leaky = detect_leaky_columns(
            feature_cols, target_column=config.target_column, df=df, corr_threshold=0.90
        )
        if leaky:
            results.append(("Auto-Leakage Detection", "WARN", f"Suspicious columns: {leaky}"))
        else:
            results.append(("Auto-Leakage Detection", "PASS", "No suspicious patterns or high correlations"))
    except Exception:
        results.append(("Auto-Leakage Detection", "SKIP", "Could not load features for analysis"))

    # 2. Naming convention check
    naming_pattern = guardrail_config.get("naming_pattern")
    if naming_pattern:
        import re
        experiments_dir = project_dir / "experiments"
        if experiments_dir.exists():
            bad_names = []
            for d in experiments_dir.iterdir():
                if d.is_dir() and not re.match(naming_pattern, d.name):
                    bad_names.append(d.name)
            if bad_names:
                results.append(("Naming Convention", "FAIL", f"Invalid names: {bad_names}"))
            else:
                results.append(("Naming Convention", "PASS", f"All experiment names match `{naming_pattern}`"))
        else:
            results.append(("Naming Convention", "SKIP", "No experiments directory"))
    else:
        results.append(("Naming Convention", "SKIP", "No naming_pattern configured"))

    # 3. Model configuration check
    active_models = [n for n, m in models.items() if m.get("active", True)]
    featureless = [n for n in active_models if not models[n].get("features")]
    if featureless:
        results.append(("Model Config", "WARN", f"Models without features: {featureless}"))
    else:
        results.append(("Model Config", "PASS", f"{len(active_models)} active models, all with features"))

    # 4. Critical paths check
    critical_paths = guardrail_config.get("critical_paths", [])
    if critical_paths:
        results.append(("Critical Paths", "INFO", f"Protected: {critical_paths}"))
    else:
        results.append(("Critical Paths", "SKIP", "No critical paths configured"))

    # Format report
    n_fail = sum(1 for _, status, _ in results if status == "FAIL")
    n_pass = sum(1 for _, status, _ in results if status == "PASS")
    n_warn = sum(1 for _, status, _ in results if status == "WARN")

    lines = ["## Guardrail Report\n"]
    lines.append(f"**Summary**: {n_pass} passed, {n_fail} failed, {n_warn} warnings\n")

    lines.append("| Guardrail | Status | Details |")
    lines.append("|-----------|--------|---------|")
    for name, status, details in results:
        lines.append(f"| {name} | {status} | {details} |")

    if n_fail > 0:
        lines.append(f"\n**{n_fail} guardrail(s) failed.** Fix violations before proceeding.")

    return "\n".join(lines)


def list_runs(project_dir: Path) -> str:
    """List all pipeline runs with key metrics."""
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "No outputs_dir configured."

    from harnessml.core.runner.workflow.run_manager import RunManager

    abs_outputs = project_dir / outputs_dir
    mgr = RunManager(abs_outputs)
    runs = mgr.list_runs()

    if not runs:
        return "No runs found."

    # Try to detect metrics from the most recent run
    metric_keys = _detect_run_metrics(runs[0]["path"])

    lines = ["## Pipeline Runs\n"]
    if metric_keys:
        header = "| Run ID | " + " | ".join(k.title() for k in metric_keys) + " | Current |"
        sep = "|--------|" + "|".join("-------" for _ in metric_keys) + "|---------|"
        lines.extend([header, sep])
        for r in runs:
            metrics = _load_run_metrics(r["path"])
            vals = " | ".join(metrics.get(k, "\u2014") for k in metric_keys)
            current = " \u2713" if r["is_current"] else ""
            lines.append(f"| {r['run_id']} | {vals} | {current} |")
    else:
        for r in runs:
            marker = " **(current)**" if r["is_current"] else ""
            lines.append(f"- `{r['run_id']}`{marker}")

    return "\n".join(lines)


def _detect_run_metrics(run_path) -> list[str]:
    """Detect which metrics exist from a run's pooled_metrics.json."""
    metrics_file = Path(run_path) / "pooled_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        return list(data.keys())
    return []


def _load_run_metrics(run_path) -> dict[str, str]:
    """Load and format metrics from a run directory."""
    metrics_file = Path(run_path) / "pooled_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        return {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in data.items()}
    return {}


def _format_backtest_result(result: dict, run_id: str | None = None) -> str:
    """Format backtest results as structured markdown."""
    lines = ["## Backtest Results\n"]

    if run_id:
        lines.append(f"- **Run ID**: `{run_id}`")

    status = result.get("status", "unknown")
    lines.append(f"- **Status**: {status}")

    if status != "success":
        error = result.get("error", "Unknown error")
        lines.append(f"- **Error**: {error}")
        return "\n".join(lines)

    # Ensemble metrics
    metrics = result.get("metrics", {})
    if metrics:
        lines.append("\n### Ensemble Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, val in sorted(metrics.items()):
            lines.append(f"| {name} | {val:.4f} |")

    # Per-model metrics (if available in per_fold)
    models_trained = result.get("models_trained", [])
    if models_trained:
        lines.append(f"\n### Models Trained ({len(models_trained)})\n")
        for m in models_trained:
            lines.append(f"- `{m}`")

    models_failed = result.get("models_failed", [])
    if models_failed:
        model_errors = result.get("model_errors", [])
        lines.append(f"\n### Models Failed ({len(models_failed)})\n")
        # Build a map of model name -> first error message
        error_by_model = {}
        for err_line in model_errors:
            # Format: "  - model_name (fold N): error message"
            stripped = err_line.strip().lstrip("- ")
            for m in models_failed:
                if stripped.startswith(m):
                    if m not in error_by_model:
                        # Extract just the error part after the colon
                        colon_idx = stripped.find("): ")
                        if colon_idx >= 0:
                            error_by_model[m] = stripped[colon_idx + 3:]
                        else:
                            error_by_model[m] = stripped
                    break
        for m in models_failed:
            if m in error_by_model:
                lines.append(f"- `{m}`: {error_by_model[m]}")
            else:
                lines.append(f"- `{m}` (failed during backtest -- check logs)")

    # Meta-learner coefficients
    meta_coeff = result.get("meta_coefficients")
    if meta_coeff:
        # Check if multiclass (values are dicts) or binary (values are floats)
        first_val = next(iter(meta_coeff.values()), None)
        if isinstance(first_val, dict):
            # Multiclass: per-class coefficient tables
            lines.append("\n### Meta-Learner Weights (Multiclass)\n")
            for class_label, class_coeffs in sorted(meta_coeff.items()):
                lines.append(f"\n**Class {class_label}:**\n")
                lines.append("| Feature | Weight |")
                lines.append("|---------|--------|")
                for feat, w in sorted(class_coeffs.items(), key=lambda x: -abs(x[1])):
                    lines.append(f"| {feat} | {w:+.4f} |")
        else:
            lines.append("\n### Meta-Learner Weights\n")
            lines.append("| Model | Weight |")
            lines.append("|-------|--------|")
            for name, weight in sorted(meta_coeff.items(), key=lambda x: -abs(x[1])):
                lines.append(f"| {name} | {weight:+.4f} |")

    # Regression model CDF scales
    cdf_scales = result.get("model_cdf_scales", {})
    if cdf_scales:
        lines.append("\n### Regression Model CDF Scales\n")
        lines.append("| Model | Avg CDF Scale |")
        lines.append("|-------|--------------|")
        for name, scale in sorted(cdf_scales.items()):
            lines.append(f"| {name} | {scale:.3f} |")

    # Per-fold breakdown
    per_fold = result.get("per_fold", {})
    if per_fold:
        lines.append(f"\n### Per-Fold Breakdown ({len(per_fold)} folds)\n")
        # Detect metric columns dynamically from first fold
        first_fold_metrics = next(iter(per_fold.values()), {})
        metric_names = [k for k in first_fold_metrics if isinstance(first_fold_metrics[k], (int, float))]
        if metric_names:
            header = "| Fold | " + " | ".join(m.replace("_", " ").title() for m in metric_names) + " |"
            sep = "|------" + "|-------" * len(metric_names) + "|"
            lines.append(header)
            lines.append(sep)
            for fold_id, fold_metrics in sorted(per_fold.items()):
                cells = [str(fold_id)]
                for m in metric_names:
                    val = fold_metrics.get(m, "N/A")
                    cells.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
                lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def run_backtest(
    project_dir: Path,
    *,
    experiment_id: str | None = None,
    variant: str | None = None,
    on_progress=None,
) -> str:
    """Run a full backtest and return formatted results.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    experiment_id : str | None
        If provided, applies the experiment's overlay before running.
    variant : str | None
        Config variant (e.g., "w" for women's).

    Returns
    -------
    str
        Markdown-formatted backtest results.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    # Load experiment overlay if specified
    overlay = None
    if experiment_id:
        exp_dir = project_dir / "experiments" / experiment_id
        overlay_path = exp_dir / "overlay.yaml"
        if overlay_path.exists():
            overlay = _load_yaml(overlay_path)

    # Create run directory
    run_dir = None
    run_id = None
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if outputs_dir:
        from harnessml.core.runner.workflow.run_manager import RunManager
        mgr = RunManager(project_dir / outputs_dir)
        run_dir = mgr.new_run()
        run_id = run_dir.name

    try:
        # Pre-backtest validation
        from harnessml.core.runner.validation.validation import Severity, format_validation_issues, validate_project
        issues = validate_project(project_dir)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        if errors:
            return format_validation_issues(errors)

        from harnessml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
            run_dir=run_dir,
        )
        runner.load()
        result = runner.backtest(on_progress=on_progress)

        return _format_backtest_result(result, run_id=run_id)
    except Exception as exc:
        return f"**Backtest failed**: {exc}"


def run_predict(
    project_dir: Path,
    fold_value: int,
    *,
    run_id: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate predictions for a target fold.

    Trains on all historical data before the target fold,
    then predicts every row in the target fold.

    Returns markdown-formatted prediction summary.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    try:
        from harnessml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        runner.load()
        preds_df = runner.predict(fold_value, run_id=run_id)

        if preds_df is None or len(preds_df) == 0:
            return f"**No predictions**: No data found for fold {fold_value}."

        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        lines = [f"## Predictions for Fold {fold_value}\n"]
        lines.append(f"- **Rows predicted**: {len(preds_df)}")
        lines.append(f"- **Models**: {len(prob_cols)}")

        if "prob_ensemble" in preds_df.columns:
            ens_probs = preds_df["prob_ensemble"]
            lines.append("\n### Ensemble Probability Distribution\n")
            lines.append(f"- Mean: {ens_probs.mean():.4f}")
            lines.append(f"- Std: {ens_probs.std():.4f}")
            lines.append(f"- Min: {ens_probs.min():.4f}")
            lines.append(f"- Max: {ens_probs.max():.4f}")

            preds_df["_confidence"] = (preds_df["prob_ensemble"] - 0.5).abs()
            top = preds_df.nlargest(10, "_confidence")

            lines.append("\n### Top 10 Most Confident Predictions\n")
            lines.append("| Row | Ensemble Prob | Confidence |")
            lines.append("|-----|---------------|------------|")
            for idx, row in top.iterrows():
                lines.append(
                    f"| {idx} | {row['prob_ensemble']:.4f} | {row['_confidence']:.4f} |"
                )

        return "\n".join(lines)

    except Exception as exc:
        return f"**Prediction failed**: {exc}"


def show_run(
    project_dir: Path,
    run_id: str | None = None,
) -> str:
    """Show results from a pipeline run.

    Parameters
    ----------
    run_id : str | None
        Run to show. If None, shows the most recent run.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    outputs_path = project_dir / outputs_dir

    if run_id:
        run_dir = outputs_path / run_id
    else:
        # Find most recent run
        if not outputs_path.exists():
            return "**Error**: No runs found."
        runs = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and d.name != "current"],
            key=lambda d: d.name,
            reverse=True,
        )
        if not runs:
            return "**Error**: No runs found."
        run_dir = runs[0]

    if not run_dir.exists():
        return f"**Error**: Run '{run_id}' not found."

    lines = [f"## Run: `{run_dir.name}`\n"]

    # Read report.md if available
    report_path = run_dir / "report.md"
    if report_path.exists():
        lines.append(report_path.read_text())
        return "\n".join(lines)

    # Read pooled_metrics.json if available
    metrics_path = run_dir / "pooled_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        lines.append("### Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, val in sorted(metrics.items()):
            if isinstance(val, (int, float)):
                lines.append(f"| {name} | {val:.4f} |")
        return "\n".join(lines)

    # List available artifacts
    artifacts = sorted(f.name for f in run_dir.rglob("*") if f.is_file())
    if artifacts:
        lines.append("### Artifacts\n")
        for a in artifacts[:20]:
            lines.append(f"- `{a}`")
        if len(artifacts) > 20:
            lines.append(f"- ... +{len(artifacts) - 20} more")
    else:
        lines.append("No artifacts found in this run.")

    return "\n".join(lines)


def compare_runs(
    project_dir: Path,
    run_id_a: str | None = None,
    run_id_b: str | None = None,
    latest: bool = False,
) -> str:
    """Compare metrics from two pipeline runs side by side with deltas.

    Parameters
    ----------
    run_id_a, run_id_b : str | None
        Explicit run IDs to compare.
    latest : bool
        If True, automatically compare the two most recent runs (ignoring
        run_id_a / run_id_b).
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    outputs_path = project_dir / outputs_dir

    if latest:
        if not outputs_path.exists():
            return "**Error**: No runs found."
        runs = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and d.name != "current"],
            key=lambda d: d.name,
        )
        if len(runs) < 2:
            return "**Error**: Need at least 2 runs to compare; found {}.".format(len(runs))
        run_dir_a, run_dir_b = runs[-2], runs[-1]
    else:
        if not run_id_a or not run_id_b:
            return "**Error**: Provide two run IDs or use latest=True."
        run_dir_a = outputs_path / run_id_a
        run_dir_b = outputs_path / run_id_b

    for rd in (run_dir_a, run_dir_b):
        if not rd.exists():
            return f"**Error**: Run '{rd.name}' not found."

    metrics_a = _load_run_metrics_raw(run_dir_a)
    metrics_b = _load_run_metrics_raw(run_dir_b)

    return _format_comparison_table(run_dir_a.name, metrics_a, run_dir_b.name, metrics_b)


def _load_run_metrics_raw(run_dir: Path) -> dict[str, float]:
    """Load numeric metrics from a run directory (raw float values)."""
    metrics_path = run_dir / "pooled_metrics.json"
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text())
        return {k: v for k, v in raw.items() if isinstance(v, (int, float))}
    return {}


def _format_comparison_table(
    id_a: str,
    metrics_a: dict[str, float],
    id_b: str,
    metrics_b: dict[str, float],
) -> str:
    """Build a markdown comparison table with delta and direction columns."""
    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

    lines = [f"## Run Comparison: `{id_a}` vs `{id_b}`\n"]

    if not all_keys:
        lines.append("No numeric metrics found in either run.")
        return "\n".join(lines)

    lines.append(f"| Metric | `{id_a}` | `{id_b}` | Delta |")
    lines.append("|--------|------|------|-------|")
    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        col_a = f"{val_a:.4f}" if val_a is not None else "\u2014"
        col_b = f"{val_b:.4f}" if val_b is not None else "\u2014"
        delta_str = ""
        if val_a is not None and val_b is not None:
            diff = val_b - val_a
            lower_better = key.lower() in _LOWER_IS_BETTER
            if diff > 0:
                arrow = "\u2193" if lower_better else "\u2191"
            elif diff < 0:
                arrow = "\u2191" if lower_better else "\u2193"
            else:
                arrow = "="
            delta_str = f"{diff:+.4f} {arrow}"
        lines.append(f"| {key} | {col_a} | {col_b} | {delta_str} |")

    return "\n".join(lines)


def show_diagnostics(
    project_dir: Path,
    run_id: str | None = None,
) -> str:
    """Show per-model diagnostics from a backtest run.

    Reads prediction artifacts from the run directory and computes
    per-model metrics (brier, accuracy, ECE, log_loss) plus model
    agreement and calibration summary.

    Returns markdown report.
    """
    import numpy as np
    import pandas as pd

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured in pipeline.yaml."

    outputs_path = project_dir / outputs_dir

    if run_id:
        run_dir = outputs_path / run_id
    else:
        if not outputs_path.exists():
            return "**Error**: No runs found."
        runs = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and d.name != "current"],
            key=lambda d: d.name,
            reverse=True,
        )
        if not runs:
            return "**Error**: No runs found."
        run_dir = runs[0]

    if not run_dir.exists():
        return f"**Error**: Run '{run_id}' not found."

    preds_dir = run_dir / "predictions"
    preds_path = preds_dir / "predictions.parquet"
    if preds_path.exists():
        preds_df = pd.read_parquet(preds_path)
    elif preds_dir.exists():
        # Predictions may be split per-fold (e.g. 2024_probabilities.parquet)
        fold_files = sorted(preds_dir.glob("*_probabilities.parquet"))
        if fold_files:
            preds_df = pd.concat(
                [pd.read_parquet(f) for f in fold_files],
                ignore_index=True,
            )
        else:
            diag_path = run_dir / "diagnostics" / "predictions.parquet"
            if diag_path.exists():
                preds_df = pd.read_parquet(diag_path)
            else:
                return f"**Error**: No predictions found in run `{run_dir.name}`."
    else:
        diag_path = run_dir / "diagnostics" / "predictions.parquet"
        if diag_path.exists():
            preds_df = pd.read_parquet(diag_path)
        else:
            return f"**Error**: No predictions found in run `{run_dir.name}`."

    target_col = pipeline_data.get("data", {}).get("target_column", "result")
    task = pipeline_data.get("data", {}).get("task", "binary")

    if target_col not in preds_df.columns:
        return f"**Error**: Predictions file missing '{target_col}' column for evaluation."

    y_true = preds_df[target_col].values
    prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]

    if not prob_cols:
        return "**Error**: No prob_* columns found in predictions."

    lines = [f"## Diagnostics: `{run_dir.name}`\n"]
    lines.append(f"- Predictions: {len(preds_df)} rows")

    if task == "multiclass":
        # Multiclass diagnostics: use per-class prob columns
        import re

        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.metrics import log_loss as sklearn_log_loss

        y_true_int = y_true.astype(int)
        pattern = re.compile(r"^prob_(.+)_c(\d+)$")
        model_class_cols: dict[str, dict[int, str]] = {}
        for col in preds_df.columns:
            m = pattern.match(col)
            if m:
                model_name = m.group(1)
                class_idx = int(m.group(2))
                model_class_cols.setdefault(model_name, {})[class_idx] = col

        model_metrics = []
        for model_name, idx_to_col in sorted(model_class_cols.items()):
            n_classes = max(idx_to_col.keys()) + 1
            if set(idx_to_col.keys()) != set(range(n_classes)):
                continue
            ordered_cols = [idx_to_col[i] for i in range(n_classes)]
            prob_matrix = preds_df[ordered_cols].values.astype(float)
            valid = ~np.isnan(prob_matrix).any(axis=1)
            if valid.sum() == 0:
                continue
            y_t = y_true_int[valid]
            y_p = prob_matrix[valid]
            y_pred = y_p.argmax(axis=1)

            entry: dict = {
                "model": model_name,
                "accuracy": float(accuracy_score(y_t, y_pred)),
                "n_samples": int(valid.sum()),
            }
            try:
                entry["log_loss"] = float(sklearn_log_loss(y_t, y_p))
            except Exception:
                pass
            try:
                entry["f1_macro"] = float(f1_score(y_t, y_pred, average="macro"))
            except Exception:
                pass

            # Per-class accuracy
            per_class = {}
            for cls in sorted(set(y_t)):
                cls_mask = y_t == cls
                if cls_mask.sum() > 0:
                    per_class[int(cls)] = float(accuracy_score(y_t[cls_mask], y_pred[cls_mask]))
            entry["per_class_accuracy"] = per_class

            model_metrics.append(entry)

        lines.append(f"- Models: {len(model_metrics)}\n")

        lines.append("### Per-Model Metrics\n")
        lines.append("| Model | Accuracy | Log Loss | F1 Macro | N |")
        lines.append("|-------|----------|----------|----------|---|")
        for entry in sorted(model_metrics, key=lambda x: -x["accuracy"]):
            ll = entry.get("log_loss", float("nan"))
            f1m = entry.get("f1_macro", float("nan"))
            lines.append(
                f"| {entry['model']} | {entry['accuracy']:.4f} "
                f"| {ll:.4f} | {f1m:.4f} | {entry['n_samples']} |"
            )

        # Per-class accuracy breakdown
        lines.append("\n### Per-Class Accuracy\n")
        # Collect all class indices across models
        all_classes = set()
        for entry in model_metrics:
            all_classes.update(entry.get("per_class_accuracy", {}).keys())
        if all_classes:
            class_headers = ["Model"] + [f"Class {c}" for c in sorted(all_classes)]
            lines.append("| " + " | ".join(class_headers) + " |")
            lines.append("|" + "|".join(["------"] * len(class_headers)) + "|")
            for entry in sorted(model_metrics, key=lambda x: -x["accuracy"]):
                cells = [entry["model"]]
                for cls in sorted(all_classes):
                    acc = entry.get("per_class_accuracy", {}).get(cls)
                    cells.append(f"{acc:.4f}" if acc is not None else "N/A")
                lines.append("| " + " | ".join(cells) + " |")

    elif task == "regression":
        # Regression diagnostics
        from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

        y_true_float = y_true.astype(float)

        model_metrics = []
        for col in sorted(prob_cols):
            model_name = col.replace("prob_", "", 1)
            y_pred = preds_df[col].values.astype(float)

            valid = ~np.isnan(y_pred) & ~np.isnan(y_true_float)
            if valid.sum() == 0:
                continue

            y_t = y_true_float[valid]
            y_p = y_pred[valid]

            model_metrics.append({
                "model": model_name,
                "rmse": float(root_mean_squared_error(y_t, y_p)),
                "mae": float(mean_absolute_error(y_t, y_p)),
                "r_squared": float(r2_score(y_t, y_p)),
                "n_samples": int(valid.sum()),
            })

        lines.append(f"- Models: {len(model_metrics)}\n")

        lines.append("### Per-Model Metrics\n")
        lines.append("| Model | RMSE | MAE | R\u00b2 | N |")
        lines.append("|-------|------|-----|-----|---|")
        for m in sorted(model_metrics, key=lambda x: x["rmse"]):
            lines.append(
                f"| {m['model']} | {m['rmse']:.4f} | {m['mae']:.4f} "
                f"| {m['r_squared']:.4f} | {m['n_samples']} |"
            )

        # Residual summary for ensemble
        ensemble_col = "prob_ensemble" if "prob_ensemble" in preds_df.columns else prob_cols[0]
        y_ens = preds_df[ensemble_col].values.astype(float)
        valid_ens = ~np.isnan(y_ens) & ~np.isnan(y_true_float)
        if valid_ens.sum() > 0:
            residuals = y_true_float[valid_ens] - y_ens[valid_ens]
            ens_name = ensemble_col.replace("prob_", "", 1)
            lines.append(f"\n### Residual Summary (`{ens_name}`)\n")
            lines.append(f"- Mean residual: {float(np.mean(residuals)):.4f}")
            lines.append(f"- Std residual: {float(np.std(residuals)):.4f}")
            lines.append(f"- Median residual: {float(np.median(residuals)):.4f}")
            lines.append(f"- Max |residual|: {float(np.max(np.abs(residuals))):.4f}")

    else:
        # Binary diagnostics
        from harnessml.core.runner.analysis.diagnostics import (
            compute_brier_score,
            compute_calibration_curve,
            compute_ece,
            compute_model_agreement,
        )

        y_true_float = y_true.astype(float)

        model_metrics = []
        for col in sorted(prob_cols):
            model_name = col.replace("prob_", "", 1)
            y_prob = preds_df[col].values.astype(float)

            valid = ~np.isnan(y_prob) & ~np.isnan(y_true_float)
            if valid.sum() == 0:
                continue

            y_t = y_true_float[valid]
            y_p = y_prob[valid]

            brier = compute_brier_score(y_t, y_p)
            ece = compute_ece(y_t, y_p)
            accuracy = float(np.mean((y_p >= 0.5).astype(float) == y_t))
            eps = 1e-15
            y_clipped = np.clip(y_p, eps, 1.0 - eps)
            log_loss_val = float(-np.mean(
                y_t * np.log(y_clipped) + (1 - y_t) * np.log(1 - y_clipped)
            ))

            model_metrics.append({
                "model": model_name,
                "brier": brier,
                "accuracy": accuracy,
                "ece": ece,
                "log_loss": log_loss_val,
                "n_samples": int(valid.sum()),
            })

        lines.append(f"- Models: {len(model_metrics)}\n")

        lines.append("### Per-Model Metrics\n")
        lines.append("| Model | Brier | Accuracy | ECE | Log Loss | N |")
        lines.append("|-------|-------|----------|-----|----------|---|")
        for m in sorted(model_metrics, key=lambda x: x["brier"]):
            lines.append(
                f"| {m['model']} | {m['brier']:.4f} | {m['accuracy']:.4f} "
                f"| {m['ece']:.4f} | {m['log_loss']:.4f} | {m['n_samples']} |"
            )

        agreement = compute_model_agreement(preds_df)
        mean_agreement = float(np.mean(agreement))
        lines.append("\n### Model Agreement\n")
        lines.append(f"- Mean agreement with ensemble: {mean_agreement:.4f}")

        lines.append("\n### Calibration Summary\n")
        for m in model_metrics:
            model_name = m["model"]
            col = f"prob_{model_name}"
            y_p = preds_df[col].values.astype(float)
            valid = ~np.isnan(y_p)
            if valid.sum() == 0:
                continue
            mean_pred = float(np.mean(y_p[valid]))
            mean_actual = float(np.mean(y_true_float[valid]))
            bias = mean_pred - mean_actual
            cal_status = "well-calibrated" if abs(bias) < 0.02 else ("over-confident" if bias > 0 else "under-confident")
            lines.append(f"- **{model_name}**: {cal_status} (bias: {bias:+.4f})")

        # Calibration curve for the ensemble (or best model)
        ensemble_col = "prob_ensemble" if "prob_ensemble" in preds_df.columns else prob_cols[0]
        y_ens = preds_df[ensemble_col].values.astype(float)
        valid_ens = ~np.isnan(y_ens) & ~np.isnan(y_true_float)
        if valid_ens.sum() > 0:
            mean_pred, mean_actual, bin_counts = compute_calibration_curve(
                y_true_float[valid_ens], y_ens[valid_ens], n_bins=10,
            )
            ens_name = ensemble_col.replace("prob_", "", 1)
            lines.append(f"\n### Calibration Curve (`{ens_name}`)\n")
            lines.append("| Predicted | Actual | Count |")
            lines.append("|-----------|--------|-------|")
            for p, a, c in zip(mean_pred, mean_actual, bin_counts):
                lines.append(f"| {p:.3f} | {a:.3f} | {c} |")

    # Feature importance (if we can load the config and data)
    try:
        from harnessml.core.runner.data.utils import get_feature_columns, get_features_df, load_data_config
        config = load_data_config(project_dir)
        df = get_features_df(project_dir, config)
        _pi_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
        _pi_fold_col = _pi_data.get("backtest", {}).get("fold_column")
        feature_cols = get_feature_columns(df, config, fold_column=_pi_fold_col)
        if feature_cols and config.target_column in df.columns:
            from harnessml.core.runner.features.discovery import compute_feature_importance
            importance_df = compute_feature_importance(
                df, feature_columns=feature_cols, top_n=15,
            )
            if len(importance_df) > 0:
                lines.append(f"\n### Feature Importance (top {len(importance_df)})\n")
                lines.append("| Feature | Importance |")
                lines.append("|---------|------------|")
                for _, row in importance_df.iterrows():
                    lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
    except Exception as fi_exc:
        logger.debug("Could not compute feature importance: %s", fi_exc)

    # Meta-learner coefficients (from pooled_metrics.json or run artifacts)
    meta_path = run_dir / "diagnostics" / "pooled_metrics.json"
    if meta_path.exists():
        try:
            pooled = json.loads(meta_path.read_text())
            meta_coeff = pooled.get("meta_coefficients")
            if meta_coeff and isinstance(meta_coeff, dict):
                lines.append("\n### Meta-Learner Weights\n")
                lines.append("| Model | Weight |")
                lines.append("|-------|--------|")
                for name, weight in sorted(meta_coeff.items(), key=lambda x: -abs(x[1])):
                    lines.append(f"| {name} | {weight:+.4f} |")
        except Exception:
            pass

    return "\n".join(lines)


def explain_model(project_dir: Path, *, name: str | None = None, run_id: str | None = None, top_n: int = 10) -> str:
    """Run SHAP explainability on a trained model from a backtest run."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return "**Error**: `shap` package is not installed. Install with `pip install shap`."

    from harnessml.core.runner.analysis.explainability import compute_shap_summary, format_shap_report
    from harnessml.core.runner.data.utils import get_feature_columns, get_features_df, load_data_config

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    # Find the run directory
    from harnessml.core.runner.workflow.run_manager import RunManager
    mgr = RunManager(project_dir / outputs_dir)

    if run_id:
        run_path = project_dir / outputs_dir / run_id
    else:
        runs = mgr.list_runs()
        if not runs:
            return "**Error**: No runs found. Run a backtest first."
        run_path = Path(runs[0]["path"])

    if not run_path.exists():
        return f"**Error**: Run directory not found: {run_path}"

    # Find model artifacts
    models_dir = run_path / "models"
    if not models_dir.exists():
        return f"**Error**: No models directory in run {run_path.name}."

    # Pick a model (specific or first available)
    import pickle
    model_files = sorted(models_dir.glob("*.pkl"))
    if not model_files:
        return "**Error**: No model artifacts (.pkl) found."

    if name:
        target_file = models_dir / f"{name}.pkl"
        if not target_file.exists():
            available = [f.stem for f in model_files]
            return f"**Error**: Model `{name}` not found. Available: {', '.join(available)}"
        model_file = target_file
    else:
        model_file = model_files[0]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load feature data
    config = load_data_config(project_dir)
    df = get_features_df(project_dir, config)
    fold_column = pipeline_data.get("backtest", {}).get("fold_column")
    feature_cols = get_feature_columns(df, config, fold_column=fold_column)
    X = df[feature_cols].values

    # Compute SHAP
    try:
        results = compute_shap_summary(model, X, feature_cols, top_n=top_n)
    except Exception as e:
        return f"**Error** computing SHAP values: {e}"

    return format_shap_report(results, model_name=model_file.stem)


def inspect_predictions(project_dir: Path, *, run_id: str | None = None, mode: str = "worst", top_n: int = 10) -> str:
    """Inspect predictions from a backtest run.

    Modes:
    - "worst": most confident wrong predictions
    - "best": most confident correct predictions
    - "uncertain": predictions closest to 0.5
    """
    import pandas as pd

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    target_col = pipeline_data.get("data", {}).get("target_column", "target")

    # Find run directory
    from harnessml.core.runner.workflow.run_manager import RunManager
    mgr = RunManager(project_dir / outputs_dir)

    if run_id:
        run_path = project_dir / outputs_dir / run_id
    else:
        runs = mgr.list_runs()
        if not runs:
            return "**Error**: No runs found. Run a backtest first."
        run_path = Path(runs[0]["path"])

    if not run_path.exists():
        return f"**Error**: Run directory not found: {run_path.name}"

    # Load prediction files
    preds_dir = run_path / "predictions"
    if not preds_dir.exists():
        return f"**Error**: No predictions directory in run `{run_path.name}`."

    parquet_files = sorted(preds_dir.glob("*.parquet"))
    if not parquet_files:
        return "**Error**: No prediction files found."

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # Find the probability column (ensemble_prob or similar)
    prob_col = None
    for candidate in ["ensemble_prob", "prob_ensemble", "probability", "prob", "pred_prob", "prediction"]:
        if candidate in df.columns:
            prob_col = candidate
            break
    if prob_col is None:
        # Try any column ending in _prob or starting with prob_
        for c in df.columns:
            if c.endswith("_prob") or c.startswith("prob_"):
                prob_col = c
                break
    if prob_col is None:
        return f"**Error**: No probability column found. Columns: {list(df.columns)}"

    if target_col not in df.columns:
        return f"**Error**: Target column `{target_col}` not found in predictions."

    # Compute metrics for sorting
    df["_confidence"] = (df[prob_col] - 0.5).abs() * 2
    df["_predicted"] = (df[prob_col] > 0.5).astype(int)
    df["_correct"] = (df["_predicted"] == df[target_col]).astype(int)

    if mode == "worst":
        # Most confident AND wrong
        wrong = df[df["_correct"] == 0].sort_values("_confidence", ascending=False)
        subset = wrong.head(top_n)
        title = "Most Confident Wrong Predictions"
    elif mode == "best":
        correct = df[df["_correct"] == 1].sort_values("_confidence", ascending=False)
        subset = correct.head(top_n)
        title = "Most Confident Correct Predictions"
    elif mode == "uncertain":
        subset = df.sort_values("_confidence", ascending=True).head(top_n)
        title = "Most Uncertain Predictions"
    else:
        return f"**Error**: Unknown mode `{mode}`. Use 'worst', 'best', or 'uncertain'."

    if subset.empty:
        return f"No predictions match mode '{mode}'."

    # Format output
    lines = [f"## {title}\n"]

    # Pick display columns: key columns + target + prob + confidence
    display_cols = []
    key_cols = pipeline_data.get("data", {}).get("key_columns", [])
    for kc in key_cols:
        if kc in df.columns:
            display_cols.append(kc)
    display_cols.extend([target_col, prob_col, "_confidence"])

    header = "| " + " | ".join(display_cols) + " |"
    sep = "|" + "|".join("------" for _ in display_cols) + "|"
    lines.extend([header, sep])

    for _, row in subset.iterrows():
        vals = []
        for c in display_cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append(f"\n*Showing {len(subset)} of {len(df)} total predictions.*")

    return "\n".join(lines)


def run_exploration(
    project_dir: Path,
    search_space: dict,
    on_progress=None,
) -> str:
    """Run a Bayesian exploration over a search space.

    Delegates to :func:`harnessml.runner.exploration.run_exploration` and
    returns the markdown report.
    """
    from harnessml.core.runner.optimization.exploration import run_exploration as _run_exploration

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    try:
        result = _run_exploration(
            project_dir=project_dir,
            search_space=search_space,
            config_dir=config_dir,
            on_progress=on_progress,
        )
        return result["report"]
    except ImportError as exc:
        return f"**Error**: {exc}"
    except Exception as exc:
        return f"**Exploration failed**: {exc}"


def format_target_comparison(results: dict[str, dict]) -> str:
    """Format multi-target comparison as markdown table."""
    if not results:
        return "No results to compare."

    # Collect all metric keys across targets
    metric_keys: list[str] = []
    for metrics in results.values():
        for k in metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    lines = ["## Target Comparison\n"]
    header = "| Target | " + " | ".join(metric_keys) + " |"
    sep = "|--------|" + "|".join("------" for _ in metric_keys) + "|"
    lines.extend([header, sep])

    _dash = "\u2014"
    for target_name, metrics in results.items():
        vals = " | ".join(
            f"{metrics.get(k, _dash):.4f}" if isinstance(metrics.get(k), (int, float)) else str(metrics.get(k, _dash))
            for k in metric_keys
        )
        lines.append(f"| {target_name} | {vals} |")

    # Highlight best per metric
    lines.append("\n**Best per metric:**")
    for k in metric_keys:
        vals = {name: m.get(k) for name, m in results.items() if k in m and isinstance(m.get(k), (int, float))}
        if not vals:
            continue
        lower_better = k in ("brier", "ece", "log_loss", "rmse", "mae")
        best_name = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        lines.append(f"- {k}: **{best_name}** ({vals[best_name]:.4f})")

    return "\n".join(lines)
