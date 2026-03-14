"""Handler for pipeline tool."""
from __future__ import annotations

from harnessml.core.logging import get_logger
from harnessml.plugin.handlers._common import resolve_project_dir
from harnessml.plugin.handlers._validation import (
    validate_enum,
    validate_required,
)
from protomcp import action, tool_group

logger = get_logger(__name__)


def _handle_run_backtest(*, experiment_id=None, variant=None, fold_values=None, ctx=None, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    def _progress_callback(current, total, message):
        if ctx is not None:
            ctx.report_progress(current, total, message)

    if ctx is not None:
        ctx.report_progress(0, 1, "Starting backtest...")

    # Parse fold_values from JSON string if needed
    parsed_fold_values = None
    if fold_values is not None:
        if isinstance(fold_values, str):
            import json
            parsed_fold_values = json.loads(fold_values)
        else:
            parsed_fold_values = fold_values

    result = cw.run_backtest(
        resolve_project_dir(project_dir),
        experiment_id=experiment_id,
        variant=variant,
        fold_values=parsed_fold_values,
        on_progress=_progress_callback,
    )

    if ctx is not None:
        ctx.report_progress(1, 1, "Backtest complete.")

    return result


def _handle_predict(*, fold_value, run_id, variant, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(fold_value, "fold_value")
    if err:
        return err
    return cw.run_predict(
        resolve_project_dir(project_dir),
        fold_value,
        run_id=run_id,
        variant=variant,
    )


def _handle_diagnostics(*, run_id, detail, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    result = cw.show_diagnostics(
        resolve_project_dir(project_dir),
        run_id=run_id,
    )
    if detail == "summary":
        return _summarize_diagnostics(result)
    return result


def _summarize_diagnostics(full_output: str) -> str:
    """Extract condensed view from full diagnostics output."""
    lines = full_output.strip().split("\n")
    summary_lines = []
    for line in lines:
        # Keep header lines and lines with key metrics
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("|") or stripped == "":
            summary_lines.append(line)
        # Keep the first few table rows (header + separator + data rows)
        # but skip detailed breakdowns
    # If there's a table, keep it but truncate to key columns
    if not summary_lines or all(ln.strip() == "" for ln in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_list_runs(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.list_runs(resolve_project_dir(project_dir))


def _handle_show_run(*, run_id, detail, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    result = cw.show_run(
        resolve_project_dir(project_dir),
        run_id=run_id,
    )
    if detail == "summary":
        return _summarize_show_run(result)
    return result


def _summarize_show_run(full_output: str) -> str:
    """Extract condensed view from full run output."""
    lines = full_output.strip().split("\n")
    summary_lines = []
    in_per_fold = False
    for line in lines:
        stripped = line.strip()
        # Detect per-fold breakdown section and skip it for summary
        if "per-fold" in stripped.lower() or "per fold" in stripped.lower():
            in_per_fold = True
            continue
        if in_per_fold:
            # Stop skipping when we hit the next section header
            if stripped.startswith("#") and "per-fold" not in stripped.lower():
                in_per_fold = False
            else:
                continue
        summary_lines.append(line)
    if not summary_lines or all(ln.strip() == "" for ln in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_compare_runs(*, run_ids, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    if not run_ids or len(run_ids) < 2:
        return "**Error**: `run_ids` must be a list of 2 run IDs to compare."
    run_a, run_b = run_ids[0], run_ids[1]

    return cw.compare_runs(
        resolve_project_dir(project_dir),
        run_id_a=run_a,
        run_id_b=run_b,
    )


def _handle_compare_latest(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.compare_runs(resolve_project_dir(project_dir), latest=True)


def _format_run_comparison(id_a: str, result_a: str, id_b: str, result_b: str) -> str:
    """Format two run results as a side-by-side comparison."""
    lines = [f"## Run Comparison: `{id_a}` vs `{id_b}`\n"]

    metrics_a = _extract_metrics_from_output(result_a)
    metrics_b = _extract_metrics_from_output(result_b)

    all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))

    if all_keys:
        lines.append("| Metric | `{}` | `{}` | Delta |".format(id_a, id_b))
        lines.append("|--------|------|------|-------|")
        for key in all_keys:
            val_a = metrics_a.get(key, "—")
            val_b = metrics_b.get(key, "—")
            delta = ""
            try:
                fa, fb = float(val_a), float(val_b)
                diff = fb - fa
                delta = f"{diff:+.4f}"
            except (ValueError, TypeError):
                pass
            lines.append(f"| {key} | {val_a} | {val_b} | {delta} |")
    else:
        lines.append(f"### Run `{id_a}`\n{result_a}\n")
        lines.append(f"### Run `{id_b}`\n{result_b}")

    return "\n".join(lines)


def _extract_metrics_from_output(output: str) -> dict[str, str]:
    """Extract key=value metric pairs from run output text."""
    metrics = {}
    for line in output.split("\n"):
        stripped = line.strip()
        # Match patterns like "- Brier: 0.1354" or "| brier | 0.1354 |"
        if ":" in stripped and not stripped.startswith("#"):
            parts = stripped.lstrip("- ").split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                val = parts[1].strip()
                # Only keep numeric-looking values
                try:
                    float(val)
                    metrics[key] = val
                except ValueError:
                    pass
        elif "|" in stripped:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if len(cells) >= 2:
                try:
                    float(cells[1])
                    metrics[cells[0].lower()] = cells[1]
                except (ValueError, IndexError):
                    pass
    return metrics


def _handle_compare_targets(*, ctx=None, project_dir, **_kwargs):
    """Compare backtest performance across all configured target profiles."""
    from harnessml.core.runner import config_writer as cw

    proj = resolve_project_dir(project_dir)

    # Load target profiles from pipeline.yaml
    import yaml
    config_path = proj / "config" / "pipeline.yaml"
    if not config_path.exists():
        return "**Error**: No pipeline.yaml found."

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    targets = cfg.get("data", {}).get("targets", {})
    if not targets:
        return "**Error**: No target profiles configured. Use configure(action='add_target') first."

    original_target = cfg.get("data", {}).get("target_column")
    results = {}

    for i, (target_name, target_def) in enumerate(targets.items()):
        if ctx is not None:
            ctx.report_progress(
                i, len(targets),
                f"Running backtest for target '{target_name}'..."
            )

        # Set active target
        cw.set_active_target(proj, target_name)

        # Run backtest
        try:
            def _progress_callback(current, total, message, _tn=target_name):
                if ctx is not None:
                    ctx.report_progress(current, total, f"[{_tn}] {message}")

            result = cw.run_backtest(proj, on_progress=_progress_callback)
            # Try to extract metrics from the result string
            metrics = _extract_metrics_from_output(result)
            # Convert string values to floats
            float_metrics = {}
            for k, v in metrics.items():
                try:
                    float_metrics[k] = float(v)
                except ValueError:
                    float_metrics[k] = v
            results[target_name] = float_metrics
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
            logger.exception("compare_targets backtest failed", action="compare_targets")
            results[target_name] = {"error": str(e)}

    # Restore original target
    if original_target:
        import yaml as _yaml
        config_path2 = proj / "config" / "pipeline.yaml"
        with open(config_path2) as f:
            cfg2 = _yaml.safe_load(f) or {}
        cfg2.setdefault("data", {})["target_column"] = original_target
        with open(config_path2, "w") as f:
            _yaml.dump(cfg2, f, default_flow_style=False, sort_keys=False)

    if ctx is not None:
        ctx.report_progress(len(targets), len(targets), "Comparison complete.")

    return cw.format_target_comparison(results)


def _handle_explain(*, name, run_id, top_n, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    return cw.explain_model(
        resolve_project_dir(project_dir),
        name=name,
        run_id=run_id,
        top_n=top_n or 10,
    )


def _handle_inspect_predictions(*, run_id, mode, top_n, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    return cw.inspect_predictions(
        resolve_project_dir(project_dir),
        run_id=run_id,
        mode=mode or "worst",
        top_n=top_n or 10,
    )


def _handle_export_notebook(*, destination, output_path, project_dir, **_kwargs):
    err = validate_required(destination, "destination")
    if err:
        return err
    err = validate_enum(destination, {"colab", "kaggle", "local"}, "destination")
    if err:
        return err
    from harnessml.core.runner.scaffold.notebook import generate_notebook

    pdir = resolve_project_dir(project_dir)
    out = None
    if output_path:
        from pathlib import Path
        out = Path(output_path)

    nb_path = generate_notebook(pdir, destination=destination, output_path=out)
    return f"Notebook generated: `{nb_path}`"


def _handle_progress(*, project_dir, **_kwargs):
    import yaml
    from harnessml.core.runner.workflow.tracker import WorkflowTracker

    proj = resolve_project_dir(project_dir)
    config_dir = proj / "config"
    pipeline_path = config_dir / "pipeline.yaml"

    workflow_config = {}
    if pipeline_path.exists():
        try:
            data = yaml.safe_load(pipeline_path.read_text()) or {}
            workflow_config = data.get("workflow", {})
        except yaml.YAMLError:
            pass

    tracker = WorkflowTracker(proj, workflow_config=workflow_config)
    status = tracker.get_status()
    return status.format_markdown()


def _handle_clear_cache(*, project_dir, **_kwargs):
    """Clear the prediction cache.  Useful after data shape changes (e.g. drop_rows)."""

    from harnessml.core.runner.training.prediction_cache import PredictionCache

    pdir = resolve_project_dir(project_dir)
    cache_dir = pdir / ".cache" / "predictions"
    if not cache_dir.exists():
        return "No prediction cache found — nothing to clear."
    cache = PredictionCache(cache_dir)
    removed = cache.clear()
    return f"Prediction cache cleared — {removed} cached entry/entries removed."


def _handle_model_correlation(*, run_id=None, project_dir, **_kwargs):
    from harnessml.core.runner.config_writer.pipeline import model_correlation

    return model_correlation(resolve_project_dir(project_dir), run_id=run_id)


def _handle_residual_analysis(*, feature=None, run_id=None, n_bins=None, project_dir, **_kwargs):
    from harnessml.core.runner.config_writer.pipeline import residual_analysis

    return residual_analysis(
        resolve_project_dir(project_dir),
        feature=feature,
        run_id=run_id,
        n_bins=int(n_bins) if n_bins is not None else 10,
    )


@tool_group("pipeline", description="Run and analyze the ML pipeline.")
class PipelineGroup:

    @action("progress", description="Show workflow progress.")
    def progress(self, *, project_dir=None, **kw):
        return _handle_progress(project_dir=project_dir, **kw)

    @action("run_backtest", description="Run a backtest.")
    def run_backtest(self, *, experiment_id=None, variant=None, fold_values=None,
                     ctx=None, project_dir=None, **kw):
        return _handle_run_backtest(experiment_id=experiment_id, variant=variant,
                                    fold_values=fold_values, ctx=ctx, project_dir=project_dir, **kw)

    @action("predict", description="Generate predictions.", requires=["fold_value"])
    def predict(self, *, fold_value=None, run_id=None, variant=None, project_dir=None, **kw):
        return _handle_predict(fold_value=fold_value, run_id=run_id, variant=variant, project_dir=project_dir, **kw)

    @action("diagnostics", description="Show run diagnostics.")
    def diagnostics(self, *, run_id=None, detail=None, project_dir=None, **kw):
        return _handle_diagnostics(run_id=run_id, detail=detail, project_dir=project_dir, **kw)

    @action("list_runs", description="List all runs.")
    def list_runs(self, *, project_dir=None, **kw):
        return _handle_list_runs(project_dir=project_dir, **kw)

    @action("show_run", description="Show details of a run.")
    def show_run(self, *, run_id=None, detail=None, project_dir=None, **kw):
        return _handle_show_run(run_id=run_id, detail=detail, project_dir=project_dir, **kw)

    @action("compare_runs", description="Compare two runs.")
    def compare_runs(self, *, run_ids=None, project_dir=None, **kw):
        return _handle_compare_runs(run_ids=run_ids, project_dir=project_dir, **kw)

    @action("compare_latest", description="Compare two most recent runs.")
    def compare_latest(self, *, project_dir=None, **kw):
        return _handle_compare_latest(project_dir=project_dir, **kw)

    @action("compare_targets", description="Compare across target profiles.")
    def compare_targets(self, *, ctx=None, project_dir=None, **kw):
        return _handle_compare_targets(ctx=ctx, project_dir=project_dir, **kw)

    @action("explain", description="Explain model predictions.")
    def explain(self, *, name=None, run_id=None, top_n=None, project_dir=None, **kw):
        return _handle_explain(name=name, run_id=run_id, top_n=top_n, project_dir=project_dir, **kw)

    @action("inspect_predictions", description="Inspect predictions.")
    def inspect_predictions(self, *, run_id=None, mode=None, top_n=None, project_dir=None, **kw):
        return _handle_inspect_predictions(run_id=run_id, mode=mode, top_n=top_n, project_dir=project_dir, **kw)

    @action("export_notebook", description="Export pipeline as notebook.", requires=["destination"])
    def export_notebook(self, *, destination=None, output_path=None, project_dir=None, **kw):
        return _handle_export_notebook(destination=destination, output_path=output_path, project_dir=project_dir, **kw)

    @action("clear_cache", description="Clear prediction cache.")
    def clear_cache(self, *, project_dir=None, **kw):
        return _handle_clear_cache(project_dir=project_dir, **kw)

    @action("model_correlation", description="Show model prediction correlations.")
    def model_correlation(self, *, run_id=None, project_dir=None, **kw):
        return _handle_model_correlation(run_id=run_id, project_dir=project_dir, **kw)

    @action("residual_analysis", description="Analyze prediction residuals.")
    def residual_analysis(self, *, feature=None, run_id=None, n_bins=None, project_dir=None, **kw):
        return _handle_residual_analysis(feature=feature, run_id=run_id, n_bins=n_bins, project_dir=project_dir, **kw)
