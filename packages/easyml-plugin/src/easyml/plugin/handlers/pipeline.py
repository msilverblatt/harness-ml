"""Handler for pipeline tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir
from easyml.plugin.handlers._validation import (
    validate_enum, validate_required, collect_hints, format_response_with_hints,
)


def _handle_run_backtest(*, experiment_id, variant, ctx, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    def _progress_callback(current, total, message):
        """Sync progress callback that logs fold progress."""
        import logging
        logging.getLogger(__name__).info("Backtest progress: %s", message)
        if ctx is not None:
            ctx.report_progress(progress=current, total=total, message=message)

    if ctx is not None:
        ctx.report_progress(progress=0, total=1, message="Starting backtest...")

    result = cw.run_backtest(
        resolve_project_dir(project_dir),
        experiment_id=experiment_id,
        variant=variant,
        on_progress=_progress_callback,
    )

    if ctx is not None:
        ctx.report_progress(progress=1, total=1, message="Backtest complete.")

    return result


def _handle_predict(*, season, run_id, variant, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(season, "season")
    if err:
        return err
    return cw.run_predict(
        resolve_project_dir(project_dir),
        season,
        run_id=run_id,
        variant=variant,
    )


def _handle_diagnostics(*, run_id, detail, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

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
    if not summary_lines or all(l.strip() == "" for l in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_list_runs(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.list_runs(resolve_project_dir(project_dir))


def _handle_show_run(*, run_id, detail, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

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
    in_per_season = False
    for line in lines:
        stripped = line.strip()
        # Detect per-season breakdown section and skip it for summary
        if "per-season" in stripped.lower() or "per season" in stripped.lower():
            in_per_season = True
            continue
        if in_per_season:
            # Stop skipping when we hit the next section header
            if stripped.startswith("#") and "per-season" not in stripped.lower():
                in_per_season = False
            else:
                continue
        summary_lines.append(line)
    if not summary_lines or all(l.strip() == "" for l in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_compare_runs(*, run_ids, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    if not run_ids or len(run_ids) < 2:
        return "**Error**: `run_ids` must be a list of 2 run IDs to compare."
    run_a, run_b = run_ids[0], run_ids[1]

    result_a = cw.show_run(resolve_project_dir(project_dir), run_id=run_a)
    result_b = cw.show_run(resolve_project_dir(project_dir), run_id=run_b)

    # Parse metrics from both runs for side-by-side comparison
    return _format_run_comparison(run_a, result_a, run_b, result_b)


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


ACTIONS = {
    "run_backtest": _handle_run_backtest,
    "predict": _handle_predict,
    "diagnostics": _handle_diagnostics,
    "list_runs": _handle_list_runs,
    "show_run": _handle_show_run,
    "compare_runs": _handle_compare_runs,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a pipeline action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="pipeline", **kwargs)
    return format_response_with_hints(result, hints)
