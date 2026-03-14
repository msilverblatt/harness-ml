"""Handler for manage_experiments tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import make_progress_callback, parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    collect_hints,
    format_response_with_hints,
    validate_enum,
    validate_required,
)


def _handle_create(*, description, hypothesis, parent_id=None, branching_reason="", phase="", project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(description, "description")
    if err:
        return err
    return cw.experiment_create(
        resolve_project_dir(project_dir),
        description,
        hypothesis=hypothesis,
        parent_id=parent_id,
        branching_reason=branching_reason or "",
        phase=phase or "",
    )


def _handle_write_overlay(*, experiment_id, overlay, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    err = validate_required(overlay, "overlay")
    if err:
        return err
    parsed = parse_json_param(overlay)
    return cw.write_overlay(
        resolve_project_dir(project_dir),
        experiment_id,
        parsed,
    )


async def _handle_run(*, experiment_id, primary_metric, variant, baseline_run_id=None, ctx, project_dir, **_kwargs):
    import asyncio

    from harnessml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err

    loop = asyncio.get_running_loop()
    _progress_callback = make_progress_callback(ctx, loop)

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message=f"Running experiment {experiment_id}...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.run_experiment(
            resolve_project_dir(project_dir),
            experiment_id,
            primary_metric=primary_metric,
            variant=variant,
            baseline_run_id=baseline_run_id,
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Experiment complete.")

    return result


def _handle_promote(*, experiment_id, primary_metric, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    return cw.promote_experiment(
        resolve_project_dir(project_dir),
        experiment_id,
        primary_metric=primary_metric,
    )


async def _handle_quick_run(*, description, overlay, hypothesis, primary_metric, baseline_run_id=None, ctx, project_dir, **_kwargs):
    import asyncio

    from harnessml.core.runner import config_writer as cw

    err = validate_required(description, "description")
    if err:
        return err
    err = validate_required(overlay, "overlay")
    if err:
        return err

    loop = asyncio.get_running_loop()
    _progress_callback = make_progress_callback(ctx, loop)

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message="Starting experiment...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.quick_run_experiment(
            resolve_project_dir(project_dir),
            description,
            overlay,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            baseline_run_id=baseline_run_id,
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Experiment complete.")

    return result


async def _handle_explore(*, search_space, detail, warm_start_from=None, ctx, project_dir, **_kwargs):
    import asyncio

    from harnessml.core.runner import config_writer as cw

    err = validate_required(search_space, "search_space")
    if err:
        return err
    parsed = parse_json_param(search_space)

    # Workflow phase gate: check if exploration is premature
    try:
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
        enforce = workflow_config.get("enforce_phases", False)
        warning = tracker.check_ready_for_tuning(enforce=enforce)
    except Exception as exc:
        if type(exc).__name__ == "WorkflowGateError":
            return f"**Blocked**: {exc}"
        warning = None

    loop = asyncio.get_running_loop()
    _progress_callback = make_progress_callback(ctx, loop)

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message="Starting exploration...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.run_exploration(
            resolve_project_dir(project_dir),
            parsed,
            on_progress=_progress_callback,
            warm_start_from=warm_start_from,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Exploration complete.")

    if detail == "summary":
        result = _summarize_exploration(result)

    if warning:
        result = f"{warning}\n\n---\n\n{result}"

    return result


def _summarize_exploration(full_output: str) -> str:
    """Return only the best trial from exploration output."""
    lines = full_output.strip().split("\n")
    summary_lines = []
    in_best_section = False
    in_all_trials = False

    for line in lines:
        stripped = line.strip().lower()
        # Capture header and best trial section
        if "best" in stripped and ("#" in stripped or "trial" in stripped):
            in_best_section = True
            in_all_trials = False
            summary_lines.append(line)
            continue
        if in_best_section:
            # Stop when we hit "all trials" or "parameter importance" sections
            if ("all trials" in stripped or "parameter importance" in stripped
                    or "ranked" in stripped):
                in_best_section = False
                in_all_trials = True
                continue
            summary_lines.append(line)
            continue
        if in_all_trials:
            # Skip all-trials and param-importance sections
            continue
        # Keep top-level headers and summary info
        if line.strip().startswith("#") or not in_all_trials:
            if "all trials" not in stripped and "parameter importance" not in stripped:
                summary_lines.append(line)

    if not summary_lines or all(ln.strip() == "" for ln in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_promote_trial(*, experiment_id, trial, primary_metric, hypothesis, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    return cw.promote_exploration_trial(
        resolve_project_dir(project_dir),
        experiment_id,
        trial=trial,
        primary_metric=primary_metric,
        hypothesis=hypothesis,
    )


def _handle_compare(*, experiment_ids, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    if not experiment_ids or len(experiment_ids) < 2:
        return "**Error**: `experiment_ids` must be a list of 2 experiment IDs to compare."

    proj = resolve_project_dir(project_dir)

    # Use journal-based comparison if available
    parsed_ids = parse_json_param(experiment_ids) if isinstance(experiment_ids, str) else experiment_ids
    return cw.compare_experiments(proj, parsed_ids)


def _handle_journal(*, last_n, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    return cw.show_journal(resolve_project_dir(project_dir), last_n=last_n or 20)


def _handle_log_result(*, experiment_id, description, hypothesis, conclusion, verdict, metrics=None, baseline_metrics=None, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    parsed_metrics = parse_json_param(metrics) if metrics else None
    parsed_baseline = parse_json_param(baseline_metrics) if baseline_metrics else None
    return cw.log_experiment_result(
        resolve_project_dir(project_dir),
        experiment_id,
        description=description or "",
        hypothesis=hypothesis or "",
        conclusion=conclusion or "",
        verdict=verdict or "",
        metrics=parsed_metrics,
        baseline_metrics=parsed_baseline,
    )


ACTIONS = {
    "create": _handle_create,
    "write_overlay": _handle_write_overlay,
    "run": _handle_run,
    "promote": _handle_promote,
    "quick_run": _handle_quick_run,
    "explore": _handle_explore,
    "promote_trial": _handle_promote_trial,
    "compare": _handle_compare,
    "journal": _handle_journal,
    "log_result": _handle_log_result,
}


async def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_experiments action."""
    import asyncio

    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    hints = collect_hints(action, tool="experiments", **kwargs)
    return format_response_with_hints(result, hints)
