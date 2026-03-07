"""Handler for manage_experiments tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum, validate_required, collect_hints, format_response_with_hints,
)


def _handle_create(*, description, hypothesis, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(description, "description")
    if err:
        return err
    return cw.experiment_create(
        resolve_project_dir(project_dir),
        description,
        hypothesis=hypothesis,
    )


def _handle_write_overlay(*, experiment_id, overlay, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

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


async def _handle_run(*, experiment_id, primary_metric, variant, ctx, project_dir, **_kwargs):
    import asyncio
    from easyml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err

    loop = asyncio.get_running_loop()

    def _progress_callback(current, total, message):
        import logging
        logging.getLogger(__name__).info("Experiment progress: %s", message)
        if ctx is not None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=current, total=total, message=message),
                loop,
            )

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message=f"Running experiment {experiment_id}...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.run_experiment(
            resolve_project_dir(project_dir),
            experiment_id,
            primary_metric=primary_metric,
            variant=variant,
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Experiment complete.")

    return result


def _handle_promote(*, experiment_id, primary_metric, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    return cw.promote_experiment(
        resolve_project_dir(project_dir),
        experiment_id,
        primary_metric=primary_metric,
    )


async def _handle_quick_run(*, description, overlay, hypothesis, primary_metric, ctx, project_dir, **_kwargs):
    import asyncio
    from easyml.core.runner import config_writer as cw

    err = validate_required(description, "description")
    if err:
        return err
    err = validate_required(overlay, "overlay")
    if err:
        return err

    loop = asyncio.get_running_loop()

    def _progress_callback(current, total, message):
        import logging
        logging.getLogger(__name__).info("Experiment progress: %s", message)
        if ctx is not None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=current, total=total, message=message),
                loop,
            )

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
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Experiment complete.")

    return result


async def _handle_explore(*, search_space, detail, ctx, project_dir, **_kwargs):
    import asyncio
    from easyml.core.runner import config_writer as cw

    err = validate_required(search_space, "search_space")
    if err:
        return err
    parsed = parse_json_param(search_space)

    loop = asyncio.get_running_loop()

    def _progress_callback(current, total, message):
        """Sync callback running in thread — schedules async progress on event loop."""
        import logging
        logging.getLogger(__name__).info("Exploration progress: %s", message)
        if ctx is not None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=current, total=total, message=message),
                loop,
            )

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message="Starting exploration...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.run_exploration(
            resolve_project_dir(project_dir),
            parsed,
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Exploration complete.")

    if detail == "summary":
        return _summarize_exploration(result)
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

    if not summary_lines or all(l.strip() == "" for l in summary_lines):
        return full_output
    return "\n".join(summary_lines)


def _handle_promote_trial(*, experiment_id, trial, primary_metric, hypothesis, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

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
    from easyml.core.runner import config_writer as cw

    if not experiment_ids or len(experiment_ids) < 2:
        return "**Error**: `experiment_ids` must be a list of 2 experiment IDs to compare."

    id_a, id_b = experiment_ids[0], experiment_ids[1]
    proj = resolve_project_dir(project_dir)

    # Run both experiments and compare results
    # For now we compare by running each and formatting side by side
    try:
        result_a = cw.run_experiment(proj, id_a)
    except Exception as e:
        return f"**Error** running experiment `{id_a}`: {e}"
    try:
        result_b = cw.run_experiment(proj, id_b)
    except Exception as e:
        return f"**Error** running experiment `{id_b}`: {e}"

    return _format_experiment_comparison(id_a, result_a, id_b, result_b)


def _format_experiment_comparison(id_a: str, result_a: str, id_b: str, result_b: str) -> str:
    """Format two experiment results as a side-by-side comparison."""
    lines = [f"## Experiment Comparison: `{id_a}` vs `{id_b}`\n"]
    lines.append(f"### `{id_a}`\n{result_a}\n")
    lines.append(f"### `{id_b}`\n{result_b}")
    return "\n".join(lines)


def _handle_journal(*, last_n, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw
    return cw.show_journal(resolve_project_dir(project_dir), last_n=last_n or 20)


def _handle_log_result(*, experiment_id, description, hypothesis, verdict, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw
    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    return cw.log_experiment_result(
        resolve_project_dir(project_dir),
        experiment_id,
        description=description or "",
        hypothesis=hypothesis or "",
        verdict=verdict or "",
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
