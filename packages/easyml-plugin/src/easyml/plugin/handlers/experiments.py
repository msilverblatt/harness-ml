"""Handler for manage_experiments tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import validate_enum, validate_required


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


def _handle_run(*, experiment_id, primary_metric, variant, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(experiment_id, "experiment_id")
    if err:
        return err
    return cw.run_experiment(
        resolve_project_dir(project_dir),
        experiment_id,
        primary_metric=primary_metric,
        variant=variant,
    )


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


def _handle_quick_run(*, description, overlay, hypothesis, primary_metric, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(description, "description")
    if err:
        return err
    err = validate_required(overlay, "overlay")
    if err:
        return err
    return cw.quick_run_experiment(
        resolve_project_dir(project_dir),
        description,
        overlay,
        hypothesis=hypothesis,
        primary_metric=primary_metric,
    )


def _handle_explore(*, search_space, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(search_space, "search_space")
    if err:
        return err
    parsed = parse_json_param(search_space)
    return cw.run_exploration(
        resolve_project_dir(project_dir),
        parsed,
    )


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


ACTIONS = {
    "create": _handle_create,
    "write_overlay": _handle_write_overlay,
    "run": _handle_run,
    "promote": _handle_promote,
    "quick_run": _handle_quick_run,
    "explore": _handle_explore,
    "promote_trial": _handle_promote_trial,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_experiments action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    return ACTIONS[action](**kwargs)
