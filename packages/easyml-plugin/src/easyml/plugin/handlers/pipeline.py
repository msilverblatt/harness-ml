"""Handler for pipeline tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir
from easyml.plugin.handlers._validation import validate_enum, validate_required


def _handle_run_backtest(*, experiment_id, variant, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.run_backtest(
        resolve_project_dir(project_dir),
        experiment_id=experiment_id,
        variant=variant,
    )


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


def _handle_diagnostics(*, run_id, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.show_diagnostics(
        resolve_project_dir(project_dir),
        run_id=run_id,
    )


def _handle_list_runs(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.list_runs(resolve_project_dir(project_dir))


def _handle_show_run(*, run_id, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.show_run(
        resolve_project_dir(project_dir),
        run_id=run_id,
    )


ACTIONS = {
    "run_backtest": _handle_run_backtest,
    "predict": _handle_predict,
    "diagnostics": _handle_diagnostics,
    "list_runs": _handle_list_runs,
    "show_run": _handle_show_run,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a pipeline action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    return ACTIONS[action](**kwargs)
