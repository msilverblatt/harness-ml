"""Handler for configure tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import validate_enum


def _handle_init(*, project_name, task, target_column, key_columns, time_column, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.scaffold_init(
        resolve_project_dir(project_dir, allow_missing=True),
        project_name,
        task=task or "classification",
        target_column=target_column or "result",
        key_columns=key_columns,
        time_column=time_column,
    )


def _handle_ensemble(*, method, temperature, exclude_models, calibration, pre_calibration, prior_feature, spline_prob_max, spline_n_bins, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    parsed_pre_cal = parse_json_param(pre_calibration)
    kw = dict(
        method=method,
        temperature=temperature,
        exclude_models=exclude_models,
        calibration=calibration,
        pre_calibration=parsed_pre_cal,
    )
    if prior_feature is not None:
        kw["prior_feature"] = prior_feature
    if spline_prob_max is not None:
        kw["spline_prob_max"] = spline_prob_max
    if spline_n_bins is not None:
        kw["spline_n_bins"] = spline_n_bins
    return cw.configure_ensemble(resolve_project_dir(project_dir), **kw)


def _handle_backtest(*, cv_strategy, seasons, metrics, min_train_folds, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.configure_backtest(
        resolve_project_dir(project_dir),
        cv_strategy=cv_strategy,
        seasons=seasons,
        metrics=metrics,
        min_train_folds=min_train_folds,
    )


def _handle_show(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.show_config(resolve_project_dir(project_dir))


def _handle_check_guardrails(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.check_guardrails(resolve_project_dir(project_dir))


def _handle_exclude_columns(*, add_columns, remove_columns, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.configure_exclude_columns(
        resolve_project_dir(project_dir),
        add_columns=add_columns,
        remove_columns=remove_columns,
    )


def _handle_set_denylist(*, add_columns, remove_columns, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.configure_denylist(
        resolve_project_dir(project_dir),
        add_columns=add_columns,
        remove_columns=remove_columns,
    )


ACTIONS = {
    "init": _handle_init,
    "ensemble": _handle_ensemble,
    "backtest": _handle_backtest,
    "show": _handle_show,
    "check_guardrails": _handle_check_guardrails,
    "exclude_columns": _handle_exclude_columns,
    "set_denylist": _handle_set_denylist,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a configure action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    return ACTIONS[action](**kwargs)
