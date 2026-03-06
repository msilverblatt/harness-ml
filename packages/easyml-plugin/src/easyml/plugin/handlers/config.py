"""Handler for configure tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum, collect_hints, format_response_with_hints,
)


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


def _handle_backtest(*, cv_strategy, fold_values, metrics, min_train_folds, fold_column, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.configure_backtest(
        resolve_project_dir(project_dir),
        cv_strategy=cv_strategy,
        fold_values=fold_values,
        metrics=metrics,
        min_train_folds=min_train_folds,
        fold_column=fold_column,
    )


def _handle_show(*, detail, section, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    full_output = cw.show_config(resolve_project_dir(project_dir))

    if section:
        return _extract_section(full_output, section)
    if detail == "summary":
        return _summarize_config(full_output)
    return full_output


def _extract_section(full_output: str, section: str) -> str:
    """Extract a specific YAML section from config output."""
    import yaml

    # The output may be markdown-wrapped YAML; try to extract the YAML block
    yaml_content = _extract_yaml_block(full_output)
    if not yaml_content:
        # Fallback: try parsing the whole thing as YAML
        yaml_content = full_output

    try:
        cfg = yaml.safe_load(yaml_content)
        if not isinstance(cfg, dict):
            return full_output
        if section not in cfg:
            available = ", ".join(sorted(cfg.keys()))
            return f"**Error**: Section `{section}` not found. Available: {available}"
        section_data = {section: cfg[section]}
        return f"## Config: `{section}`\n\n```yaml\n{yaml.dump(section_data, default_flow_style=False, sort_keys=False)}```"
    except yaml.YAMLError:
        # Fallback to text-based extraction
        return _extract_section_text(full_output, section)


def _extract_yaml_block(text: str) -> str | None:
    """Extract YAML content from a markdown code block."""
    lines = text.split("\n")
    in_block = False
    yaml_lines = []
    for line in lines:
        if line.strip().startswith("```yaml") or line.strip().startswith("```yml"):
            in_block = True
            continue
        if in_block and line.strip() == "```":
            break
        if in_block:
            yaml_lines.append(line)
    return "\n".join(yaml_lines) if yaml_lines else None


def _extract_section_text(full_output: str, section: str) -> str:
    """Text-based fallback for section extraction."""
    lines = full_output.split("\n")
    section_lines = []
    in_section = False
    section_indent = None

    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith(f"{section}:") or stripped.startswith(f"  {section}:"):
            in_section = True
            section_indent = len(line) - len(line.lstrip())
            section_lines.append(stripped)
            continue
        if in_section:
            if stripped == "" or stripped.startswith(" "):
                current_indent = len(line) - len(line.lstrip()) if stripped else section_indent + 1
                if current_indent > section_indent or stripped == "":
                    section_lines.append(stripped)
                else:
                    break
            else:
                break

    if section_lines:
        return f"## Config: `{section}`\n\n```yaml\n" + "\n".join(section_lines) + "\n```"
    return f"**Error**: Section `{section}` not found in config output."


def _summarize_config(full_output: str) -> str:
    """Return a condensed config summary with key settings only."""
    import yaml

    yaml_content = _extract_yaml_block(full_output)
    if not yaml_content:
        yaml_content = full_output

    try:
        cfg = yaml.safe_load(yaml_content)
        if not isinstance(cfg, dict):
            return full_output
    except yaml.YAMLError:
        return full_output

    lines = ["## Config Summary\n"]

    # Project info
    if "project" in cfg:
        proj = cfg["project"]
        lines.append(f"- **Project**: {proj.get('name', '?')}")
        lines.append(f"- **Task**: {proj.get('task', '?')}")
        lines.append(f"- **Target**: {proj.get('target_column', '?')}")

    # Model count
    models = cfg.get("models", {})
    active_count = sum(1 for m in models.values() if isinstance(m, dict) and m.get("active", True))
    lines.append(f"- **Models**: {active_count} active / {len(models)} total")

    # Ensemble
    ens = cfg.get("ensemble", {})
    if ens:
        lines.append(f"- **Ensemble**: method={ens.get('method', '?')}, calibration={ens.get('calibration', 'none')}")

    # Backtest
    bt = cfg.get("backtest", {})
    if bt:
        fold_values = bt.get("fold_values", [])
        lines.append(f"- **Backtest**: {len(fold_values)} folds, cv={bt.get('cv_strategy', '?')}")

    # Features count
    features = cfg.get("features", {})
    if isinstance(features, dict):
        lines.append(f"- **Features**: {len(features)} defined")

    return "\n".join(lines)


def _handle_check_guardrails(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.check_guardrails(resolve_project_dir(project_dir))


def _handle_update_data(*, target_column, key_columns, time_column, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.update_data_config(
        resolve_project_dir(project_dir),
        target_column=target_column,
        key_columns=key_columns,
        time_column=time_column,
    )


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


def _handle_add_target(*, name, target_column, task, metrics, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    if not name:
        return "**Error**: `name` is required for add_target."
    if not target_column:
        return "**Error**: `target_column` is required for add_target."

    parsed_metrics = parse_json_param(metrics) if isinstance(metrics, str) else metrics
    return cw.add_target(
        resolve_project_dir(project_dir),
        name,
        column=target_column,
        task=task or "binary",
        metrics=parsed_metrics,
    )


def _handle_list_targets(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.list_targets(resolve_project_dir(project_dir))


def _handle_set_target(*, name, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    if not name:
        return "**Error**: `name` is required for set_target."
    return cw.set_active_target(resolve_project_dir(project_dir), name)


ACTIONS = {
    "init": _handle_init,
    "update_data": _handle_update_data,
    "ensemble": _handle_ensemble,
    "backtest": _handle_backtest,
    "show": _handle_show,
    "check_guardrails": _handle_check_guardrails,
    "exclude_columns": _handle_exclude_columns,
    "set_denylist": _handle_set_denylist,
    "add_target": _handle_add_target,
    "list_targets": _handle_list_targets,
    "set_target": _handle_set_target,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a configure action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="config", **kwargs)
    return format_response_with_hints(result, hints)
