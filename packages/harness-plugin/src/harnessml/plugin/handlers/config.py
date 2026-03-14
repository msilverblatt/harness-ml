"""Handler for configure tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    validate_required,
)
from protomcp import action, tool_group


def _handle_init(*, project_name, task, target_column, key_columns, time_column, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    resolved = resolve_project_dir(project_dir, allow_missing=True)
    defaulted_task = task is None
    result = cw.scaffold_init(
        resolved,
        project_name,
        task=task or "classification",
        target_column=target_column or "result",
        key_columns=key_columns,
        time_column=time_column,
    )
    # Register project in the global registry so Studio can find it
    from harnessml.plugin.handlers._common import get_active_emitter
    emitter = get_active_emitter()
    if emitter is not None:
        emitter.set_project(str(resolved))
    if defaulted_task and not result.startswith("**Error"):
        result += "\n\n**Note**: No task specified — defaulted to 'classification' (binary). Options: regression, multiclass, ranking, survival, probabilistic."
    return result


def _handle_ensemble(*, method, temperature, exclude_models, calibration, pre_calibration, prior_feature, spline_prob_max, spline_n_bins, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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


def _handle_backtest(*, cv_strategy, fold_values, metrics, min_train_folds, fold_column,
                     n_folds, window_size, group_column, eval_filter, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.configure_backtest(
        resolve_project_dir(project_dir),
        cv_strategy=cv_strategy,
        fold_values=fold_values,
        metrics=metrics,
        min_train_folds=min_train_folds,
        fold_column=fold_column,
        n_folds=n_folds,
        window_size=window_size,
        group_column=group_column,
        eval_filter=eval_filter,
    )


def _handle_show(*, detail, section, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    full_output = cw.show_config(resolve_project_dir(project_dir))

    if section:
        return _extract_section(full_output, section)
    if detail == "summary":
        return _summarize_config(full_output)
    return full_output


def _extract_section(full_output: str, section: str) -> str:
    """Extract a specific section from config output.

    The show_config output uses markdown headers like ``### Backtest``.
    We first try matching those headers (title-cased), then fall back to
    YAML-based extraction for backwards compatibility.
    """
    import re

    import yaml

    # --- Primary path: match markdown ### headers ---
    title = section.replace("_", " ").title()
    # Match e.g. "### Backtest" or "### Models (5)" — header may have trailing info
    header_pattern = re.compile(r"^###\s+" + re.escape(title) + r"(?:\s.*)?$", re.MULTILINE)
    match = header_pattern.search(full_output)
    if match:
        start = match.start()
        # Find the next ### header (or end of string)
        next_header = re.search(r"^###\s+", full_output[match.end():], re.MULTILINE)
        if next_header:
            end = match.end() + next_header.start()
        else:
            end = len(full_output)
        return full_output[start:end].rstrip()

    # --- Fallback: YAML-based extraction ---
    yaml_content = _extract_yaml_block(full_output)
    if not yaml_content:
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
    from harnessml.core.runner import config_writer as cw

    return cw.check_guardrails(resolve_project_dir(project_dir))


def _handle_update_data(*, target_column, key_columns, time_column, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.update_data_config(
        resolve_project_dir(project_dir),
        target_column=target_column,
        key_columns=key_columns,
        time_column=time_column,
    )


def _handle_exclude_columns(*, add_columns, remove_columns, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.configure_exclude_columns(
        resolve_project_dir(project_dir),
        add_columns=add_columns,
        remove_columns=remove_columns,
    )


def _handle_set_denylist(*, add_columns, remove_columns, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.configure_denylist(
        resolve_project_dir(project_dir),
        add_columns=add_columns,
        remove_columns=remove_columns,
    )


def _handle_add_target(*, name, target_column, task, metrics, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    err = validate_required(target_column, "target_column")
    if err:
        return err

    parsed_metrics = parse_json_param(metrics) if isinstance(metrics, str) else metrics
    return cw.add_target(
        resolve_project_dir(project_dir),
        name,
        column=target_column,
        task=task or "binary",
        metrics=parsed_metrics,
    )


def _handle_list_targets(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.list_targets(resolve_project_dir(project_dir))


def _handle_set_target(*, name, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    return cw.set_active_target(resolve_project_dir(project_dir), name)


def _handle_studio(*, project_dir, **_kwargs):
    """Return the Studio dashboard URL."""
    import os
    port = int(os.environ.get("HARNESS_STUDIO_PORT", "8421"))
    base_url = f"http://localhost:{port}"
    if project_dir:
        from pathlib import Path
        project_name = Path(project_dir).resolve().name
        return f"**Harness Studio** → {base_url}/{project_name}/dashboard"
    return f"**Harness Studio** → {base_url}"


def _handle_suggest_cv(*, project_dir, **_kwargs):
    from harnessml.core.runner.config_writer.pipeline import suggest_cv

    return suggest_cv(resolve_project_dir(project_dir))


@tool_group("configure", description="Configure the ML pipeline.")
class ConfigureGroup:

    @action("init", description="Initialize a new project.")
    def init(self, *, project_name=None, task=None, target_column=None,
             key_columns=None, time_column=None, project_dir=None, **kw):
        return _handle_init(project_name=project_name, task=task, target_column=target_column,
                            key_columns=key_columns, time_column=time_column, project_dir=project_dir, **kw)

    @action("update_data", description="Update data configuration.")
    def update_data(self, *, target_column=None, key_columns=None, time_column=None,
                    project_dir=None, **kw):
        return _handle_update_data(target_column=target_column, key_columns=key_columns,
                                   time_column=time_column, project_dir=project_dir, **kw)

    @action("ensemble", description="Configure ensemble settings.")
    def ensemble(self, *, method=None, temperature=None, exclude_models=None,
                 calibration=None, pre_calibration=None, prior_feature=None,
                 spline_prob_max=None, spline_n_bins=None, project_dir=None, **kw):
        return _handle_ensemble(method=method, temperature=temperature,
                                exclude_models=exclude_models, calibration=calibration,
                                pre_calibration=pre_calibration, prior_feature=prior_feature,
                                spline_prob_max=spline_prob_max, spline_n_bins=spline_n_bins,
                                project_dir=project_dir, **kw)

    @action("backtest", description="Configure backtest settings.")
    def backtest(self, *, cv_strategy=None, fold_values=None, metrics=None,
                 min_train_folds=None, fold_column=None, n_folds=None,
                 window_size=None, group_column=None, eval_filter=None,
                 project_dir=None, **kw):
        return _handle_backtest(cv_strategy=cv_strategy, fold_values=fold_values,
                                metrics=metrics, min_train_folds=min_train_folds,
                                fold_column=fold_column, n_folds=n_folds,
                                window_size=window_size, group_column=group_column,
                                eval_filter=eval_filter, project_dir=project_dir, **kw)

    @action("show", description="Show pipeline configuration.")
    def show(self, *, detail=None, section=None, project_dir=None, **kw):
        return _handle_show(detail=detail, section=section, project_dir=project_dir, **kw)

    @action("check_guardrails", description="Check configuration guardrails.")
    def check_guardrails(self, *, project_dir=None, **kw):
        return _handle_check_guardrails(project_dir=project_dir, **kw)

    @action("exclude_columns", description="Configure excluded columns.")
    def exclude_columns(self, *, add_columns=None, remove_columns=None, project_dir=None, **kw):
        return _handle_exclude_columns(add_columns=add_columns, remove_columns=remove_columns, project_dir=project_dir, **kw)

    @action("set_denylist", description="Configure denylist columns.")
    def set_denylist(self, *, add_columns=None, remove_columns=None, project_dir=None, **kw):
        return _handle_set_denylist(add_columns=add_columns, remove_columns=remove_columns, project_dir=project_dir, **kw)

    @action("add_target", description="Add a target profile.", requires=["name", "target_column"])
    def add_target(self, *, name=None, target_column=None, task=None, metrics=None,
                   project_dir=None, **kw):
        return _handle_add_target(name=name, target_column=target_column, task=task,
                                  metrics=metrics, project_dir=project_dir, **kw)

    @action("list_targets", description="List target profiles.")
    def list_targets(self, *, project_dir=None, **kw):
        return _handle_list_targets(project_dir=project_dir, **kw)

    @action("set_target", description="Set active target.", requires=["name"])
    def set_target(self, *, name=None, project_dir=None, **kw):
        return _handle_set_target(name=name, project_dir=project_dir, **kw)

    @action("studio", description="Get Studio dashboard URL.")
    def studio(self, *, project_dir=None, **kw):
        return _handle_studio(project_dir=project_dir, **kw)

    @action("suggest_cv", description="Suggest CV strategy.")
    def suggest_cv(self, *, project_dir=None, **kw):
        return _handle_suggest_cv(project_dir=project_dir, **kw)
