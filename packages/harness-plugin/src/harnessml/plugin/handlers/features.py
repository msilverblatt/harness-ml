"""Handler for manage_features tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    validate_required,
)
from protomcp import action, tool_group


def _handle_add(*, name, formula, type, source, column, condition, pairwise_mode, category, description, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    if not formula and not type and not source and not condition:
        return (
            "**Error**: Provide at least one of: type, formula, source, or condition."
        )
    return cw.add_feature(
        resolve_project_dir(project_dir),
        name,
        formula,
        type=type,
        source=source,
        column=column,
        condition=condition,
        pairwise_mode=pairwise_mode,
        category=category,
        description=description,
    )


def _handle_add_batch(*, features, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.add_features_batch(resolve_project_dir(project_dir), parsed)


def _handle_test_transformations(*, features, test_interactions, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.test_feature_transformations(
        resolve_project_dir(project_dir),
        parsed,
        test_interactions=test_interactions,
    )


def _handle_discover(*, top_n, method, ctx=None, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    def _progress_callback(current, total, message):
        if ctx is not None:
            ctx.report_progress(current, total, message)

    if ctx is not None:
        ctx.report_progress(0, 1, "Starting feature discovery...")

    result = cw.discover_features(
        resolve_project_dir(project_dir),
        top_n=top_n,
        method=method,
        on_progress=_progress_callback,
    )

    if ctx is not None:
        ctx.report_progress(1, 1, "Feature discovery complete.")

    return result


def _handle_diversity(*, project_dir, **_kwargs):
    import yaml
    from harnessml.core.runner.features.diversity import format_diversity_report

    proj = resolve_project_dir(project_dir)
    config_path = proj / "config" / "pipeline.yaml"
    if not config_path.exists():
        return "**Error**: No pipeline.yaml found. Run `configure(action='init')` first."

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models", {})
    if not models:
        return "**Error**: No models configured. Add models first with `models(action='add', ...)`."

    return format_diversity_report(models)


def _handle_auto_search(*, features, search_types, top_n, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    parsed_features = parse_json_param(features) if features else None
    parsed_search_types = parse_json_param(search_types) if search_types else None
    return cw.auto_search_features(
        resolve_project_dir(project_dir),
        features=parsed_features,
        search_types=parsed_search_types,
        top_n=top_n,
    )


def _handle_prune(*, threshold=None, method=None, dry_run=None, project_dir, **_kwargs):
    from harnessml.core.runner.config_writer.features import prune_features

    return prune_features(
        resolve_project_dir(project_dir),
        threshold=float(threshold) if threshold is not None else 0.01,
        method=method or "xgboost",
        dry_run=dry_run if dry_run is not None else True,
    )


@tool_group("features", description="Manage ML features.")
class FeaturesGroup:

    @action("add", description="Add a feature.", requires=["name"])
    def add(self, *, name=None, formula=None, type=None, source=None, column=None,
            condition=None, pairwise_mode=None, category=None, description=None,
            project_dir=None, **kw):
        return _handle_add(name=name, formula=formula, type=type, source=source,
                           column=column, condition=condition, pairwise_mode=pairwise_mode,
                           category=category, description=description, project_dir=project_dir, **kw)

    @action("add_batch", description="Add multiple features.", requires=["features"])
    def add_batch(self, *, features=None, project_dir=None, **kw):
        return _handle_add_batch(features=features, project_dir=project_dir, **kw)

    @action("test_transformations", description="Test feature transformations.", requires=["features"])
    def test_transformations(self, *, features=None, test_interactions=None, project_dir=None, **kw):
        return _handle_test_transformations(features=features, test_interactions=test_interactions, project_dir=project_dir, **kw)

    @action("discover", description="Discover new features.")
    def discover(self, *, top_n=20, method="xgboost", ctx=None, project_dir=None, **kw):
        return _handle_discover(top_n=top_n, method=method, ctx=ctx, project_dir=project_dir, **kw)

    @action("diversity", description="Show feature diversity report.")
    def diversity(self, *, project_dir=None, **kw):
        return _handle_diversity(project_dir=project_dir, **kw)

    @action("auto_search", description="Auto-search for features.")
    def auto_search(self, *, features=None, search_types=None, top_n=None, project_dir=None, **kw):
        return _handle_auto_search(features=features, search_types=search_types, top_n=top_n, project_dir=project_dir, **kw)

    @action("prune", description="Prune low-importance features.")
    def prune(self, *, threshold=None, method=None, dry_run=None, project_dir=None, **kw):
        return _handle_prune(threshold=threshold, method=method, dry_run=dry_run, project_dir=project_dir, **kw)
