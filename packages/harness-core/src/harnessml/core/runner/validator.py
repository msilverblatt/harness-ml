"""YAML loading and validation against ProjectConfig schema.

Loads a split config directory (pipeline.yaml, models.yaml, ensemble.yaml,
etc.), applies overlays, and validates against :class:`ProjectConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from harnessml.core.config.merge import deep_merge, resolve_feature_mutations
from harnessml.core.runner.schema import ProjectConfig
from pydantic import ValidationError

# Top-level pipeline.yaml keys that map directly to ProjectConfig fields.
# These are passed through as-is during multi-section parsing.
_PIPELINE_PASSTHROUGH_KEYS = {
    "data", "backtest", "models", "ensemble",
    "sources", "experiments", "guardrails", "server",
    "feature_config",
}
# The "features" key in pipeline.yaml maps to "feature_config" in ProjectConfig
# (pipeline-level feature settings, NOT FeatureDecl registrations).
_PIPELINE_FEATURES_KEY = "features"


@dataclass
class ValidationResult:
    """Outcome of validating a project config directory."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: ProjectConfig | None = None

    def format(self) -> str:
        """Return a human-readable string summarising errors and warnings."""
        lines: list[str] = []
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if not lines:
            return ""
        return "\n".join(lines)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict for empty files."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve_variant_path(config_dir: Path, filename: str, variant: str | None) -> Path:
    """Resolve a filename with optional variant suffix.

    For example, variant="w" turns "pipeline.yaml" into "pipeline_w.yaml"
    if that file exists.
    """
    base = config_dir / filename
    if variant is not None:
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        variant_name = f"{stem}_{variant}{suffix}"
        variant_path = config_dir / variant_name
        if variant_path.exists():
            return variant_path
    return base


def _load_section(config_dir: Path, filename: str, variant: str | None) -> dict:
    """Load a section YAML file if it exists, with variant resolution."""
    path = _resolve_variant_path(config_dir, filename, variant)
    if path.exists():
        return _load_yaml(path)
    return {}


def _load_models_dir(config_dir: Path, variant: str | None) -> dict:
    """Load all YAML files from a models/ subdirectory, merging them together."""
    models_dir = config_dir / "models"
    if not models_dir.is_dir():
        return {}

    merged: dict = {}
    for yaml_file in sorted(models_dir.glob("*.yaml")):
        # Apply variant resolution per-file
        data = _load_yaml(yaml_file)
        merged = deep_merge(merged, data)
    return merged


def _validate_views(config: ProjectConfig, warnings: list[str]) -> list[str]:
    """Validate view DAG for consistency.

    Returns list of error strings (empty = valid).
    """
    errors: list[str] = []
    data = config.data

    if not data.views:
        return errors

    # Collect all valid names (sources + views)
    source_names = set(data.sources.keys())
    view_names = set(data.views.keys())
    all_names = source_names | view_names

    # 1. Check each view's source reference
    for name, view_def in data.views.items():
        if view_def.source not in all_names:
            errors.append(
                f"View '{name}' references unknown source '{view_def.source}'. "
                f"Available: {sorted(all_names)}"
            )

        # 2. Check join/union other references in steps
        for i, step in enumerate(view_def.steps):
            if hasattr(step, 'other') and step.other not in all_names:
                errors.append(
                    f"View '{name}' step {i} ({step.op}) references unknown "
                    f"'{step.other}'. Available: {sorted(all_names)}"
                )

    # 3. Check for cycles using detect_cycle from dag.py
    # Build dependency graph: view -> set of views/sources it depends on
    deps: dict[str, set[str]] = {}
    for name, view_def in data.views.items():
        view_deps: set[str] = set()
        if view_def.source in view_names:
            view_deps.add(view_def.source)
        for step in view_def.steps:
            if hasattr(step, 'other') and step.other in view_names:
                view_deps.add(step.other)
        deps[name] = view_deps

    # Add views with no view-dependencies for completeness
    for name in view_names:
        if name not in deps:
            deps[name] = set()

    from harnessml.core.runner.dag import detect_cycle
    cycle = detect_cycle(deps)
    if cycle:
        errors.append(f"Cycle detected in view DAG: {cycle}")

    # 4. Check features_view references an existing view
    if data.features_view and data.features_view not in view_names:
        errors.append(
            f"features_view '{data.features_view}' is not a defined view. "
            f"Available views: {sorted(view_names)}"
        )

    # 5. Check group_by aggs use supported functions
    supported_aggs = {"mean", "sum", "count", "std", "min", "max", "median", "first", "last", "nunique", "var"}
    for name, view_def in data.views.items():
        for i, step in enumerate(view_def.steps):
            if step.op == "group_by":
                for col, agg_spec in step.aggs.items():
                    aggs_list = [agg_spec] if isinstance(agg_spec, str) else agg_spec
                    for agg in aggs_list:
                        if agg not in supported_aggs:
                            errors.append(
                                f"View '{name}' step {i}: unsupported aggregation '{agg}' "
                                f"for column '{col}'. Supported: {sorted(supported_aggs)}"
                            )

    return errors


def validate_project(
    config_dir: str | Path,
    overlay: dict | None = None,
    variant: str | None = None,
) -> ValidationResult:
    """Validate a project config directory.

    Parameters
    ----------
    config_dir:
        Path to the config directory containing pipeline.yaml and
        optional section files.
    overlay:
        Optional dict to deep-merge on top of the resolved config.
    variant:
        Optional variant suffix (e.g. ``"w"``).  Each file is tried
        with the variant suffix first (e.g. ``pipeline_w.yaml``), falling
        back to the base filename.

    Returns
    -------
    ValidationResult
        Contains ``valid``, ``errors``, ``warnings``, and the parsed
        ``config`` (None if validation failed).
    """
    config_dir = Path(config_dir)
    errors: list[str] = []
    warnings: list[str] = []

    # --- Check pipeline.yaml exists (required) ---
    pipeline_path = _resolve_variant_path(config_dir, "pipeline.yaml", variant)
    if not pipeline_path.exists():
        errors.append(
            f"Required file pipeline.yaml not found in {config_dir}"
        )
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # --- Load and parse pipeline.yaml (multi-section file) ---
    pipeline_raw: dict[str, Any] = _load_yaml(pipeline_path)
    merged: dict[str, Any] = {}

    # Extract known sections from pipeline.yaml into the merged dict.
    # Keys that map directly to ProjectConfig fields are passed through.
    for key in _PIPELINE_PASSTHROUGH_KEYS:
        if key in pipeline_raw:
            merged[key] = pipeline_raw[key]

    # The "features" key in pipeline.yaml maps to "feature_config" in
    # ProjectConfig (pipeline-level feature settings like first_period,
    # momentum_window).  This is distinct from the features.yaml file
    # which contains FeatureDecl registrations.
    if _PIPELINE_FEATURES_KEY in pipeline_raw:
        merged["feature_config"] = pipeline_raw[_PIPELINE_FEATURES_KEY]

    # Unknown top-level keys (e.g. "bracket") are silently ignored.

    # --- Load models from models.yaml and/or models/ subdirectory ---
    models_from_file = _load_section(config_dir, "models.yaml", variant)
    models_from_dir = _load_models_dir(config_dir, variant)

    if models_from_file:
        merged = deep_merge(merged, models_from_file)
    if models_from_dir:
        merged = deep_merge(merged, models_from_dir)

    # --- Load optional section files ---
    section_files = [
        "ensemble.yaml",
        "features.yaml",
        "sources.yaml",
        "experiments.yaml",
        "guardrails.yaml",
        "server.yaml",
    ]
    for section_file in section_files:
        section_data = _load_section(config_dir, section_file, variant)
        if section_data:
            merged = deep_merge(merged, section_data)

    # --- Apply overlay ---
    if overlay is not None:
        merged = deep_merge(merged, overlay)
        merged = resolve_feature_mutations(merged)

    # --- Validate against ProjectConfig ---
    try:
        config = ProjectConfig(**merged)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            msg = err["msg"]
            # Include the input value for clarity
            inp = err.get("input")
            if inp is not None:
                errors.append(f"{loc}: {msg} (got {inp!r})")
            else:
                errors.append(f"{loc}: {msg}")
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # --- Validate view DAG ---
    view_errors = _validate_views(config, warnings)
    if view_errors:
        errors.extend(view_errors)
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    return ValidationResult(valid=True, errors=errors, warnings=warnings, config=config)
