"""Shared validation utilities for MCP handlers."""
from __future__ import annotations

from difflib import get_close_matches


def validate_enum(value: str, valid: set[str], param_name: str) -> str | None:
    """Validate value against allowed set, return error message or None."""
    if value in valid:
        return None
    closest = get_close_matches(value, sorted(valid), n=1, cutoff=0.6)
    msg = f"**Error**: Invalid `{param_name}` '{value}'. Valid: {', '.join(sorted(valid))}"
    if closest:
        msg += f"\n\nDid you mean **{closest[0]}**?"
    return msg


def validate_required(value, param_name: str) -> str | None:
    """Return error message if value is None/empty, else None."""
    if value is None or value == "":
        return f"**Error**: `{param_name}` is required."
    return None


# -----------------------------------------------------------------------
# Cross-parameter hints
# -----------------------------------------------------------------------


def collect_hints(action: str, tool: str = "", **kwargs) -> list[str]:
    """Generate helpful hints based on parameter combinations.

    These are non-blocking suggestions appended after successful operations
    to guide the user toward best practices.
    """
    hints = []

    if tool == "models":
        hints.extend(_model_hints(action, **kwargs))
    elif tool == "data":
        hints.extend(_data_hints(action, **kwargs))
    elif tool == "config":
        hints.extend(_config_hints(action, **kwargs))
    elif tool == "experiments":
        hints.extend(_experiment_hints(action, **kwargs))

    return hints


def _model_hints(action: str, **kwargs) -> list[str]:
    """Hints for models actions."""
    hints = []
    if action == "add":
        mode = kwargs.get("mode")
        if mode == "regressor" and kwargs.get("cdf_scale") is None:
            hints.append(
                "**Hint**: Regressor models often need `cdf_scale` to convert "
                "outputs to probabilities. Consider setting it (e.g. 15.0)."
            )
        if kwargs.get("include_in_ensemble") is True and not kwargs.get("features"):
            hints.append(
                "**Hint**: Model has no features specified. It will use all "
                "available features by default."
            )
        if kwargs.get("preset") and kwargs.get("params"):
            hints.append(
                "**Hint**: You specified both a `preset` and `params`. The params "
                "will override matching preset defaults."
            )
    if action == "update":
        if kwargs.get("active") is False and kwargs.get("include_in_ensemble") is None:
            hints.append(
                "**Hint**: Setting `active=false` disables training. If you only "
                "want to exclude from the ensemble, use `include_in_ensemble=false` instead."
            )
    return hints


def _data_hints(action: str, **kwargs) -> list[str]:
    """Hints for data actions."""
    hints = []
    if action == "add" and not kwargs.get("join_on"):
        hints.append(
            "**Hint**: No `join_on` specified. Data will be appended as new rows. "
            "If merging with existing data, set `join_on` to the key columns."
        )
    if action == "add_view" and not kwargs.get("steps"):
        hints.append(
            "**Hint**: View created with no transform steps. It will pass through "
            "the source data unchanged. Add steps later with `update_view`."
        )
    return hints


def _config_hints(action: str, **kwargs) -> list[str]:
    """Hints for configure actions."""
    hints = []
    if action == "ensemble":
        cal = kwargs.get("calibration")
        if cal == "spline" and kwargs.get("spline_n_bins") is None:
            hints.append(
                "**Hint**: Spline calibration uses 20 bins by default. "
                "Adjust with `spline_n_bins` if you have fewer samples."
            )
    if action == "backtest":
        fold_values = kwargs.get("fold_values")
        min_folds = kwargs.get("min_train_folds")
        if fold_values and min_folds and len(fold_values) <= min_folds:
            hints.append(
                "**Hint**: `min_train_folds` is >= number of fold values. "
                "Early folds will have no training data. Consider reducing it."
            )
    return hints


def _experiment_hints(action: str, **kwargs) -> list[str]:
    """Hints for experiments actions."""
    hints = []
    if action == "quick_run" and not kwargs.get("hypothesis"):
        hints.append(
            "**Hint**: Consider adding a `hypothesis` to document what you "
            "expect this experiment to show. Helps with later analysis."
        )
    return hints


def format_response_with_hints(response: str, hints: list[str]) -> str:
    """Append hints to a response if any exist."""
    if not hints:
        return response
    return response + "\n\n---\n" + "\n".join(hints)


# -----------------------------------------------------------------------
# Actionable error helpers
# -----------------------------------------------------------------------


def actionable_error(message: str, *, suggestion: str | None = None) -> str:
    """Format an error with an actionable suggestion."""
    result = f"**Error**: {message}"
    if suggestion:
        result += f"\n\n**Suggestion**: {suggestion}"
    return result


def missing_config_error(project_dir: str = "") -> str:
    """Standard error when pipeline.yaml is missing."""
    return actionable_error(
        "No pipeline.yaml found." + (f" (looked in {project_dir})" if project_dir else ""),
        suggestion="Run `configure(action='init')` to initialize the project.",
    )


def missing_data_error() -> str:
    """Standard error when feature store has no data."""
    return actionable_error(
        "Feature store is empty — no data has been loaded.",
        suggestion="Run `data(action='add', data_path='path/to/data.csv')` to load data.",
    )


def unknown_model_type_error(given: str, valid_types: list[str]) -> str:
    """Error with 'did you mean?' for model types."""
    closest = get_close_matches(given, valid_types, n=1, cutoff=0.5)
    msg = f"Unknown model type `{given}`. Valid types: {', '.join(sorted(valid_types))}"
    suggestion = None
    if closest:
        suggestion = f"Did you mean `{closest[0]}`?"
    return actionable_error(msg, suggestion=suggestion)
