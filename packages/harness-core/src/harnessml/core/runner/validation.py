"""Pre-backtest validation framework.

Runs a suite of checks before a backtest to catch configuration and data
issues early.  Returns structured ValidationIssue objects with severity
levels so the caller can decide whether to block or warn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationCode(Enum):
    FOLD_COLUMN_MISSING = "FOLD_COLUMN_MISSING"
    TEMPORAL_UNSORTED = "TEMPORAL_UNSORTED"
    FEATURE_NOT_FOUND = "FEATURE_NOT_FOUND"
    CLASS_IMBALANCE = "CLASS_IMBALANCE"
    HIGH_MISSING_RATE = "HIGH_MISSING_RATE"
    NO_ACTIVE_MODELS = "NO_ACTIVE_MODELS"
    TARGET_COLUMN_MISSING = "TARGET_COLUMN_MISSING"


@dataclass
class ValidationIssue:
    code: ValidationCode
    severity: Severity
    message: str
    detail: str = ""


def validate_project(
    project_dir: Path,
    data_df: Any = None,
) -> list[ValidationIssue]:
    """Run pre-backtest validation checks on a project.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    data_df : DataFrame, optional
        Pre-loaded data DataFrame.  If not provided, validation
        skips data-dependent checks.

    Returns
    -------
    list[ValidationIssue]
        List of issues found.  Empty list means all checks passed.
    """
    from harnessml.core.runner.config_writer._helpers import _get_config_dir, _load_yaml

    issues: list[ValidationIssue] = []

    try:
        config_dir = _get_config_dir(Path(project_dir))
    except FileNotFoundError:
        issues.append(ValidationIssue(
            code=ValidationCode.FEATURE_NOT_FOUND,
            severity=Severity.ERROR,
            message="Config directory not found.",
            detail=f"Expected config directory at {project_dir}/config/",
        ))
        return issues

    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    models_data = _load_yaml(config_dir / "models.yaml")

    data_section = pipeline_data.get("data", {})
    backtest_section = pipeline_data.get("backtest", {})

    # Check: target column (defaults to "result" if not set)
    target_col = data_section.get("target_column", "result")

    # Check: active models
    models = models_data.get("models", {})
    active_models = {k: v for k, v in models.items() if v.get("active", True)}
    if not active_models:
        issues.append(ValidationIssue(
            code=ValidationCode.NO_ACTIVE_MODELS,
            severity=Severity.ERROR,
            message="No active models found in models.yaml.",
        ))

    # Check: fold column
    fold_col = backtest_section.get("fold_column")
    if fold_col and data_df is not None:
        if fold_col not in data_df.columns:
            issues.append(ValidationIssue(
                code=ValidationCode.FOLD_COLUMN_MISSING,
                severity=Severity.ERROR,
                message=f"Fold column `{fold_col}` not found in data.",
                detail=f"Available columns: {', '.join(sorted(data_df.columns)[:20])}",
            ))

    # Data-dependent checks
    if data_df is not None:
        # Check: target column exists in data
        if target_col and target_col not in data_df.columns:
            issues.append(ValidationIssue(
                code=ValidationCode.TARGET_COLUMN_MISSING,
                severity=Severity.ERROR,
                message=f"Target column `{target_col}` not found in data.",
            ))

        # Check: class imbalance (for classification tasks)
        if target_col and target_col in data_df.columns:

            target = data_df[target_col].dropna()
            if len(target) > 0:
                value_counts = target.value_counts(normalize=True)
                min_class_pct = value_counts.min()
                if min_class_pct < 0.05:
                    issues.append(ValidationIssue(
                        code=ValidationCode.CLASS_IMBALANCE,
                        severity=Severity.WARNING,
                        message=f"Class imbalance detected: minority class is {min_class_pct:.1%}.",
                        detail="Consider class weighting or resampling.",
                    ))

        # Check: high missing rate on model features
        feature_defs = data_section.get("feature_defs", {})
        all_model_features: set[str] = set()
        for m in active_models.values():
            all_model_features.update(m.get("features", []))

        for feat in all_model_features:
            if feat in data_df.columns:
                missing_rate = data_df[feat].isna().mean()
                if missing_rate > 0.5:
                    issues.append(ValidationIssue(
                        code=ValidationCode.HIGH_MISSING_RATE,
                        severity=Severity.WARNING,
                        message=f"Feature `{feat}` has {missing_rate:.0%} missing values.",
                    ))
            elif feat in feature_defs:
                # Declarative feature — will be computed, skip check
                pass
            else:
                issues.append(ValidationIssue(
                    code=ValidationCode.FEATURE_NOT_FOUND,
                    severity=Severity.WARNING,
                    message=f"Feature `{feat}` not found in data or feature_defs.",
                ))

        # Check: temporal ordering (if fold column present)
        if fold_col and fold_col in data_df.columns:
            folds = data_df[fold_col].dropna().unique()
            sorted_folds = sorted(folds)
            if list(folds) != sorted_folds and len(folds) > 1:
                issues.append(ValidationIssue(
                    code=ValidationCode.TEMPORAL_UNSORTED,
                    severity=Severity.WARNING,
                    message="Data is not sorted by fold column.",
                    detail=f"Fold column `{fold_col}` values are not in ascending order.",
                ))

    return issues


def format_validation_issues(issues: list[ValidationIssue]) -> str:
    """Format validation issues as markdown."""
    if not issues:
        return "All pre-backtest validations passed."

    errors = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]

    lines: list[str] = []

    if errors:
        lines.append(f"## Validation Errors ({len(errors)})\n")
        for issue in errors:
            lines.append(f"- **[{issue.code.value}]** {issue.message}")
            if issue.detail:
                lines.append(f"  {issue.detail}")

    if warnings:
        lines.append(f"\n## Validation Warnings ({len(warnings)})\n")
        for issue in warnings:
            lines.append(f"- **[{issue.code.value}]** {issue.message}")
            if issue.detail:
                lines.append(f"  {issue.detail}")

    return "\n".join(lines)
