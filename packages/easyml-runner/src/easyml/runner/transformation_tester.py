from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default transformations to test
_DEFAULT_TRANSFORMATIONS: dict[str, callable] = {
    "raw": lambda x: x,
    "log": lambda x: np.log(np.abs(x) + 1),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "cbrt": lambda x: np.cbrt(x),
    "square": lambda x: x ** 2,
    "reciprocal": lambda x: 1.0 / (x + np.sign(x) * 1e-6),
    "rank": lambda x: x.rank(pct=True),
    "zscore": lambda x: (x - x.mean()) / (x.std() + 1e-8),
}

# Formulas corresponding to transformations (for feature creation)
_TRANSFORMATION_FORMULAS: dict[str, str] = {
    "log": "log({col})",
    "sqrt": "sqrt({col})",
    "cbrt": "cbrt({col})",
    "square": "{col} ** 2",
    "reciprocal": "1.0 / ({col} + 0.000001)",
    "rank": "{col}",  # rank transform needs special handling
    "zscore": "{col}",  # zscore needs special handling
}


@dataclass
class TransformationResult:
    """Result for a single transformation test."""

    feature: str
    transformation: str
    formula: str
    correlation: float
    abs_correlation: float
    improvement: float  # vs raw baseline
    null_rate: float


@dataclass
class TransformationReport:
    """Full report of transformation testing."""

    results: list[TransformationResult] = field(default_factory=list)
    best_per_feature: dict[str, TransformationResult] = field(default_factory=dict)
    suggested_features: list[dict] = field(default_factory=list)

    def format_summary(self) -> str:
        """Markdown table: feature | best transform | corr | improvement."""
        if not self.best_per_feature:
            return "No transformation results."

        lines = ["## Transformation Test Results\n"]
        lines.append("| Feature | Best Transform | Correlation | Improvement |")
        lines.append("|---------|---------------|-------------|-------------|")

        for feature, best in sorted(
            self.best_per_feature.items(),
            key=lambda x: x[1].abs_correlation,
            reverse=True,
        ):
            lines.append(
                f"| {feature} "
                f"| {best.transformation} "
                f"| {best.correlation:+.4f} "
                f"| {best.improvement:+.4f} |"
            )

        if self.suggested_features:
            lines.append("\n### Suggested Features\n")
            for feat in self.suggested_features:
                lines.append(
                    f"- **{feat['name']}**: `{feat['formula']}` "
                    f"(corr={feat.get('correlation', 0):+.4f})"
                )

        return "\n".join(lines)

    def get_create_commands(self) -> list[dict]:
        """Return feature creation dicts for best non-raw transformations.

        Ready to pass to create_features_batch().
        """
        commands = []
        for feat in self.suggested_features:
            commands.append({
                "name": feat["name"],
                "formula": feat["formula"],
                "description": feat.get("description", ""),
            })
        return commands


def _compute_correlation(series: pd.Series, target: pd.Series) -> float:
    """Compute correlation, handling NaN."""
    try:
        valid = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(valid) < 10:
            return 0.0
        corr = float(valid["x"].corr(valid["y"]))
        return corr if not np.isnan(corr) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _find_top_interaction_partners(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    n: int = 5,
) -> list[str]:
    """Find the top N features most correlated with the given feature."""
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != feature_col and c != target_col and c.startswith("diff_")
    ]

    correlations = []
    series = df[feature_col]
    for col in numeric_cols:
        try:
            corr = abs(float(series.corr(df[col])))
            if not np.isnan(corr):
                correlations.append((col, corr))
        except (TypeError, ValueError):
            continue

    correlations.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in correlations[:n]]


def run_transformation_tests(
    project_dir: Path,
    features: list[str],
    *,
    target_col: str = "result",
    transformations: list[str] | None = None,
    test_interactions: bool = True,
    top_interaction_partners: int = 5,
    features_dir: str | None = None,
) -> TransformationReport:
    """Automatically test mathematical transformations of features.

    Default transformations tested per feature:
    - raw (baseline)
    - log(|x| + 1)
    - sqrt(|x|)
    - cbrt(x) (preserves sign)
    - x^2 (squared)
    - 1/x (reciprocal, with epsilon)
    - rank (rank transform)
    - z-score (standardized)

    If test_interactions=True, also tests:
    - feature * top_partner (for top N correlated features)
    - feature / top_partner
    - feature - top_partner

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    features : list[str]
        Column names to test transformations on.
    target_col : str
        Target column.
    transformations : list[str] | None
        Subset of transformations to test. None = all defaults.
    test_interactions : bool
        Whether to test interactions with correlated features.
    top_interaction_partners : int
        Number of partners for interaction testing.
    features_dir : str | None
        Override features directory.

    Returns
    -------
    TransformationReport
    """
    project_dir = Path(project_dir)

    if features_dir is not None:
        feat_dir = Path(features_dir)
    else:
        feat_dir = project_dir / "data" / "features"

    parquet_path = feat_dir / "matchup_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Features not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    target = df[target_col].astype(float)

    # Determine which transformations to test
    if transformations is not None:
        transform_funcs = {
            k: v for k, v in _DEFAULT_TRANSFORMATIONS.items()
            if k in transformations
        }
    else:
        transform_funcs = dict(_DEFAULT_TRANSFORMATIONS)

    all_results: list[TransformationResult] = []
    best_per_feature: dict[str, TransformationResult] = {}
    suggested_features: list[dict] = []

    for feature_col in features:
        if feature_col not in df.columns:
            logger.warning("Feature %s not found in dataset, skipping", feature_col)
            continue

        series = df[feature_col].astype(float)
        raw_corr = _compute_correlation(series, target)

        feature_results: list[TransformationResult] = []

        for trans_name, trans_func in transform_funcs.items():
            try:
                transformed = trans_func(series)
                corr = _compute_correlation(transformed, target)
                null_rate = float(transformed.isna().mean()) if hasattr(transformed, 'isna') else 0.0
                improvement = abs(corr) - abs(raw_corr)

                formula = _TRANSFORMATION_FORMULAS.get(trans_name, feature_col)
                formula = formula.format(col=feature_col)

                result = TransformationResult(
                    feature=feature_col,
                    transformation=trans_name,
                    formula=formula,
                    correlation=corr,
                    abs_correlation=abs(corr),
                    improvement=improvement,
                    null_rate=null_rate,
                )
                feature_results.append(result)
                all_results.append(result)
            except Exception as exc:
                logger.warning(
                    "Transform %s failed on %s: %s",
                    trans_name, feature_col, exc,
                )

        # Test interactions
        if test_interactions:
            partners = _find_top_interaction_partners(
                df, feature_col, target_col, n=top_interaction_partners,
            )
            for partner in partners:
                partner_series = df[partner].astype(float)
                for op_name, op_func, op_formula in [
                    ("multiply", lambda a, b: a * b, "{col} * {partner}"),
                    ("divide", lambda a, b: a / (b + np.sign(b) * 1e-6), "{col} / ({partner} + 0.000001)"),
                    ("subtract", lambda a, b: a - b, "{col} - {partner}"),
                ]:
                    try:
                        interaction = op_func(series, partner_series)
                        corr = _compute_correlation(interaction, target)
                        null_rate = float(interaction.isna().mean())
                        improvement = abs(corr) - abs(raw_corr)

                        formula = op_formula.format(col=feature_col, partner=partner)
                        result = TransformationResult(
                            feature=feature_col,
                            transformation=f"{op_name}({partner})",
                            formula=formula,
                            correlation=corr,
                            abs_correlation=abs(corr),
                            improvement=improvement,
                            null_rate=null_rate,
                        )
                        feature_results.append(result)
                        all_results.append(result)
                    except Exception:
                        continue

        # Find best for this feature
        if feature_results:
            best = max(feature_results, key=lambda r: r.abs_correlation)
            best_per_feature[feature_col] = best

            # Suggest non-raw transformations that improve correlation
            if best.transformation != "raw" and best.improvement > 0:
                # Generate a clean feature name
                trans_suffix = best.transformation.replace("(", "_").replace(")", "")
                feat_base = feature_col.replace("diff_", "")
                suggested_features.append({
                    "name": f"{trans_suffix}_{feat_base}",
                    "formula": best.formula,
                    "description": (
                        f"{best.transformation} of {feature_col} "
                        f"(corr improvement: {best.improvement:+.4f})"
                    ),
                    "correlation": best.correlation,
                })

    return TransformationReport(
        results=all_results,
        best_per_feature=best_per_feature,
        suggested_features=suggested_features,
    )
