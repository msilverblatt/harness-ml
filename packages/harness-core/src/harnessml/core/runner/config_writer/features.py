"""Feature operations: add, batch, discover, search, transform."""
from __future__ import annotations

import logging
from pathlib import Path

from harnessml.core.runner.config_writer._helpers import (
    _get_config_dir,
    _load_yaml,
    _persist_feature_defs,
)

logger = logging.getLogger(__name__)


def _validate_formula_syntax(formula: str) -> str | None:
    """Validate formula syntax by attempting to compile it.

    Returns an error message string if the formula is invalid, None if valid.
    """
    try:
        compile(formula, "<formula>", "eval")
    except SyntaxError as exc:
        detail = f"line {exc.lineno}, col {exc.offset}" if exc.lineno else ""
        msg = exc.msg if exc.msg else "invalid syntax"
        parts = [f"Invalid formula syntax: {msg}"]
        if detail:
            parts.append(f" ({detail})")
        parts.append(f"\n  Formula: `{formula}`")
        return "".join(parts)
    return None


def add_feature(
    project_dir: Path,
    name: str,
    formula: str | None = None,
    *,
    type: str | None = None,
    source: str | None = None,
    column: str | None = None,
    condition: str | None = None,
    pairwise_mode: str = "diff",
    category: str = "general",
    description: str = "",
) -> str:
    """Create a new feature via the declarative FeatureStore.

    All features go through the FeatureStore. If type is not specified,
    it is inferred: formula -> pairwise, condition -> regime, source -> team.
    """
    project_dir = Path(project_dir)

    # Validate formula syntax before proceeding
    if formula is not None:
        error = _validate_formula_syntax(formula)
        if error is not None:
            return f"**Error**: {error}"

    from harnessml.core.runner.data.utils import load_data_config
    from harnessml.core.runner.features.store import FeatureStore
    from harnessml.core.runner.schema import FeatureDef, FeatureType, PairwiseMode

    config = load_data_config(project_dir)
    store = FeatureStore(project_dir, config)

    # Infer type if not specified
    if type is None:
        if formula is not None:
            type = "pairwise"
        elif condition is not None:
            type = "regime"
        elif source is not None:
            type = "entity"
        else:
            raise ValueError(
                "Must provide type, formula, condition, or source."
            )

    feature_type = FeatureType(type)
    pw_mode = PairwiseMode(pairwise_mode)

    feature_def = FeatureDef(
        name=name,
        type=feature_type,
        source=source,
        column=column,
        formula=formula,
        condition=condition,
        pairwise_mode=pw_mode,
        category=category,
        description=description,
    )

    result = store.add(feature_def)

    # Persist feature_defs to pipeline.yaml
    store.save_registry()
    _persist_feature_defs(project_dir, config)

    # Format response
    lines = [f"## Added {type} feature: {name}\n"]
    if description:
        lines.append(f"_{description}_\n")

    if feature_type == FeatureType.ENTITY:
        cache_entry = store._cache._entries.get(name)
        if cache_entry and cache_entry.derivatives:
            lines.append("**Auto-generated pairwise:**")
            matchup_df = store._load_matchup_data()
            target_col = config.target_column
            for deriv in cache_entry.derivatives:
                try:
                    deriv_series = store.compute(deriv)
                    corr = 0.0
                    if target_col in matchup_df.columns:
                        corr = float(deriv_series.corr(matchup_df[target_col].astype(float)))
                        if not isinstance(corr, float) or corr != corr:
                            corr = 0.0
                    lines.append(f"- `{deriv}` (r={corr:+.4f})")
                except Exception:
                    lines.append(f"- `{deriv}`")

    lines.append(f"\n- **Correlation**: {result.correlation:+.4f}")
    lines.append(f"- **Null rate**: {result.null_rate:.1%}")
    if result.stats:
        for k, v in result.stats.items():
            lines.append(f"- **{k.title()}**: {v:.4f}")
    lines.append(f"- **Category**: {category}")

    # Check for redundant formulas
    if formula and config.feature_defs:
        warnings = []
        for fname, fdef in config.feature_defs.items():
            if fname != name and getattr(fdef, "formula", None) == formula:
                warnings.append(
                    f"**Warning**: Formula is identical to existing feature `{fname}`"
                )
        if warnings:
            lines.append("")
            lines.extend(warnings)

    return "\n".join(lines)


def add_features_batch(
    project_dir: Path,
    features: list[dict],
) -> str:
    """Create multiple features via the declarative FeatureStore.

    Each dict can include: name, formula, type, source, column, condition,
    pairwise_mode, category, description. Handles @-references between
    features via topological ordering.
    """
    from harnessml.core.runner.data.utils import load_data_config
    from harnessml.core.runner.features.engine import _topological_sort_features
    from harnessml.core.runner.features.store import FeatureStore
    from harnessml.core.runner.schema import FeatureDef, FeatureType, PairwiseMode

    project_dir = Path(project_dir)
    config = load_data_config(project_dir)
    store = FeatureStore(project_dir, config)

    # Resolve dependency order
    feature_names = {f["name"] for f in features}
    ordered = _topological_sort_features(features, feature_names)

    results = []
    for feat_dict in ordered:
        feat_name = feat_dict["name"]
        feat_formula = feat_dict.get("formula")
        feat_type = feat_dict.get("type")
        feat_condition = feat_dict.get("condition")
        feat_source = feat_dict.get("source")

        # Infer type if not specified
        if feat_type is None:
            if feat_formula is not None:
                feat_type = "pairwise"
            elif feat_condition is not None:
                feat_type = "regime"
            elif feat_source is not None:
                feat_type = "entity"
            else:
                feat_type = "pairwise"

        feature_def = FeatureDef(
            name=feat_name,
            type=FeatureType(feat_type),
            formula=feat_formula,
            source=feat_source,
            column=feat_dict.get("column"),
            condition=feat_condition,
            pairwise_mode=PairwiseMode(feat_dict.get("pairwise_mode", "diff")),
            category=feat_dict.get("category", "general"),
            description=feat_dict.get("description", ""),
        )
        result = store.add(feature_def)
        results.append(result)

    # Persist all at once
    store.save_registry()
    _persist_feature_defs(project_dir, config)

    lines = [f"## Created {len(results)} Features\n"]
    for r in results:
        lines.append(f"- **{r.column_added}** (corr={r.correlation:+.4f})")
    return "\n".join(lines)


def test_feature_transformations(
    project_dir: Path,
    features: list[str],
    *,
    test_interactions: bool = True,
) -> str:
    """Test mathematical transformations of features."""
    from harnessml.core.runner.data.utils import get_features_df, load_data_config
    from harnessml.core.runner.transformation_tester import run_transformation_tests

    project_dir = Path(project_dir)

    # Load config and features DataFrame
    feat_defs = None
    try:
        config = load_data_config(project_dir)
        if config.feature_defs:
            feat_defs = dict(config.feature_defs)
    except Exception:
        config = None

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    report = run_transformation_tests(
        project_dir=project_dir,
        features=features,
        test_interactions=test_interactions,
        feature_defs=feat_defs,
        df=df,
    )
    return report.format_summary()


def discover_features(
    project_dir: Path,
    *,
    top_n: int = 20,
    method: str = "xgboost",
    on_progress: callable | None = None,
) -> str:
    """Run feature discovery analysis."""
    from harnessml.core.runner.data.utils import get_feature_columns, get_features_df, load_data_config
    from harnessml.core.runner.schema import DataConfig

    def _report(step, total, msg):
        if on_progress is not None:
            on_progress(step, total, msg)

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()


    _report(0, 5, "Loading feature data...")
    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    # Get feature columns and feature_defs from config if available
    feature_cols = None
    feat_defs = None
    pipeline_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
    backtest_data = pipeline_data.get("backtest", {})
    fold_col = backtest_data.get("fold_column")
    target_col = pipeline_data.get("data", {}).get("target_column", "result")
    if config is not None:
        feature_cols = get_feature_columns(df, config, fold_column=fold_col)
        if config.feature_defs:
            feat_defs = dict(config.feature_defs)

    # Exclude denylist columns from feature discovery
    _sources_data = _load_yaml(_get_config_dir(project_dir) / "sources.yaml")
    _denylist = set(_sources_data.get("guardrails", {}).get("feature_leakage_denylist", []))
    if _denylist and feature_cols:
        feature_cols = [c for c in feature_cols if c not in _denylist]

    from harnessml.core.runner.features.discovery import (
        compute_feature_correlations,
        compute_feature_importance,
        detect_redundant_features,
        format_discovery_report,
        suggest_feature_groups,
    )

    _report(1, 5, "Computing feature correlations...")
    correlations = compute_feature_correlations(
        df, target_col=target_col, top_n=top_n, feature_columns=feature_cols, feature_defs=feat_defs,
    )
    _report(2, 5, "Computing feature importance (method=%s)..." % method)
    importance = compute_feature_importance(
        df, target_col=target_col, method=method, top_n=top_n, feature_columns=feature_cols, feature_defs=feat_defs,
    )
    _report(3, 5, "Detecting redundant features...")
    redundant = detect_redundant_features(df, feature_columns=feature_cols)
    _report(4, 5, "Suggesting feature groups...")
    groups = suggest_feature_groups(df, feature_columns=feature_cols, feature_defs=feat_defs)

    return format_discovery_report(correlations, importance, redundant, groups)


def auto_search_features(
    project_dir: Path,
    features: list[str] | None = None,
    *,
    search_types: list[str] | None = None,
    top_n: int = 20,
) -> str:
    """Run automated feature search over given columns.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    features : list[str] | None
        Column names to search over. If None, uses all feature columns.
    search_types : list[str] | None
        Which search types to run: "interactions", "lags", "rolling".
        Defaults to all three.
    top_n : int
        Number of top results to return.

    Returns
    -------
    str
        Markdown-formatted report of top candidates.
    """
    from harnessml.core.runner.data.utils import get_feature_columns, get_features_df, load_data_config
    from harnessml.core.runner.features.auto_search import auto_search, format_auto_search_report
    from harnessml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    # Resolve feature columns
    pipeline_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
    backtest_data = pipeline_data.get("backtest", {})
    fold_col = backtest_data.get("fold_column")
    if features:
        feature_cols = [c for c in features if c in df.columns]
        missing = [c for c in features if c not in df.columns]
        if missing:
            logger.warning("Columns not found in dataset, skipping: %s", missing)
    else:
        feature_cols = get_feature_columns(df, config, fold_column=fold_col)

    if not feature_cols:
        return "**Error**: No feature columns found to search over."

    target_col = config.target_column

    results = auto_search(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        search_types=search_types,
        top_n=top_n,
    )
    return format_auto_search_report(results)
