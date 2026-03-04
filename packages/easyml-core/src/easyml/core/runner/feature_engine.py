"""Feature formula evaluation and computation helpers.

Provides safe formula evaluation (no eval/exec) using pd.eval with a
restricted namespace.  Column names resolve to DataFrame columns,
@feature_name references resolve to previously created features, and a
whitelist of numpy functions is exposed.

Used internally by FeatureStore for formula-based features, regime
conditions, and pairwise computations.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Whitelist of allowed functions in formulas
_SAFE_FUNCTIONS: dict[str, object] = {
    "abs": np.abs,
    "log": lambda x: np.log(np.abs(x) + 1),  # safe log: log(|x| + 1)
    "sqrt": lambda x: np.sqrt(np.abs(x)),     # safe sqrt: sqrt(|x|)
    "cbrt": np.cbrt,                           # preserves sign
    "clip": np.clip,
    "log1p": np.log1p,
    "sign": np.sign,
    "square": np.square,
}

# Pattern to match @feature references
_FEATURE_REF_PATTERN = re.compile(r"@(\w+)")

# Pattern to match function calls: func(...)
_FUNC_CALL_PATTERN = re.compile(r"(\w+)\s*\(")

# Pattern to match whole identifiers: word boundary, then letters/digits/underscores
# This captures function names too -- we filter them out afterward.
_IDENTIFIER_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\b")


@dataclass
class FeatureResult:
    """Result of feature creation."""

    name: str
    column_added: str
    correlation: float
    abs_correlation: float
    redundant_with: list[str] = field(default_factory=list)
    null_rate: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def format_summary(self) -> str:
        """Markdown summary for tool response."""
        lines = [f"## Created: {self.column_added}\n"]
        if self.description:
            lines.append(f"_{self.description}_\n")

        lines.append(f"- **Correlation with target**: {self.correlation:+.4f}")
        lines.append(f"- **Null rate**: {self.null_rate:.1%}")

        if self.stats:
            lines.append(f"- **Mean**: {self.stats.get('mean', 0):.4f}")
            lines.append(f"- **Std**: {self.stats.get('std', 0):.4f}")
            lines.append(
                f"- **Range**: [{self.stats.get('min', 0):.4f}, "
                f"{self.stats.get('max', 0):.4f}]"
            )

        if self.redundant_with:
            lines.append(
                f"\n**Warning**: Redundant (r > 0.95) with: "
                f"{', '.join(self.redundant_with)}"
            )

        return "\n".join(lines)


def _resolve_formula(
    formula: str,
    df: pd.DataFrame,
    created_features: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Safely evaluate a formula against a DataFrame.

    The approach: replace column references and @-references with
    actual Series, then use pandas eval with a restricted namespace.
    No eval() or exec() is used -- only pd.eval with engine="python".
    """
    if created_features is None:
        created_features = {}

    # Build namespace with safe functions
    namespace: dict[str, object] = dict(_SAFE_FUNCTIONS)

    # Track which tokens have been mapped to variable names so we don't
    # double-replace overlapping identifiers.
    token_map: dict[str, str] = {}

    # Resolve @feature references first
    resolved = formula
    for match in _FEATURE_REF_PATTERN.finditer(formula):
        ref_name = match.group(1)
        if ref_name in created_features:
            var_name = f"_ref_{ref_name}"
            namespace[var_name] = created_features[ref_name]
            token_map[f"@{ref_name}"] = var_name
        elif ref_name in df.columns:
            var_name = f"_ref_{ref_name}"
            namespace[var_name] = df[ref_name]
            token_map[f"@{ref_name}"] = var_name
        else:
            raise ValueError(
                f"Feature reference @{ref_name} not found. "
                f"No column '{ref_name}' and no created feature '{ref_name}'."
            )

    # Apply @-reference replacements (longest first to avoid partial matches)
    for token in sorted(token_map, key=len, reverse=True):
        resolved = resolved.replace(token, token_map[token])

    # Now find all column name references that still need resolving.
    # We use _FUNC_CALL_PATTERN to identify which identifiers are function
    # calls (followed by '(') and skip those.
    func_names = set(_SAFE_FUNCTIONS.keys())
    already_mapped: set[str] = set()

    # Identify identifiers that are used as function calls in this formula
    called_funcs = set(_FUNC_CALL_PATTERN.findall(resolved))

    # Collect unique identifiers from the resolved formula
    all_tokens: list[str] = []
    for match in _IDENTIFIER_PATTERN.finditer(resolved):
        token = match.group(1)
        if token not in all_tokens:
            all_tokens.append(token)

    # Sort longest first for greedy replacement
    all_tokens.sort(key=len, reverse=True)

    for token in all_tokens:
        # Skip known function names, already-resolved refs/cols, and numeric
        if token in func_names or token in called_funcs:
            continue
        if token.startswith("_ref_") or token.startswith("_col_"):
            continue
        # Skip numeric constants
        try:
            float(token)
            continue
        except ValueError:
            pass
        if token in already_mapped:
            continue

        if token in df.columns:
            var_name = f"_col_{token}"
            namespace[var_name] = df[token]
            already_mapped.add(token)
        else:
            raise ValueError(
                f"Column '{token}' not found in dataset. "
                f"Available columns (first 20): {list(df.columns)[:20]}"
            )

    # Replace column tokens in the resolved string (longest first)
    for token in sorted(already_mapped, key=len, reverse=True):
        var_name = f"_col_{token}"
        resolved = resolved.replace(token, var_name)

    # Evaluate using pd.eval with local namespace
    try:
        result = pd.eval(resolved, local_dict=namespace, engine="python")
    except Exception as exc:
        raise ValueError(f"Formula evaluation failed: {exc}") from exc

    if isinstance(result, (int, float, np.number)):
        result = pd.Series(result, index=df.index)

    return pd.Series(result, index=df.index, dtype=float)


def _compute_feature_stats(series: pd.Series) -> dict[str, float]:
    """Compute basic statistics for a feature."""
    valid = series.dropna()
    if len(valid) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _check_redundancy(
    series: pd.Series,
    df: pd.DataFrame,
    threshold: float = 0.95,
    max_check: int = 50,
) -> list[str]:
    """Check for high correlation with existing numeric columns."""
    redundant: list[str] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_check]
    for col in numeric_cols:
        try:
            corr = series.corr(df[col])
            if abs(corr) > threshold:
                redundant.append(col)
        except (TypeError, ValueError):
            continue
    return redundant


def _topological_sort_features(
    features: list[dict],
    feature_names: set[str],
) -> list[dict]:
    """Sort features so dependencies (@-references) come first."""
    name_to_feat = {f["name"]: f for f in features}
    deps: dict[str, set[str]] = {}

    for f in features:
        formula = f.get("formula") or ""
        refs = set(_FEATURE_REF_PATTERN.findall(formula))
        # Only track deps on features in this batch
        deps[f["name"]] = refs & feature_names

    # Kahn's algorithm
    in_degree = {n: len(d) for n, d in deps.items()}
    queue = [n for n, d in in_degree.items() if d == 0]
    ordered: list[dict] = []

    while queue:
        node = queue.pop(0)
        ordered.append(name_to_feat[node])
        for n, d in deps.items():
            if node in d:
                d.remove(node)
                in_degree[n] -= 1
                if in_degree[n] == 0:
                    queue.append(n)

    if len(ordered) != len(features):
        missing = set(name_to_feat) - {f["name"] for f in ordered}
        raise ValueError(f"Circular dependency detected among features: {missing}")

    return ordered
