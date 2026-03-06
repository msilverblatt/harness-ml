"""Feature diversity analysis for model ensembles.

Analyzes feature overlap and diversity across models to help ensure
ensemble members use sufficiently different feature sets.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any


def _active_models(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Filter to only active models."""
    return {k: v for k, v in models.items() if v.get("active", True)}


def _jaccard(features_a: list[str], features_b: list[str]) -> float:
    """Compute Jaccard similarity between two feature lists."""
    set_a = set(features_a)
    set_b = set(features_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_overlap_matrix(
    models: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], float]:
    """Compute pairwise Jaccard overlap between all active models.

    Returns a dict mapping (model_a, model_b) -> overlap score.
    Includes diagonal (self-overlap = 1.0) and both directions for symmetry.
    """
    active = _active_models(models)
    names = sorted(active.keys())
    matrix: dict[tuple[str, str], float] = {}

    for name in names:
        matrix[(name, name)] = 1.0

    for a, b in combinations(names, 2):
        overlap = _jaccard(active[a]["features"], active[b]["features"])
        matrix[(a, b)] = overlap
        matrix[(b, a)] = overlap

    return matrix


def compute_diversity_score(models: dict[str, dict[str, Any]]) -> float:
    """Compute overall diversity score (1 - mean pairwise overlap).

    Returns 1.0 for perfectly diverse (no overlap), 0.0 for identical features.
    Returns 1.0 for 0 or 1 models (trivially diverse).
    """
    active = _active_models(models)
    names = sorted(active.keys())

    if len(names) <= 1:
        return 1.0

    overlaps = []
    for a, b in combinations(names, 2):
        overlaps.append(_jaccard(active[a]["features"], active[b]["features"]))

    if not overlaps:
        return 1.0

    return 1.0 - (sum(overlaps) / len(overlaps))


def find_redundant_features(
    models: dict[str, dict[str, Any]],
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """Find model pairs with feature overlap above the given threshold.

    Returns a list of dicts sorted by overlap descending, each containing:
    - model_a, model_b: model names
    - overlap: Jaccard similarity
    - shared_features: list of shared feature names
    """
    active = _active_models(models)
    names = sorted(active.keys())
    results: list[dict[str, Any]] = []

    for a, b in combinations(names, 2):
        set_a = set(active[a]["features"])
        set_b = set(active[b]["features"])
        union = set_a | set_b
        if not union:
            continue
        intersection = set_a & set_b
        overlap = len(intersection) / len(union)
        if overlap > threshold:
            results.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "overlap": overlap,
                    "shared_features": sorted(intersection),
                }
            )

    results.sort(key=lambda x: x["overlap"], reverse=True)
    return results


def suggest_removal(
    models: dict[str, dict[str, Any]],
    target_score: float = 0.7,
) -> list[dict[str, Any]]:
    """Suggest feature removals to improve diversity toward a target score.

    Returns a list of suggestion dicts with:
    - model: which model to remove from
    - feature: which feature to remove

    Prefers removing shared features from the model with more features.
    """
    import copy

    active = _active_models(models)

    if compute_diversity_score(active) >= target_score:
        return []

    working = copy.deepcopy(active)
    suggestions: list[dict[str, Any]] = []

    max_iterations = sum(len(m["features"]) for m in working.values())
    for _ in range(max_iterations):
        if compute_diversity_score(working) >= target_score:
            break

        # Find the most overlapping pair
        names = sorted(working.keys())
        worst_pair = None
        worst_overlap = -1.0
        for a, b in combinations(names, 2):
            overlap = _jaccard(working[a]["features"], working[b]["features"])
            if overlap > worst_overlap:
                worst_overlap = overlap
                worst_pair = (a, b)

        if worst_pair is None or worst_overlap == 0.0:
            break

        a, b = worst_pair
        shared = set(working[a]["features"]) & set(working[b]["features"])
        if not shared:
            break

        # Pick the model with more features to remove from
        target_model = a if len(working[a]["features"]) >= len(working[b]["features"]) else b

        # Don't remove if it would leave the model with no features
        if len(working[target_model]["features"]) <= 1:
            break

        feature_to_remove = sorted(shared)[0]
        working[target_model]["features"].remove(feature_to_remove)
        suggestions.append({"model": target_model, "feature": feature_to_remove})

    return suggestions


def format_diversity_report(models: dict[str, dict[str, Any]]) -> str:
    """Format a markdown report of feature diversity analysis.

    Includes diversity score, overlap matrix, pass/fail status,
    and any redundant pairs found.
    """
    active = _active_models(models)
    names = sorted(active.keys())
    score = compute_diversity_score(active)
    matrix = compute_overlap_matrix(active)
    redundant = find_redundant_features(active)

    lines: list[str] = []
    lines.append("# Feature Diversity Report")
    lines.append("")

    # Score and status
    status = "PASS" if score >= 0.5 else "FAIL"
    lines.append(f"**Diversity Score:** {score:.2f}")
    lines.append(f"**Status:** {status}")
    lines.append("")

    # Overlap matrix
    if names:
        lines.append("## Overlap Matrix")
        lines.append("")
        header = "| | " + " | ".join(names) + " |"
        sep = "|---" * (len(names) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for a in names:
            row_vals = [f"{matrix[(a, b)]:.2f}" for b in names]
            lines.append(f"| {a} | " + " | ".join(row_vals) + " |")
        lines.append("")

    # Redundant pairs
    if redundant:
        lines.append("## Redundant Pairs (overlap > 0.80)")
        lines.append("")
        for r in redundant:
            lines.append(
                f"- **{r['model_a']}** / **{r['model_b']}**: "
                f"overlap={r['overlap']:.2f}, "
                f"shared={r['shared_features']}"
            )
        lines.append("")

    return "\n".join(lines)
