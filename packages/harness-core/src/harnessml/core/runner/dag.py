"""Model dependency graph — DAG resolution for provider features.

Builds a dependency graph from model definitions where provider models
declare output features consumed by downstream models. Uses topological
sort (Kahn's algorithm) to determine correct training order, grouped
into parallel waves.
"""
from __future__ import annotations

from collections import defaultdict, deque

from harnessml.core.runner.schema import ModelDef


def build_provider_map(models: dict[str, ModelDef]) -> dict[str, str]:
    """Map each provided feature column to its provider model name.

    Parameters
    ----------
    models : dict[str, ModelDef]
        All model definitions keyed by name.

    Returns
    -------
    dict[str, str]
        Mapping from column name (e.g. "surv_e8") to provider model name.

    Raises
    ------
    ValueError
        If two models provide the same feature column.
    """
    provider_map: dict[str, str] = {}
    for model_name, model_def in models.items():
        for col in model_def.provides:
            if col in provider_map:
                raise ValueError(
                    f"Feature {col!r} is provided by both "
                    f"{provider_map[col]!r} and {model_name!r}"
                )
            provider_map[col] = model_name
    return provider_map


def infer_dependencies(
    models: dict[str, ModelDef],
    provider_map: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    """Infer model dependencies from feature references.

    A model depends on a provider if any of its features match
    ``col`` or ``diff_{col}`` where ``col`` is in the provider map.

    Parameters
    ----------
    models : dict[str, ModelDef]
        All model definitions keyed by name.
    provider_map : dict[str, str] | None
        Pre-computed provider map. Built from models if not supplied.

    Returns
    -------
    dict[str, set[str]]
        Mapping from model name to set of model names it depends on.
    """
    if provider_map is None:
        provider_map = build_provider_map(models)

    deps: dict[str, set[str]] = {name: set() for name in models}

    for model_name, model_def in models.items():
        for feat in model_def.features:
            # Strip diff_ prefix to find the raw column name
            raw = feat[len("diff_"):] if feat.startswith("diff_") else feat
            if raw in provider_map:
                provider = provider_map[raw]
                if provider != model_name:
                    deps[model_name].add(provider)

    return deps


def topological_waves(deps: dict[str, set[str]]) -> list[list[str]]:
    """Topological sort into parallel waves using Kahn's algorithm.

    Wave 0 contains models with no dependencies. Wave 1 contains models
    whose only dependencies are in wave 0, etc. Models within a wave
    can be trained in any order.

    Parameters
    ----------
    deps : dict[str, set[str]]
        Dependency graph: model name -> set of dependency model names.

    Returns
    -------
    list[list[str]]
        Waves of model names in dependency order.

    Raises
    ------
    ValueError
        If a cycle is detected in the dependency graph.
    """
    if not deps:
        return []

    # Build in-degree counts and reverse adjacency
    in_degree: dict[str, int] = {name: 0 for name in deps}
    reverse: dict[str, list[str]] = defaultdict(list)

    for model_name, model_deps in deps.items():
        in_degree[model_name] = len(model_deps)
        for dep in model_deps:
            reverse[dep].append(model_name)

    # Seed queue with zero-dependency models
    queue = deque(sorted(name for name, deg in in_degree.items() if deg == 0))
    waves: list[list[str]] = []
    processed = 0

    while queue:
        # Current wave = all models currently in the queue
        wave = sorted(queue)
        waves.append(wave)
        queue.clear()

        for model_name in wave:
            processed += 1
            for dependent in reverse.get(model_name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    if processed < len(deps):
        cycle_members = sorted(
            name for name, deg in in_degree.items() if deg > 0
        )
        raise ValueError(
            f"Dependency cycle detected in model graph. "
            f"Models in cycle: {cycle_members}"
        )

    return waves


def detect_cycle(deps: dict[str, set[str]]) -> list[str] | None:
    """Check for cycles in the dependency graph.

    Returns
    -------
    list[str] | None
        Names of models involved in the cycle, or None if acyclic.
    """
    try:
        topological_waves(deps)
        return None
    except ValueError:
        # Find cycle members
        in_degree: dict[str, int] = {}
        for model_name, model_deps in deps.items():
            in_degree.setdefault(model_name, 0)
            in_degree[model_name] = len(model_deps)
            for dep in model_deps:
                in_degree.setdefault(dep, 0)

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        visited: set[str] = set()

        while queue:
            node = queue.popleft()
            visited.add(node)
            for model_name, model_deps in deps.items():
                if node in model_deps:
                    in_degree[model_name] -= 1
                    if in_degree[model_name] == 0:
                        queue.append(model_name)

        return sorted(name for name in deps if name not in visited)
