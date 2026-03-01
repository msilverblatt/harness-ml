"""Typed artifact declarations for pipeline stages.

Artifacts are declared via ArtifactDecl (from easyml.schemas.core) and resolved
into concrete paths for dependency tracking between stages.
"""
from __future__ import annotations

from easyml.schemas.core import ArtifactDecl, StageConfig


def resolve_artifact_paths(
    stages: dict[str, StageConfig],
) -> dict[str, str]:
    """Build a mapping from artifact name to its declared path.

    Scans all stages' produces lists and returns {artifact_name: path}.
    Raises ValueError on duplicate artifact names.
    """
    artifact_map: dict[str, str] = {}
    for stage_name, stage in stages.items():
        for artifact in stage.produces:
            if artifact.name in artifact_map:
                raise ValueError(
                    f"Duplicate artifact '{artifact.name}' declared in "
                    f"stage '{stage_name}' (already declared with path "
                    f"'{artifact_map[artifact.name]}')"
                )
            artifact_map[artifact.name] = artifact.path
    return artifact_map
