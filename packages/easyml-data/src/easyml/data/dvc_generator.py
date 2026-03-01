"""Generate DVC pipeline YAML from typed stage configurations."""
from __future__ import annotations

from typing import Any

import yaml

from easyml.schemas.core import StageConfig
from easyml.data.artifacts import resolve_artifact_paths


def generate_dvc_yaml(stages: dict[str, StageConfig]) -> dict[str, Any]:
    """Build a DVC-compatible pipeline dict from StageConfig declarations.

    Each stage gets:
    - cmd: ``uv run python <script>``
    - deps: script path + paths of consumed artifacts
    - outs: paths of produced artifacts

    Artifact names in ``consumes`` are resolved to paths via the produces
    declarations across all stages.
    """
    artifact_map = resolve_artifact_paths(stages)
    dvc_stages: dict[str, Any] = {}

    for stage_name, stage in stages.items():
        deps: list[str] = [stage.script]
        for consumed_name in stage.consumes:
            if consumed_name not in artifact_map:
                raise KeyError(
                    f"Stage '{stage_name}' consumes artifact '{consumed_name}' "
                    f"which is not produced by any stage"
                )
            deps.append(artifact_map[consumed_name])

        outs: list[str] = [artifact.path for artifact in stage.produces]

        dvc_stages[stage_name] = {
            "cmd": f"uv run python {stage.script}",
            "deps": deps,
            "outs": outs,
        }

    return {"stages": dvc_stages}


def generate_dvc_string(stages: dict[str, StageConfig]) -> str:
    """Generate a DVC YAML string from stage configurations."""
    dvc_dict = generate_dvc_yaml(stages)
    return yaml.dump(dvc_dict, default_flow_style=False, sort_keys=False)
