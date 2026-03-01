"""Tests for typed artifacts and DVC config generator."""
from __future__ import annotations

from easyml.schemas.core import ArtifactDecl, StageConfig
from easyml.data.dvc_generator import generate_dvc_yaml, generate_dvc_string


def test_generate_dvc_yaml_basic():
    stages = {
        "ingest": StageConfig(
            script="pipelines/ingest.py",
            consumes=[],
            produces=[
                ArtifactDecl(name="raw_data", type="data", path="data/processed/")
            ],
        ),
        "featurize": StageConfig(
            script="pipelines/featurize.py",
            consumes=["raw_data"],
            produces=[
                ArtifactDecl(name="features", type="features", path="data/features/")
            ],
        ),
    }
    dvc = generate_dvc_yaml(stages)
    assert "stages" in dvc
    assert "ingest" in dvc["stages"]
    assert "featurize" in dvc["stages"]
    # ingest has no deps (except its script), featurize depends on raw_data
    assert "data/processed/" in dvc["stages"]["featurize"]["deps"]
    assert "data/features/" in dvc["stages"]["featurize"]["outs"]


def test_generate_dvc_yaml_chain():
    stages = {
        "a": StageConfig(
            script="a.py",
            consumes=[],
            produces=[ArtifactDecl(name="x", type="data", path="out/x/")],
        ),
        "b": StageConfig(
            script="b.py",
            consumes=["x"],
            produces=[ArtifactDecl(name="y", type="features", path="out/y/")],
        ),
        "c": StageConfig(
            script="c.py",
            consumes=["y"],
            produces=[ArtifactDecl(name="z", type="model", path="out/z/")],
        ),
    }
    dvc = generate_dvc_yaml(stages)
    assert "out/x/" in dvc["stages"]["b"]["deps"]
    assert "out/y/" in dvc["stages"]["c"]["deps"]


def test_generate_dvc_yaml_to_string():
    stages = {
        "test": StageConfig(
            script="test.py",
            consumes=[],
            produces=[ArtifactDecl(name="out", type="data", path="data/out/")],
        ),
    }
    yaml_str = generate_dvc_string(stages)
    assert "stages:" in yaml_str
    assert "test:" in yaml_str
