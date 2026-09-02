import json
import zipfile

import cloudpickle
import pytest

from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.config.models import ModelsConfig
from harness.ml.config.project import ProjectConfig
from harness.ml.runners.artifacts import (
    ArtifactError,
    ArtifactIntegrityError,
    UntrustedArtifactError,
    inspect_artifact,
    load_artifact,
    save_artifact,
)
from harness.ml.runners.production import ProductionBundle


def metadata():
    return {
        "task": {"task_type": "binary"},
        "models": [],
        "training_features": [{"name": "x", "dtype": "float64"}],
        "ensemble_columns": [],
        "fingerprints": {"training_data": "abc"},
        "output": {"kind": "scalar"},
    }


def rewrite_archive(source, destination, transform):
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    transform(members)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)


def test_save_inspect_and_trusted_load(tmp_path):
    path = tmp_path / "model.bundle"
    save_artifact(path, {"answer": 42}, metadata())

    manifest = inspect_artifact(path)
    assert manifest["artifact_type"] == "harness.production_bundle"
    assert manifest["format_version"] == 1
    assert manifest["payload"]["size_bytes"] > 0
    assert len(manifest["payload"]["sha256"]) == 64
    assert load_artifact(path, trusted=True) == {"answer": 42}


def test_load_requires_explicit_trust_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "model.bundle"
    save_artifact(path, {"answer": 42}, metadata())
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(cloudpickle, "load", forbidden_load)
    with pytest.raises(UntrustedArtifactError, match="trusted=True"):
        load_artifact(path, trusted=False)
    assert not called


def test_checksum_corruption_is_rejected_before_deserialization(tmp_path, monkeypatch):
    source = tmp_path / "source.bundle"
    corrupt = tmp_path / "corrupt.bundle"
    save_artifact(source, {"answer": 42}, metadata())
    rewrite_archive(
        source,
        corrupt,
        lambda members: members.__setitem__(
            "payload.pkl", members["payload.pkl"] + b"x"
        ),
    )
    called = False

    def forbidden_loads(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("deserializer must not run")

    monkeypatch.setattr(cloudpickle, "load", forbidden_loads)
    with pytest.raises(ArtifactIntegrityError, match="size|checksum"):
        load_artifact(corrupt, trusted=True)
    assert not called


def test_unsupported_newer_format_is_rejected(tmp_path):
    source = tmp_path / "source.bundle"
    newer = tmp_path / "newer.bundle"
    save_artifact(source, {"answer": 42}, metadata())

    def change_version(members):
        manifest = json.loads(members["manifest.json"])
        manifest["format_version"] = 999
        members["manifest.json"] = json.dumps(manifest).encode()

    rewrite_archive(source, newer, change_version)
    with pytest.raises(ArtifactError, match="newer than supported"):
        inspect_artifact(newer)


def test_missing_member_is_rejected(tmp_path):
    path = tmp_path / "missing.bundle"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", b"{}")
    with pytest.raises(ArtifactIntegrityError, match="missing required members"):
        inspect_artifact(path)


def test_failed_save_preserves_existing_artifact(tmp_path, monkeypatch):
    path = tmp_path / "model.bundle"
    path.write_bytes(b"existing")

    def fail_serialization(value):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(cloudpickle, "dumps", fail_serialization)
    with pytest.raises(RuntimeError, match="serialization failed"):
        save_artifact(path, {"answer": 42}, metadata())
    assert path.read_bytes() == b"existing"
    assert not (tmp_path / "model.bundle.tmp").exists()


def test_legacy_production_bundle_can_be_migrated(tmp_path):
    legacy_path = tmp_path / "legacy-production.bundle"
    migrated_path = tmp_path / "migrated.bundle"
    bundle = ProductionBundle(
        project_config=ProjectConfig(target_column="target"),
        models_config=ModelsConfig(models={}),
        ensemble_config=EnsembleConfig(),
        feature_set=None,
        models={},
    )
    del bundle.training_feature_schema
    del bundle.training_data_fingerprint
    legacy_path.write_bytes(cloudpickle.dumps(bundle))

    with pytest.warns(FutureWarning, match="legacy"):
        loaded = ProductionBundle.load(legacy_path, trusted=True)
    assert loaded.training_feature_schema == []
    assert loaded.training_data_fingerprint == ""
    loaded.save(migrated_path)
    assert ProductionBundle.inspect(migrated_path)["format_version"] == 1


def test_trusted_legacy_pickle_loads_with_migration_warning(tmp_path):
    path = tmp_path / "legacy.bundle"
    path.write_bytes(cloudpickle.dumps({"legacy": True}))
    manifest = inspect_artifact(path)
    assert manifest["format_version"] == 0
    with pytest.warns(FutureWarning, match="legacy"):
        assert load_artifact(path, trusted=True) == {"legacy": True}
