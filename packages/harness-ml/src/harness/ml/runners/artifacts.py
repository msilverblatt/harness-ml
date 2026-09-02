from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cloudpickle

ARTIFACT_TYPE = "harness.production_bundle"
FORMAT_VERSION = 1
MANIFEST_MEMBER = "manifest.json"
PAYLOAD_MEMBER = "payload.pkl"
MAX_MANIFEST_BYTES = 1024 * 1024
_REQUIRED_MANIFEST_FIELDS = {
    "artifact_type",
    "format_version",
    "created_at",
    "payload",
    "runtime",
    "packages",
    "task",
    "models",
    "training_features",
    "ensemble_columns",
    "fingerprints",
    "output",
}


class ArtifactError(ValueError):
    """Base error for an invalid or incompatible Harness artifact."""


class UntrustedArtifactError(ArtifactError):
    """Raised before an executable artifact is deserialized without consent."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact does not match its manifest."""


def package_versions() -> dict[str, str]:
    packages = {}
    for distribution in (
        "harness-ml",
        "numpy",
        "pandas",
        "scikit-learn",
        "cloudpickle",
    ):
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "unknown"
    return packages


def runtime_metadata() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def save_artifact(path: str | Path, value: Any, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = cloudpickle.dumps(value)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "payload": {
            "member": PAYLOAD_MEMBER,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "runtime": runtime_metadata(),
        "packages": package_versions(),
        **metadata,
    }
    _validate_manifest_shape(manifest)
    encoded_manifest = json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(MANIFEST_MEMBER, encoded_manifest)
            archive.writestr(PAYLOAD_MEMBER, payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
        if os.name != "nt":
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not zipfile.is_zipfile(source):
        return {
            "artifact_type": "harness.production_bundle.legacy_pickle",
            "format_version": 0,
            "legacy": True,
            "warning": "Legacy bundles require trusted pickle deserialization to inspect further.",
        }
    with zipfile.ZipFile(source, mode="r") as archive:
        return _inspect_archive(archive)


def load_artifact(path: str | Path, *, trusted: bool) -> Any:
    if not trusted:
        raise UntrustedArtifactError(
            "Production bundles contain executable pickle data. Load only artifacts "
            "from a trusted source and pass trusted=True explicitly."
        )
    source = Path(path)
    if not zipfile.is_zipfile(source):
        warnings.warn(
            "Loading a legacy raw-pickle production bundle; re-save it to upgrade "
            "to the versioned artifact format.",
            FutureWarning,
            stacklevel=2,
        )
        with source.open("rb") as handle:
            return cloudpickle.load(handle)

    # Validate and deserialize through the same open file descriptor so the path
    # cannot be swapped between integrity verification and pickle loading.
    with zipfile.ZipFile(source, mode="r") as archive:
        _inspect_archive(archive)
        with archive.open(PAYLOAD_MEMBER) as payload:
            return cloudpickle.load(payload)


def _inspect_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = set(archive.namelist())
    required_members = {MANIFEST_MEMBER, PAYLOAD_MEMBER}
    missing = required_members - names
    if missing:
        raise ArtifactIntegrityError(
            f"Production bundle is missing required members: {sorted(missing)}"
        )
    unexpected = names - required_members
    if unexpected:
        raise ArtifactIntegrityError(
            f"Production bundle has unexpected members: {sorted(unexpected)}"
        )
    if archive.getinfo(MANIFEST_MEMBER).file_size > MAX_MANIFEST_BYTES:
        raise ArtifactIntegrityError("Production bundle manifest exceeds 1 MiB")
    try:
        manifest = json.loads(archive.read(MANIFEST_MEMBER))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactIntegrityError(
            "Production bundle manifest is invalid JSON"
        ) from error
    _validate_manifest_shape(manifest)
    expected = manifest["payload"]
    if expected.get("member") != PAYLOAD_MEMBER:
        raise ArtifactIntegrityError("Manifest names an unsupported payload member")
    payload_info = archive.getinfo(PAYLOAD_MEMBER)
    if expected.get("size_bytes") != payload_info.file_size:
        raise ArtifactIntegrityError(
            "Production bundle payload size does not match manifest"
        )
    digest = hashlib.sha256()
    with archive.open(PAYLOAD_MEMBER) as payload:
        for chunk in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(chunk)
    if expected.get("sha256") != digest.hexdigest():
        raise ArtifactIntegrityError(
            "Production bundle payload checksum does not match manifest"
        )
    return manifest


def _validate_manifest_shape(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactIntegrityError("Production bundle manifest must be a JSON object")
    missing = _REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ArtifactIntegrityError(
            f"Production bundle manifest is missing fields: {sorted(missing)}"
        )
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactError(
            f"Unsupported artifact type: {manifest.get('artifact_type')!r}"
        )
    format_version = manifest.get("format_version")
    if not isinstance(format_version, int):
        raise ArtifactError("Production bundle format_version must be an integer")
    if format_version > FORMAT_VERSION:
        raise ArtifactError(
            f"Production bundle format {format_version} is newer than supported "
            f"format {FORMAT_VERSION}"
        )
    if format_version < 1:
        raise ArtifactError(f"Unsupported production bundle format: {format_version}")
    if not isinstance(manifest.get("payload"), dict):
        raise ArtifactIntegrityError("Production bundle payload metadata is invalid")
