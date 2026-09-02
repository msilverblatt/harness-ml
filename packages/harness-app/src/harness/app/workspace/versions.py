from dataclasses import dataclass, field
from pathlib import Path
import shutil
import uuid
import yaml

from harness.app.workspace.locking import atomic_write_json, atomic_write_text


@dataclass
class VersionMeta:
    id: str
    parent: str | None = None
    experiment_type: str | None = None
    hypothesis: str = ""
    conclusion: str = ""
    verdict: str = ""
    timestamp: str = ""
    data_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


class VersionTree:
    def __init__(self, workspace_dir: Path):
        self._root = Path(workspace_dir)
        self._versions_dir = self._root / "versions"

    def create_version(
        self, meta: VersionMeta, config_manager, diff: dict | None = None
    ) -> str:
        """Atomically create a version directory with metadata and config snapshot."""
        staging = self.stage_version(meta, config_manager, diff)
        atomic_write_json(staging / "run" / "state.json", {"status": "complete"})
        self.publish_version(staging, meta.id)
        return meta.id

    def stage_version(
        self, meta: VersionMeta, config_manager, diff: dict | None = None
    ) -> Path:
        """Create a hidden version directory that can receive run artifacts."""
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        version_dir = self._versions_dir / meta.id
        if version_dir.exists():
            raise ValueError(f"Version already exists: {meta.id}")
        staging = self._versions_dir / f".{meta.id}.{uuid.uuid4().hex}.tmp"
        try:
            staging.mkdir(parents=True)
            meta_dict = {k: v for k, v in vars(meta).items() if v}
            (staging / "meta.yaml").write_text(
                yaml.dump(meta_dict, default_flow_style=False, sort_keys=False)
            )
            (staging / "diff.yaml").write_text(
                yaml.dump(diff or {}, default_flow_style=False, sort_keys=False)
            )
            config_manager.snapshot_config(staging / "config")
            (staging / "run").mkdir()
            atomic_write_json(staging / "run" / "state.json", {"status": "running"})
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return staging

    def publish_version(self, staging: Path, version_id: str) -> None:
        version_dir = self._versions_dir / version_id
        if version_dir.exists():
            raise ValueError(f"Version already exists: {version_id}")
        staging.replace(version_dir)

    def delete_version(self, version_id: str) -> None:
        version_dir = self._versions_dir / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)

    def get_version(self, version_id: str) -> VersionMeta | None:
        meta_path = self._versions_dir / version_id / "meta.yaml"
        if not meta_path.exists():
            return None
        data = yaml.safe_load(meta_path.read_text()) or {}
        return VersionMeta(**data)

    def update_version(self, version_id: str, **kwargs):
        meta = self.get_version(version_id)
        if meta is None:
            raise ValueError(f"Version not found: {version_id}")
        for k, v in kwargs.items():
            setattr(meta, k, v)
        meta_path = self._versions_dir / version_id / "meta.yaml"
        meta_dict = {k: v for k, v in vars(meta).items() if v}
        atomic_write_text(
            meta_path,
            yaml.dump(meta_dict, default_flow_style=False, sort_keys=False),
        )

    def get_current(self) -> str | None:
        pointer = self._root / "current"
        if not pointer.exists():
            return None
        return pointer.read_text().strip()

    def set_current(self, version_id: str, config_manager):
        version_dir = self._versions_dir / version_id
        if not version_dir.exists():
            raise ValueError(f"Version not found: {version_id}")
        config_manager.restore_config(version_dir / "config")
        atomic_write_text(self._root / "current", version_id)

    def list_versions(self) -> list[VersionMeta]:
        if not self._versions_dir.exists():
            return []
        versions = []
        for d in sorted(self._versions_dir.iterdir()):
            if d.is_dir() and (d / "meta.yaml").exists():
                data = yaml.safe_load((d / "meta.yaml").read_text()) or {}
                versions.append(VersionMeta(**data))
        return versions

    def next_version_id(self) -> str:
        existing = self.list_versions()
        if not existing:
            return "v001"
        nums = []
        for v in existing:
            try:
                nums.append(int(v.id.lstrip("v")))
            except ValueError:
                pass
        return f"v{max(nums) + 1:03d}" if nums else "v001"

    def compare(self, v1: str, v2: str) -> dict:
        m1 = self.get_version(v1)
        m2 = self.get_version(v2)
        if m1 is None or m2 is None:
            raise ValueError("Version not found")
        deltas = {}
        for key in set(list(m1.metrics.keys()) + list(m2.metrics.keys())):
            val1 = m1.metrics.get(key, float("nan"))
            val2 = m2.metrics.get(key, float("nan"))
            deltas[key] = {"v1": val1, "v2": val2, "delta": val2 - val1}
        return deltas

    def ancestry(self, version_id: str) -> list[VersionMeta]:
        chain = []
        current = version_id
        while current:
            meta = self.get_version(current)
            if meta is None:
                break
            chain.append(meta)
            current = meta.parent
        chain.reverse()
        return chain
