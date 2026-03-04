"""Versioned output management with timestamped directories, manifest, and symlink.

Each output category (models, predictions, backtest, etc.) gets a ``runs/``
subdirectory containing timestamped entries.  A ``manifest.json`` tracks
metadata and a ``latest`` symlink points to the promoted run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunManager:
    """Manage versioned output directories with manifest and symlink.

    Parameters
    ----------
    base_dir : Path | str
        Root directory for this output category (e.g. ``models/``,
        ``predictions/``).
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.manifest_path = self.base_dir / "manifest.json"
        self.symlink_path = self.base_dir / "latest"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def new_run(self, run_id: str | None = None) -> Path:
        """Create a new timestamped run directory.

        Parameters
        ----------
        run_id : str | None
            Explicit run ID (``YYYYMMDD_HHMMSS`` format).  Generated
            from the current UTC time if omitted.

        Returns
        -------
        Path
            The newly created run directory.
        """
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def promote(self, run_name: str, label: str = "current") -> None:
        """Point a manifest label (and the symlink) at a run.

        Parameters
        ----------
        run_name : str
            Run directory name (the run_id).
        label : str
            Manifest label (default ``"current"``).  The ``latest``
            symlink always tracks the ``"current"`` label.
        """
        run_dir = self.runs_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        manifest = self._load_manifest()
        manifest[label] = run_name

        if "runs" not in manifest:
            manifest["runs"] = {}
        if run_name not in manifest["runs"]:
            manifest["runs"][run_name] = {
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

        self._save_manifest(manifest)

        if label == "current":
            self._update_symlink(run_name)

    def get_latest(self) -> Path | None:
        """Return the path to the ``current`` run, or ``None``.

        Reads the manifest first, then falls back to the symlink.
        """
        manifest = self._load_manifest()
        current_id = manifest.get("current")
        if current_id:
            run_dir = self.runs_dir / current_id
            if run_dir.exists():
                return run_dir

        if self.symlink_path.is_symlink():
            target = self.symlink_path.resolve()
            if target.exists():
                return target

        return None

    def list_runs(self) -> list[dict]:
        """List all runs sorted by name.

        Returns
        -------
        list[dict]
            Each dict contains ``run_id``, ``created``, ``labels``,
            and ``path``.
        """
        if not self.runs_dir.exists():
            return []

        manifest = self._load_manifest()
        runs_meta = manifest.get("runs", {})

        label_map: dict[str, list[str]] = {}
        for key, value in manifest.items():
            if key != "runs" and isinstance(value, str):
                label_map.setdefault(value, []).append(key)

        results = []
        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            meta = runs_meta.get(run_id, {})
            results.append({
                "run_id": run_id,
                "created": meta.get("created", ""),
                "labels": label_map.get(run_id, []),
                "path": run_dir,
            })

        return results

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def write_manifest(self, run_path: Path, **kwargs: object) -> None:
        """Write arbitrary metadata into a run-level manifest.

        Parameters
        ----------
        run_path : Path
            The run directory to write the manifest into.
        **kwargs
            Arbitrary key-value pairs to store.
        """
        manifest_file = run_path / "manifest.json"
        existing = {}
        if manifest_file.exists():
            existing = json.loads(manifest_file.read_text())
        existing.update(kwargs)
        manifest_file.write_text(json.dumps(existing, indent=2, default=str) + "\n")

    def read_manifest(self, run_path: Path) -> dict:
        """Read the run-level manifest.

        Parameters
        ----------
        run_path : Path
            The run directory containing the manifest.

        Returns
        -------
        dict
        """
        manifest_file = run_path / "manifest.json"
        if not manifest_file.exists():
            return {}
        return json.loads(manifest_file.read_text())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        tmp.rename(self.manifest_path)

    def _update_symlink(self, run_id: str) -> None:
        target = Path("runs") / run_id  # relative symlink
        if self.symlink_path.is_symlink() or self.symlink_path.exists():
            self.symlink_path.unlink()
        self.symlink_path.symlink_to(target)
