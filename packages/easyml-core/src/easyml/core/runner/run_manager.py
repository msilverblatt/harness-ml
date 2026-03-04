"""Run management for versioned pipeline output directories.

Manages timestamped run directories with promotion via symlinks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class RunManager:
    """Manages versioned pipeline output directories.

    Directory structure:
        outputs_base/
            YYYYMMDD_HHMMSS/      # run directory
                predictions/
                diagnostics/
            current -> YYYYMMDD_HHMMSS  # symlink to promoted run

    Parameters
    ----------
    outputs_base : Path
        Base directory for all run outputs.
    """

    def __init__(self, outputs_base: Path) -> None:
        self.outputs_base = Path(outputs_base)

    def new_run(self, run_id: str | None = None) -> Path:
        """Create a new run directory.

        Parameters
        ----------
        run_id : str | None
            Optional custom run identifier. If None, a timestamp-based
            ID (YYYYMMDD_HHMMSS) is generated.

        Returns
        -------
        Path
            Path to the newly created run directory.
        """
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        run_dir = self.outputs_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        (run_dir / "predictions").mkdir(exist_ok=True)
        (run_dir / "diagnostics").mkdir(exist_ok=True)

        logger.info("Created run directory: %s", run_dir)
        return run_dir

    def promote(self, run_id: str) -> None:
        """Promote a run by updating the 'current' symlink.

        Parameters
        ----------
        run_id : str
            The run directory name to promote.

        Raises
        ------
        FileNotFoundError
            If the run directory does not exist.
        """
        run_dir = self.outputs_base / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        current_link = self.outputs_base / "current"

        # Remove existing symlink if present
        if current_link.is_symlink():
            current_link.unlink()
        elif current_link.exists():
            # current exists but is not a symlink — don't overwrite
            raise ValueError(
                f"'current' exists at {current_link} but is not a symlink. "
                "Remove it manually before promoting."
            )

        current_link.symlink_to(run_id)
        logger.info("Promoted run %s -> current", run_id)

    def list_runs(self) -> list[dict]:
        """List all run directories, sorted by name (newest first).

        Returns
        -------
        list of dict
            Each dict has 'run_id', 'path', and 'is_current' keys.
        """
        if not self.outputs_base.exists():
            return []

        current_target = self._resolve_current()

        runs = []
        for entry in sorted(self.outputs_base.iterdir(), reverse=True):
            if entry.name == "current":
                continue
            if not entry.is_dir():
                continue

            is_current = (current_target is not None and entry.name == current_target)
            runs.append({
                "run_id": entry.name,
                "path": str(entry),
                "is_current": is_current,
            })

        return runs

    def get_current(self) -> Path | None:
        """Get the path to the currently promoted run.

        Returns
        -------
        Path | None
            Path to the current run directory, or None if no run is promoted.
        """
        current_link = self.outputs_base / "current"
        if not current_link.is_symlink():
            return None

        target = current_link.resolve()
        if target.exists():
            return target
        return None

    def _resolve_current(self) -> str | None:
        """Resolve the 'current' symlink target name.

        Returns
        -------
        str | None
            The run_id that 'current' points to, or None.
        """
        current_link = self.outputs_base / "current"
        if not current_link.is_symlink():
            return None

        # readlink gives the symlink target (relative or absolute)
        target = Path(current_link.readlink())
        return target.name
