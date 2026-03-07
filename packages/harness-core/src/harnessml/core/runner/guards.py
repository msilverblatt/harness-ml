"""Stage guards with staleness detection.

Guards validate preconditions before a pipeline stage runs:
- Required files exist
- Parquet files meet minimum row counts
- Artifacts are not stale relative to config/source files
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from harnessml.core.schemas.contracts import GuardrailViolation


class GuardrailViolationError(Exception):
    """Exception raised when a stage guard check fails.

    Wraps a GuardrailViolation model with the violation details.
    """

    def __init__(self, violation: GuardrailViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


class StageGuard:
    """Validates preconditions before a pipeline stage executes.

    Parameters
    ----------
    name:
        Human-readable guard name for error messages.
    requires:
        List of file/directory paths that must exist.
    min_rows:
        If set, any .parquet file in ``requires`` must have at least this
        many rows.
    stale_if_older_than:
        If set, any file in ``requires`` must be newer than every file in
        this list (i.e., artifacts must be regenerated after config changes).
    """

    def __init__(
        self,
        name: str,
        requires: list[str],
        min_rows: int | None = None,
        stale_if_older_than: list[str] | None = None,
    ) -> None:
        self.name = name
        self.requires = requires
        self.min_rows = min_rows
        self.stale_if_older_than = stale_if_older_than or []

    def check(self) -> None:
        """Run all guard checks. Raises GuardrailViolationError on failure."""
        self._check_existence()
        if self.min_rows is not None:
            self._check_min_rows()
        if self.stale_if_older_than:
            self._check_staleness()

    def _check_existence(self) -> None:
        """Verify all required paths exist."""
        for path_str in self.requires:
            p = Path(path_str)
            if not p.exists():
                raise GuardrailViolationError(
                    GuardrailViolation(
                        blocked=True,
                        rule=f"guard:{self.name}:exists",
                        message=f"Required path does not exist: {path_str}",
                        source=self.name,
                    )
                )

    def _check_min_rows(self) -> None:
        """Verify parquet files meet minimum row count."""
        for path_str in self.requires:
            p = Path(path_str)
            if p.is_file() and p.suffix == ".parquet":
                df = pd.read_parquet(p)
                if len(df) < self.min_rows:
                    raise GuardrailViolationError(
                        GuardrailViolation(
                            blocked=True,
                            rule=f"guard:{self.name}:min_rows",
                            message=(
                                f"{path_str} has {len(df)} rows, "
                                f"expected at least {self.min_rows}"
                            ),
                            source=self.name,
                        )
                    )

    def _check_staleness(self) -> None:
        """Verify required files are newer than reference files."""
        # Find the newest reference file mtime
        newest_ref_mtime: float = 0
        newest_ref_path: str = ""
        for ref_path_str in self.stale_if_older_than:
            ref_p = Path(ref_path_str)
            if ref_p.exists():
                ref_mtime = ref_p.stat().st_mtime
                if ref_mtime > newest_ref_mtime:
                    newest_ref_mtime = ref_mtime
                    newest_ref_path = ref_path_str

        if newest_ref_mtime == 0:
            return  # No reference files exist, nothing to compare

        # Check that all required files are newer than the newest reference
        for path_str in self.requires:
            p = Path(path_str)
            if p.exists():
                artifact_mtime = p.stat().st_mtime
                if artifact_mtime < newest_ref_mtime:
                    raise GuardrailViolationError(
                        GuardrailViolation(
                            blocked=True,
                            rule=f"guard:{self.name}:stale",
                            message=(
                                f"{path_str} is stale — older than "
                                f"{newest_ref_path}"
                            ),
                            source=self.name,
                        )
                    )
