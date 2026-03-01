"""Failure-tolerant refresh orchestrator for data sources."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from easyml.data.sources import SourceRegistry


class RefreshOrchestrator:
    """Runs all registered data sources with failure tolerance.

    Individual source failures are captured and reported but do not
    stop other sources from running (scrapers may be temporarily down).
    """

    def __init__(self, sources: SourceRegistry) -> None:
        self._sources = sources

    def refresh_all(
        self,
        config: dict[str, Any],
        category: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run all registered sources and return a status report.

        Parameters
        ----------
        config:
            Configuration dict passed to each source function.
        category:
            If set, only run sources in this category.

        Returns
        -------
        Dict mapping source name to status dict with keys:
        - status: "success" or "error"
        - error: error message (only if status == "error")
        """
        report: dict[str, dict[str, Any]] = {}

        for meta in self._sources.list_sources(category=category):
            try:
                # Use the first output path as the default output_dir
                output_dir = Path(meta.outputs[0]) if meta.outputs else Path(".")
                self._sources.run(meta.name, output_dir=output_dir, config=config)
                report[meta.name] = {"status": "success"}
            except Exception as exc:
                report[meta.name] = {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

        return report

    def refresh_source(
        self,
        name: str,
        config: dict[str, Any],
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run a single source by name.

        Parameters
        ----------
        name:
            Registered source name.
        config:
            Configuration dict passed to the source function.
        output_dir:
            Override output directory. If None, uses the first declared output.
        """
        meta = self._sources.get_metadata(name)
        if output_dir is None:
            output_dir = Path(meta.outputs[0]) if meta.outputs else Path(".")
        else:
            output_dir = Path(output_dir)

        try:
            self._sources.run(name, output_dir=output_dir, config=config)
            return {"status": "success"}
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
