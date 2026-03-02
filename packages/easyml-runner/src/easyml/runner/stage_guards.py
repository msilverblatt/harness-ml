"""Pipeline stage guards for PipelineRunner.

Builds pre-configured StageGuard instances for each pipeline stage
using DataConfig paths. Guards validate that prerequisite artifacts
exist and are not stale before execution.
"""
from __future__ import annotations

import logging
from pathlib import Path

from easyml.data.guards import GuardrailViolationError, StageGuard
from easyml.runner.schema import DataConfig

logger = logging.getLogger(__name__)


class PipelineGuards:
    """Factory for pipeline stage guards.

    Parameters
    ----------
    data_config : DataConfig
        Data directory layout from the project config.
    project_dir : Path
        Root project directory (used for relative path resolution).
    enabled : bool
        If False, all guard checks are no-ops.
    """

    def __init__(
        self,
        data_config: DataConfig,
        project_dir: Path,
        enabled: bool = True,
    ) -> None:
        self.data_config = data_config
        self.project_dir = project_dir
        self.enabled = enabled

    def _resolve_features_path(self) -> Path:
        """Resolve the features parquet file path from config.

        Uses features_dir + features_file from DataConfig.
        If the config's features_dir is absolute, use it directly.
        Otherwise resolve relative to project_dir.
        """
        features_dir = Path(self.data_config.features_dir)
        if not features_dir.is_absolute():
            features_dir = self.project_dir / features_dir
        return features_dir / self.data_config.features_file

    def _uses_view(self) -> bool:
        """Check if the config uses a features_view instead of a parquet file."""
        return bool(self.data_config.features_view)

    def guard_train(self) -> None:
        """Validate prerequisites for training: feature files must exist."""
        if not self.enabled or self._uses_view():
            return
        features_path = self._resolve_features_path()
        guard = StageGuard(name="train_ready", requires=[str(features_path)])
        guard.check()

    def guard_predict(self, models_dir: Path | None = None) -> None:
        """Validate prerequisites for prediction: features + models must exist."""
        if not self.enabled:
            return
        requires: list[str] = []
        if not self._uses_view():
            features_path = self._resolve_features_path()
            requires.append(str(features_path))
        if models_dir is not None:
            requires.append(str(models_dir))
        if requires:
            guard = StageGuard(name="predict_ready", requires=requires)
            guard.check()

    def guard_backtest(self) -> None:
        """Validate prerequisites for backtesting: features with sufficient rows."""
        if not self.enabled or self._uses_view():
            return
        features_path = self._resolve_features_path()
        guard = StageGuard(
            name="backtest_ready",
            requires=[str(features_path)],
            min_rows=10,
        )
        guard.check()

    def warn_if_stale(
        self,
        artifacts: list[str],
        sources: list[str],
        stage_name: str,
    ) -> bool:
        """Check if artifacts are stale relative to sources.

        Returns True if stale, False otherwise.  Staleness is a warning,
        not a blocking error — the caller decides what to do.
        """
        if not self.enabled:
            return False
        try:
            guard = StageGuard(
                name=f"{stage_name}_staleness",
                requires=artifacts,
                stale_if_older_than=sources,
            )
            guard.check()
            return False
        except GuardrailViolationError:
            logger.warning("Staleness detected for stage %s", stage_name)
            return True
