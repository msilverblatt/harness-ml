"""DataWorkspace — manages the data layer of a harness workspace."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from harness.data.runner import PipelineResult, PipelineRunner
from harness.data.sources.protocol import SourceConfig
from harness.data.sources.registry import SourceRegistry


class DataWorkspace:
    """Manages the data layer of a harness workspace."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self._root = Path(workspace_dir)
        self._data_dir = self._root / "data"
        self._sources_yaml = self._data_dir / "sources.yaml"
        self._transforms_yaml = self._data_dir / "transforms.yaml"

    def init(self) -> None:
        """Create directory structure and empty config files."""
        (self._data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self._data_dir / "clean").mkdir(parents=True, exist_ok=True)

        if not self._sources_yaml.exists():
            self._sources_yaml.write_text(
                yaml.dump({"sources": {}}, default_flow_style=False)
            )
        if not self._transforms_yaml.exists():
            self._transforms_yaml.write_text(
                yaml.dump({"transforms": []}, default_flow_style=False)
            )

    def add_source(self, name: str, path: str, **kwargs: Any) -> None:
        """Register a source via SourceRegistry."""
        registry = SourceRegistry(self._data_dir)
        config = SourceConfig(name=name, path=path, **kwargs)
        registry.add(config)

    def list_sources(self) -> list[SourceConfig]:
        """List all registered sources."""
        registry = SourceRegistry(self._data_dir)
        return registry.list_all()

    def add_transform(self, step: dict) -> None:
        """Append a transform step to transforms.yaml."""
        steps = self.load_transforms()
        steps.append(step)
        self._transforms_yaml.write_text(
            yaml.dump({"transforms": steps}, default_flow_style=False, sort_keys=False)
        )

    def load_transforms(self) -> list[dict]:
        """Read transform steps from transforms.yaml."""
        if not self._transforms_yaml.exists():
            return []
        content = yaml.safe_load(self._transforms_yaml.read_text()) or {}
        return content.get("transforms", []) or []

    def run_pipeline(self) -> PipelineResult:
        """Execute: load sources + transforms → run PipelineRunner."""
        sources = [
            {"name": cfg.name, "source_type": cfg.source_type, "path": cfg.path}
            | {k: v for k, v in cfg.params.items()}
            for cfg in self.list_sources()
            if cfg.enabled
        ]
        transforms = self.load_transforms()
        runner = PipelineRunner(self._root)
        return runner.run(sources=sources, transforms=transforms)

    def load_clean_data(self) -> pd.DataFrame:
        """Read data/clean/dataset.parquet."""
        path = self._data_dir / "clean" / "dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {path}")
        return pd.read_parquet(str(path))

    def load_schema(self) -> dict:
        """Read data/clean/schema.json."""
        path = self._data_dir / "clean" / "schema.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        return json.loads(path.read_text())
