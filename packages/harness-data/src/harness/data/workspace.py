"""DataWorkspace — manages the data layer of a harness workspace."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager

import pandas as pd
import yaml

from harness.data.sources.protocol import SourceConfig
from harness.data.sources.registry import SourceRegistry
from harness.data.io import atomic_write_text
from harness.data.runner import PipelineResult, PipelineRunner


class DataWorkspace:
    """Manages the data layer of a harness workspace."""

    def __init__(
        self,
        workspace_dir: str | Path,
        mutation_guard: Callable[[str], ContextManager] | None = None,
    ) -> None:
        self._root = Path(workspace_dir)
        self._mutation_guard = mutation_guard
        self._data_dir = self._root / "data"
        self._sources_yaml = self._data_dir / "sources.yaml"
        self._transforms_yaml = self._data_dir / "transforms.yaml"

    def init(self) -> None:
        """Create directory structure and empty config files."""
        with self._guard("data_init"):
            (self._data_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self._data_dir / "clean").mkdir(parents=True, exist_ok=True)

            if not self._sources_yaml.exists():
                atomic_write_text(
                    self._sources_yaml,
                    yaml.dump({"sources": {}}, default_flow_style=False),
                )
            if not self._transforms_yaml.exists():
                atomic_write_text(
                    self._transforms_yaml,
                    yaml.dump({"transforms": []}, default_flow_style=False),
                )

    def add_source(self, name: str, path: str, **kwargs: Any) -> None:
        """Register a source via SourceRegistry."""
        with self._guard("data_add_source"):
            registry = SourceRegistry(self._data_dir)
            config = SourceConfig(name=name, path=path, **kwargs)
            registry.add(config)

    def list_sources(self) -> list[SourceConfig]:
        """List all registered sources."""
        registry = SourceRegistry(self._data_dir)
        return registry.list_all()

    def add_transform(self, step: dict) -> None:
        """Append a transform step to transforms.yaml."""
        with self._guard("data_add_transform"):
            steps = self.load_transforms()
            steps.append(step)
            atomic_write_text(
                self._transforms_yaml,
                yaml.dump(
                    {"transforms": steps}, default_flow_style=False, sort_keys=False
                ),
            )

    def load_transforms(self) -> list[dict]:
        """Read transform steps from transforms.yaml."""
        if not self._transforms_yaml.exists():
            return []
        content = yaml.safe_load(self._transforms_yaml.read_text()) or {}
        return content.get("transforms", []) or []

    def run_pipeline(self) -> PipelineResult:
        """Execute: load sources + transforms → run PipelineRunner."""
        with self._guard("data_run_pipeline"):
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
        schema = json.loads(path.read_text())
        dataset = self._data_dir / "clean" / "dataset.parquet"
        if dataset.exists() and schema.get("data_hash") != _hash_file(dataset):
            raise RuntimeError(
                "Clean dataset and schema are inconsistent; rerun the data pipeline"
            )
        return schema

    def _guard(self, operation: str) -> ContextManager:
        if self._mutation_guard is None:
            return nullcontext()
        return self._mutation_guard(operation)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
