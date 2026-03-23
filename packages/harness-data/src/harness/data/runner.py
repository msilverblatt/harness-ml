"""Pipeline runner — loads sources, applies transforms, writes parquet + schema."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from harness.data.sources.file import FileSource
from harness.data.sources.url import UrlSource
from harness.data.sources.api import ApiSource
from harness.data.sources.protocol import SourceConfig
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig

SOURCE_ADAPTERS: dict[str, Any] = {
    "file": FileSource(),
    "url": UrlSource(),
    "api": ApiSource(),
}


@dataclass
class PipelineResult:
    row_count: int
    column_count: int
    columns: list[str]
    output_path: str
    schema_path: str
    data_hash: str


class PipelineRunner:
    """Stateless runner: sources + transforms → parquet + schema.json."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self._workspace = Path(workspace_dir)
        self._engine = TransformEngine()

    def run(
        self,
        sources: list[dict],
        transforms: list[dict],
    ) -> PipelineResult:
        if not sources:
            raise ValueError("At least one source is required")

        frames = self._load_sources(sources)
        df = self._merge_frames(frames)
        df = self._apply_transforms(df, transforms)

        output_dir = self._workspace / "data" / "clean"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "dataset.parquet"
        schema_path = output_dir / "schema.json"

        df.to_parquet(str(output_path), index=False)
        data_hash = self._hash_parquet(output_path)

        schema = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "data_hash": data_hash,
        }
        schema_path.write_text(json.dumps(schema, indent=2))

        return PipelineResult(
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            output_path=str(output_path),
            schema_path=str(schema_path),
            data_hash=data_hash,
        )

    def _load_sources(self, sources: list[dict]) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for src in sources:
            src_type = src.get("source_type", "file")
            adapter = SOURCE_ADAPTERS.get(src_type)
            if adapter is None:
                raise ValueError(f"Unknown source_type: '{src_type}'")
            config = SourceConfig(**{k: v for k, v in src.items()})
            base_dir = str(self._workspace) if not Path(config.path or "").is_absolute() else None
            df = adapter.load(config, base_dir=base_dir)
            frames.append(df)
        return frames

    def _merge_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if len(frames) == 1:
            return frames[0].copy()
        result = frames[0]
        for other in frames[1:]:
            common = list(set(result.columns) & set(other.columns))
            if common:
                result = result.merge(other, on=common, how="left")
            else:
                result = pd.concat([result, other], ignore_index=True)
        return result

    def _apply_transforms(self, df: pd.DataFrame, transforms: list[dict]) -> pd.DataFrame:
        if not transforms:
            return df
        steps = [StepConfig(op=t["op"], params=t.get("params", {})) for t in transforms]
        return self._engine.run_pipeline(df, steps)

    def _hash_parquet(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
