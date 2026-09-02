from dataclasses import asdict

from harness.data.profiling.profiler import DataProfiler
from harness.server.context import require_workspace
from protomcp import action, tool_group


@tool_group("data", description="Manage sources and the declarative data pipeline.")
class DataTools:
    @action("add_source", description="Register a data source.")
    def add_source(
        self, name: str, path: str, source_type: str = "file", params: dict | None = None
    ) -> dict:
        workspace = require_workspace()
        workspace.data.add_source(
            name, path, source_type=source_type, params=params or {}
        )
        return {"name": name, "path": path, "source_type": source_type}

    @action("transform", description="Append a declarative transform step.")
    def transform(self, op: str, params: dict | None = None) -> dict:
        workspace = require_workspace()
        step = {"op": op, "params": params or {}}
        workspace.data.add_transform(step)
        return step

    @action("run", description="Execute the data pipeline.")
    def run(self) -> dict:
        result = require_workspace().data.run_pipeline()
        return asdict(result)

    @action("profile", description="Profile the clean dataset.")
    def profile(self) -> dict:
        workspace = require_workspace()
        return asdict(DataProfiler().profile(workspace.data.load_clean_data()))

    @action("inspect", description="Preview clean data and its schema.")
    def inspect(self, rows: int = 10) -> dict:
        if rows < 1 or rows > 1000:
            raise ValueError("rows must be between 1 and 1000")
        workspace = require_workspace()
        frame = workspace.data.load_clean_data()
        try:
            schema = workspace.data.load_schema()
        except FileNotFoundError:
            schema = {
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "columns": list(frame.columns),
            }
        return {
            "schema": schema,
            "rows": frame.head(rows).where(frame.notna(), None).to_dict(orient="records"),
        }
