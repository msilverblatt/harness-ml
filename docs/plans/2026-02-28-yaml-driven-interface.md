# YAML-Driven Orchestration Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single `easyml-runner` package that is the only consumer of the 7 library packages. Users interact exclusively with validated YAML + CLI commands + MCP tools. The existing packages are untouched.

**Architecture:** `easyml-runner` is a thin orchestration layer. It owns: (1) `ProjectConfig` — a Pydantic model that validates the entire YAML config tree, (2) auto-loaders that dynamically import feature/source modules declared in YAML, (3) a `PipelineRunner` that wires library APIs together from config, (4) a Click CLI (`easyml` command), (5) an MCP server generator that creates tools from a YAML server definition, and (6) a project scaffold generator. The 7 library packages remain pure libraries with stable APIs — `easyml-runner` is the only thing that imports them.

**Tech Stack:** Python 3.11+, Click (CLI), Pydantic v2 (validation), FastMCP (server generation), existing easyml-* packages as runtime engine.

---

## Design Principle: Decoupled Orchestration

```
                         YAML Config
                              |
                              v
                    ┌─────────────────┐
                    │  easyml-runner   │  <-- the ONLY consumer
                    │  (orchestration) │
                    └─────┬───────────┘
                          │ imports
          ┌───────────────┼───────────────┐
          v               v               v
   ┌─────────────┐ ┌───────────┐ ┌──────────────┐
   │ easyml-     │ │ easyml-   │ │ easyml-      │
   │ models      │ │ features  │ │ guardrails   │
   └─────────────┘ └───────────┘ └──────────────┘
          │               │               │
          └───────┬───────┘───────────────┘
                  v
          ┌─────────────┐
          │ easyml-     │
          │ schemas     │  <-- shared contracts
          └─────────────┘
```

The 7 library packages are **never modified** by this plan. They stay as stable, independently testable libraries. All new code goes in `easyml-runner`.

---

## Phase 10: Project Config Schema (in easyml-runner)

### Task 10.1: ProjectConfig Pydantic model + validation

**Files:**
- Create: `packages/easyml-runner/pyproject.toml`
- Create: `packages/easyml-runner/src/easyml/runner/__init__.py`
- Create: `packages/easyml-runner/src/easyml/runner/schema.py`
- Create: `packages/easyml-runner/src/easyml/runner/validator.py`
- Create: `packages/easyml-runner/tests/__init__.py`
- Create: `packages/easyml-runner/tests/test_schema.py`
- Create: `packages/easyml-runner/tests/test_validator.py`

The runner owns ALL the project-level Pydantic models. These are NOT in easyml-schemas — they're orchestration-layer concerns.

**Step 1: Create package**

`packages/easyml-runner/pyproject.toml`:
```toml
[project]
name = "easyml-runner"
version = "0.1.0"
description = "YAML-driven orchestration layer for easyml"
requires-python = ">=3.11"
dependencies = [
    "easyml-schemas",
    "easyml-config",
    "easyml-features",
    "easyml-models",
    "easyml-data",
    "easyml-experiments",
    "easyml-guardrails",
    "click>=8.0",
    "pyyaml>=6.0",
]

[project.scripts]
easyml = "easyml.runner.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/easyml"]

[tool.uv.sources]
easyml-schemas = { workspace = true }
easyml-config = { workspace = true }
easyml-features = { workspace = true }
easyml-models = { workspace = true }
easyml-data = { workspace = true }
easyml-experiments = { workspace = true }
easyml-guardrails = { workspace = true }
```

Add `"click>=8.0"` to root `pyproject.toml` dev dependencies.

No `__init__.py` at `src/easyml/` level (namespace package).

**Step 2: Write failing tests for schema.py**

```python
# packages/easyml-runner/tests/test_schema.py
import pytest
from easyml.runner.schema import (
    ProjectConfig, DataConfig, BacktestConfig, ModelDef,
    EnsembleDef, FeatureDecl, SourceDecl, ExperimentDef,
    GuardrailDef, ServerToolDef, ServerDef,
)


class TestProjectConfig:

    def test_minimal_valid(self):
        """Minimal config: data + one model + ensemble + backtest."""
        config = ProjectConfig(
            data=DataConfig(
                raw_dir="data/raw",
                processed_dir="data/processed",
                features_dir="data/features",
            ),
            models={
                "logreg_seed": ModelDef(
                    type="logistic_regression",
                    features=["diff_seed_num"],
                    params={"C": 1.0},
                ),
            },
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(
                cv_strategy="leave_one_season_out",
                seasons=[2023, 2024],
                metrics=["brier", "accuracy"],
            ),
        )
        assert config.models["logreg_seed"].type == "logistic_regression"

    def test_with_features(self):
        """Feature declarations point to Python modules."""
        config = ProjectConfig(
            data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
            features={
                "win_rate": FeatureDecl(
                    module="my_features.resume",
                    function="compute_win_rate",
                    category="resume",
                    level="team",
                    columns=["win_rate"],
                ),
            },
            models={"m": ModelDef(type="logistic_regression", features=["diff_win_rate"], params={})},
            ensemble=EnsembleDef(method="average"),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2024], metrics=["brier"]),
        )
        assert config.features["win_rate"].module == "my_features.resume"

    def test_with_sources(self):
        """Data source declarations."""
        config = ProjectConfig(
            data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
            sources={
                "kenpom": SourceDecl(
                    module="scrapers.kenpom",
                    function="scrape",
                    category="external",
                    temporal_safety="pre_tournament",
                    outputs=["data/external/kenpom/"],
                ),
            },
            models={"m": ModelDef(type="logistic_regression", features=["x"], params={})},
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2024], metrics=["brier"]),
        )
        assert config.sources["kenpom"].temporal_safety == "pre_tournament"

    def test_invalid_model_type_rejects(self):
        with pytest.raises(Exception):
            ProjectConfig(
                data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
                models={"bad": ModelDef(type="", features=[], params={})},
                ensemble=EnsembleDef(method="stacked"),
                backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2024], metrics=["brier"]),
            )

    def test_invalid_cv_strategy_rejects(self):
        with pytest.raises(Exception):
            ProjectConfig(
                data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
                models={"m": ModelDef(type="logistic_regression", features=["x"], params={})},
                ensemble=EnsembleDef(method="stacked"),
                backtest=BacktestConfig(cv_strategy="invalid_strategy", seasons=[2024], metrics=["brier"]),
            )

    def test_invalid_ensemble_method_rejects(self):
        with pytest.raises(Exception):
            ProjectConfig(
                data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
                models={"m": ModelDef(type="logistic_regression", features=["x"], params={})},
                ensemble=EnsembleDef(method="invalid"),
                backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2024], metrics=["brier"]),
            )

    def test_serialization_roundtrip(self):
        config = ProjectConfig(
            data=DataConfig(raw_dir="data/raw", processed_dir="data/processed", features_dir="data/features"),
            models={"m": ModelDef(type="logistic_regression", features=["x"], params={"C": 1.0})},
            ensemble=EnsembleDef(method="stacked", meta_learner={"type": "logistic", "C": 2.5}),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2023, 2024], metrics=["brier"]),
        )
        d = config.model_dump()
        config2 = ProjectConfig(**d)
        assert config2.models["m"].params["C"] == 1.0

    def test_guardrail_config(self):
        gc = GuardrailDef(
            feature_leakage_denylist=["leaky_col"],
            critical_paths=["config/", "models/"],
            naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
            rate_limit_seconds=60,
        )
        assert "leaky_col" in gc.feature_leakage_denylist

    def test_server_config(self):
        sc = ServerDef(
            name="my-pipeline",
            tools={
                "train": ServerToolDef(
                    command="easyml run train",
                    args=["gender", "run_id"],
                    guardrails=["sanity_check"],
                    description="Train all models",
                ),
            },
            inspection=["show_config", "list_models"],
        )
        assert "train" in sc.tools

    def test_full_config(self):
        """A complete config with all sections populated."""
        config = ProjectConfig(
            data=DataConfig(raw_dir="data/raw", processed_dir="data/processed", features_dir="data/features", gender="M"),
            features={
                "win_rate": FeatureDecl(module="feats.resume", category="resume", level="team", columns=["win_rate"]),
            },
            sources={
                "kenpom": SourceDecl(module="scrapers.kp", category="external", temporal_safety="pre_tournament", outputs=["data/kp/"]),
            },
            models={
                "logreg": ModelDef(type="logistic_regression", features=["diff_win_rate"], params={"C": 1.0}),
                "xgb": ModelDef(type="xgboost", features=["diff_win_rate"], params={"max_depth": 3}),
            },
            ensemble=EnsembleDef(method="stacked", meta_learner={"C": 2.5}, calibration="spline"),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out", seasons=[2023, 2024], metrics=["brier", "accuracy"]),
            experiments=ExperimentDef(naming_pattern=r"exp-\d{3}-[a-z0-9-]+$"),
            guardrails=GuardrailDef(feature_leakage_denylist=["leaky"]),
            server=ServerDef(
                name="test",
                tools={"train": ServerToolDef(command="easyml run train", description="Train")},
                inspection=["show_config"],
            ),
        )
        assert len(config.models) == 2
        assert config.server.name == "test"
```

**Step 3: Run tests to verify they fail**

Run: `cd ~/easyml && uv sync && uv run pytest packages/easyml-runner/tests/test_schema.py -v`
Expected: FAIL (module doesn't exist yet)

**Step 4: Implement schema.py**

```python
"""Project-level Pydantic models for YAML validation.

These schemas live in the runner (orchestration layer), NOT in easyml-schemas.
They define the validated shape of the YAML config tree that drives the pipeline.
"""

from __future__ import annotations
from typing import Any, ClassVar, Literal
from pydantic import BaseModel, field_validator


# --- Declarative registration ---

class FeatureDecl(BaseModel):
    """Declares a feature by pointing to a Python module."""
    module: str              # e.g. "my_features.resume"
    function: str = "compute"
    category: str
    level: Literal["team", "matchup"]
    columns: list[str]
    nan_strategy: str = "median"


class SourceDecl(BaseModel):
    """Declares a data source by pointing to a Python module."""
    module: str
    function: str = "scrape"
    category: str
    temporal_safety: Literal["pre_tournament", "post_tournament", "mixed", "unknown"]
    outputs: list[str]
    leakage_notes: str = ""


# --- Config sections ---

class DataConfig(BaseModel):
    """Data directory layout."""
    raw_dir: str
    processed_dir: str
    features_dir: str
    gender: str = "M"
    extra: dict[str, Any] = {}


class ModelDef(BaseModel):
    """One model definition from YAML."""
    type: str
    features: list[str]
    params: dict[str, Any] = {}
    active: bool = True
    mode: Literal["classifier", "regressor"] = "classifier"
    n_seeds: int = 1
    pre_calibration: str | None = None
    training_filter: dict[str, Any] | None = None

    _KNOWN_TYPES: ClassVar[set[str]] = {
        "logistic_regression", "elastic_net", "xgboost", "catboost",
        "lightgbm", "random_forest", "mlp", "tabnet",
    }
    _extra_types: ClassVar[set[str]] = set()

    @classmethod
    def register_type(cls, name: str) -> None:
        cls._extra_types.add(name)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        all_types = cls._KNOWN_TYPES | cls._extra_types
        if v not in all_types:
            raise ValueError(
                f"Unknown model type '{v}'. "
                f"Valid types: {sorted(all_types)}"
            )
        return v


class EnsembleDef(BaseModel):
    """Ensemble configuration."""
    method: Literal["stacked", "average"]
    meta_learner: dict[str, Any] | None = None
    pre_calibration: dict[str, str] | None = None
    calibration: str | None = None
    temperature: float = 1.0
    clip_floor: float = 0.0
    availability_adjustment: float = 0.1
    exclude_models: list[str] = []


class BacktestConfig(BaseModel):
    """Backtest evaluation configuration."""
    cv_strategy: Literal["leave_one_season_out", "expanding_window", "sliding_window", "purged_kfold"]
    seasons: list[int]
    metrics: list[str]
    min_train_folds: int = 1


class ExperimentDef(BaseModel):
    """Experiment protocol configuration."""
    naming_pattern: str = r"exp-\d{3}-[a-z0-9-]+$"
    log_path: str = "EXPERIMENT_LOG.md"
    experiments_dir: str = "experiments/"
    do_not_retry_path: str | None = None


class GuardrailDef(BaseModel):
    """Guardrail configuration."""
    feature_leakage_denylist: list[str] = []
    critical_paths: list[str] = []
    naming_pattern: str | None = None
    rate_limit_seconds: int = 60


class ServerToolDef(BaseModel):
    """One MCP tool definition."""
    command: str
    args: list[str] = []
    guardrails: list[str] = []
    description: str = ""
    timeout: int = 600


class ServerDef(BaseModel):
    """MCP server configuration."""
    name: str
    tools: dict[str, ServerToolDef] = {}
    inspection: list[str] = []


# --- Top-level ---

class ProjectConfig(BaseModel):
    """
    The ONE validated schema for an entire easyml project.
    Every YAML file maps to a section of this model.
    """
    data: DataConfig
    models: dict[str, ModelDef]
    ensemble: EnsembleDef
    backtest: BacktestConfig
    features: dict[str, FeatureDecl] = {}
    sources: dict[str, SourceDecl] = {}
    experiments: ExperimentDef = ExperimentDef()
    guardrails: GuardrailDef = GuardrailDef()
    server: ServerDef | None = None
```

**Step 5: Write failing tests for validator.py**

```python
# packages/easyml-runner/tests/test_validator.py
import pytest
from pathlib import Path
from easyml.runner.validator import validate_project, ValidationResult


class TestValidator:

    def test_valid_config(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: data/raw\n  processed_dir: data/processed\n  features_dir: data/features\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models.yaml").write_text(
            "logreg:\n  type: logistic_regression\n  features: [diff_x]\n  params: {C: 1.0}\n"
        )
        (config / "ensemble.yaml").write_text("method: stacked\n")

        result = validate_project(config)
        assert result.valid
        assert "logreg" in result.config.models

    def test_missing_pipeline_yaml(self, tmp_path):
        result = validate_project(tmp_path)
        assert not result.valid
        assert any("pipeline.yaml" in e for e in result.errors)

    def test_invalid_model_type(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models.yaml").write_text(
            "bad:\n  type: nonexistent_algo\n  features: [x]\n  params: {}\n"
        )
        (config / "ensemble.yaml").write_text("method: stacked\n")

        result = validate_project(config)
        assert not result.valid
        assert any("nonexistent_algo" in e for e in result.errors)

    def test_with_overlay(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models.yaml").write_text(
            "m:\n  type: logistic_regression\n  features: [x]\n  params: {C: 1.0}\n"
        )
        (config / "ensemble.yaml").write_text("method: stacked\n")

        overlay = {"models": {"m": {"params": {"C": 5.0}}}}
        result = validate_project(config, overlay=overlay)
        assert result.valid
        assert result.config.models["m"].params["C"] == 5.0

    def test_with_features_yaml(self, tmp_path):
        config = tmp_path / "config"
        config.mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models.yaml").write_text("m:\n  type: logistic_regression\n  features: [x]\n  params: {}\n")
        (config / "ensemble.yaml").write_text("method: stacked\n")
        (config / "features.yaml").write_text(
            "win_rate:\n  module: feats.resume\n  function: compute\n  category: resume\n  level: team\n  columns: [win_rate]\n"
        )

        result = validate_project(config)
        assert result.valid
        assert "win_rate" in result.config.features

    def test_models_subdir(self, tmp_path):
        """Models loaded from config/models/ directory."""
        config = tmp_path / "config"
        config.mkdir()
        (config / "models").mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models" / "production.yaml").write_text(
            "logreg:\n  type: logistic_regression\n  features: [x]\n  params: {}\n"
        )
        (config / "ensemble.yaml").write_text("method: stacked\n")

        result = validate_project(config)
        assert result.valid
        assert "logreg" in result.config.models

    def test_format_errors(self):
        result = ValidationResult(
            valid=False,
            errors=["models -> bad -> type: Unknown model type 'foo'"],
        )
        text = result.format()
        assert "models -> bad -> type" in text

    def test_variant_loading(self, tmp_path):
        """Variant files (e.g. pipeline_w.yaml) take priority."""
        config = tmp_path / "config"
        config.mkdir()
        (config / "pipeline.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n  gender: M\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "pipeline_w.yaml").write_text(
            "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: f\n  gender: W\n"
            "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
        )
        (config / "models.yaml").write_text("m:\n  type: logistic_regression\n  features: [x]\n  params: {}\n")
        (config / "ensemble.yaml").write_text("method: stacked\n")

        result = validate_project(config, variant="w")
        assert result.valid
        assert result.config.data.gender == "W"
```

**Step 6: Implement validator.py**

```python
"""Load YAML files from a config directory and validate against ProjectConfig."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from easyml.config.merge import deep_merge
from easyml.runner.schema import ProjectConfig


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: ProjectConfig | None = None

    def format(self) -> str:
        if self.valid:
            return "Config valid."
        lines = ["Config validation failed:", ""]
        for i, err in enumerate(self.errors, 1):
            lines.append(f"  {i}. {err}")
        return "\n".join(lines)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _pick_file(config_dir: Path, base: str, variant: str | None) -> Path | None:
    """Pick variant file if it exists, otherwise base file."""
    if variant:
        vpath = config_dir / base.replace(".yaml", f"_{variant}.yaml")
        if vpath.exists():
            return vpath
    bpath = config_dir / base
    return bpath if bpath.exists() else None


def validate_project(
    config_dir: str | Path,
    overlay: dict | None = None,
    variant: str | None = None,
) -> ValidationResult:
    config_dir = Path(config_dir)
    errors = []

    # Pipeline.yaml is required
    pipeline_path = _pick_file(config_dir, "pipeline.yaml", variant)
    if pipeline_path is None or not pipeline_path.exists():
        return ValidationResult(valid=False, errors=["Required file missing: pipeline.yaml"])

    # Load pipeline as base
    merged: dict[str, Any] = _load_yaml(pipeline_path)

    # Load models (from models.yaml and/or config/models/ directory)
    models_data: dict[str, Any] = {}
    models_dir = config_dir / "models"
    if models_dir.is_dir():
        for mf in sorted(models_dir.glob("*.yaml")):
            if variant:
                variant_file = mf.parent / mf.name.replace(".yaml", f"_{variant}.yaml")
                if variant_file.exists() and mf != variant_file:
                    continue
            models_data = deep_merge(models_data, _load_yaml(mf))

    models_path = _pick_file(config_dir, "models.yaml", variant)
    if models_path and models_path.exists():
        models_data = deep_merge(models_data, _load_yaml(models_path))

    if models_data:
        merged["models"] = deep_merge(merged.get("models", {}), models_data)

    # Load section files
    for section in ["ensemble", "features", "sources", "experiments", "guardrails", "server"]:
        path = _pick_file(config_dir, f"{section}.yaml", variant)
        if path and path.exists():
            data = _load_yaml(path)
            if data:
                merged[section] = deep_merge(merged.get(section, {}), data)

    # Apply overlay
    if overlay:
        merged = deep_merge(merged, overlay)

    # Validate
    try:
        config = ProjectConfig(**merged)
        return ValidationResult(valid=True, config=config)
    except ValidationError as e:
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return ValidationResult(valid=False, errors=errors)
```

**Step 7: Run tests**

Run: `cd ~/easyml && uv sync && uv run pytest packages/easyml-runner/tests/test_schema.py packages/easyml-runner/tests/test_validator.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
cd ~/easyml && git add packages/easyml-runner/ pyproject.toml
git commit -m "feat(runner): add ProjectConfig schema and YAML validator"
```

---

## Phase 11: Auto-Loaders

These live in the runner, not in the feature/data packages.

### Task 11.1: Feature and source auto-loaders

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/loaders.py`
- Test: `packages/easyml-runner/tests/test_loaders.py`

**Step 1: Write failing tests**

```python
# packages/easyml-runner/tests/test_loaders.py
import pytest
import sys
import types
from easyml.runner.loaders import load_features, load_sources
from easyml.runner.schema import FeatureDecl, SourceDecl
from easyml.features.registry import FeatureRegistry
from easyml.data.sources import SourceRegistry


def _make_module(module_name: str, func_name: str, func=None):
    """Inject a fake module into sys.modules."""
    mod = types.ModuleType(module_name)
    if func is None:
        def func(df, config):
            result = df[["entity_id", "period_id"]].copy()
            result["col"] = 1.0
            return result
    setattr(mod, func_name, func)
    sys.modules[module_name] = mod
    return mod


class TestFeatureLoader:

    def test_loads_features(self):
        _make_module("test_feats.a", "compute_a")

        decls = {
            "feat_a": FeatureDecl(
                module="test_feats.a",
                function="compute_a",
                category="resume",
                level="team",
                columns=["col"],
            ),
        }
        registry = FeatureRegistry()
        load_features(decls, registry)
        assert "feat_a" in [m.name for m in registry.list_features()]

    def test_loads_multiple(self):
        _make_module("test_feats.b", "compute_b")
        _make_module("test_feats.c", "compute_c")

        decls = {
            "b": FeatureDecl(module="test_feats.b", function="compute_b", category="cat1", level="team", columns=["b"]),
            "c": FeatureDecl(module="test_feats.c", function="compute_c", category="cat2", level="matchup", columns=["c"]),
        }
        registry = FeatureRegistry()
        load_features(decls, registry)
        names = [m.name for m in registry.list_features()]
        assert "b" in names and "c" in names

    def test_bad_module_raises(self):
        decls = {"bad": FeatureDecl(module="nonexistent.x", category="x", level="team", columns=["x"])}
        with pytest.raises(ImportError, match="nonexistent.x"):
            load_features(decls, FeatureRegistry())

    def test_bad_function_raises(self):
        _make_module("test_feats.d", "existing_fn")
        decls = {"bad": FeatureDecl(module="test_feats.d", function="wrong_fn", category="x", level="team", columns=["x"])}
        with pytest.raises(AttributeError, match="wrong_fn"):
            load_features(decls, FeatureRegistry())


class TestSourceLoader:

    def test_loads_sources(self):
        _make_module("test_scrapers.kp", "scrape", func=lambda out, cfg: None)

        decls = {
            "kenpom": SourceDecl(
                module="test_scrapers.kp",
                function="scrape",
                category="external",
                temporal_safety="pre_tournament",
                outputs=["data/kp/"],
            ),
        }
        registry = SourceRegistry()
        load_sources(decls, registry)
        assert "kenpom" in registry

    def test_bad_source_module_raises(self):
        decls = {"bad": SourceDecl(module="nope", category="x", temporal_safety="unknown", outputs=[])}
        with pytest.raises(ImportError, match="nope"):
            load_sources(decls, SourceRegistry())
```

**Step 2: Run tests to verify fail**

Run: `cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_loaders.py -v`

**Step 3: Implement loaders.py**

```python
"""Auto-load features and sources from YAML declarations by dynamic import."""

from __future__ import annotations
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easyml.features.registry import FeatureRegistry
    from easyml.data.sources import SourceRegistry
    from easyml.runner.schema import FeatureDecl, SourceDecl


def _import_function(module_path: str, func_name: str, context: str):
    """Import a module and get a function from it."""
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise ImportError(f"{context}: cannot import module '{module_path}'")
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(
            f"{context}: module '{module_path}' has no function '{func_name}'. "
            f"Available: {[a for a in dir(mod) if not a.startswith('_')]}"
        )
    return fn


def load_features(
    declarations: dict[str, FeatureDecl],
    registry: FeatureRegistry,
) -> None:
    """Register features from YAML declarations by dynamically importing modules."""
    for name, decl in declarations.items():
        fn = _import_function(decl.module, decl.function, f"Feature '{name}'")
        decorator = registry.register(
            name=name,
            category=decl.category,
            level=decl.level,
            output_columns=decl.columns,
            nan_strategy=decl.nan_strategy,
        )
        decorator(fn)


def load_sources(
    declarations: dict[str, SourceDecl],
    registry: SourceRegistry,
) -> None:
    """Register data sources from YAML declarations by dynamically importing modules."""
    for name, decl in declarations.items():
        fn = _import_function(decl.module, decl.function, f"Source '{name}'")
        decorator = registry.register(
            name=name,
            category=decl.category,
            outputs=decl.outputs,
            temporal_safety=decl.temporal_safety,
            leakage_notes=decl.leakage_notes,
        )
        decorator(fn)
```

**Step 4: Run tests, commit**

```bash
cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_loaders.py -v
git add packages/easyml-runner/
git commit -m "feat(runner): add YAML-driven feature and source auto-loaders"
```

---

## Phase 12: CLI

### Task 12.1: Click CLI skeleton + validate command

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/cli.py`
- Test: `packages/easyml-runner/tests/test_cli.py`

**Step 1: Write failing tests**

```python
# packages/easyml-runner/tests/test_cli.py
import pytest
from pathlib import Path
from click.testing import CliRunner
from easyml.runner.cli import main


@pytest.fixture
def valid_config(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "pipeline.yaml").write_text(
        "data:\n  raw_dir: data/raw\n  processed_dir: data/processed\n  features_dir: data/features\n"
        "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2024]\n  metrics: [brier]\n"
    )
    (config / "models.yaml").write_text("m:\n  type: logistic_regression\n  features: [x]\n  params: {C: 1.0}\n")
    (config / "ensemble.yaml").write_text("method: stacked\n")
    return config


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "easyml" in result.output


def test_validate_success(valid_config):
    result = CliRunner().invoke(main, ["--config-dir", str(valid_config), "validate"])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_fail(tmp_path):
    result = CliRunner().invoke(main, ["--config-dir", str(tmp_path), "validate"])
    assert result.exit_code == 1
    assert "pipeline.yaml" in result.output


def test_inspect_config(valid_config):
    result = CliRunner().invoke(main, ["--config-dir", str(valid_config), "inspect", "config"])
    assert result.exit_code == 0
    assert "logistic_regression" in result.output


def test_inspect_models(valid_config):
    result = CliRunner().invoke(main, ["--config-dir", str(valid_config), "inspect", "models"])
    assert result.exit_code == 0
    assert "m:" in result.output
```

**Step 2: Implement cli.py**

```python
"""easyml CLI — the YAML-driven interface to the ML framework."""

import json
import click
from pathlib import Path


@click.group()
@click.option("--config-dir", "-c", default="config", help="Config directory path")
@click.option("--gender", "-g", default=None, help="Gender variant (e.g., 'w' for women's)")
@click.pass_context
def main(ctx, config_dir, gender):
    """easyml -- AI-driven ML framework with guardrails."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = Path(config_dir)
    ctx.obj["variant"] = gender


@main.command()
@click.pass_context
def validate(ctx):
    """Validate all YAML config files."""
    from easyml.runner.validator import validate_project
    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    click.echo(result.format())
    if not result.valid:
        raise SystemExit(1)


# --- Run commands ---

@main.group()
def run():
    """Run pipeline stages."""
    pass


@run.command("train")
@click.option("--run-id", default=None)
@click.pass_context
def run_train(ctx, run_id):
    """Train all models from config."""
    from easyml.runner.pipeline import PipelineRunner
    runner = PipelineRunner(project_dir=Path.cwd(), config_dir=ctx.obj["config_dir"], variant=ctx.obj["variant"])
    result = runner.train(run_id=run_id)
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("backtest")
@click.pass_context
def run_backtest(ctx):
    """Run backtest evaluation."""
    from easyml.runner.pipeline import PipelineRunner
    runner = PipelineRunner(project_dir=Path.cwd(), config_dir=ctx.obj["config_dir"], variant=ctx.obj["variant"])
    result = runner.backtest()
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("pipeline")
@click.pass_context
def run_pipeline(ctx):
    """Run full pipeline: train + backtest."""
    from easyml.runner.pipeline import PipelineRunner
    runner = PipelineRunner(project_dir=Path.cwd(), config_dir=ctx.obj["config_dir"], variant=ctx.obj["variant"])
    result = runner.run_full()
    click.echo(json.dumps(result, indent=2, default=str))


# --- Experiment commands ---

@main.group()
def experiment():
    """Manage experiments."""
    pass


@experiment.command("create")
@click.argument("experiment_id")
@click.pass_context
def exp_create(ctx, experiment_id):
    """Create a new experiment."""
    from easyml.runner.validator import validate_project
    from easyml.experiments import ExperimentManager

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo(result.format())
        raise SystemExit(1)

    ec = result.config.experiments
    mgr = ExperimentManager(
        experiments_dir=ec.experiments_dir,
        naming_pattern=ec.naming_pattern,
        log_path=ec.log_path,
    )
    path = mgr.create(experiment_id)
    click.echo(f"Created experiment at {path}")


@experiment.command("log")
@click.argument("experiment_id")
@click.option("--hypothesis", required=True)
@click.option("--changes", required=True)
@click.option("--verdict", required=True, type=click.Choice(["keep", "revert", "partial"]))
@click.option("--notes", default="")
@click.pass_context
def exp_log(ctx, experiment_id, hypothesis, changes, verdict, notes):
    """Log experiment results."""
    from easyml.runner.validator import validate_project
    from easyml.experiments import ExperimentManager

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    ec = result.config.experiments
    mgr = ExperimentManager(experiments_dir=ec.experiments_dir, log_path=ec.log_path)
    mgr.log(experiment_id=experiment_id, hypothesis=hypothesis, changes=changes, verdict=verdict, notes=notes)
    click.echo(f"Logged experiment {experiment_id}")


@experiment.command("list")
@click.pass_context
def exp_list(ctx):
    """List all experiments."""
    from easyml.runner.validator import validate_project

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo("No valid config.")
        raise SystemExit(1)

    exp_dir = Path(result.config.experiments.experiments_dir)
    if not exp_dir.exists():
        click.echo("No experiments found.")
        return
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir() and (d / "overlay.yaml").exists():
            click.echo(f"  {d.name}")


# --- Inspect commands ---

@main.group()
def inspect():
    """Inspect configuration, models, features."""
    pass


@inspect.command("config")
@click.option("--section", default=None)
@click.pass_context
def inspect_config(ctx, section):
    """Show resolved configuration."""
    from easyml.runner.validator import validate_project

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo(result.format())
        raise SystemExit(1)

    data = result.config.model_dump()
    if section:
        data = data.get(section, f"Section '{section}' not found")
    click.echo(json.dumps(data, indent=2, default=str))


@inspect.command("models")
@click.pass_context
def inspect_models(ctx):
    """List all configured models."""
    from easyml.runner.validator import validate_project

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo(result.format())
        raise SystemExit(1)

    for name, md in result.config.models.items():
        status = "active" if md.active else "excluded"
        click.echo(f"  {name}: {md.type} ({status}) -- {len(md.features)} features")


@inspect.command("features")
@click.pass_context
def inspect_features(ctx):
    """List declared features."""
    from easyml.runner.validator import validate_project

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo(result.format())
        raise SystemExit(1)

    if not result.config.features:
        click.echo("No features declared in config.")
        return
    for name, fd in result.config.features.items():
        click.echo(f"  {name}: {fd.category}/{fd.level} -- {fd.columns} (from {fd.module})")


# --- Serve ---

@main.command()
@click.pass_context
def serve(ctx):
    """Start MCP server from server config."""
    from easyml.runner.validator import validate_project
    from easyml.runner.server_gen import generate_server

    result = validate_project(ctx.obj["config_dir"], variant=ctx.obj["variant"])
    if not result.valid:
        click.echo(result.format())
        raise SystemExit(1)

    if result.config.server is None:
        click.echo("No 'server:' section in config. Add server.yaml to config directory.")
        raise SystemExit(1)

    server = generate_server(result.config.server, config_dir=ctx.obj["config_dir"])
    mcp = server.to_fastmcp()
    mcp.run()


# --- Init ---

@main.command()
@click.argument("project_dir")
@click.option("--name", default=None, help="Project name")
def init(project_dir, name):
    """Initialize a new easyml project."""
    from easyml.runner.scaffold import scaffold_project
    path = Path(project_dir)
    try:
        scaffold_project(path, project_name=name)
        click.echo(f"Project initialized at {path}")
        click.echo("Next steps:")
        click.echo(f"  1. cd {path}")
        click.echo("  2. Edit config/models.yaml to define your models")
        click.echo("  3. Add feature modules to features/")
        click.echo("  4. Run: easyml validate")
    except FileExistsError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
```

**Step 3: Run tests, commit**

```bash
cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_cli.py -v
git add packages/easyml-runner/
git commit -m "feat(runner): add Click CLI with validate, run, experiment, inspect, serve, init"
```

### Task 12.2: PipelineRunner — wires library APIs from config

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/pipeline.py`
- Test: `packages/easyml-runner/tests/test_pipeline.py`

**Step 1: Write failing tests**

```python
# packages/easyml-runner/tests/test_pipeline.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from easyml.runner.pipeline import PipelineRunner


@pytest.fixture
def project(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "pipeline.yaml").write_text(
        "data:\n  raw_dir: data/raw\n  processed_dir: data/processed\n  features_dir: data/features\n"
        "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2023, 2024]\n  metrics: [brier, accuracy]\n"
    )
    (config / "models.yaml").write_text(
        "logreg:\n  type: logistic_regression\n  features: [diff_x]\n  params: {C: 1.0}\n"
    )
    (config / "ensemble.yaml").write_text("method: average\n")

    # Mock data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "season": np.repeat([2022, 2023, 2024], [80, 60, 60]),
        "diff_x": np.random.randn(n),
        "result": np.random.randint(0, 2, n),
    })
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)
    df.to_parquet(feat_dir / "matchup_features.parquet", index=False)
    return tmp_path


def test_load(project):
    runner = PipelineRunner(project_dir=project, config_dir=project / "config")
    config = runner.load()
    assert "logreg" in config.models


def test_train(project):
    runner = PipelineRunner(project_dir=project, config_dir=project / "config")
    runner.load()
    result = runner.train()
    assert result["status"] == "success"
    assert "logreg" in result["models_trained"]


def test_backtest(project):
    runner = PipelineRunner(project_dir=project, config_dir=project / "config")
    runner.load()
    result = runner.backtest()
    assert result["status"] == "success"
    assert "brier" in result["metrics"]


def test_run_full(project):
    runner = PipelineRunner(project_dir=project, config_dir=project / "config")
    result = runner.run_full()
    assert result["status"] == "success"
    assert "brier" in result["metrics"]
    assert result["metrics"]["brier"] < 0.5


def test_two_models(tmp_path):
    """Multiple models train and ensemble."""
    config = tmp_path / "config"
    config.mkdir()
    (config / "pipeline.yaml").write_text(
        "data:\n  raw_dir: r\n  processed_dir: p\n  features_dir: data/features\n"
        "backtest:\n  cv_strategy: leave_one_season_out\n  seasons: [2023, 2024]\n  metrics: [brier]\n"
    )
    (config / "models.yaml").write_text(
        "logreg:\n  type: logistic_regression\n  features: [diff_x, diff_y]\n  params: {C: 1.0}\n"
        "enet:\n  type: elastic_net\n  features: [diff_x]\n  params: {C: 0.5}\n"
    )
    (config / "ensemble.yaml").write_text("method: average\n")

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "season": np.repeat([2022, 2023, 2024], [80, 60, 60]),
        "diff_x": np.random.randn(n),
        "diff_y": np.random.randn(n),
        "result": np.random.randint(0, 2, n),
    })
    (tmp_path / "data" / "features").mkdir(parents=True)
    df.to_parquet(tmp_path / "data" / "features" / "matchup_features.parquet", index=False)

    runner = PipelineRunner(project_dir=tmp_path, config_dir=config)
    result = runner.run_full()
    assert result["status"] == "success"
    assert len(result["models_trained"]) == 2
```

**Step 2: Implement pipeline.py**

This is the core orchestration module. It imports from the library packages and wires them together based on ProjectConfig.

```python
"""Pipeline orchestration from validated YAML config.

This module is the sole consumer of the easyml library packages.
It reads ProjectConfig and wires FeatureRegistry, ModelRegistry,
TrainOrchestrator, StackedEnsemble, BacktestRunner together.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from easyml.runner.validator import validate_project
from easyml.runner.schema import ProjectConfig
from easyml.models.registry import ModelRegistry
from easyml.models.orchestrator import TrainOrchestrator
from easyml.models.ensemble import StackedEnsemble
from easyml.models.cv import LeaveOneSeasonOut, ExpandingWindow, SlidingWindow, PurgedKFold
from easyml.models.backtest import BacktestRunner


_CV_MAP = {
    "leave_one_season_out": LeaveOneSeasonOut,
    "expanding_window": ExpandingWindow,
    "sliding_window": SlidingWindow,
    "purged_kfold": PurgedKFold,
}


class PipelineRunner:

    def __init__(
        self,
        project_dir: str | Path,
        config_dir: str | Path | None = None,
        variant: str | None = None,
        overlay: dict | None = None,
    ):
        self.project_dir = Path(project_dir)
        self.config_dir = Path(config_dir) if config_dir else self.project_dir / "config"
        self.variant = variant
        self.overlay = overlay
        self.config: ProjectConfig | None = None
        self._model_registry: ModelRegistry | None = None

    def load(self) -> ProjectConfig:
        result = validate_project(self.config_dir, overlay=self.overlay, variant=self.variant)
        if not result.valid:
            raise ValueError(f"Config validation failed:\n{result.format()}")
        self.config = result.config
        self._model_registry = ModelRegistry.with_defaults()

        # Load declared features if any
        if self.config.features:
            from easyml.features.registry import FeatureRegistry
            from easyml.runner.loaders import load_features
            self._feature_registry = FeatureRegistry()
            load_features(self.config.features, self._feature_registry)

        return self.config

    def _load_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
        features_dir = self.project_dir / self.config.data.features_dir
        matchup_path = features_dir / "matchup_features.parquet"
        if not matchup_path.exists():
            raise FileNotFoundError(f"Matchup features not found at {matchup_path}")

        df = pd.read_parquet(matchup_path)
        all_features = set()
        for mc in self.config.models.values():
            if mc.active:
                all_features.update(mc.features)

        feature_cols = sorted(all_features & set(df.columns))
        X = df[feature_cols].values
        y = df["result"].values
        seasons = df["season"].values
        return df, X, y, feature_cols, seasons

    def train(self, run_id: str | None = None) -> dict[str, Any]:
        if self.config is None:
            self.load()

        df, X, y, feature_cols, seasons = self._load_data()

        model_configs = {}
        for name, mc in self.config.models.items():
            model_configs[name] = {
                "type": mc.type,
                "features": mc.features,
                "params": mc.params,
                "active": mc.active,
            }

        output_dir = self.project_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        orch = TrainOrchestrator(
            model_registry=self._model_registry,
            model_configs=model_configs,
            output_dir=output_dir,
        )
        trained = orch.train_all(X, y, feature_columns=feature_cols)
        self._trained_models = trained

        return {"status": "success", "models_trained": list(trained.keys())}

    def backtest(self) -> dict[str, Any]:
        if self.config is None:
            self.load()

        df, X, y, feature_cols, seasons = self._load_data()

        cv_cls = _CV_MAP[self.config.backtest.cv_strategy]
        cv = cv_cls(min_train_folds=self.config.backtest.min_train_folds)
        folds = cv.split(X, fold_ids=seasons)

        fold_predictions: dict[int, dict[str, np.ndarray]] = {}
        fold_y: dict[int, np.ndarray] = {}

        for fold in folds:
            fold_id = int(seasons[fold.test_idx[0]])
            if fold_id not in self.config.backtest.seasons:
                continue

            X_train, y_train = X[fold.train_idx], y[fold.train_idx]
            X_test, y_test = X[fold.test_idx], y[fold.test_idx]

            model_preds = {}
            for name, mc in self.config.models.items():
                if not mc.active:
                    continue
                model = self._model_registry.create(mc.type, mc.params)
                feat_idx = [feature_cols.index(f) for f in mc.features if f in feature_cols]
                if not feat_idx:
                    continue

                model.fit(X_train[:, feat_idx], y_train)
                preds = model.predict_proba(X_test[:, feat_idx])
                model_preds[name] = preds

            if model_preds:
                fold_predictions[fold_id] = model_preds
                fold_y[fold_id] = y_test

        bt = BacktestRunner()
        result = bt.run(predictions=fold_predictions, y=fold_y, metrics=self.config.backtest.metrics)

        return {
            "status": "success",
            "metrics": result.pooled_metrics,
            "per_fold": result.per_fold_metrics,
        }

    def run_full(self) -> dict[str, Any]:
        self.load()
        train_result = self.train()
        if train_result["status"] != "success":
            return train_result
        bt_result = self.backtest()
        bt_result["models_trained"] = train_result["models_trained"]
        return bt_result
```

Note: The implementation must match the actual APIs of the existing library packages. Read the existing `TrainOrchestrator.train_all()`, `BacktestRunner.run()`, and `StackedEnsemble` signatures before implementing. The code above is the intended design — adjust to match actual APIs.

**Step 3: Run tests, commit**

```bash
cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_pipeline.py -v
git add packages/easyml-runner/
git commit -m "feat(runner): add PipelineRunner wiring library APIs from YAML config"
```

---

## Phase 13: MCP Server Generator + Project Scaffold

### Task 13.1: Server generator

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/server_gen.py`
- Test: `packages/easyml-runner/tests/test_server_gen.py`

**Step 1: Write failing tests**

```python
# packages/easyml-runner/tests/test_server_gen.py
import pytest
from pathlib import Path
from easyml.runner.server_gen import generate_server, GeneratedServer
from easyml.runner.schema import ServerDef, ServerToolDef


def test_creates_execution_tools():
    config = ServerDef(
        name="test",
        tools={
            "train": ServerToolDef(command="easyml run train", args=["gender", "run_id"], description="Train models"),
            "backtest": ServerToolDef(command="easyml run backtest", args=["gender"], description="Run backtest"),
        },
    )
    server = generate_server(config, config_dir=Path("/tmp"))
    assert "train" in server.tools
    assert "backtest" in server.tools


def test_creates_inspection_tools():
    config = ServerDef(
        name="test",
        inspection=["show_config", "list_models", "list_features"],
    )
    server = generate_server(config, config_dir=Path("/tmp"))
    assert "show_config" in server.tools
    assert "list_models" in server.tools
    assert "list_features" in server.tools


def test_tool_has_description():
    config = ServerDef(
        name="test",
        tools={"train": ServerToolDef(command="easyml run train", description="Train all models")},
    )
    server = generate_server(config, config_dir=Path("/tmp"))
    assert server.tools["train"].description == "Train all models"


def test_tool_has_guardrails():
    config = ServerDef(
        name="test",
        tools={"train": ServerToolDef(command="easyml run train", guardrails=["sanity_check", "feature_leakage"])},
    )
    server = generate_server(config, config_dir=Path("/tmp"))
    assert server.tools["train"].guardrails == ["sanity_check", "feature_leakage"]
```

**Step 2: Implement server_gen.py**

```python
"""Generate MCP server from YAML ServerDef configuration."""

from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from easyml.runner.schema import ServerDef, ServerToolDef


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable
    args: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)


@dataclass
class GeneratedServer:
    name: str
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def to_fastmcp(self):
        from fastmcp import FastMCP
        mcp = FastMCP(self.name)
        for name, spec in self.tools.items():
            mcp.tool(name=name, description=spec.description)(spec.fn)
        return mcp


def _make_execution_tool(tool_def: ServerToolDef) -> Callable:
    cmd_parts = tool_def.command.split()

    async def execute(**kwargs) -> str:
        cmd = list(cmd_parts)
        for arg_name in tool_def.args:
            val = kwargs.get(arg_name)
            if val is not None:
                flag = f"--{arg_name.replace('_', '-')}"
                if isinstance(val, bool):
                    if val:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(val)])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=tool_def.timeout)
            return json.dumps({"status": "success" if result.returncode == 0 else "error",
                              "stdout": result.stdout, "stderr": result.stderr[-500:] if result.stderr else ""})
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "timeout"})

    return execute


def _make_inspection_tool(name: str, config_dir: Path) -> Callable | None:
    if name == "show_config":
        async def fn(section: str | None = None) -> str:
            from easyml.runner.validator import validate_project
            result = validate_project(config_dir)
            if not result.valid:
                return result.format()
            data = result.config.model_dump()
            if section:
                data = data.get(section, f"Section '{section}' not found")
            return json.dumps(data, indent=2, default=str)
        return fn
    elif name == "list_models":
        async def fn() -> str:
            from easyml.runner.validator import validate_project
            result = validate_project(config_dir)
            if not result.valid:
                return result.format()
            lines = []
            for n, mc in result.config.models.items():
                lines.append(f"{n}: {mc.type} ({'active' if mc.active else 'excluded'}) -- {len(mc.features)} features")
            return "\n".join(lines)
        return fn
    elif name == "list_features":
        async def fn() -> str:
            from easyml.runner.validator import validate_project
            result = validate_project(config_dir)
            if not result.valid:
                return result.format()
            if not result.config.features:
                return "No features declared."
            lines = []
            for n, fd in result.config.features.items():
                lines.append(f"{n}: {fd.category}/{fd.level} -- {fd.columns}")
            return "\n".join(lines)
        return fn
    elif name == "list_experiments":
        async def fn() -> str:
            from easyml.runner.validator import validate_project
            result = validate_project(config_dir)
            if not result.valid:
                return result.format()
            exp_dir = Path(result.config.experiments.experiments_dir)
            if not exp_dir.exists():
                return "No experiments."
            dirs = [d.name for d in sorted(exp_dir.iterdir()) if d.is_dir()]
            return "\n".join(dirs) if dirs else "No experiments."
        return fn
    return None


def generate_server(config: ServerDef, config_dir: Path) -> GeneratedServer:
    server = GeneratedServer(name=config.name)

    for name, tool_def in config.tools.items():
        server.tools[name] = ToolSpec(
            name=name,
            description=tool_def.description,
            fn=_make_execution_tool(tool_def),
            args=tool_def.args,
            guardrails=tool_def.guardrails,
        )

    for tool_name in config.inspection:
        fn = _make_inspection_tool(tool_name, config_dir)
        if fn:
            server.tools[tool_name] = ToolSpec(
                name=tool_name,
                description=f"Inspect: {tool_name.replace('_', ' ')}",
                fn=fn,
            )

    return server
```

**Step 3: Run tests, commit**

```bash
cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_server_gen.py -v
git add packages/easyml-runner/
git commit -m "feat(runner): add MCP server generator from YAML config"
```

### Task 13.2: Project scaffold

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/scaffold.py`
- Test: `packages/easyml-runner/tests/test_scaffold.py`

**Step 1: Write failing tests**

```python
# packages/easyml-runner/tests/test_scaffold.py
import pytest
from pathlib import Path
from click.testing import CliRunner
from easyml.runner.cli import main
from easyml.runner.validator import validate_project


def test_init_creates_project(tmp_path):
    result = CliRunner().invoke(main, ["init", str(tmp_path / "my-project")])
    assert result.exit_code == 0
    project = tmp_path / "my-project"
    assert (project / "config" / "pipeline.yaml").exists()
    assert (project / "config" / "models.yaml").exists()
    assert (project / "config" / "ensemble.yaml").exists()
    assert (project / "config" / "server.yaml").exists()
    assert (project / "CLAUDE.md").exists()
    assert (project / "features").is_dir()
    assert (project / "data").is_dir()


def test_scaffolded_config_validates(tmp_path):
    CliRunner().invoke(main, ["init", str(tmp_path / "test")])
    result = validate_project(tmp_path / "test" / "config")
    assert result.valid, f"Scaffold invalid: {result.format()}"


def test_init_refuses_nonempty_dir(tmp_path):
    project = tmp_path / "existing"
    project.mkdir()
    (project / "something.txt").write_text("content")
    result = CliRunner().invoke(main, ["init", str(project)])
    assert result.exit_code != 0
```

**Step 2: Implement scaffold.py**

The scaffold generates template YAML files that pass `validate_project()`. Include a starter model (logreg_baseline) so the project is immediately runnable.

```python
"""Project scaffold generator."""

from pathlib import Path

_PIPELINE = """\
data:
  raw_dir: data/raw
  processed_dir: data/processed
  features_dir: data/features

backtest:
  cv_strategy: leave_one_season_out
  seasons: []
  metrics:
    - brier
    - accuracy
    - log_loss
    - ece

experiments:
  naming_pattern: "exp-\\\\d{{3}}-[a-z0-9-]+$"
  log_path: EXPERIMENT_LOG.md
  experiments_dir: experiments/
"""

_MODELS = """\
# Model definitions. Available types:
#   logistic_regression, elastic_net, xgboost, catboost,
#   lightgbm, random_forest, mlp, tabnet

logreg_baseline:
  type: logistic_regression
  features:
    - diff_seed_num
  params:
    C: 1.0
"""

_ENSEMBLE = """\
method: stacked
# meta_learner:
#   C: 2.5
# calibration: spline
# temperature: 1.0
# exclude_models: []
"""

_SERVER = """\
name: {name}
tools:
  run_pipeline:
    command: easyml run pipeline
    args: [gender]
    description: Run full train + backtest pipeline
  train_models:
    command: easyml run train
    args: [gender, run-id]
    description: Train all models
  run_backtest:
    command: easyml run backtest
    args: [gender]
    description: Run backtest evaluation
inspection:
  - show_config
  - list_models
  - list_features
  - list_experiments
"""

_CLAUDE_MD = """\
# {name}

## Commands
- `easyml validate` -- validate all YAML config
- `easyml run pipeline` -- full train + backtest
- `easyml run train` -- train models only
- `easyml run backtest` -- backtest only
- `easyml experiment create <id>` -- create experiment
- `easyml experiment log <id> --hypothesis "..." --changes "..." --verdict keep`
- `easyml inspect config` -- show resolved config
- `easyml inspect models` -- list models
- `easyml serve` -- start MCP server

## Config
All config is YAML in `config/`:
- `pipeline.yaml` -- data paths, backtest seasons, metrics
- `models.yaml` -- model definitions (type, features, params)
- `ensemble.yaml` -- ensemble method, calibration
- `server.yaml` -- MCP server tool definitions

## Rules
- Never modify production config directly -- use experiments + promote
- Always use `uv run` for Python commands
- Log every experiment, regardless of outcome
"""


def scaffold_project(project_dir: Path, project_name: str | None = None) -> None:
    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(f"Directory already exists and is not empty: {project_dir}")

    name = project_name or project_dir.name

    for d in ["config", "data/raw", "data/processed", "data/features",
              "features", "experiments", "models"]:
        (project_dir / d).mkdir(parents=True, exist_ok=True)

    (project_dir / "config" / "pipeline.yaml").write_text(_PIPELINE)
    (project_dir / "config" / "models.yaml").write_text(_MODELS)
    (project_dir / "config" / "ensemble.yaml").write_text(_ENSEMBLE)
    (project_dir / "config" / "server.yaml").write_text(_SERVER.format(name=name))
    (project_dir / "CLAUDE.md").write_text(_CLAUDE_MD.format(name=name))
    (project_dir / "EXPERIMENT_LOG.md").write_text(f"# {name} Experiment Log\n")
```

**Step 3: Run tests, commit**

```bash
cd ~/easyml && uv run pytest packages/easyml-runner/tests/test_scaffold.py -v
git add packages/easyml-runner/
git commit -m "feat(runner): add easyml init project scaffold"
```

---

## Phase 14: Integration + Docs

### Task 14.1: End-to-end YAML-driven test

**Files:**
- Create: `packages/easyml-runner/tests/test_e2e.py`

Test the full workflow: scaffold → validate → train → backtest → experiment.

```python
# packages/easyml-runner/tests/test_e2e.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from click.testing import CliRunner
from easyml.runner.cli import main
from easyml.runner.pipeline import PipelineRunner


@pytest.fixture
def live_project(tmp_path):
    """Scaffolded project with mock matchup data."""
    project = tmp_path / "test-project"
    CliRunner().invoke(main, ["init", str(project)])

    # Update backtest seasons
    pipeline_yaml = project / "config" / "pipeline.yaml"
    text = pipeline_yaml.read_text().replace("seasons: []", "seasons: [2023, 2024]")
    pipeline_yaml.write_text(text)

    # Generate mock data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "season": np.repeat([2022, 2023, 2024], [80, 60, 60]),
        "diff_seed_num": np.random.randn(n),
        "result": np.random.randint(0, 2, n),
    })
    df.to_parquet(project / "data" / "features" / "matchup_features.parquet", index=False)
    return project


def test_validate(live_project):
    result = CliRunner().invoke(main, ["--config-dir", str(live_project / "config"), "validate"])
    assert result.exit_code == 0


def test_inspect(live_project):
    result = CliRunner().invoke(main, ["--config-dir", str(live_project / "config"), "inspect", "config"])
    assert result.exit_code == 0
    assert "logistic_regression" in result.output


def test_full_pipeline(live_project):
    runner = PipelineRunner(project_dir=live_project, config_dir=live_project / "config")
    result = runner.run_full()
    assert result["status"] == "success"
    assert result["metrics"]["brier"] < 0.5
    assert "logreg_baseline" in result["models_trained"]


def test_experiment_create(live_project):
    result = CliRunner().invoke(main, [
        "--config-dir", str(live_project / "config"),
        "experiment", "create", "exp-001-test",
    ])
    assert result.exit_code == 0
    assert (live_project / "experiments" / "exp-001-test" / "overlay.yaml").exists()
```

**Commit:** `test: add end-to-end YAML-driven integration test`

### Task 14.2: Package exports + full test suite + documentation

**Step 1: Update `__init__.py`**

```python
# packages/easyml-runner/src/easyml/runner/__init__.py
from easyml.runner.schema import (
    ProjectConfig, DataConfig, BacktestConfig, ModelDef, EnsembleDef,
    FeatureDecl, SourceDecl, ExperimentDef, GuardrailDef,
    ServerDef, ServerToolDef,
)
from easyml.runner.validator import validate_project, ValidationResult
from easyml.runner.pipeline import PipelineRunner
from easyml.runner.loaders import load_features, load_sources
from easyml.runner.server_gen import generate_server, GeneratedServer
from easyml.runner.scaffold import scaffold_project

__all__ = [
    "ProjectConfig", "DataConfig", "BacktestConfig", "ModelDef", "EnsembleDef",
    "FeatureDecl", "SourceDecl", "ExperimentDef", "GuardrailDef",
    "ServerDef", "ServerToolDef",
    "validate_project", "ValidationResult",
    "PipelineRunner",
    "load_features", "load_sources",
    "generate_server", "GeneratedServer",
    "scaffold_project",
]
```

**Step 2: Run full test suite**

```bash
cd ~/easyml && uv run pytest -v
```

Expected: All tests pass (325 existing + ~60 new runner tests).

**Step 3: Create README.md for easyml-runner**

`packages/easyml-runner/README.md`:
```markdown
# easyml-runner

YAML-driven orchestration layer for easyml. Operates the full ML pipeline through validated config + CLI commands + MCP tools.

## Install

pip install easyml-runner

## Quick Start

# Initialize a project
easyml init my-project
cd my-project

# Edit config/models.yaml, config/pipeline.yaml
easyml validate
easyml run pipeline
easyml experiment create exp-001-test
easyml serve

## Key APIs

- `ProjectConfig` — Pydantic model validating entire YAML config tree
- `validate_project(config_dir)` — Load + validate YAML files
- `PipelineRunner` — Wire library APIs from config, run train/backtest
- `generate_server(config)` — Auto-generate MCP server from YAML
- `scaffold_project(path)` — Generate new project skeleton
```

**Step 4: Update root README with YAML-driven section**

Add to `~/easyml/README.md`:

```markdown
## YAML-Driven Interface

easyml projects are operated entirely through YAML config and CLI commands:

    easyml init my-project && cd my-project
    easyml validate
    easyml run pipeline
    easyml experiment create exp-001-test
    easyml serve

No Python imports needed. Edit YAML, run commands.
```

**Step 5: Commit**

```bash
cd ~/easyml && git add .
git commit -m "feat(runner): finalize package exports, README, integration tests"
```

---

## Task Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 10 | 1 | ProjectConfig schema + YAML validator (in easyml-runner) |
| 11 | 1 | Feature + source auto-loaders (in easyml-runner) |
| 12 | 2 | CLI (validate, run, experiment, inspect) + PipelineRunner |
| 13 | 2 | MCP server generator + project scaffold |
| 14 | 2 | E2E test + exports + docs |
| **Total** | **8 tasks** | **1 new package, 0 modifications to existing packages** |

## Dependency Order

```
Task 10.1 (schema + validator) → Task 11.1 (loaders) → Task 12.1 (CLI skeleton)
                                                              ↓
                                      Task 12.2 (PipelineRunner) → Task 13.1 (server gen)
                                                                          ↓
                                                              Task 13.2 (scaffold) → Task 14.1 (E2E test) → Task 14.2 (exports + docs)
```

All tasks are sequential within easyml-runner. The existing 7 packages are never touched.
