"""Project-level Pydantic models for YAML-driven orchestration.

These schemas define the shape of a project's configuration — models,
ensemble, backtest, features, sources, experiments, guardrails, and
MCP server definitions.  They live in the runner package (not
harnessml-schemas) because they are orchestration concerns.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Tag, field_validator, model_validator

# -----------------------------------------------------------------------
# Feature & source declarations
# -----------------------------------------------------------------------

class FeatureType(str, Enum):
    """Semantic type of a declarative feature."""
    ENTITY = "entity"
    PAIRWISE = "pairwise"
    INSTANCE = "instance"
    REGIME = "regime"


class PairwiseMode(str, Enum):
    """How to derive pairwise features from entity features."""
    DIFF = "diff"
    RATIO = "ratio"
    BOTH = "both"
    NONE = "none"


class FeatureDef(BaseModel):
    """Declarative feature definition.

    Supports four semantic types:
    - entity: Per-entity per-period metric. Auto-generates pairwise.
    - pairwise: Per-instance (A vs B). Derived from entity or custom formula.
    - instance: Per-instance context property (column or formula).
    - regime: Temporal/contextual boolean flag.

    Also supports legacy FeatureDecl fields (module, function, columns, level)
    for backward compatibility.
    """
    name: str = ""
    type: FeatureType = FeatureType.ENTITY
    source: str | None = None
    column: str | None = None
    formula: str | None = None
    condition: str | None = None
    pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    description: str = ""
    nan_strategy: str = "median"
    category: str = "general"
    enabled: bool = True

    # Legacy FeatureDecl fields (optional for backward compat)
    module: str | None = None
    function: str | None = None
    columns: list[str] | None = None
    level: str | None = None


# Backward-compatible alias
FeatureDecl = FeatureDef


class FeatureStoreConfig(BaseModel):
    """Configuration for the declarative feature store."""
    cache_dir: str = "data/features/cache"
    auto_pairwise: bool = True
    default_pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    entity_a_column: str = "entity_a_id"
    entity_b_column: str = "entity_b_id"
    entity_column: str = "entity_id"
    period_column: str = "period_id"


class SourceDecl(BaseModel):
    """Declares a data source."""

    module: str
    function: str
    category: str
    temporal_safety: Literal["pre_event", "post_event", "mixed", "unknown"]
    outputs: list[str]
    leakage_notes: str = ""


# -----------------------------------------------------------------------
# Features config (pipeline-level feature settings)
# -----------------------------------------------------------------------

class FeaturesConfig(BaseModel):
    """Pipeline-level feature computation settings."""

    first_period: int = 2003
    momentum_window: int = 10


# -----------------------------------------------------------------------
# Data cleaning & source config
# -----------------------------------------------------------------------

class ColumnCleaningRule(BaseModel):
    """Per-column cleaning configuration with cascade support.

    Rules cascade: column-level > source-level > global-level.
    """

    null_strategy: Literal["median", "mode", "zero", "drop", "ffill", "constant"] = "median"
    null_fill_value: Any | None = None
    coerce_numeric: bool = False
    clip_outliers: tuple[float, float] | None = None
    log_transform: bool = False
    normalize: Literal["none", "zscore", "minmax"] = "none"


class SourceConfig(BaseModel):
    """A declared data source in the pipeline."""

    name: str
    path: str | None = None
    format: Literal["csv", "parquet", "excel", "auto"] = "auto"
    join_on: list[str] | None = None
    columns: dict[str, ColumnCleaningRule] | None = None
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()
    temporal_safety: Literal["pre_event", "post_event", "mixed", "unknown"] = "unknown"
    enabled: bool = True


# -----------------------------------------------------------------------
# Transform steps (declarative ETL)
# -----------------------------------------------------------------------


class FilterStep(BaseModel):
    """Keep rows matching an expression."""

    op: Literal["filter"] = "filter"
    expr: str  # e.g. "DayNum < 134", "status == 'active'"


class SelectStep(BaseModel):
    """Keep or rename columns."""

    op: Literal["select"] = "select"
    columns: dict[str, str] | list[str]  # list=keep, dict={new: old}


class DeriveStep(BaseModel):
    """Create new columns from expressions."""

    op: Literal["derive"] = "derive"
    columns: dict[str, str]  # {new_col: expression}


class GroupByStep(BaseModel):
    """Group and aggregate."""

    op: Literal["group_by"] = "group_by"
    keys: list[str]
    aggs: dict[str, str | list[str]]  # {col: "mean"} or {col: ["mean","std"]}


class JoinStep(BaseModel):
    """Join with another source or view."""

    op: Literal["join"] = "join"
    other: str  # name of source or view
    on: list[str] | dict[str, str]  # same-name or {left: right}
    how: Literal["left", "inner", "right", "outer"] = "left"
    select: list[str] | None = None  # columns to take from other (None=all)
    prefix: str | None = None  # prefix for other's columns


class UnionStep(BaseModel):
    """Vertically concatenate with another source or view."""

    op: Literal["union"] = "union"
    other: str


class UnpivotStep(BaseModel):
    """Melt wide columns into long format."""

    op: Literal["unpivot"] = "unpivot"
    id_columns: list[str]  # columns to keep as-is
    unpivot_columns: dict[str, list[str]]  # {new_col: [src_col_1, src_col_2]}
    names_column: str | None = None  # column storing which source the value came from
    names_map: dict[str, str] | None = None  # rename source identifiers


class CastStep(BaseModel):
    """Cast column types."""

    op: Literal["cast"] = "cast"
    columns: dict[str, str]  # {col: "int"} or {col: "int:str[1:3]"}


class SortStep(BaseModel):
    """Sort rows."""

    op: Literal["sort"] = "sort"
    by: list[str] | str
    ascending: bool | list[bool] = True


class DistinctStep(BaseModel):
    """Deduplicate rows."""

    op: Literal["distinct"] = "distinct"
    columns: list[str] | None = None
    keep: Literal["first", "last"] = "first"


class RollingStep(BaseModel):
    """Rolling/windowed aggregation partitioned by keys."""

    op: Literal["rolling"] = "rolling"
    keys: list[str]  # partition columns
    order_by: str  # column to sort within groups
    window: int  # window size
    aggs: dict[str, str]  # {new_col: "source_col:func"} e.g. {"avg_pts_3": "points:mean"}
    min_periods: int | None = None  # minimum observations; defaults to window


class HeadStep(BaseModel):
    """Take first or last N rows per group."""

    op: Literal["head"] = "head"
    keys: list[str]
    n: int = 1
    order_by: str | list[str] | None = None
    ascending: bool | list[bool] = True
    position: Literal["first", "last"] = "first"


class RankStep(BaseModel):
    """Add rank columns, optionally within groups."""

    op: Literal["rank"] = "rank"
    columns: dict[str, str]  # {new_col: source_col}
    keys: list[str] | None = None  # optional partition
    method: Literal["average", "min", "max", "first", "dense"] = "average"
    ascending: bool = True
    pct: bool = False


class ConditionalAggStep(BaseModel):
    """Group and aggregate with optional per-agg filter conditions.

    Agg format: ``{new_col: "source_col:func"} | {new_col: "source_col:func:where_expr"}``

    Example::

        aggs:
          win_avg_pts: "points:mean:result == 1"
          total_games: "game_id:count"
    """

    op: Literal["cond_agg"] = "cond_agg"
    keys: list[str]
    aggs: dict[str, str]  # {new_col: "col:func" or "col:func:condition"}


class IsInStep(BaseModel):
    """Filter rows where a column's value is (or is not) in a list."""

    op: Literal["isin"] = "isin"
    column: str
    values: list[Any]
    negate: bool = False


class LagStep(BaseModel):
    """Shift column values within groups (lag/lead)."""

    op: Literal["lag"] = "lag"
    keys: list[str]           # group columns
    order_by: str             # sort column
    columns: dict[str, str]   # {new_col: "source_col:lag_periods"}


class EwmStep(BaseModel):
    """Exponentially weighted moving statistic within groups."""

    op: Literal["ewm"] = "ewm"
    keys: list[str]
    order_by: str
    span: float               # EWM span parameter
    aggs: dict[str, str]      # {new_col: "source_col:stat"} where stat is mean/std/var


class DiffStep(BaseModel):
    """First/second differences or percent change within groups."""

    op: Literal["diff"] = "diff"
    keys: list[str]
    order_by: str
    columns: dict[str, str]   # {new_col: "source_col:periods"}
    pct: bool = False         # if True, compute pct_change instead


class TrendStep(BaseModel):
    """OLS slope over a rolling window within groups."""

    op: Literal["trend"] = "trend"
    keys: list[str]
    order_by: str
    window: int
    columns: dict[str, str]   # {new_col: "source_col"}


class EncodeStep(BaseModel):
    """Categorical encoding (frequency, ordinal, target LOO, target temporal)."""

    op: Literal["encode"] = "encode"
    column: str
    method: str               # "target_loo", "target_temporal", "frequency", "ordinal"
    output: str | None = None # output column name (defaults to f"{column}_encoded")


class BinStep(BaseModel):
    """Discretize a continuous column into bins."""

    op: Literal["bin"] = "bin"
    column: str
    method: str               # "quantile", "uniform", "custom", "kmeans"
    n_bins: int = 10
    output: str | None = None
    boundaries: list[float] | None = None  # for custom method


class DatetimeStep(BaseModel):
    """Extract calendar features from a datetime column."""

    op: Literal["datetime"] = "datetime"
    column: str
    extract: list[str] | None = None    # ["year", "month", "day", "dayofweek", "hour", "quarter", "weekofyear"]
    cyclical: list[str] | None = None   # ["month", "dayofweek", "hour"] -- adds sin/cos pairs


class NullIndicatorStep(BaseModel):
    """Create binary indicators for missing values."""

    op: Literal["null_indicator"] = "null_indicator"
    columns: list[str]
    prefix: str = "missing_"


TransformStep = Annotated[
    Union[
        Annotated[FilterStep, Tag("filter")],
        Annotated[SelectStep, Tag("select")],
        Annotated[DeriveStep, Tag("derive")],
        Annotated[GroupByStep, Tag("group_by")],
        Annotated[JoinStep, Tag("join")],
        Annotated[UnionStep, Tag("union")],
        Annotated[UnpivotStep, Tag("unpivot")],
        Annotated[CastStep, Tag("cast")],
        Annotated[SortStep, Tag("sort")],
        Annotated[DistinctStep, Tag("distinct")],
        Annotated[RollingStep, Tag("rolling")],
        Annotated[HeadStep, Tag("head")],
        Annotated[RankStep, Tag("rank")],
        Annotated[ConditionalAggStep, Tag("cond_agg")],
        Annotated[IsInStep, Tag("isin")],
        Annotated[LagStep, Tag("lag")],
        Annotated[EwmStep, Tag("ewm")],
        Annotated[DiffStep, Tag("diff")],
        Annotated[TrendStep, Tag("trend")],
        Annotated[EncodeStep, Tag("encode")],
        Annotated[BinStep, Tag("bin")],
        Annotated[DatetimeStep, Tag("datetime")],
        Annotated[NullIndicatorStep, Tag("null_indicator")],
    ],
    Discriminator("op"),
]


# -----------------------------------------------------------------------
# View definitions
# -----------------------------------------------------------------------


class ViewDef(BaseModel):
    """A named view: a source plus a chain of transform steps."""

    source: str  # name of a source or another view
    steps: list[TransformStep] = []
    description: str = ""
    cache: bool = True
    depends_on: list[str] = []  # explicit view dependencies
    cache_ttl_seconds: int | None = None  # cache expiry


# -----------------------------------------------------------------------
# Target profiles
# -----------------------------------------------------------------------

class TargetProfile(BaseModel):
    """Named target definition with task type and metrics."""
    column: str
    task: str = "binary"
    metrics: list[str] = []


# -----------------------------------------------------------------------
# Data sub-configs
# -----------------------------------------------------------------------

class DataPathsConfig(BaseModel):
    """Path-related data configuration."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    features_file: str = "features.parquet"
    outputs_dir: str | None = None
    entity_features_path: str | None = None


class MLProblemConfig(BaseModel):
    """ML problem definition."""
    task: str = "classification"
    target_column: str = "result"
    key_columns: list[str] = []
    time_column: str | None = None
    exclude_columns: list[str] = []


class DataCleaningConfig(BaseModel):
    """Column rename / cleaning configuration."""
    column_renames: dict[str, str] = {}


# -----------------------------------------------------------------------
# Data config
# -----------------------------------------------------------------------

# Fields belonging to each sub-config, used for routing flat kwargs
_PATHS_FIELDS = set(DataPathsConfig.model_fields.keys())
_ML_PROBLEM_FIELDS = set(MLProblemConfig.model_fields.keys())
_CLEANING_FIELDS = set(DataCleaningConfig.model_fields.keys())


class DataConfig(BaseModel):
    """Data configuration — paths + ML problem definition."""

    # Composed sub-configs
    paths: DataPathsConfig = DataPathsConfig()
    ml_problem: MLProblemConfig = MLProblemConfig()
    cleaning: DataCleaningConfig = DataCleaningConfig()

    # Named target profiles
    targets: dict[str, TargetProfile] = {}

    # Data pipeline
    sources: dict[str, SourceConfig] = {}
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()

    # Declarative feature store
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    feature_defs: dict[str, FeatureDef] = {}

    # Declarative views (ETL)
    views: dict[str, ViewDef] = {}
    features_view: str | None = None  # which view becomes the prediction table

    @model_validator(mode="before")
    @classmethod
    def _route_flat_kwargs(cls, data: Any) -> Any:
        """Route flat kwargs to sub-configs for backward compatibility."""
        if not isinstance(data, dict):
            return data
        paths_vals: dict[str, Any] = {}
        ml_vals: dict[str, Any] = {}
        clean_vals: dict[str, Any] = {}
        for field_name in list(data.keys()):
            if field_name in _PATHS_FIELDS and field_name not in ("paths",):
                paths_vals[field_name] = data.pop(field_name)
            elif field_name in _ML_PROBLEM_FIELDS and field_name not in ("ml_problem",):
                ml_vals[field_name] = data.pop(field_name)
            elif field_name in _CLEANING_FIELDS and field_name not in ("cleaning",):
                clean_vals[field_name] = data.pop(field_name)
        if paths_vals:
            existing = data.get("paths", {})
            if isinstance(existing, dict):
                existing.update(paths_vals)
                data["paths"] = existing
            else:
                data["paths"] = paths_vals
        if ml_vals:
            existing = data.get("ml_problem", {})
            if isinstance(existing, dict):
                existing.update(ml_vals)
                data["ml_problem"] = existing
            else:
                data["ml_problem"] = ml_vals
        if clean_vals:
            existing = data.get("cleaning", {})
            if isinstance(existing, dict):
                existing.update(clean_vals)
                data["cleaning"] = existing
            else:
                data["cleaning"] = clean_vals
        return data

    # --- Backward-compat property accessors for DataPathsConfig fields ---
    @property
    def raw_dir(self) -> str:
        return self.paths.raw_dir

    @raw_dir.setter
    def raw_dir(self, value: str) -> None:
        self.paths.raw_dir = value

    @property
    def processed_dir(self) -> str:
        return self.paths.processed_dir

    @processed_dir.setter
    def processed_dir(self, value: str) -> None:
        self.paths.processed_dir = value

    @property
    def features_dir(self) -> str:
        return self.paths.features_dir

    @features_dir.setter
    def features_dir(self, value: str) -> None:
        self.paths.features_dir = value

    @property
    def features_file(self) -> str:
        return self.paths.features_file

    @features_file.setter
    def features_file(self, value: str) -> None:
        self.paths.features_file = value

    @property
    def outputs_dir(self) -> str | None:
        return self.paths.outputs_dir

    @outputs_dir.setter
    def outputs_dir(self, value: str | None) -> None:
        self.paths.outputs_dir = value

    @property
    def entity_features_path(self) -> str | None:
        return self.paths.entity_features_path

    @entity_features_path.setter
    def entity_features_path(self, value: str | None) -> None:
        self.paths.entity_features_path = value

    # --- Backward-compat property accessors for MLProblemConfig fields ---
    @property
    def task(self) -> str:
        return self.ml_problem.task

    @task.setter
    def task(self, value: str) -> None:
        self.ml_problem.task = value

    @property
    def target_column(self) -> str:
        return self.ml_problem.target_column

    @target_column.setter
    def target_column(self, value: str) -> None:
        self.ml_problem.target_column = value

    @property
    def key_columns(self) -> list[str]:
        return self.ml_problem.key_columns

    @key_columns.setter
    def key_columns(self, value: list[str]) -> None:
        self.ml_problem.key_columns = value

    @property
    def time_column(self) -> str | None:
        return self.ml_problem.time_column

    @time_column.setter
    def time_column(self, value: str | None) -> None:
        self.ml_problem.time_column = value

    @property
    def exclude_columns(self) -> list[str]:
        return self.ml_problem.exclude_columns

    @exclude_columns.setter
    def exclude_columns(self, value: list[str]) -> None:
        self.ml_problem.exclude_columns = value

    # --- Backward-compat property accessors for DataCleaningConfig fields ---
    @property
    def column_renames(self) -> dict[str, str]:
        return self.cleaning.column_renames

    @column_renames.setter
    def column_renames(self, value: dict[str, str]) -> None:
        self.cleaning.column_renames = value

    def resolve_target(self, name: str | None = None) -> tuple[str, str, list[str]]:
        """Resolve a target profile by name. Returns (column, task, metrics)."""
        if name is None:
            return self.target_column, self.task, []
        if name not in self.targets:
            available = ", ".join(sorted(self.targets.keys())) if self.targets else "(none defined)"
            raise ValueError(f"Unknown target '{name}'. Available targets: {available}")
        tp = self.targets[name]
        return tp.column, tp.task, list(tp.metrics)


# -----------------------------------------------------------------------
# Interaction & injection definitions
# -----------------------------------------------------------------------

class InteractionDef(BaseModel):
    """Defines a feature computed from two existing columns at predict time."""

    left: str
    right: str
    op: Literal["multiply", "add", "subtract", "divide", "abs_diff"]


class InjectionDef(BaseModel):
    """Defines an external feature source to merge into prediction data."""

    source_type: Literal["parquet", "csv", "callable"]
    path_pattern: str | None = None
    merge_keys: list[str]
    columns: list[str]
    fill_na: float = 0.0
    callable_module: str | None = None
    callable_function: str | None = None


# -----------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------

# Known model types — extensible via register_type() / unregister_type()
_KNOWN_MODEL_TYPES: set[str] = {
    "xgboost",
    "xgboost_regression",
    "catboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
    "elastic_net",
    "mlp",
    "tabnet",
    "gnn",
    "survival",
}


class ModelDef(BaseModel):
    """Single model definition."""

    type: str
    features: list[str] = []
    feature_sets: list[str] = []
    params: dict[str, Any] = {}
    active: bool = True
    mode: Literal["classifier", "regressor"] = "classifier"
    n_seeds: int = 1
    prediction_type: str | None = None
    train_folds: str = "all"
    pre_calibration: str | None = None
    cdf_scale: float | None = None
    training_filter: dict[str, Any] | None = None

    # Class imbalance handling
    class_weight: str | dict | None = None  # None, "balanced", or {label: weight}

    # Per-model NaN handling
    zero_fill_features: list[str] = []  # fill these with 0 before dropna

    # Provider fields — model A's output becomes features for model B
    provides: list[str] = []
    provides_level: Literal["instance", "entity"] = "instance"
    include_in_ensemble: bool = True
    provider_isolation: Literal["none", "per_fold"] = "none"

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _KNOWN_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type {v!r}. "
                f"Known types: {sorted(_KNOWN_MODEL_TYPES)}. "
                f"Use ModelDef.register_type() to add custom types."
            )
        return v

    @classmethod
    def register_type(cls, type_name: str) -> None:
        """Register a custom model type so it passes validation."""
        _KNOWN_MODEL_TYPES.add(type_name)

    @classmethod
    def unregister_type(cls, type_name: str) -> None:
        """Remove a custom model type from the known set."""
        _KNOWN_MODEL_TYPES.discard(type_name)


# -----------------------------------------------------------------------
# Ensemble definition
# -----------------------------------------------------------------------

class LogitAdjustment(BaseModel):
    """A single logit-space post-processing adjustment.

    Modes:
      - paired: 2 columns (entity A value, entity B value), each 0-1.
        Penalty = strength * (1 - value). Applied as logit -= penalty_a,
        logit += penalty_b.
      - diff: 1 column, already a signed difference.
        Applied as logit += strength * value.
    """

    columns: list[str]
    strength: float = 0.1
    default: float = 1.0
    mode: Literal["paired", "diff"] = "paired"

    @field_validator("columns")
    @classmethod
    def _check_columns(cls, v: list[str], info) -> list[str]:
        mode = info.data.get("mode", "paired")
        if mode == "paired" and len(v) != 2:
            raise ValueError("paired mode requires exactly 2 columns")
        if mode == "diff" and len(v) != 1:
            raise ValueError("diff mode requires exactly 1 column")
        return v


class CalibrationConfig(BaseModel):
    """Calibration settings for the ensemble."""
    method: str = "spline"
    spline_prob_max: float = 0.985
    spline_n_bins: int = 20


class PostProcessingConfig(BaseModel):
    """Post-processing settings for the ensemble."""
    temperature: float = 1.0
    clip_floor: float = 0.0
    prior_compression: float = 0.0
    prior_compression_threshold: int = 4
    prior_feature: str | None = None
    logit_adjustments: list[LogitAdjustment] = []


# Fields belonging to each sub-config, used for routing flat kwargs
_CALIBRATION_FIELDS = {"spline_prob_max", "spline_n_bins"}
_POST_PROCESSING_FIELDS = {"temperature", "clip_floor", "prior_compression",
                            "prior_compression_threshold", "prior_feature",
                            "logit_adjustments"}


class EnsembleDef(BaseModel):
    """Ensemble configuration."""

    method: Literal["stacked", "average"]
    meta_learner: dict[str, Any] = {}
    pre_calibration: dict[str, str] = {}
    meta_features: list[str] = []
    exclude_models: list[str] = []

    # Composed sub-configs
    calibration_config: CalibrationConfig = CalibrationConfig()
    post_processing: PostProcessingConfig = PostProcessingConfig()

    @model_validator(mode="before")
    @classmethod
    def _route_flat_kwargs(cls, data: Any) -> Any:
        """Route flat kwargs to sub-configs for backward compatibility."""
        if not isinstance(data, dict):
            return data
        cal_vals: dict[str, Any] = {}
        pp_vals: dict[str, Any] = {}

        # Handle "calibration" string -> calibration_config.method
        if "calibration" in data and "calibration_config" not in data:
            cal_val = data.pop("calibration")
            if isinstance(cal_val, str):
                cal_vals["method"] = cal_val
            elif isinstance(cal_val, dict):
                data["calibration_config"] = cal_val

        for field_name in list(data.keys()):
            if field_name in _CALIBRATION_FIELDS:
                cal_vals[field_name] = data.pop(field_name)
            elif field_name in _POST_PROCESSING_FIELDS:
                pp_vals[field_name] = data.pop(field_name)

        if cal_vals:
            existing = data.get("calibration_config", {})
            if isinstance(existing, dict):
                existing.update(cal_vals)
                data["calibration_config"] = existing
            else:
                data["calibration_config"] = cal_vals
        if pp_vals:
            existing = data.get("post_processing", {})
            if isinstance(existing, dict):
                existing.update(pp_vals)
                data["post_processing"] = existing
            else:
                data["post_processing"] = pp_vals
        return data

    # --- Backward-compat property: calibration (string) ---
    @property
    def calibration(self) -> str:
        return self.calibration_config.method

    @calibration.setter
    def calibration(self, value: str) -> None:
        self.calibration_config.method = value

    @property
    def spline_prob_max(self) -> float:
        return self.calibration_config.spline_prob_max

    @spline_prob_max.setter
    def spline_prob_max(self, value: float) -> None:
        self.calibration_config.spline_prob_max = value

    @property
    def spline_n_bins(self) -> int:
        return self.calibration_config.spline_n_bins

    @spline_n_bins.setter
    def spline_n_bins(self, value: int) -> None:
        self.calibration_config.spline_n_bins = value

    # --- Backward-compat properties: PostProcessingConfig ---
    @property
    def temperature(self) -> float:
        return self.post_processing.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.post_processing.temperature = value

    @property
    def clip_floor(self) -> float:
        return self.post_processing.clip_floor

    @clip_floor.setter
    def clip_floor(self, value: float) -> None:
        self.post_processing.clip_floor = value

    @property
    def prior_compression(self) -> float:
        return self.post_processing.prior_compression

    @prior_compression.setter
    def prior_compression(self, value: float) -> None:
        self.post_processing.prior_compression = value

    @property
    def prior_compression_threshold(self) -> int:
        return self.post_processing.prior_compression_threshold

    @prior_compression_threshold.setter
    def prior_compression_threshold(self, value: int) -> None:
        self.post_processing.prior_compression_threshold = value

    @property
    def prior_feature(self) -> str | None:
        return self.post_processing.prior_feature

    @prior_feature.setter
    def prior_feature(self, value: str | None) -> None:
        self.post_processing.prior_feature = value

    @property
    def logit_adjustments(self) -> list[LogitAdjustment]:
        return self.post_processing.logit_adjustments

    @logit_adjustments.setter
    def logit_adjustments(self, value: list[LogitAdjustment]) -> None:
        self.post_processing.logit_adjustments = value


# -----------------------------------------------------------------------
# Backtest config
# -----------------------------------------------------------------------

_CV_STRATEGIES = {"leave_one_out", "expanding_window", "sliding_window", "purged_kfold"}

_CV_STRATEGY_ALIASES = {
    "loso": "leave_one_out",
    "loo": "leave_one_out",
    "expanding": "expanding_window",
    "sliding": "sliding_window",
    "purged": "purged_kfold",
}


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    cv_strategy: str
    fold_column: str = "fold"       # column used for CV fold splitting
    fold_values: list[int] = []     # which values to use as test folds
    metrics: list[str] = ["brier", "accuracy", "ece", "log_loss"]
    min_train_folds: int = 1
    window_size: int | None = None
    n_folds: int | None = None
    purge_gap: int = 1
    eval_filter: str | None = None  # pandas query expression to filter test folds before metrics

    @field_validator("cv_strategy")
    @classmethod
    def _validate_cv_strategy(cls, v: str) -> str:
        v = _CV_STRATEGY_ALIASES.get(v, v)
        if v not in _CV_STRATEGIES:
            raise ValueError(
                f"Invalid cv_strategy {v!r}. "
                f"Must be one of: {sorted(_CV_STRATEGIES)} "
                f"(aliases: {sorted(_CV_STRATEGY_ALIASES.keys())})"
            )
        return v

    @model_validator(mode="after")
    def _validate_strategy_params(self) -> BacktestConfig:
        if self.cv_strategy == "sliding_window" and self.window_size is None:
            raise ValueError("sliding_window strategy requires window_size")
        if self.cv_strategy == "purged_kfold" and self.n_folds is None:
            raise ValueError("purged_kfold strategy requires n_folds")
        return self


# -----------------------------------------------------------------------
# Experiment definition
# -----------------------------------------------------------------------

class ExperimentDef(BaseModel):
    """Experiment protocol configuration."""

    naming_pattern: str | None = None
    log_path: str | None = None
    experiments_dir: str | None = None
    do_not_retry_path: str | None = None


# -----------------------------------------------------------------------
# Guardrail definition
# -----------------------------------------------------------------------

class GuardrailDef(BaseModel):
    """Guardrail configuration."""

    feature_leakage_denylist: list[str] = []
    critical_paths: list[str] = []
    naming_pattern: str | None = None
    rate_limit_seconds: int | None = None


# -----------------------------------------------------------------------
# Server / MCP tool definitions
# -----------------------------------------------------------------------

class ServerToolDef(BaseModel):
    """One MCP tool definition."""

    command: str
    args: list[str] = []
    guardrails: list[str] = []
    description: str | None = None
    timeout: int | None = None


class ServerDef(BaseModel):
    """MCP server configuration."""

    name: str
    tools: dict[str, ServerToolDef] = {}
    inspection: list[str] = []


# -----------------------------------------------------------------------
# Top-level project config
# -----------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level project configuration, assembled from YAML files."""

    config_version: str = "1.0"

    data: DataConfig
    models: dict[str, ModelDef]
    ensemble: EnsembleDef
    backtest: BacktestConfig
    feature_config: FeaturesConfig | None = None
    features: dict[str, FeatureDecl] | None = None
    sources: dict[str, SourceDecl] | None = None
    interactions: dict[str, InteractionDef] | None = None
    injections: dict[str, InjectionDef] | None = None
    experiments: ExperimentDef | None = None
    guardrails: GuardrailDef | None = None
    server: ServerDef | None = None

    def compute_config_hash(self) -> str:
        """Compute a deterministic SHA-256 hash of the config (excluding version)."""
        import hashlib
        import json

        data = self.model_dump(exclude={"config_version"})
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
