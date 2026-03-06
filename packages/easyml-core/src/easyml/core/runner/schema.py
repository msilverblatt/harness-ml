"""Project-level Pydantic models for YAML-driven orchestration.

These schemas define the shape of a project's configuration — models,
ensemble, backtest, features, sources, experiments, guardrails, and
MCP server definitions.  They live in the runner package (not
easyml-schemas) because they are orchestration concerns.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Tag, field_validator


# -----------------------------------------------------------------------
# Feature & source declarations
# -----------------------------------------------------------------------

class FeatureDecl(BaseModel):
    """Declares a feature pointing to a Python module."""

    module: str
    function: str
    category: str
    level: str  # free-form: "entity", "interaction", "regime", "query", etc.
    columns: list[str]
    nan_strategy: str = "median"


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
    """
    name: str
    type: FeatureType
    source: str | None = None
    column: str | None = None
    formula: str | None = None
    condition: str | None = None
    pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    description: str = ""
    nan_strategy: str = "median"
    category: str = "general"
    enabled: bool = True


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


# -----------------------------------------------------------------------
# Data config
# -----------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data configuration — paths + ML problem definition."""

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    features_file: str = "features.parquet"     # relative to features_dir
    outputs_dir: str | None = None

    # ML problem definition
    task: str = "classification"                # classification, regression, ranking
    target_column: str = "result"
    key_columns: list[str] = []                 # row identifiers (game_id, customer_id, etc.)
    time_column: str | None = None              # for temporal CV splits
    exclude_columns: list[str] = []             # columns to never use as features
    entity_features_path: str | None = None      # path to entity-level features parquet

    # Column name normalization
    column_renames: dict[str, str] = {}  # {old_name: new_name}

    # Data pipeline
    sources: dict[str, SourceConfig] = {}
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()

    # Declarative feature store
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    feature_defs: dict[str, FeatureDef] = {}

    # Declarative views (ETL)
    views: dict[str, ViewDef] = {}
    features_view: str | None = None  # which view becomes the prediction table


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

class EnsembleDef(BaseModel):
    """Ensemble configuration."""

    method: Literal["stacked", "average"]
    meta_learner: dict[str, Any] = {}
    pre_calibration: dict[str, str] = {}
    calibration: str = "spline"
    spline_prob_max: float = 0.985
    spline_n_bins: int = 20
    meta_features: list[str] = []
    prior_compression: float = 0.0
    prior_compression_threshold: int = 4
    temperature: float = 1.0
    clip_floor: float = 0.0
    availability_adjustment: float = 0.1
    exclude_models: list[str] = []
    prior_feature: str | None = None  # data column to use as prior (mapped to diff_prior)


# -----------------------------------------------------------------------
# Backtest config
# -----------------------------------------------------------------------

_CV_STRATEGIES = {"leave_one_out", "expanding_window", "sliding_window", "purged_kfold"}


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

    @field_validator("cv_strategy")
    @classmethod
    def _validate_cv_strategy(cls, v: str) -> str:
        if v not in _CV_STRATEGIES:
            raise ValueError(
                f"Invalid cv_strategy {v!r}. "
                f"Must be one of: {sorted(_CV_STRATEGIES)}"
            )
        return v


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
