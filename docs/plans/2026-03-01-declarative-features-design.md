# Declarative Feature System Design

## Problem

The current feature system has two disconnected paths:

1. **Formula-based (MCP)**: Flat `pd.eval` expressions against columns already in parquet.
   The AI must know exact column names, manually create pairwise diffs, and reason about
   the computation pipeline.

2. **Programmatic (dev-facing)**: Decorator-based registry with caching in `harnessml-features`.
   Requires writing Python functions, not usable from MCP tools.

Neither path supports what we need: the AI says "I want a model with the diff of adj_em"
and the system resolves that this means source a team-level metric → compute pairwise diff
→ cache → ready for models. No pipeline thinking required.

Additionally:
- No concept of feature *type* (team, pairwise, matchup, regime) — everything is a flat column
- No automatic pairwise generation from team features
- No regime feature support (temporal/contextual flags)
- `PairwiseFeatureBuilder` exists in `harnessml-features` but isn't wired to MCP or the runner
- `feature_sets` on `ModelDef` exists but isn't implemented

## Approach

**Type-Driven Feature Ontology (Approach A)**: Each feature declares its semantic type.
Types drive computation rules automatically. Team features auto-generate pairwise variants.
The AI declares features by type, the system handles derivation, caching, and resolution.

---

## Feature Type Ontology

Four semantic types, each with a defined scope and computation model:

| Type | Scope | Example | Computation |
|------|-------|---------|-------------|
| **team** | Per-entity per-period | adj_em, win_rate | Sourced from entity-level data. Auto-generates pairwise. |
| **pairwise** | Per-matchup (A vs B) | diff_adj_em, ratio_tempo | Derived from team features via diff/ratio/custom. |
| **matchup** | Per-game context | is_neutral, venue_altitude | Sourced from matchup-level data. No derivation. |
| **regime** | Contextual flag | late_season, conference_tourney | Boolean condition on existing columns. |

**Cascade rule**: Team features automatically produce pairwise variants unless opted out.
When you add a team feature, the system registers its pairwise derivatives without being asked.

---

## Schema

### New Types (in `schema.py`)

```python
class FeatureType(str, Enum):
    TEAM = "team"
    PAIRWISE = "pairwise"
    MATCHUP = "matchup"
    REGIME = "regime"

class PairwiseMode(str, Enum):
    DIFF = "diff"        # entity_a - entity_b
    RATIO = "ratio"      # entity_a / entity_b
    BOTH = "both"        # generates diff and ratio
    NONE = "none"        # no auto-derivation

class FeatureDef(BaseModel):
    """Declarative feature definition."""
    name: str
    type: FeatureType
    source: str | None = None           # SourceConfig name to pull from
    column: str | None = None           # column in source data (for direct sourcing)
    formula: str | None = None          # computation formula (pd.eval safe)
    condition: str | None = None        # boolean condition (for regime features)
    pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    description: str = ""
    nan_strategy: str = "median"
    category: str = "general"           # for feature_sets grouping
    enabled: bool = True
```

### Feature Store Config (added to DataConfig)

```python
class FeatureStoreConfig(BaseModel):
    """Configuration for the declarative feature store."""
    cache_dir: str = "data/features/cache"
    auto_pairwise: bool = True           # auto-generate pairwise from team features
    default_pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    entity_a_column: str = "entity_a_id" # matchup-level entity A identifier
    entity_b_column: str = "entity_b_id" # matchup-level entity B identifier
    entity_column: str = "entity_id"     # team-level entity identifier
    period_column: str = "period_id"     # temporal identifier (season, etc.)

class DataConfig(BaseModel):
    # ... existing fields ...
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    feature_defs: dict[str, FeatureDef] = {}   # declared features
```

### Feature Sets (implemented via `ModelDef.feature_sets`)

Feature sets group features by category. Models reference sets instead of individual columns:

```yaml
models:
  xgb_v1:
    feature_sets: ["efficiency", "tempo", "matchup_context"]
    # Resolves to all enabled features in those categories
```

Resolution: `FeatureStore.resolve_sets(["efficiency", "tempo"])` → `list[str]` of column names.

---

## FeatureStore Architecture

### Class Design

```python
class FeatureStore:
    """Central feature management: declaration, computation, caching, resolution."""

    def __init__(self, project_dir: Path, config: DataConfig):
        self.project_dir = project_dir
        self.config = config
        self.store_config = config.feature_store
        self._registry: dict[str, FeatureDef] = dict(config.feature_defs)
        self._cache = FeatureCache(project_dir / store_config.cache_dir)
        self._matchup_df: pd.DataFrame | None = None   # loaded matchup data
        self._entity_dfs: dict[str, pd.DataFrame] = {}  # per-source entity data

    # --- Declaration ---
    def add(self, feature: FeatureDef) -> FeatureResult:
        """Declare and compute a feature. Auto-generates pairwise if team type."""

    def remove(self, name: str) -> None:
        """Remove a feature and its derivatives from registry and cache."""

    # --- Computation ---
    def compute(self, name: str) -> pd.Series:
        """Compute a single feature. Checks cache first."""

    def compute_all(self, names: list[str] | None = None) -> pd.DataFrame:
        """Compute all (or specified) features. Returns matchup-level DataFrame."""

    # --- Resolution ---
    def resolve(self, names: list[str]) -> pd.DataFrame:
        """Resolve feature names to a training-ready DataFrame."""

    def resolve_sets(self, set_names: list[str]) -> list[str]:
        """Resolve feature set names to column name list."""

    def available(self, type_filter: FeatureType | None = None) -> list[FeatureDef]:
        """List all registered features, optionally filtered by type."""

    # --- Cache Management ---
    def refresh(self, sources: list[str] | None = None) -> dict[str, str]:
        """Invalidate and recompute features from specified sources."""

    # --- Persistence ---
    def save_registry(self) -> None:
        """Persist feature_defs back to pipeline.yaml."""
```

### Computation Flow by Type

**Team features**:
1. Load source data (from `SourceConfig` or existing parquet)
2. Extract `column` or evaluate `formula` at entity+period level
3. Cache entity-level result
4. If `auto_pairwise` and `pairwise_mode != NONE`:
   - Use `PairwiseFeatureBuilder` to generate matchup-level diff/ratio
   - Register derived pairwise features automatically
   - Cache pairwise results separately

**Pairwise features**:
1. If derived from team feature: computed automatically (see above)
2. If custom formula: evaluate at matchup level using `team_a.*` / `team_b.*` references
3. Cache at matchup level

**Matchup features**:
1. Load from matchup-level source or evaluate formula
2. No derivation — stored directly
3. Cache at matchup level

**Regime features**:
1. Evaluate `condition` as a boolean expression against matchup data
2. Store as 0/1 integer column
3. Cache at matchup level

### Formula Resolution (extended)

Current `_resolve_formula()` in `feature_engine.py` handles column references and
`@feature_name` references. Extended for the new system:

- **`team_a.{col}` / `team_b.{col}`**: Resolved to entity A/B values at matchup level.
  Used for custom pairwise formulas.
- **`@{feature_name}`**: Resolved to a previously computed feature in the store.
  If not yet computed, triggers on-demand computation (lazy resolution).
- **`{column_name}`**: Direct column reference in current scope (matchup or entity level).

---

## Caching

### Content-Addressed Cache

Each computed feature gets a cache key derived from:

```python
def _compute_cache_key(self, feature: FeatureDef) -> str:
    """Deterministic cache key from feature definition + source state."""
    components = [
        feature.name,
        feature.type.value,
        feature.source or "",
        feature.column or "",
        feature.formula or "",
        feature.condition or "",
        feature.pairwise_mode.value,
        self._source_fingerprint(feature.source),  # hash of source data shape+sample
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
```

### Cache Structure

```
data/features/cache/
├── manifest.json           # {feature_name: {cache_key, path, type, computed_at}}
├── team/
│   ├── adj_em.parquet      # entity_id + period_id + adj_em
│   └── win_rate.parquet
├── pairwise/
│   ├── diff_adj_em.parquet # matchup-level with entity_a_id, entity_b_id, period_id
│   └── ratio_win_rate.parquet
├── matchup/
│   └── is_neutral.parquet
└── regime/
    └── late_season.parquet
```

### Invalidation Cascade

When source data changes:
1. All team features from that source are invalidated
2. All pairwise features derived from those team features are invalidated
3. Regime features referencing invalidated columns are invalidated

Implemented via dependency tracking in the manifest:

```json
{
  "adj_em": {
    "cache_key": "a1b2c3d4",
    "path": "team/adj_em.parquet",
    "type": "team",
    "source": "kenpom",
    "derived_from": [],
    "derivatives": ["diff_adj_em", "ratio_adj_em"]
  },
  "diff_adj_em": {
    "cache_key": "e5f6g7h8",
    "path": "pairwise/diff_adj_em.parquet",
    "type": "pairwise",
    "source": null,
    "derived_from": ["adj_em"],
    "derivatives": []
  }
}
```

---

## Auto-Derivation Rules

When a team feature is added with `pairwise_mode != NONE`:

| Mode | Generated Feature | Formula |
|------|-------------------|---------|
| `diff` | `diff_{name}` | `entity_a.{col} - entity_b.{col}` |
| `ratio` | `ratio_{name}` | `entity_a.{col} / (entity_b.{col} + epsilon)` |
| `both` | `diff_{name}` + `ratio_{name}` | Both of the above |

The derived pairwise features:
- Are auto-registered in `_registry` with `type=PAIRWISE`
- Have `source=None` and `derived_from=[parent_name]`
- Inherit `category` and `nan_strategy` from parent
- Are cached independently (so they don't recompute when parent is still valid)

---

## MCP Interface

### Updated Tools

```python
def add_feature(
    project_dir: Path,
    name: str,
    *,
    type: str = "pairwise",           # "team", "pairwise", "matchup", "regime"
    source: str | None = None,         # SourceConfig name
    column: str | None = None,         # column in source
    formula: str | None = None,        # computation formula
    condition: str | None = None,      # regime condition
    pairwise_mode: str = "diff",       # "diff", "ratio", "both", "none"
    category: str = "general",
    description: str = "",
) -> str:
    """Add a declarative feature to the project."""
```

**Backward compatibility**: If `type` is not specified and `formula` is provided,
falls back to the current formula-based path (creates a pairwise-level column via pd.eval).
This preserves the existing `add_feature(name, formula)` contract.

### Example MCP Interactions

```
# AI adds a team feature — pairwise auto-generated
add_feature(name="adj_em", type="team", source="kenpom", column="AdjEM")
→ "Added team feature 'adj_em' from kenpom.AdjEM
   Auto-generated: diff_adj_em (r=0.42), ratio_adj_em (r=0.38)
   Category: general | Cache: team/adj_em.parquet"

# AI adds a matchup feature
add_feature(name="is_neutral", type="matchup", source="game_info", column="neutral_site")
→ "Added matchup feature 'is_neutral' from game_info.neutral_site
   Correlation with result: 0.03 | Null rate: 0.0%"

# AI adds a regime feature
add_feature(name="late_season", type="regime", condition="day_num > 100")
→ "Added regime feature 'late_season': day_num > 100
   Applies to 45.2% of matchups | Correlation with result: 0.01"

# AI adds a custom pairwise feature
add_feature(name="seed_product", type="pairwise", formula="team_a.seed * team_b.seed")
→ "Added pairwise feature 'seed_product'
   Correlation with result: -0.31 | Null rate: 0.0%"

# AI uses formula shorthand (backward-compatible)
add_feature(name="em_tempo", formula="diff_adj_em * diff_adj_tempo")
→ "Added feature 'em_tempo' (formula)
   Correlation with result: 0.29 | Null rate: 0.0%"
```

### Updated `available_features()`

Returns features grouped by type with stats:

```markdown
## Team Features (5)
| Name | Source | Pairwise | Correlation | Category |
|------|--------|----------|-------------|----------|
| adj_em | kenpom | diff (r=0.42) | - | efficiency |
| win_rate | team_stats | diff (r=0.38) | - | record |

## Pairwise Features (12)
| Name | Derived From | Correlation | Category |
|------|-------------|-------------|----------|
| diff_adj_em | adj_em | 0.42 | efficiency |
| diff_win_rate | win_rate | 0.38 | record |
| seed_product | (custom) | -0.31 | seeding |

## Matchup Features (2)
| Name | Source | Correlation | Category |
|------|--------|-------------|----------|
| is_neutral | game_info | 0.03 | context |

## Regime Features (1)
| Name | Condition | Coverage | Correlation |
|------|-----------|----------|-------------|
| late_season | day_num > 100 | 45.2% | 0.01 |
```

---

## Pipeline Runner Integration

### `_load_data()` Changes

Currently loads a single parquet. With the feature store:

```python
def _load_data(self):
    store = FeatureStore(self.project_dir, self.config.data)
    # Resolve all features needed by active models
    needed = set()
    for name, model in self._get_active_models():
        needed.update(model.features)
        needed.update(store.resolve_sets(model.feature_sets))
    # Compute and assemble
    self._df = store.resolve(list(needed))
```

### `feature_sets` Implementation

`ModelDef.feature_sets` references category names. Resolution:

```python
store.resolve_sets(["efficiency", "tempo"])
# → ["diff_adj_em", "diff_adj_oe", "diff_adj_de", "diff_adj_tempo", "ratio_adj_tempo"]
```

This allows models to declare intent ("use all efficiency features") instead of
listing individual columns.

### Provider Integration

Existing `provides` / `provides_level` on `ModelDef` remains unchanged.
Provider outputs inject into the feature store as synthetic matchup or team features,
available to downstream models.

---

## Existing Systems: Integration Plan

### Formula Engine (`feature_engine.py`)

Becomes a **computation backend** for the feature store. `FeatureStore.compute()` delegates
formula evaluation to `_resolve_formula()` for formula-based features. No breaking changes
to the function — it gains a caller.

### harnessml-features Package

`FeatureRegistry`, `FeatureBuilder`, `PairwiseFeatureBuilder` become **optional backends**.
Projects that use programmatic features (Python functions registered via decorators) continue
to work. The feature store can delegate to the builder for registered features.

### Current `add_feature()` MCP Tool

Backward compatible. If called without `type`, assumes formula-based pairwise feature
(current behavior). If called with `type`, uses the declarative system.

---

## Migration

Existing projects need:
1. No immediate changes — the formula path continues to work as-is
2. To opt into declarative features: add `feature_store:` config and start using `type=` in `add_feature()`
3. Existing formula-created columns in `features.parquet` remain accessible as raw columns
4. `feature_defs:` in YAML is optional — features can be added purely via MCP tools
