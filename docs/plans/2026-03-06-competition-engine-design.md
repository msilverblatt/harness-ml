# Competition Engine Design

## Overview

Generic competition simulation engine in `easyml-sports` supporting arbitrary
competitive formats: single-elimination brackets, double-elimination, round-robin
leagues, Swiss-system tournaments, and group-stage-to-knockout hybrids.

Config-driven (YAML/dict), with pluggable strategies, hook-based narrative
generation, and MCP tool integration.

## Formats

| Format | Use Cases | Output |
|--------|-----------|--------|
| `single_elimination` | March Madness, any knockout | Bracket picks, round probs |
| `double_elimination` | Esports, wrestling | Bracket picks (winners + losers) |
| `round_robin` | League seasons, group stages | Standings distributions |
| `swiss` | Chess, some esports | Standings distributions |
| `group_knockout` | World Cup, Champions League | Group standings + bracket picks |

## Package Layout

```
packages/easyml-sports/src/easyml/sports/
├── __init__.py                 # existing (hook registration)
├── hooks.py                    # existing + COMPETITION_NARRATIVE hook
├── matchups.py                 # existing
├── pairwise.py                 # existing
├── competitions/
│   ├── __init__.py             # public API re-exports
│   ├── schemas.py              # Pydantic configs + result types
│   ├── structure.py            # Build structures from config (any format)
│   ├── simulator.py            # Monte Carlo engine (vectorized)
│   ├── optimizer.py            # Pool-aware generation + pluggable strategies
│   ├── scorer.py               # Configurable scoring engine
│   ├── adjustments.py          # Post-model probability adjustments
│   ├── explainer.py            # Generic differentials + hook-based narratives
│   ├── confidence.py           # Pre-competition diagnostics
│   └── export.py               # Multi-format output (markdown, JSON, CSV)
```

MCP handler:

```
packages/easyml-plugin/src/easyml/plugin/handlers/competitions.py
```

## Config Schema

All competitions defined declaratively via `CompetitionConfig`:

```yaml
# Single-elimination
competition:
  format: single_elimination
  n_participants: 64
  regions: 4
  seeding: ranked           # ranked | random | manual
  byes: auto                # auto | none
  scoring:
    type: per_round
    values: [10, 20, 40, 80, 160, 320]

# Round-robin league
competition:
  format: round_robin
  n_participants: 20
  rounds: 38
  scoring:
    win: 3
    draw: 1
    loss: 0

# Swiss-system
competition:
  format: swiss
  n_participants: 32
  n_rounds: 5
  scoring:
    win: 1.0
    draw: 0.5
    loss: 0.0

# Group stage to knockout
competition:
  format: group_knockout
  groups:
    n_groups: 8
    group_size: 4
    format: round_robin
    advance: 2
  knockout:
    format: single_elimination
    scoring:
      type: per_round
      values: [10, 20, 40, 80]

# Double-elimination
competition:
  format: double_elimination
  n_participants: 16
  grand_final: true
  scoring:
    type: per_round
    values: [10, 20, 40, 80, 160]
```

## Data Types

### Core schemas (Pydantic models in `schemas.py`)

```python
class CompetitionConfig(BaseModel):
    format: CompetitionFormat
    n_participants: int
    regions: int | None = None
    seeding: SeedingMode = "ranked"
    scoring: ScoringConfig
    rounds: int | None = None           # round_robin
    n_rounds: int | None = None         # swiss
    byes: str = "auto"                  # single/double elimination
    groups: GroupConfig | None = None    # group_knockout
    knockout: KnockoutConfig | None = None
    grand_final: bool = True            # double_elimination

class MatchupContext(BaseModel):
    slot: str
    round_num: int
    entity_a: str
    entity_b: str
    prob_a: float
    model_probs: dict[str, float] = {}
    model_agreement: float = 1.0
    pick: str = ""
    strategy: str = ""
    upset: bool = False

class CompetitionResult(BaseModel):
    picks: dict[str, str]               # slot -> winner
    matchups: dict[str, MatchupContext]
    expected_points: float = 0.0
    win_probability: float = 0.0
    top10_probability: float = 0.0
    strategy: str = ""

class StandingsEntry(BaseModel):
    entity: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    goal_diff: float = 0.0

class CompetitionStructure(BaseModel):
    config: CompetitionConfig
    slots: list[str]
    slot_matchups: dict[str, tuple[str, str]]
    slot_to_round: dict[str, int]
    round_slots: dict[int, list[str]]
    seed_to_entity: dict[str, str]
    entity_to_seed: dict[str, str]

class ScoreResult(BaseModel):
    total_points: float
    round_points: dict[int, float]
    round_correct: dict[int, int]
    round_total: dict[int, int]
    picks_detail: list[dict]

class AdjustmentConfig(BaseModel):
    entity_multipliers: dict[str, float] = {}
    external_probs: pd.DataFrame | None = None  # uses arbitrary model
    external_weight: float = 0.0
    probability_overrides: dict[str, tuple[float, float]] = {}
```

## Simulator

```python
class CompetitionSimulator:
    def __init__(self, config: CompetitionConfig,
                 structure: CompetitionStructure,
                 probabilities: pd.DataFrame):
        # Builds dense probability matrix for O(1) lookups
        # Stores per-model probs for agreement analysis

    def get_win_prob(self, entity_a: str, entity_b: str) -> float: ...
    def get_model_agreement(self, entity_a: str, entity_b: str) -> float: ...
    def get_matchup_context(self, ...) -> MatchupContext: ...

    # Simulation
    def simulate_once(self, rng) -> dict: ...
    def simulate_many(self, n: int, seed: int) -> np.ndarray: ...
    def pick_most_likely(self) -> dict: ...

    # Analytics
    def entity_round_probabilities(self, n_sims, seed) -> pd.DataFrame: ...
```

Format-specific simulation dispatched internally:

- **Single/double elimination**: resolve bracket slots round-by-round, vectorized
  across all simulations
- **Round-robin**: simulate each fixture, accumulate standings
- **Swiss**: simulate round-by-round with pairing based on current standings
- **Group -> knockout**: simulate group stage (round-robin), build knockout bracket
  from qualifiers, simulate knockout

## Optimizer

Pool-size-aware bracket generation for elimination formats. Pluggable strategies.

```python
class CompetitionOptimizer:
    def __init__(self, simulator: CompetitionSimulator,
                 strategies: dict[str, StrategyFn] | None = None):
        # Merges custom strategies with built-in defaults

    def generate_brackets(self, pool_size: int, n_brackets: int,
                          n_sims: int, seed: int) -> list[CompetitionResult]:
        # 1. Generate candidates via strategy mix (scaled by pool_size)
        # 2. Score against simulated outcomes + opponent field (vectorized)
        # 3. Select top N diverse brackets (round-weighted overlap)
```

### Built-in strategies

| Strategy | Description |
|----------|-------------|
| `chalk` | Always pick favorite |
| `near_chalk` | Chalk with close-matchup flips (underdog >40%) |
| `random_sim` | Pure Monte Carlo sample |
| `contrarian` | Uniform upset boost (compress probs toward 0.5) |
| `late_contrarian` | Chalk early rounds, upset-boosted late rounds |
| `champion_anchor` | Condition on sampled champion, simulate rest |

Custom strategies registered via `StrategyFn` protocol:

```python
class StrategyFn(Protocol):
    def __call__(self, simulator: CompetitionSimulator,
                 rng: np.random.Generator, **kwargs) -> dict[str, str]: ...
```

Strategy mix interpolated continuously as function of `log10(pool_size)`:
small pools skew toward chalk, large pools toward contrarian/champion_anchor.

Diversity selection uses round-weighted overlap (late rounds weighted higher),
greedy selection with configurable overlap threshold (default 90%).

## Scorer

```python
class CompetitionScorer:
    def __init__(self, scoring: ScoringConfig): ...

    def score_bracket(self, picks: dict, actuals: dict,
                      structure: CompetitionStructure) -> ScoreResult: ...

    def score_standings(self, predicted: list[StandingsEntry],
                        actual: list[StandingsEntry]) -> ScoreResult: ...
```

## Adjustments

Three adjustment types applied post-model, pre-simulation:

1. **Entity multipliers** — scale an entity's win probability (e.g., injuries)
2. **External probability blending** — weighted blend with external source
   (e.g., betting lines)
3. **Hard overrides** — direct probability assignment for specific matchups

All clamped to `[0.01, 0.99]`, all logged for auditability.

```python
def apply_adjustments(probabilities: pd.DataFrame,
                      adjustments: AdjustmentConfig
                      ) -> tuple[pd.DataFrame, list[dict]]: ...
```

## Explainer

Generic feature-differential engine with hook-based narrative customization.

```python
class CompetitionExplainer:
    def __init__(self, entity_features: pd.DataFrame,
                 feature_display_names: dict[str, str] | None = None,
                 narrative_hook: Callable | None = None): ...

    def compute_differentials(self, entity_a, entity_b,
                              top_n=5) -> list[dict]: ...

    def generate_pick_stories(self, result: CompetitionResult) -> list[dict]:
        # Generic: differentials + model consensus + probability
        # Calls narrative_hook if registered for domain-specific prose

    def generate_entity_profiles(self, simulator, round_probs,
                                 top_n=20) -> list[dict]:
        # Round-by-round progression (elimination) or
        # standings probability (league)
```

Domain plugins register narrative hooks via `HookRegistry`:

```python
HookRegistry.register("COMPETITION_NARRATIVE", basketball_narrative_fn)
```

## Confidence

Pre-competition diagnostics (informational only, does not affect predictions):

1. **Feature outliers** — entities with features >2 sigma from training distribution
2. **Model disagreement** — matchups where models disagree most
3. **Historical context** — historical win rates for similar matchups
4. **Thin samples** — matchup types with limited historical data

## Export

Multi-format output:

- `export_bracket_markdown()` — human-readable bracket picks
- `export_standings_markdown()` — standings tables
- `export_json()` — machine-readable with full matchup context
- `export_csv()` — probabilities in submission format
- `export_analysis_report()` — comprehensive markdown combining all analyses

## MCP Tools

Single handler `handlers/competitions.py` with action dispatch:

| Action | Description |
|--------|-------------|
| `create` | Create competition from config dict |
| `list_formats` | Show available formats |
| `simulate` | Run Monte Carlo simulations |
| `standings` | Get standings distributions (league/swiss) |
| `round_probs` | Entity progression probabilities (elimination) |
| `generate_brackets` | Pool-aware bracket generation |
| `score_bracket` | Score picks against actuals |
| `adjust` | Apply probability adjustments |
| `explain` | Generate pick explanations |
| `profiles` | Entity profiles |
| `confidence` | Pre-competition diagnostics |
| `export` | Export results in various formats |
| `list_strategies` | Show available strategies |
| `add_strategy` | Register custom strategy |

## Testing

- Unit tests per module in `packages/easyml-sports/tests/competitions/`
- Each format gets dedicated test cases
- Simulator tested with known probabilities and deterministic seeds
- Optimizer tested for diversity and strategy mix properties
- Scorer tested against hand-computed results
- Integration test: full pipeline from config -> simulate -> optimize -> score -> export
