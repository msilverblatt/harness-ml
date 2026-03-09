# Harness Sports

Optional domain plugin for [HarnessML](https://github.com/msilverblatt/harness-ml) that adds sports matchup prediction capabilities. Extends the core framework through the hook system without modifying harness-core.

## Architecture

```
harnessml.sports
├── hooks.py       # Registers sports-specific hooks into core via HookRegistry
├── matchups.py    # Pairwise diff-feature matchup generation for tournaments
└── pairwise.py    # Auto-pairwise feature generation for head-to-head prediction
```

## How It Works

Harness Sports uses the core hook system (`HookRegistry`) to inject sports-domain behavior. On import, it registers hooks that teach core components about sports-specific patterns:

```python
from harnessml.core.runner.hooks import HookRegistry, COLUMN_CANDIDATES, COLUMN_RENAMES

HookRegistry.register(COLUMN_CANDIDATES, _sports_column_candidates)
HookRegistry.register(COLUMN_RENAMES, _sports_column_renames)
HookRegistry.register(COMPETITION_NARRATIVE, _default_competition_narrative)
```

### Hook Points

| Hook | Purpose |
|------|---------|
| `COLUMN_CANDIDATES` | Tells core about sports column patterns (TeamA, TeamB, TeamAWon, etc.) |
| `COLUMN_RENAMES` | Maps sports columns to canonical names (TeamAWon -> result) |
| `COMPETITION_NARRATIVE` | Provides domain-specific narrative for competition results |

## Column Candidates

Sports data follows a team/opponent pattern. The plugin registers candidate column names so that core pipeline, reporting, and profiling components can detect them automatically:

- **Team A**: `TeamA`, `team_a`
- **Team B**: `TeamB`, `team_b`
- **Label**: `TeamAWon`
- **Margin**: `TeamAMargin`
- **ID patterns**: `TeamA`, `TeamB`

## Matchup Generation

`matchups.py` generates all pairwise diff-feature matchups from team-season features:

- For each pair of seeded teams (A < B by seed) in a season
- Looks up team-season features for both teams
- Computes `diff_*` features (TeamA value - TeamB value)
- Imputes NaN values with feature medians
- Computes interaction features when configured
- Outputs include TeamA, TeamB, season, and all diff/interaction columns

This is used for tournament bracket prediction where you need to predict the outcome of every possible matchup.

## Matchup Symmetry

The diff-feature approach enforces matchup symmetry: swapping TeamA and TeamB negates all diff features, producing a complementary probability. This is a fundamental property for valid pairwise prediction.

## Key Principle

Harness Sports does **not** modify harness-core. It extends core through the hook system and provides domain-specific utilities that integrate with the standard pipeline. Any project can use harness-core without harness-sports installed.

## Quick Start

```bash
# From the monorepo root
uv sync

# Run tests
uv run pytest packages/harness-sports/tests/ -v
```

## Testing

```bash
uv run pytest packages/harness-sports/tests/ -v
```
