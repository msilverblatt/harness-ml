# easyml-features

Feature registry and engineering for EasyML -- decorator-based registration,
incremental caching via source hashing, and pairwise matchup features.

## Installation

```bash
pip install easyml-features
```

## Quick Start

```python
from easyml.features import FeatureRegistry, FeatureBuilder, PairwiseFeatureBuilder
from pathlib import Path

registry = FeatureRegistry()

@registry.register(
    name="win_rate", category="resume", level="team",
    output_columns=["win_rate"],
)
def compute_win_rate(df, config):
    result = df[["entity_id", "period_id"]].copy()
    result["win_rate"] = df["wins"] / (df["wins"] + df["losses"])
    return result

# Build features with incremental caching
builder = FeatureBuilder(
    registry=registry,
    cache_dir=Path("data/cache/"),
    manifest_path=Path("data/manifest.json"),
)
features_df = builder.build_all(raw_data, config={})

# Pairwise (matchup-level) features
pairwise = PairwiseFeatureBuilder(methods=["diff", "ratio"])
matchup_df = pairwise.build(features_df, matchups, feature_columns=["win_rate"])
```

## Key APIs

- `FeatureRegistry` -- Decorator-based registration with source hashing and auto-discovery
- `FeatureBuilder` -- Builds all registered features with manifest-based incremental caching
- `FeatureResolver` -- Resolves feature names to column lists from registry metadata
- `PairwiseFeatureBuilder` -- Computes diff/ratio features for entity matchups
