# harness-ml Plan 2b: Features + Evals

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature definition system (entity, pairwise, instance, model_output types with formula resolution via harness-data's expression engine) and the generic eval framework (threshold checks, comparisons, YAML-driven eval definitions with presets).

**Architecture:** Features are declarative definitions in `features.yaml` that resolve to DataFrame columns at training time. The feature resolver calls harness-data's expression engine for computed features. The eval system is a generic framework — no domain-specific code. Eval dimensions are pure YAML. Presets provide sensible defaults per task type.

**Tech Stack:** Python 3.11+, pandas, numpy, pydantic, pyyaml, harness-data (expression engine dependency)

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Sections 9 (Eval System) + 10 (Pairwise Features)

**E2E testing mandate:** After every 2-3 tasks, write real e2e tests that exercise the full chain with realistic sports data, verify actual computed values, and catch integration bugs.

---

## File Structure

```
packages/harness-ml/src/harness/ml/
├── features/
│   ├── __init__.py
│   ├── schema.py              # FeatureDefinition, FeatureType enum, FeatureSet
│   ├── resolver.py            # FeatureResolver — resolve definitions to DataFrame columns
│   ├── pairwise.py            # Pairwise derivative generation (diff, ratio)
│   └── augmentation.py        # Symmetric data augmentation for pairwise models
├── evals/
│   ├── __init__.py
│   ├── schema.py              # EvalDefinition, EvalCheck, EvalComparison, EvalReport
│   ├── runner.py              # Generic eval runner — load defs → check → compare → report
│   ├── checks.py              # Threshold operators (<, >, between, !=)
│   ├── comparisons.py         # vs_parent, vs_baseline comparisons
│   └── presets/
│       ├── binary.yaml
│       ├── regression.yaml
│       └── multiclass.yaml

tests/
├── test_features/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_resolver.py
│   ├── test_pairwise.py
│   └── test_augmentation.py
├── test_evals/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_runner.py
│   ├── test_checks.py
│   └── test_comparisons.py
└── test_e2e.py                # Extended with feature + eval e2e tests
```

---

### Task 1: Feature Schema + Types

**Files:**
- Create: `src/harness/ml/features/__init__.py`
- Create: `src/harness/ml/features/schema.py`
- Create: `tests/test_features/__init__.py`
- Create: `tests/test_features/test_schema.py`

- [ ] **Step 1: Write failing tests for feature schema**

```python
# tests/test_features/test_schema.py
import pytest
from harness.ml.features.schema import (
    FeatureDefinition, FeatureType, FeatureSet,
)


class TestFeatureType:
    def test_entity_type(self):
        assert FeatureType.ENTITY == "entity"

    def test_pairwise_type(self):
        assert FeatureType.PAIRWISE == "pairwise"

    def test_instance_type(self):
        assert FeatureType.INSTANCE == "instance"

    def test_model_output_type(self):
        assert FeatureType.MODEL_OUTPUT == "model_output"


class TestFeatureDefinition:
    def test_entity_feature(self):
        f = FeatureDefinition(
            name="seed",
            feature_type=FeatureType.ENTITY,
            source_column="seed",
        )
        assert f.name == "seed"
        assert f.feature_type == FeatureType.ENTITY
        assert f.auto_pairwise is True  # default for entity features

    def test_pairwise_feature_with_formula(self):
        f = FeatureDefinition(
            name="rating_diff",
            feature_type=FeatureType.PAIRWISE,
            formula="rating_a - rating_b",
        )
        assert f.formula == "rating_a - rating_b"

    def test_instance_feature(self):
        f = FeatureDefinition(
            name="tournament_stage",
            feature_type=FeatureType.INSTANCE,
            source_column="stage",
        )
        assert f.source_column == "stage"

    def test_model_output_feature(self):
        f = FeatureDefinition(
            name="pred_strength",
            feature_type=FeatureType.MODEL_OUTPUT,
            model="team_strength",
            auto_pairwise=True,
        )
        assert f.model == "team_strength"

    def test_feature_active_default(self):
        f = FeatureDefinition(name="x", feature_type=FeatureType.INSTANCE, source_column="x")
        assert f.active is True

    def test_feature_inactive(self):
        f = FeatureDefinition(name="x", feature_type=FeatureType.INSTANCE, source_column="x", active=False)
        assert f.active is False


class TestFeatureSet:
    def test_create_feature_set(self):
        fs = FeatureSet(features={
            "seed": FeatureDefinition(name="seed", feature_type=FeatureType.ENTITY, source_column="seed"),
            "rating_diff": FeatureDefinition(name="rating_diff", feature_type=FeatureType.PAIRWISE, formula="rating_a - rating_b"),
        })
        assert len(fs.features) == 2

    def test_active_features(self):
        fs = FeatureSet(features={
            "a": FeatureDefinition(name="a", feature_type=FeatureType.INSTANCE, source_column="a", active=True),
            "b": FeatureDefinition(name="b", feature_type=FeatureType.INSTANCE, source_column="b", active=False),
        })
        active = fs.active_features()
        assert len(active) == 1
        assert "a" in active

    def test_features_by_type(self):
        fs = FeatureSet(features={
            "seed": FeatureDefinition(name="seed", feature_type=FeatureType.ENTITY, source_column="seed"),
            "stage": FeatureDefinition(name="stage", feature_type=FeatureType.INSTANCE, source_column="stage"),
            "diff": FeatureDefinition(name="diff", feature_type=FeatureType.PAIRWISE, formula="a - b"),
        })
        entities = fs.features_by_type(FeatureType.ENTITY)
        assert len(entities) == 1
        assert "seed" in entities

    def test_from_yaml_dict(self):
        yaml_data = {
            "seed": {"type": "entity", "source_column": "seed"},
            "rating_diff": {"type": "pairwise", "formula": "rating_a - rating_b"},
        }
        fs = FeatureSet.from_yaml_dict(yaml_data)
        assert len(fs.features) == 2
        assert fs.features["seed"].feature_type == FeatureType.ENTITY
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement feature schema**

```python
# src/harness/ml/features/schema.py
"""Feature definitions — declarative feature types and sets."""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any


class FeatureType(str, Enum):
    ENTITY = "entity"
    PAIRWISE = "pairwise"
    INSTANCE = "instance"
    MODEL_OUTPUT = "model_output"


class FeatureDefinition(BaseModel):
    """A single feature definition."""
    name: str
    feature_type: FeatureType
    source_column: str | None = None     # Column name in clean data
    formula: str | None = None           # Expression to compute (uses expression engine)
    model: str | None = None             # Provider model name (for model_output type)
    auto_pairwise: bool = True           # Auto-generate diff/ratio for entity + model_output
    pairwise_methods: list[str] = Field(default_factory=lambda: ["diff", "ratio"])
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureSet(BaseModel):
    """Collection of feature definitions."""
    features: dict[str, FeatureDefinition] = Field(default_factory=dict)

    def active_features(self) -> dict[str, FeatureDefinition]:
        return {k: v for k, v in self.features.items() if v.active}

    def features_by_type(self, feature_type: FeatureType) -> dict[str, FeatureDefinition]:
        return {k: v for k, v in self.features.items() if v.feature_type == feature_type}

    @classmethod
    def from_yaml_dict(cls, data: dict) -> FeatureSet:
        features = {}
        for name, defn in data.items():
            defn = dict(defn)
            defn["name"] = name
            if "type" in defn:
                defn["feature_type"] = defn.pop("type")
            features[name] = FeatureDefinition(**defn)
        return cls(features=features)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): feature schema with entity, pairwise, instance, model_output types"
```

---

### Task 2: Pairwise Derivative Generation

**Files:**
- Create: `src/harness/ml/features/pairwise.py`
- Create: `tests/test_features/test_pairwise.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_pairwise.py
import pandas as pd
import numpy as np
import pytest
from harness.ml.features.pairwise import generate_pairwise_derivatives


class TestPairwiseDerivatives:
    def test_diff_derivative(self):
        """Entity feature generates diff: entity_a - entity_b."""
        df = pd.DataFrame({
            "entity_a_seed": [1, 3, 5],
            "entity_b_seed": [2, 1, 4],
        })
        result = generate_pairwise_derivatives(df, "seed", ["diff"])
        assert "diff_seed" in result.columns
        assert list(result["diff_seed"]) == [-1, 2, 1]

    def test_ratio_derivative(self):
        """Entity feature generates ratio: entity_a / entity_b."""
        df = pd.DataFrame({
            "entity_a_rating": [80.0, 90.0],
            "entity_b_rating": [40.0, 90.0],
        })
        result = generate_pairwise_derivatives(df, "rating", ["ratio"])
        assert "ratio_rating" in result.columns
        assert result["ratio_rating"].iloc[0] == 2.0
        assert result["ratio_rating"].iloc[1] == 1.0

    def test_both_derivatives(self):
        df = pd.DataFrame({
            "entity_a_wins": [20, 15],
            "entity_b_wins": [10, 20],
        })
        result = generate_pairwise_derivatives(df, "wins", ["diff", "ratio"])
        assert "diff_wins" in result.columns
        assert "ratio_wins" in result.columns
        assert list(result["diff_wins"]) == [10, -5]

    def test_ratio_with_zero_denominator(self):
        """Ratio should handle zero denominator safely."""
        df = pd.DataFrame({
            "entity_a_x": [10.0],
            "entity_b_x": [0.0],
        })
        result = generate_pairwise_derivatives(df, "x", ["ratio"])
        assert np.isfinite(result["ratio_x"].iloc[0]) or result["ratio_x"].iloc[0] == 0.0

    def test_custom_column_prefix(self):
        """Support custom entity column naming."""
        df = pd.DataFrame({
            "team_a_seed": [1, 3],
            "team_b_seed": [2, 1],
        })
        result = generate_pairwise_derivatives(
            df, "seed", ["diff"], entity_a_prefix="team_a_", entity_b_prefix="team_b_"
        )
        assert "diff_seed" in result.columns
        assert list(result["diff_seed"]) == [-1, 2]
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement pairwise derivative generation**

```python
# src/harness/ml/features/pairwise.py
"""Pairwise derivative generation — diff, ratio from entity features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_pairwise_derivatives(
    df: pd.DataFrame,
    feature_name: str,
    methods: list[str],
    entity_a_prefix: str = "entity_a_",
    entity_b_prefix: str = "entity_b_",
) -> pd.DataFrame:
    """Generate pairwise derivative columns from entity columns.

    For feature 'seed' with entity columns 'entity_a_seed' and 'entity_b_seed':
    - diff: diff_seed = entity_a_seed - entity_b_seed
    - ratio: ratio_seed = entity_a_seed / entity_b_seed (safe division)
    """
    result = df.copy()
    col_a = f"{entity_a_prefix}{feature_name}"
    col_b = f"{entity_b_prefix}{feature_name}"

    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(
            f"Entity columns not found: '{col_a}' and/or '{col_b}' "
            f"not in {list(df.columns)}"
        )

    a = df[col_a].astype(float)
    b = df[col_b].astype(float)

    for method in methods:
        if method == "diff":
            result[f"diff_{feature_name}"] = a - b
        elif method == "ratio":
            result[f"ratio_{feature_name}"] = np.where(b != 0, a / b, 0.0)
        else:
            raise ValueError(f"Unknown pairwise method: {method}")

    return result
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): pairwise derivative generation (diff, ratio)"
```

---

### Task 3: Feature Resolver

**Files:**
- Create: `src/harness/ml/features/resolver.py`
- Create: `tests/test_features/test_resolver.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_resolver.py
import pandas as pd
import numpy as np
import pytest
from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
from harness.ml.features.resolver import FeatureResolver


class TestFeatureResolver:
    @pytest.fixture
    def clean_data(self):
        """Realistic matchup dataset."""
        return pd.DataFrame({
            "game_id": [1, 2, 3, 4, 5],
            "entity_a_seed": [1, 3, 5, 2, 4],
            "entity_b_seed": [2, 1, 4, 6, 3],
            "entity_a_rating": [90.0, 75.0, 85.0, 80.0, 70.0],
            "entity_b_rating": [80.0, 85.0, 70.0, 65.0, 88.0],
            "neutral_site": [0, 1, 0, 1, 0],
            "season": [2024, 2024, 2024, 2024, 2024],
            "target": [1, 0, 1, 1, 0],
        })

    def test_resolve_instance_feature(self, clean_data):
        """Instance feature resolves to existing column."""
        features = FeatureSet(features={
            "neutral_site": FeatureDefinition(
                name="neutral_site", feature_type=FeatureType.INSTANCE,
                source_column="neutral_site",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(clean_data, features)
        assert "neutral_site" in result.columns

    def test_resolve_entity_feature_generates_derivatives(self, clean_data):
        """Entity feature auto-generates diff and ratio columns."""
        features = FeatureSet(features={
            "seed": FeatureDefinition(
                name="seed", feature_type=FeatureType.ENTITY,
                source_column="seed", auto_pairwise=True,
                pairwise_methods=["diff", "ratio"],
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(clean_data, features)
        assert "diff_seed" in result.columns
        assert "ratio_seed" in result.columns
        # Verify values: game 1: entity_a_seed=1, entity_b_seed=2 → diff=-1
        assert result["diff_seed"].iloc[0] == -1

    def test_resolve_pairwise_formula(self, clean_data):
        """Pairwise feature with formula resolves via expression engine."""
        features = FeatureSet(features={
            "rating_gap": FeatureDefinition(
                name="rating_gap", feature_type=FeatureType.PAIRWISE,
                formula="entity_a_rating - entity_b_rating",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(clean_data, features)
        assert "rating_gap" in result.columns
        assert result["rating_gap"].iloc[0] == 10.0  # 90 - 80

    def test_resolve_only_active_features(self, clean_data):
        """Inactive features are skipped."""
        features = FeatureSet(features={
            "active_feat": FeatureDefinition(
                name="active_feat", feature_type=FeatureType.INSTANCE,
                source_column="neutral_site", active=True,
            ),
            "inactive_feat": FeatureDefinition(
                name="inactive_feat", feature_type=FeatureType.INSTANCE,
                source_column="season", active=False,
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(clean_data, features)
        resolved_names = resolver.resolved_feature_names
        assert "active_feat" in resolved_names
        assert "inactive_feat" not in resolved_names

    def test_resolve_multiple_types(self, clean_data):
        """Mix of entity + pairwise + instance resolves together."""
        features = FeatureSet(features={
            "seed": FeatureDefinition(
                name="seed", feature_type=FeatureType.ENTITY,
                source_column="seed", pairwise_methods=["diff"],
            ),
            "rating_gap": FeatureDefinition(
                name="rating_gap", feature_type=FeatureType.PAIRWISE,
                formula="entity_a_rating - entity_b_rating",
            ),
            "neutral": FeatureDefinition(
                name="neutral", feature_type=FeatureType.INSTANCE,
                source_column="neutral_site",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(clean_data, features)
        assert "diff_seed" in result.columns
        assert "rating_gap" in result.columns
        assert "neutral" in result.columns

    def test_resolve_missing_column_raises(self, clean_data):
        """Formula referencing nonexistent column gives clear error."""
        features = FeatureSet(features={
            "bad": FeatureDefinition(
                name="bad", feature_type=FeatureType.PAIRWISE,
                formula="nonexistent_col * 2",
            ),
        })
        resolver = FeatureResolver()
        with pytest.raises((ValueError, KeyError)):
            resolver.resolve(clean_data, features)

    def test_get_feature_columns(self, clean_data):
        """resolved_feature_names returns only the resolved feature column names."""
        features = FeatureSet(features={
            "seed": FeatureDefinition(
                name="seed", feature_type=FeatureType.ENTITY,
                source_column="seed", pairwise_methods=["diff", "ratio"],
            ),
            "neutral": FeatureDefinition(
                name="neutral", feature_type=FeatureType.INSTANCE,
                source_column="neutral_site",
            ),
        })
        resolver = FeatureResolver()
        resolver.resolve(clean_data, features)
        names = resolver.resolved_feature_names
        assert "diff_seed" in names
        assert "ratio_seed" in names
        assert "neutral" in names
        # Original data columns should NOT be in the feature names
        assert "game_id" not in names
        assert "target" not in names
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement FeatureResolver**

The resolver:
1. Iterates active features by type
2. For INSTANCE: maps source_column or uses the name directly
3. For ENTITY: calls `generate_pairwise_derivatives` from `pairwise.py`
4. For PAIRWISE: evaluates formula via harness-data's `ExpressionEngine`
5. For MODEL_OUTPUT: skips (handled by the training pipeline's ProviderContext)
6. Tracks resolved column names in `resolved_feature_names`

```python
# src/harness/ml/features/resolver.py
"""Feature resolver — resolves feature definitions to DataFrame columns."""
from __future__ import annotations

import pandas as pd
from harness.data.expressions.engine import ExpressionEngine
from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
from harness.ml.features.pairwise import generate_pairwise_derivatives


class FeatureResolver:
    """Resolves feature definitions against a DataFrame."""

    def __init__(self):
        self._expr_engine = ExpressionEngine()
        self._resolved_names: list[str] = []

    @property
    def resolved_feature_names(self) -> list[str]:
        return list(self._resolved_names)

    def resolve(self, df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
        """Resolve all active features, adding computed columns to the DataFrame."""
        result = df.copy()
        self._resolved_names = []

        for name, defn in feature_set.active_features().items():
            if defn.feature_type == FeatureType.INSTANCE:
                self._resolve_instance(result, defn)
            elif defn.feature_type == FeatureType.ENTITY:
                self._resolve_entity(result, defn)
            elif defn.feature_type == FeatureType.PAIRWISE:
                self._resolve_pairwise(result, defn)
            elif defn.feature_type == FeatureType.MODEL_OUTPUT:
                pass  # Handled by ProviderContext in training pipeline

        return result

    def _resolve_instance(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        col = defn.source_column or defn.name
        if col not in df.columns:
            raise ValueError(f"Instance feature '{defn.name}': column '{col}' not found")
        if col != defn.name:
            df[defn.name] = df[col]
        self._resolved_names.append(defn.name)

    def _resolve_entity(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        if defn.auto_pairwise:
            feature_name = defn.source_column or defn.name
            updated = generate_pairwise_derivatives(
                df, feature_name, defn.pairwise_methods
            )
            for method in defn.pairwise_methods:
                col_name = f"{method}_{feature_name}"
                df[col_name] = updated[col_name]
                self._resolved_names.append(col_name)

    def _resolve_pairwise(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        if defn.formula:
            df[defn.name] = self._expr_engine.evaluate(df, defn.formula)
        elif defn.source_column and defn.source_column in df.columns:
            if defn.source_column != defn.name:
                df[defn.name] = df[defn.source_column]
        else:
            raise ValueError(f"Pairwise feature '{defn.name}': needs formula or source_column")
        self._resolved_names.append(defn.name)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): feature resolver with entity, pairwise, instance resolution"
```

---

### Task 4: Symmetric Data Augmentation

**Files:**
- Create: `src/harness/ml/features/augmentation.py`
- Create: `tests/test_features/test_augmentation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features/test_augmentation.py
import pandas as pd
import numpy as np
import pytest
from harness.ml.features.augmentation import augment_symmetric


class TestSymmetricAugmentation:
    def test_doubles_rows(self):
        df = pd.DataFrame({
            "diff_seed": [1, -2, 3],
            "diff_rating": [10.0, -5.0, 15.0],
            "neutral": [1, 0, 1],
            "target": [1, 0, 1],
        })
        result = augment_symmetric(df, target_col="target")
        assert len(result) == 6  # doubled

    def test_diff_features_negated(self):
        df = pd.DataFrame({
            "diff_seed": [3],
            "ratio_rating": [2.0],
            "neutral": [1],
            "target": [1],
        })
        result = augment_symmetric(df, target_col="target")
        # Original row + reversed row
        original = result.iloc[0]
        reversed_row = result.iloc[1]
        assert original["diff_seed"] == 3
        assert reversed_row["diff_seed"] == -3

    def test_binary_target_flipped(self):
        df = pd.DataFrame({
            "diff_seed": [1],
            "target": [1],
        })
        result = augment_symmetric(df, target_col="target", task_type="binary")
        assert result["target"].iloc[0] == 1
        assert result["target"].iloc[1] == 0  # flipped

    def test_regression_target_negated(self):
        df = pd.DataFrame({
            "diff_seed": [1],
            "target": [5.0],  # score spread
        })
        result = augment_symmetric(df, target_col="target", task_type="regression")
        assert result["target"].iloc[0] == 5.0
        assert result["target"].iloc[1] == -5.0

    def test_non_diff_features_unchanged(self):
        df = pd.DataFrame({
            "diff_seed": [1],
            "neutral": [1],
            "target": [1],
        })
        result = augment_symmetric(df, target_col="target")
        assert result["neutral"].iloc[0] == 1
        assert result["neutral"].iloc[1] == 1  # unchanged

    def test_ratio_features_inverted(self):
        df = pd.DataFrame({
            "ratio_rating": [2.0],
            "target": [1],
        })
        result = augment_symmetric(df, target_col="target")
        assert result["ratio_rating"].iloc[0] == 2.0
        assert abs(result["ratio_rating"].iloc[1] - 0.5) < 0.001  # inverted
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement augmentation**

```python
# src/harness/ml/features/augmentation.py
"""Symmetric data augmentation for pairwise models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def augment_symmetric(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary",
    diff_prefix: str = "diff_",
    ratio_prefix: str = "ratio_",
) -> pd.DataFrame:
    """Double training data with reversed pairwise rows.

    - diff_* features are negated
    - ratio_* features are inverted (1/x, with safe division)
    - Binary targets: 1-y
    - Regression targets (signed spreads): -y
    - Non-pairwise features are unchanged
    """
    reversed_df = df.copy()

    # Negate diff features
    for col in reversed_df.columns:
        if col.startswith(diff_prefix):
            reversed_df[col] = -reversed_df[col]

    # Invert ratio features
    for col in reversed_df.columns:
        if col.startswith(ratio_prefix):
            reversed_df[col] = np.where(
                reversed_df[col] != 0, 1.0 / reversed_df[col], 0.0
            )

    # Flip target
    if task_type == "binary":
        reversed_df[target_col] = 1 - reversed_df[target_col]
    elif task_type == "regression":
        reversed_df[target_col] = -reversed_df[target_col]

    return pd.concat([df, reversed_df], ignore_index=True)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): symmetric data augmentation for pairwise models"
```

---

### Task 5: Features E2E Tests

**Files:**
- Update: `tests/test_e2e.py`

- [ ] **Step 1: Write e2e tests for the full feature pipeline**

```python
class TestE2EFeatures:
    def test_full_feature_resolution_sports_data(self):
        """Realistic sports matchup feature resolution."""
        from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
        from harness.ml.features.resolver import FeatureResolver
        from harness.ml.features.augmentation import augment_symmetric

        # Realistic matchup data
        df = pd.DataFrame({
            "game_id": range(1, 11),
            "entity_a_seed": [1, 3, 5, 2, 4, 1, 3, 5, 2, 4],
            "entity_b_seed": [2, 1, 4, 6, 3, 3, 2, 1, 5, 6],
            "entity_a_rating": [90, 75, 85, 80, 70, 88, 77, 83, 82, 72],
            "entity_b_rating": [80, 85, 70, 65, 88, 75, 82, 90, 68, 80],
            "neutral_site": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            "target": [1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        })

        features = FeatureSet(features={
            "seed": FeatureDefinition(
                name="seed", feature_type=FeatureType.ENTITY,
                source_column="seed", pairwise_methods=["diff", "ratio"],
            ),
            "rating_gap": FeatureDefinition(
                name="rating_gap", feature_type=FeatureType.PAIRWISE,
                formula="entity_a_rating - entity_b_rating",
            ),
            "neutral": FeatureDefinition(
                name="neutral", feature_type=FeatureType.INSTANCE,
                source_column="neutral_site",
            ),
        })

        # Resolve features
        resolver = FeatureResolver()
        resolved = resolver.resolve(df, features)

        # Verify derivatives
        assert resolved["diff_seed"].iloc[0] == -1  # 1 - 2
        assert resolved["rating_gap"].iloc[0] == 10  # 90 - 80
        assert resolved["neutral"].iloc[0] == 0

        # Augment for pairwise training
        augmented = augment_symmetric(resolved, "target", "binary")
        assert len(augmented) == 20  # doubled
        # Reversed row should have negated diff
        assert augmented["diff_seed"].iloc[10] == 1  # negated: -(-1)

        # Feature names
        names = resolver.resolved_feature_names
        assert "diff_seed" in names
        assert "ratio_seed" in names
        assert "rating_gap" in names
        assert "neutral" in names
        assert "game_id" not in names
```

- [ ] **Step 2: Run e2e tests, verify pass**
- [ ] **Step 3: Commit**

```bash
git commit -m "test(harness-ml): e2e tests for feature resolution + augmentation"
```

---

### Task 6: Eval Schema

**Files:**
- Create: `src/harness/ml/evals/__init__.py`
- Create: `src/harness/ml/evals/schema.py`
- Create: `tests/test_evals/__init__.py`
- Create: `tests/test_evals/test_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evals/test_schema.py
import pytest
from harness.ml.evals.schema import (
    EvalCheck, EvalComparison, EvalDimension, EvalReport, CheckResult, ComparisonResult,
)


class TestEvalCheck:
    def test_less_than_pass(self):
        check = EvalCheck(metric="brier", op="<", value=0.25, severity="error")
        assert check.evaluate(0.20) is True

    def test_less_than_fail(self):
        check = EvalCheck(metric="brier", op="<", value=0.25, severity="error")
        assert check.evaluate(0.30) is False

    def test_greater_than(self):
        check = EvalCheck(metric="auroc", op=">", value=0.65, severity="warning")
        assert check.evaluate(0.80) is True
        assert check.evaluate(0.50) is False

    def test_between(self):
        check = EvalCheck(metric="slope", op="between", value=[0.8, 1.2], severity="warning")
        assert check.evaluate(1.0) is True
        assert check.evaluate(0.5) is False
        assert check.evaluate(1.5) is False


class TestEvalComparison:
    def test_expect_lower(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.20, baseline=0.25)
        assert result.improved is True
        assert result.delta == pytest.approx(-0.05)

    def test_expect_higher(self):
        comp = EvalComparison(vs="parent", metric="auroc", expect="higher")
        result = comp.evaluate(current=0.85, baseline=0.80)
        assert result.improved is True

    def test_regression_detected(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.30, baseline=0.20)
        assert result.improved is False
        assert result.delta == pytest.approx(0.10)


class TestEvalDimension:
    def test_from_yaml_dict(self):
        data = {
            "description": "Probability accuracy",
            "checks": [
                {"metric": "brier", "op": "<", "value": 0.25, "severity": "error"},
            ],
            "comparisons": [
                {"vs": "parent", "metric": "brier", "expect": "lower"},
            ],
            "judgment": "Check calibration curve",
        }
        dim = EvalDimension.from_yaml_dict("prob_accuracy", data)
        assert dim.name == "prob_accuracy"
        assert len(dim.checks) == 1
        assert len(dim.comparisons) == 1
        assert dim.judgment == "Check calibration curve"


class TestEvalReport:
    def test_summary_counts(self):
        report = EvalReport(
            dimensions={
                "dim1": {
                    "checks": [
                        CheckResult(metric="a", value=0.1, op="<", threshold=0.2, passed=True),
                        CheckResult(metric="b", value=0.5, op="<", threshold=0.3, passed=False, severity="error"),
                    ],
                    "comparisons": [
                        ComparisonResult(vs="parent", metric="a", current=0.1, baseline=0.2, delta=-0.1, improved=True),
                    ],
                },
            }
        )
        summary = report.summary()
        assert summary["checks_passed"] == 1
        assert summary["checks_total"] == 2
        assert summary["checks_failed_error"] == 1
        assert summary["improvements"] == 1
        assert summary["regressions"] == 0
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement eval schema**

Pydantic models: EvalCheck (with `evaluate(value) -> bool`), EvalComparison (with `evaluate(current, baseline) -> ComparisonResult`), EvalDimension, CheckResult, ComparisonResult, EvalReport (with `summary() -> dict`).

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): eval schema with checks, comparisons, dimensions, report"
```

---

### Task 7: Eval Runner + Checks + Comparisons

**Files:**
- Create: `src/harness/ml/evals/runner.py`
- Create: `src/harness/ml/evals/checks.py`
- Create: `src/harness/ml/evals/comparisons.py`
- Create: `tests/test_evals/test_runner.py`
- Create: `tests/test_evals/test_checks.py`
- Create: `tests/test_evals/test_comparisons.py`

- [ ] **Step 1: Write failing tests for runner**

```python
# tests/test_evals/test_runner.py
import pytest
import yaml
from pathlib import Path
from harness.ml.evals.runner import EvalRunner
from harness.ml.evals.schema import EvalDimension


class TestEvalRunner:
    def test_run_with_all_passing(self):
        dims = {
            "accuracy": EvalDimension.from_yaml_dict("accuracy", {
                "description": "Model accuracy",
                "checks": [{"metric": "brier", "op": "<", "value": 0.3, "severity": "error"}],
                "comparisons": [],
            }),
        }
        metrics = {"brier": 0.20}
        runner = EvalRunner(dims)
        report = runner.run(metrics)
        assert report.summary()["checks_passed"] == 1
        assert report.summary()["checks_failed_error"] == 0

    def test_run_with_failure(self):
        dims = {
            "accuracy": EvalDimension.from_yaml_dict("accuracy", {
                "description": "Model accuracy",
                "checks": [{"metric": "brier", "op": "<", "value": 0.15, "severity": "error"}],
                "comparisons": [],
            }),
        }
        metrics = {"brier": 0.20}
        runner = EvalRunner(dims)
        report = runner.run(metrics)
        assert report.summary()["checks_failed_error"] == 1

    def test_run_with_comparisons(self):
        dims = {
            "accuracy": EvalDimension.from_yaml_dict("accuracy", {
                "description": "Model accuracy",
                "checks": [],
                "comparisons": [{"vs": "parent", "metric": "brier", "expect": "lower"}],
            }),
        }
        metrics = {"brier": 0.18}
        parent_metrics = {"brier": 0.22}
        runner = EvalRunner(dims)
        report = runner.run(metrics, parent_metrics=parent_metrics)
        assert report.summary()["improvements"] == 1

    def test_load_from_yaml(self, tmp_path):
        evals_yaml = tmp_path / "evals.yaml"
        evals_yaml.write_text(yaml.dump({
            "evals": {
                "calibration": {
                    "description": "Calibration quality",
                    "checks": [
                        {"metric": "ece", "op": "<", "value": 0.05, "severity": "error"},
                    ],
                    "comparisons": [
                        {"vs": "parent", "metric": "ece", "expect": "lower"},
                    ],
                    "judgment": "Check the curve",
                },
            }
        }))
        runner = EvalRunner.from_yaml(evals_yaml)
        report = runner.run(
            metrics={"ece": 0.03},
            parent_metrics={"ece": 0.05},
        )
        assert report.summary()["checks_passed"] == 1
        assert report.summary()["improvements"] == 1

    def test_missing_metric_skipped(self):
        dims = {
            "test": EvalDimension.from_yaml_dict("test", {
                "description": "test",
                "checks": [{"metric": "nonexistent", "op": "<", "value": 0.5, "severity": "warning"}],
                "comparisons": [],
            }),
        }
        runner = EvalRunner(dims)
        report = runner.run({"brier": 0.2})
        # Missing metrics should be skipped, not error
        assert report.summary()["checks_total"] == 0
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement runner, checks, comparisons**

The runner:
1. Loads eval dimensions from YAML or dict
2. For each dimension, runs checks and comparisons against provided metrics
3. Produces an EvalReport

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-ml): eval runner with threshold checks + parent comparisons"
```

---

### Task 8: Eval Presets + Integration

**Files:**
- Create: `src/harness/ml/evals/presets/binary.yaml`
- Create: `src/harness/ml/evals/presets/regression.yaml`
- Create: `src/harness/ml/evals/presets/multiclass.yaml`

- [ ] **Step 1: Create preset YAML files**

```yaml
# src/harness/ml/evals/presets/binary.yaml
evals:
  probability_accuracy:
    description: "Are predicted probabilities trustworthy?"
    checks:
      - metric: ece
        op: "<"
        value: 0.05
        severity: error
      - metric: brier
        op: "<"
        value: 0.25
        severity: warning
    comparisons:
      - vs: parent
        metric: brier
        expect: lower
      - vs: parent
        metric: ece
        expect: lower
    judgment: |
      Review the calibration curve. Are there probability ranges where
      the model is systematically over- or under-confident?

  discrimination:
    description: "Can the model distinguish between outcomes?"
    checks:
      - metric: auroc
        op: ">"
        value: 0.55
        severity: error
      - metric: accuracy
        op: ">"
        value: 0.52
        severity: warning
    comparisons:
      - vs: parent
        metric: auroc
        expect: higher
    judgment: |
      Is the model separating positive and negative cases effectively?

  stability:
    description: "Consistent across evaluation folds?"
    checks:
      - metric: fold_std_brier
        op: "<"
        value: 0.05
        severity: warning
    comparisons:
      - vs: parent
        metric: fold_std_brier
        expect: lower
    judgment: |
      Large fold variance suggests the model is unstable or overfitting.
```

```yaml
# src/harness/ml/evals/presets/regression.yaml
evals:
  accuracy:
    description: "How close are predictions to actual values?"
    checks:
      - metric: rmse
        op: "<"
        value: 10.0
        severity: warning
      - metric: r2
        op: ">"
        value: 0.3
        severity: error
    comparisons:
      - vs: parent
        metric: rmse
        expect: lower
      - vs: parent
        metric: r2
        expect: higher
    judgment: |
      Is the model explaining meaningful variance or just fitting noise?
```

```yaml
# src/harness/ml/evals/presets/multiclass.yaml
evals:
  classification:
    description: "Classification accuracy across classes"
    checks:
      - metric: accuracy
        op: ">"
        value: 0.4
        severity: error
      - metric: f1_macro
        op: ">"
        value: 0.3
        severity: warning
    comparisons:
      - vs: parent
        metric: accuracy
        expect: higher
    judgment: |
      Are any classes being systematically misclassified?
```

- [ ] **Step 2: Write test that loads presets**

```python
def test_load_binary_preset():
    from harness.ml.evals.runner import EvalRunner
    preset_path = Path(__file__).parent.parent.parent / "src" / "harness" / "ml" / "evals" / "presets" / "binary.yaml"
    runner = EvalRunner.from_yaml(preset_path)
    report = runner.run({"brier": 0.20, "ece": 0.03, "auroc": 0.75, "accuracy": 0.70})
    summary = report.summary()
    assert summary["checks_total"] > 0
    assert summary["checks_passed"] > 0
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(harness-ml): eval presets for binary, regression, multiclass"
```

---

### Task 9: Features + Evals E2E Tests

**Files:**
- Update: `tests/test_e2e.py`

- [ ] **Step 1: Write comprehensive e2e tests**

```python
class TestE2EFullPipeline:
    def test_features_to_model_to_eval(self, binary_dataset):
        """Full chain: define features → resolve → train model → compute metrics → run evals."""
        from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
        from harness.ml.features.resolver import FeatureResolver
        from harness.ml.models.registry import ModelRegistry
        from harness.ml.tasks.registry import TaskRegistry
        from harness.ml.evals.runner import EvalRunner
        from harness.ml.evals.schema import EvalDimension

        X, y = binary_dataset

        # Train a model
        model = ModelRegistry.get("logistic")
        task = TaskRegistry.get("binary")
        result = model.fit(X, y, None, None, model.default_params("binary"))
        preds = model.predict(result.model, X)

        # Compute metrics
        metrics = task.compute_metrics(y.values, preds, ["brier", "accuracy", "auroc", "ece"])

        # Run evals
        dims = {
            "accuracy": EvalDimension.from_yaml_dict("accuracy", {
                "description": "Model quality",
                "checks": [
                    {"metric": "brier", "op": "<", "value": 0.25, "severity": "error"},
                    {"metric": "auroc", "op": ">", "value": 0.6, "severity": "error"},
                ],
                "comparisons": [],
            }),
        }
        runner = EvalRunner(dims)
        report = runner.run(metrics)

        summary = report.summary()
        assert summary["checks_passed"] == 2  # both should pass on correlated data
        assert summary["checks_failed_error"] == 0
```

- [ ] **Step 2: Run e2e tests**
- [ ] **Step 3: Update `__init__.py` with feature + eval exports**

```python
# Add to src/harness/ml/__init__.py
from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
from harness.ml.features.resolver import FeatureResolver
from harness.ml.evals.runner import EvalRunner
from harness.ml.evals.schema import EvalReport
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(harness-ml): features + evals e2e tests + public API (Plan 2b complete)"
```
