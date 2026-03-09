# Example: Feature Engineering with Experiment Discipline

A walkthrough of adding engineered features and running disciplined experiments to validate them.

---

## Starting point

Assume you have a project with baseline models already trained and evaluated. You want to improve performance through feature engineering.

Check your current workflow phase:

```
pipeline(action="progress")
```

Feature engineering should come after model diversity — make sure you have at least 3 different model types in your ensemble before investing in features.

---

## Step 1: Understand your data

```
data(action="profile")
features(action="diversity")
```

The diversity check tells you if your current features are too correlated or if you are missing coverage in certain categories (e.g., all ratio features, no interaction features).

---

## Step 2: Add engineered features

### Formula features

Derive new columns from existing ones using pandas expressions:

```
features(action="add", name="price_per_unit",
  formula="total_price / quantity",
  type="instance",
  category="ratio"
)
```

### Interaction features

Auto-discover useful feature interactions:

```
features(action="auto_search",
  features=["age", "income", "tenure", "usage_hours"],
  search_types=["interactions"],
  top_n=10
)
```

### Grouped (entity) features

For entity-level aggregations (e.g., per-customer stats):

```
features(action="add_batch", features=[
  {"name": "avg_monthly_spend", "source": "monthly_charges", "type": "entity", "category": "spending"},
  {"name": "max_monthly_spend", "source": "monthly_charges", "type": "entity", "category": "spending"},
  {"name": "spend_volatility", "source": "monthly_charges", "type": "entity", "category": "spending"}
])
```

Entity features with `pairwise_mode` generate difference and/or ratio columns automatically:

```
features(action="add", name="income_level",
  source="annual_income",
  type="entity",
  pairwise_mode="both"
)
```

This creates `income_level_diff` and `income_level_ratio` columns.

### Regime features

Boolean flags that indicate a specific condition:

```
features(action="add", name="is_high_value",
  condition="total_charges > 5000",
  type="regime"
)
```

---

## Step 3: Test features before committing

Check which transformations add predictive value:

```
features(action="test_transformations",
  features=["price_per_unit", "avg_monthly_spend", "spend_volatility"],
  test_interactions=true
)
```

---

## Step 4: Run a disciplined experiment

Create an experiment with a hypothesis:

```
experiments(action="create",
  description="Add spending ratio and volatility features",
  hypothesis="Adding price_per_unit and spend_volatility should capture purchasing behavior patterns not represented by raw columns. Expect Brier improvement of 0.002-0.005, primarily on folds with higher customer diversity."
)
```

Write the overlay (features are additive, so overlay adds them to all models):

```
experiments(action="write_overlay",
  experiment_id="exp-003",
  overlay={
    "models.xgb_main.append_features": ["price_per_unit", "spend_volatility"],
    "models.lgb_main.append_features": ["price_per_unit", "spend_volatility"],
    "models.cat_main.append_features": ["price_per_unit", "spend_volatility"]
  }
)
```

Run:

```
experiments(action="run", experiment_id="exp-003", primary_metric="brier")
```

---

## Step 5: Analyze and conclude

Check diagnostics:

```
pipeline(action="diagnostics")
```

Log the conclusion regardless of outcome:

```
experiments(action="log_result",
  experiment_id="exp-003",
  conclusion="price_per_unit improved Brier by 0.003 across all folds. spend_volatility was neutral — high correlation with existing tenure feature likely makes it redundant. Will keep price_per_unit and drop spend_volatility. Next: test interaction between price_per_unit and contract_type.",
  verdict="partial"
)
```

---

## Step 6: Iterate

Based on the conclusion, create a follow-up experiment. The key discipline rules:

1. **Change one thing at a time** — do not add 5 features and change hyperparameters simultaneously
2. **Try 3-5 configurations** before abandoning a feature engineering strategy
3. **Diagnose between attempts** — check per-fold breakdowns, feature importance shifts
4. **Log every result** — failed experiments narrow the search space

```
experiments(action="create",
  description="Test price_per_unit x contract_type interaction",
  hypothesis="Interaction between price_per_unit and contract_type should capture that high-spend monthly customers churn differently than high-spend annual customers. Expect Brier improvement of 0.001-0.003."
)
```

---

## Quick reference: feature types

| Type | Use case | Example |
|------|----------|---------|
| `instance` | Direct column or row-level formula | `log(income)` |
| `entity` | Grouped aggregation (generates pairwise columns) | Per-team batting average |
| `regime` | Boolean condition flag | `is_weekend`, `is_playoff` |
| `pairwise` | Explicit instance-level formula between entities | `home_score - away_score` |
