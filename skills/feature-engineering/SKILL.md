# HarnessML: Feature Engineering

Use when creating and testing new features. Every feature is a hypothesis about the data. Treat it that way.

---

## The Principle

A feature encodes an assumption: "this signal is predictive because of this mechanism." If you can't state the mechanism, you're adding noise and hoping the model sorts it out. Sometimes it does. Usually it doesn't.

---

## The Feature-as-Hypothesis Cycle

### 1. Hypothesize

Start with a reason. Not "let me try a ratio of A/B" — but "A/B captures efficiency because absolute A is confounded by scale, and normalizing by B controls for that."

Sources of feature hypotheses:
- Domain knowledge (`domain-research` skill)
- EDA findings (`eda` skill)
- Diagnosis of model errors (`diagnosis` skill) — "the model fails on subgroup X, what feature captures X's behavior?"
- Feature interaction patterns — "A and B are both weak, but A*B would capture [specific phenomenon]"

### 2. Check for Redundancy

Before adding anything:

```
features(action="discover")
```

If an existing feature already captures the same signal (correlation >0.8 with your proposed feature), either skip yours or articulate what yours captures that the existing one doesn't.

### 3. Add the Feature

```
features(action="add", name="...", formula="...", type="...")
```

Feature types:
- **instance**: Computed per row from that row's columns
- **grouped**: Requires aggregation over a group (e.g., mean by category)
- **interaction**: Product or combination of two+ features
- **ratio**: Normalized comparison
- **indicator**: Binary flag for a condition
- **regime**: Flags different operating modes

### 4. Test in Isolation

One feature per experiment. If you batch-add five features, you learn nothing about which one helped.

```
experiments(action="create",
  description="Test [feature name]",
  hypothesis="[Feature] should improve [metric] because [mechanism]. Expected gain: [range]."
)
experiments(action="run", experiment_id="...")
```

### 5. Interpret the Result

Not just "did the metric go up?"

| Result | What it means | What to do |
|--------|---------------|------------|
| Feature improves metric as predicted | Hypothesis confirmed. Domain reasoning is sound. | Look for related features capturing the same phenomenon from different angles. |
| Feature improves metric but for different reason than predicted | The signal exists but your explanation was wrong. Investigate the actual mechanism. | This often leads to better hypotheses. |
| Feature has no effect | Either redundant with existing features, or hypothesis is wrong. | Check correlation with existing features. If redundant, move on. If not, refine the hypothesis — maybe the functional form is wrong (continuous vs binned, linear vs interaction). |
| Feature hurts metric | It's adding noise that the model can't separate from signal. | The hypothesis may still be correct but the operationalization is wrong. Try a different formula or transformation. |
| Feature helps one model but not others | Signal exists but is model-dependent. | Trees might not need binned versions of continuous features. Linear models can't learn interactions natively. Match feature form to model type. |

### 6. Log the Learning

The experiment conclusion should update your understanding of what drives the target. A feature that doesn't work teaches you something about the data — document what.

---

## Feature Engineering Strategies

### Interactions

When two features are individually weak but mechanically related:

```
features(action="add", name="leverage_x_rates", formula="leverage * rate_change", type="interaction")
```

Auto-search can help discover interactions you haven't thought of:

```
features(action="auto_search", features=[...], search_types=["interactions"])
```

But review auto-discovered interactions through a domain lens. If you can't explain why A*B should matter, it's probably a statistical artifact.

### Temporal Features

For time-dependent data:

```
features(action="auto_search", features=[...], search_types=["lags", "rolling"])
```

Lag features capture momentum and reversion. Rolling features capture trends and volatility. But be careful about leakage — a lag of 0 periods is just the current value, and rolling windows must not include future data.

### Grouped Aggregates

When row-level data needs context:

```
features(action="add", name="category_avg_price", formula="mean(price) group_by category", type="grouped")
```

These capture "how does this row compare to its group?" signals. But they require careful leakage protection — the aggregation must exclude the current row's target in supervised settings.

### Binning

When a continuous feature has a non-linear relationship with the target:

```
features(action="add", name="age_group", formula="bin(age, edges=[18, 35, 50, 65])", type="instance")
```

Use domain knowledge for bin edges, not arbitrary quantiles. "Clinically meaningful age groups" is better than "quartiles."

---

## Per-Model Feature Sets

Different models benefit from different features. This is a key source of ensemble diversity.

- **Linear models**: Need explicit interactions, bins, and polynomial terms because they can't learn non-linear relationships
- **Tree models**: Learn interactions natively but benefit from ratio features that make splits more efficient
- **Neural networks**: Can learn complex relationships but benefit from normalized inputs and meaningful encodings

Consider giving different feature subsets to different models:

```
models(action="update", name="lr_main", features=[...])  # linear-friendly features
models(action="update", name="xgb_main", features=[...])  # tree-friendly features
```

---

## Anti-Patterns

- **Batch-adding features**: Adding 10 features at once makes it impossible to learn what works
- **Features without hypotheses**: "Let me try log(X)" without a reason is noise generation
- **Ignoring negative results**: A feature that doesn't work narrows the search space. Log it.
- **Same features for every model**: This reduces ensemble diversity. Model-specific feature sets are a powerful lever.
- **Over-relying on auto-search**: Automated discovery finds patterns. Domain reasoning finds signals. The distinction matters.
