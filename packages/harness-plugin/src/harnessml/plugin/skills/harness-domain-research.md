# Skill: Hypothesis-Driven Domain Research

## Purpose

Guide structured domain research before and during ML experimentation. The best ML results come from deep domain understanding, not random parameter sweeps. In practice, the biggest gains come from domain-driven hypotheses — combining individually weak signals that capture real phenomena — rather than blind feature generation or hyperparameter tuning.

This skill applies to any ML domain: healthcare, finance, sports, retail, manufacturing, climate, etc.

---

## Step 1: Domain Literature Review

Before touching any data, invest time understanding the domain.

### Actions

- Use web search and Wikipedia to study the problem domain
- Identify known predictive factors from academic literature, industry reports, and practitioner knowledge
- List domain-specific phenomena worth modeling
- For each phenomenon, document:
  - **What it is**: A clear description of the phenomenon
  - **Why it might be predictive**: The causal or correlational reasoning
  - **What data would capture it**: Raw columns, external datasets, or derived signals

### Output

A ranked list of domain hypotheses with expected signal strength (Strong / Medium / Weak).

### Examples Across Domains

| Domain | Phenomenon | Why Predictive | Data Needed |
|--------|-----------|----------------|-------------|
| Healthcare | Medication non-adherence | Missed doses lead to readmission | Prescription fill gaps, appointment no-shows |
| Finance | Earnings surprise momentum | Market underreacts to consecutive surprises | Quarterly EPS vs consensus, trailing surprise direction |
| Retail | Basket complementarity | Co-purchased items predict return likelihood | Transaction-level item pairs, category adjacency |
| Manufacturing | Tool wear degradation curve | Non-linear wear predicts part failure timing | Vibration sensor readings, cumulative usage hours |
| Sports | Rookie coach + luck signal | New coaches with unsustainable win rates regress | Coach tenure, pythagorean win differential |
| Climate | Soil moisture memory | Prior season moisture affects drought prediction | Lagged soil moisture, precipitation anomaly |

---

## Step 2: Map Hypotheses to Features

Translate each domain hypothesis into concrete, implementable features.

### For Each Hypothesis, Identify

1. **Raw data columns needed** — what must exist in the dataset or be sourced externally
2. **Feature transformation** — the formula or logic that captures the phenomenon
3. **Feature type**:
   - **Instance**: computed per row from that row's columns (e.g., `age > 65`)
   - **Interaction**: product or combination of two+ features (e.g., `dosage * weight`)
   - **Ratio**: normalized comparison (e.g., `revenue / employees`)
   - **Indicator**: binary flag for a condition (e.g., `has_prior_claim`)
   - **Grouped aggregate**: requires group-level computation (e.g., `mean(price) by category`)

### Check for Redundancy

Before adding features, inspect what already exists:

```
features(action="discover")
```

If an existing feature already captures the same signal, skip it or refine the hypothesis to capture something the existing feature misses.

### Prioritize By

1. **Expected signal strength** — how strongly domain reasoning suggests predictive power
2. **Novelty vs existing features** — does this capture something the model cannot already learn?
3. **Feasibility** — is the required data available and reliable?

### Example Mapping

**Hypothesis**: In credit risk, borrowers who max out multiple cards simultaneously are higher risk than those who max out one card gradually.

| Element | Detail |
|---------|--------|
| Raw columns | `credit_limit`, `current_balance`, `account_id`, `statement_date` |
| Feature | `count of accounts with utilization > 0.9 in same month` |
| Type | Grouped aggregate (group by borrower + month) |
| Formula | `"sum(utilization > 0.9) group_by borrower_id, statement_month"` |
| Novelty | Existing `avg_utilization` misses the simultaneous multi-account pattern |

---

## Step 3: Implement and Test Features (Single-Variable)

Add and evaluate features one at a time to isolate each hypothesis's contribution.

### Process

1. **Add the feature**:
   ```
   features(action="add", name="simultaneous_maxout_count", formula="sum(utilization > 0.9) group_by borrower_id, statement_month")
   ```

2. **Test correlation with target** before adding to any model — a feature with zero univariate signal may still matter in interactions, but start by checking direct relationships.

3. **Run a single-variable experiment** using the `harness-run-experiment` skill for proper experiment discipline.

4. **Document the full cycle**: hypothesis, feature definition, correlation, experiment result, learning.

### Key Discipline

- Do NOT batch-add ten features and run one experiment. You will not know which feature helped or hurt.
- A feature with weak univariate correlation may still be valuable in combination (Step 4), but document the weak signal now.
- A feature with strong correlation but no experiment lift may be redundant with existing features — that is useful information.

---

## Step 4: Signal Combination

After testing individual hypotheses, look for meaningful combinations driven by domain reasoning.

### Domain-Driven Interactions

The most valuable interactions come from domain understanding, not blind search:

- **"Feature A alone is weak, Feature B alone is weak, but A*B captures [phenomenon]"**
  - Example (healthcare): `is_diabetic` alone is a weak predictor of surgical complications. `bmi > 35` alone is weak. But `is_diabetic * high_bmi` captures a specific high-risk population.
  - Example (finance): `short_interest_ratio` alone is noisy. `earnings_surprise` alone is moderate. But `high_short_interest * positive_surprise` captures short squeeze dynamics.

### Conditional Effects

Look for features that only matter in certain contexts:

- Example (retail): `days_since_last_purchase` predicts churn for subscription products but is meaningless for one-time purchases. Create `days_since_last * is_subscription`.
- Example (manufacturing): `ambient_temperature` only affects failure rate above a threshold. Create `temp_above_critical * usage_intensity`.

### Complement with Auto-Search

Use automated feature search to find combinations you might miss, but treat it as a supplement:

```
features(action="auto_search", features=[...], search_types=["interactions", "ratios"])
```

Review auto-discovered features through a domain lens — discard those without a plausible mechanism, keep those that suggest a real phenomenon you had not considered.

---

## Step 5: Iterate

Results from experiments should generate NEW hypotheses.

### Signals that Results Send

| Result | What It Means | Next Action |
|--------|--------------|-------------|
| Feature works as expected | Domain reasoning is sound | Try related features capturing the same phenomenon from different angles |
| Feature works but in opposite direction | Confounding or counter-intuitive mechanism | Investigate why — this often reveals the most interesting insights |
| Feature has zero signal | Model already captures this, or hypothesis is wrong | Check correlation with existing features; if redundant, move on; if not, refine the hypothesis |
| Feature helps one model but not others | Signal exists but is model-dependent | Consider whether the feature's functional form matches the model (e.g., tree vs linear) |
| Interaction works but components do not | Genuine conditional effect | Look for other conditional effects in the same family |

### Update Domain Understanding

Maintain a running model of what drives the target variable. Each experiment either confirms or challenges that model. The goal is not just better features but a deeper understanding of the domain that compounds across experiments.

---

## Types of Domain Signals

When brainstorming hypotheses, consider these categories:

### 1. Direct Predictors

Features that directly measure the outcome driver.

- Hospital readmission: `count_of_comorbidities`
- House price: `square_footage`
- Customer churn: `support_tickets_last_30d`

### 2. Proxy Signals

Features that indirectly indicate something predictive when the direct measure is unavailable.

- Financial distress proxy: `days_payable_outstanding` (when cash flow data is missing)
- Health risk proxy: `pharmacy_visit_frequency` (when medical records are incomplete)
- Product quality proxy: `return_rate_by_sku` (when defect data is not tracked)

### 3. Interaction Effects

Two features that are weak alone but strong together.

- `high_leverage * rising_rates` — leverage is fine until rates move
- `new_employee * complex_task` — experience only matters for hard work
- `rural_location * low_income` — geographic isolation compounds economic disadvantage

### 4. Conditional Effects

A feature that only matters in certain contexts.

- `marketing_spend` only predicts sales for products with existing brand awareness
- `study_hours` only predicts exam scores above a minimum aptitude threshold
- `rainfall` only affects crop yield during the growing season

### 5. Regime Indicators

Features that signal different operating modes where relationships change.

- `vix_above_30` — volatility regime where correlations break down
- `pandemic_flag` — demand patterns shift fundamentally
- `product_lifecycle_stage` — growth vs maturity vs decline dynamics differ
- `is_holiday_season` — consumer behavior follows different rules

### 6. Contrarian Signals

Features whose predictive direction is counter-intuitive, often the most valuable.

- Higher employee satisfaction surveys sometimes predict imminent turnover (dissatisfied employees do not bother responding)
- More safety incidents can predict fewer fatalities (reporting culture means problems get caught early)
- Lower marketing spend can predict higher sales (companies cut spend when organic demand is strong)

---

## Research Log Template

Use this template to maintain a structured record of domain research and hypothesis testing. Keep this log in your project's experiment directory or notes.

```
## Domain Research Log

### Hypothesis 1: [Name]
- **Domain reasoning**: [Why this should be predictive — the mechanism, not just correlation]
- **Literature/source**: [Where you found evidence for this hypothesis]
- **Feature(s)**: [Name and formula for how to operationalize]
- **Feature type**: [Instance / Interaction / Ratio / Indicator / Grouped aggregate]
- **Expected signal**: [Strong / Medium / Weak, with reasoning]
- **Data requirements**: [What raw columns are needed, availability status]
- **Result**: [Correlation with target, experiment outcome, lift over baseline]
- **Learning**: [What this tells us about the domain — not just "it worked" or "it didn't"]
- **Follow-up**: [Next hypothesis generated by this result]

### Hypothesis 2: [Name]
...
```

### Example Filled Entry

```
### Hypothesis 3: Ticket sharing indicates family groups
- **Domain reasoning**: Passengers sharing a ticket number are likely traveling together
  as a family. Family groups may have different survival patterns than solo travelers
  because they stay together during evacuation rather than acting individually.
- **Literature/source**: Maritime disaster research on group decision-making under stress
- **Feature(s)**: ticket_freq = "count() group_by ticket"; is_shared_ticket = "ticket_freq > 1"
- **Feature type**: Grouped aggregate + indicator
- **Expected signal**: Medium — captures group dynamics but overlaps with family_size
- **Data requirements**: ticket column (available), passenger_id for grouping (available)
- **Result**: ticket_freq correlation with target: 0.094. Experiment showed +0.003 AUC
  over baseline. Modest but additive with family_size (correlation between them: 0.72).
- **Learning**: Group membership signal exists but is largely captured by SibSp/Parch.
  The residual signal may come from non-family groups (friends, servants) traveling together.
- **Follow-up**: Try ticket_freq for passengers with family_size=0 — captures non-family
  group membership that SibSp/Parch misses entirely.
```

---

## When to Use This Skill

- **At project start**: Before writing any features, spend time in Steps 1-2
- **After initial baseline**: Use experiment results to drive Steps 3-5
- **When progress stalls**: Return to Step 1 with new questions informed by what the model already captures
- **When adding a new data source**: Map the new data through Steps 1-2 before feature engineering

## Anti-Patterns to Avoid

- **Blind feature generation**: Adding features without a hypothesis for why they should work
- **Skipping single-variable testing**: Batch-adding features makes it impossible to learn what works
- **Ignoring negative results**: A feature that does not work is information about the domain
- **Over-relying on auto-search**: Automated interaction search finds statistical artifacts; domain reasoning finds real signals
- **Hypothesis-free iteration**: Running experiments without updating your domain model wastes cycles
