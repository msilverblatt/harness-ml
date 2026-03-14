# HarnessML: Exploratory Data Analysis

Use when you need to build intuition about the data. This isn't a checkbox — it's how you generate your first hypotheses and catch problems before they poison your models.

---

## The Goal of EDA

EDA produces **hypotheses**, not charts. Every observation should lead to a question: "If X is true, then feature Y should be predictive because Z." Those hypotheses feed into `domain-research` and `experiment-design`.

---

## Target Analysis

Start with the thing you're predicting.

```
data(action="inspect")
```

**Ask:**
- What's the distribution? (Class balance for classification, skew for regression)
- Are there temporal patterns? (Does the target rate change over time?)
- Are there segments where the target behaves differently? (Target rate by category, region, time period)
- Is the target noisy or clean? (If noisy, your metric ceiling may be lower than you think)

**What this tells you:**
- Class imbalance → you may need stratified CV, class weights, or a metric that handles imbalance (F1 over accuracy)
- Temporal drift → you need temporal CV (expanding window, not random k-fold), and drift-aware features
- Segment differences → features that capture segment membership may be valuable
- Noise → calibrate your expectations. Don't chase the last 0.001 on a noisy target.

---

## Feature-Target Relationships

```
features(action="discover")
```

**Ask:**
- Which features have the strongest univariate signal?
- Are the strong features surprising or expected?
- Are there features with near-zero signal that you expected to matter? (This is interesting — why don't they matter?)
- Are there features with suspiciously strong signal? (Possible leakage)

**What this tells you:**
- Strong expected features confirm domain knowledge — good, build on this
- Strong unexpected features need investigation — is it leakage? A proxy for something else?
- Weak expected features are the most interesting — either the relationship is non-linear (trees will find it), conditional (need an interaction), or your expectation was wrong (update your mental model)

---

## Feature-Feature Relationships

**Ask:**
- Are there highly correlated feature pairs? (>0.9 correlation means one is redundant)
- Are there feature clusters? (Groups of features measuring the same underlying thing)
- Do categorical features have high cardinality? (May need encoding strategy)

**What this tells you:**
- Correlated features inflate importance scores and can destabilize linear models
- Feature clusters suggest a latent variable — sometimes a single derived feature capturing that latent variable is better than the cluster
- High cardinality categoricals need careful handling — frequency encoding, target encoding (with leakage protection), or grouping rare levels

---

## Missing Data Patterns

**Ask:**
- Is missingness random or patterned?
- Are certain features always missing together? (Suggests a data source issue)
- Is missingness correlated with the target? (If so, missingness itself is a feature)

**What this tells you:**
- Random missingness → impute and move on
- Patterned missingness → the *pattern* is information. A `null_indicator` feature for these columns may be predictive.
- Target-correlated missingness → important signal, but also a leakage risk if the missingness is caused by the outcome

---

## Segments and Subgroups

**Ask:**
- Does the target behave differently across natural segments? (Time periods, categories, geographies)
- Are there subgroups where prediction will be harder? (Small segments, unusual distributions)
- Would separate models or segment-specific features help?

**What this tells you:**
- Segment-dependent target behavior → regime indicators or interaction features
- Hard-to-predict subgroups → these will dominate your error. Understanding them is where the biggest gains come from.

---

## What EDA Is Not

- It is not running `data(action="inspect")` and moving on
- It is not a phase to "complete" before modeling
- It is something you return to throughout the project — every surprising experiment result is an invitation to look at the data again
- The best data scientists spend more time here than anywhere else

---

## Output

By the end of initial EDA, you should have:
1. A written description of the data landscape (target, features, quality issues)
2. A list of 5-10 hypotheses about what might be predictive and why
3. Identified risks (leakage candidates, class imbalance, temporal drift)
4. Questions you can't answer yet that will guide early experiments
