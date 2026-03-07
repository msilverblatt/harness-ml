---
name: harness-run-experiment
description: |
  Enforces disciplined experiment execution in HarnessML projects.
  Use when running any ML experiment — hypothesis required before starting,
  conclusion required after finishing, and exhaustive iteration required
  before abandoning any strategy.
---

# HarnessML: Disciplined Experiment Execution

**System Prompt for AI Agents:**

You are running ML experiments in HarnessML. Every experiment follows a strict protocol: hypothesis before action, diagnosis between attempts, conclusion after completion. You never abandon a strategy after a single attempt. You exhaust it methodically, diagnose failures, and only revert when the entire strategy is provably flawed.

---

## The Non-Negotiable Rules

### Rule 1: Required Hypothesis

Every experiment starts with a hypothesis. No exceptions.

A hypothesis is NOT "try lower learning rate." A hypothesis IS:

> "Reducing learning_rate from 0.1 to 0.01 should reduce overfitting on folds 3 and 7, where train-test metric divergence is largest. I expect Brier to improve by 0.005-0.01 overall, with the biggest gains on those high-variance folds."

The hypothesis must contain:
- **What** you are changing (the single variable)
- **Why** you expect it to help (the mechanism)
- **How much** improvement you expect (quantified success criteria)

### Rule 2: Required Conclusion

Every experiment ends with a conclusion. No exceptions.

A conclusion is NOT "it got worse." A conclusion IS:

> "Reducing learning_rate to 0.01 improved fold 3 by 0.008 but degraded folds 1, 4, 5 by 0.003-0.006 each. The overfitting hypothesis was partially correct — high-variance folds improved — but the model became underpowered on well-behaved folds. Next attempt: try 0.03 as a middle ground, or increase n_estimators to compensate."

The conclusion must contain:
- **What happened** (per-fold or per-metric breakdown, not just aggregate)
- **Why it happened** (diagnosis, not just observation)
- **What it means** for the next attempt

### Rule 3: Exhaust the Strategy

**This is the most important rule.** Do not try one configuration, see it fail, and revert.

When pursuing a strategy (e.g., "add regularization," "try neural network," "engineer interaction features"):

1. Try 3-5 meaningfully different configurations minimum
2. Diagnose WHY each attempt failed before the next one
3. Let each diagnosis inform the next attempt
4. Only revert when you can articulate why the ENTIRE STRATEGY is flawed, not just one parameterization

**What "meaningfully different" means:**
- Changing learning_rate from 0.01 to 0.011 is NOT meaningfully different
- Changing learning_rate from 0.01 to 0.05 while also adjusting n_estimators to compensate IS meaningfully different
- Adding one feature is NOT exhaustive; adding 5 related features in different combinations IS

**The experiment log should show a TRAIL OF ATTEMPTS:**
```
Experiment: Add regularization to XGBoost
  Attempt 1: reg_alpha=1.0 -> Brier +0.003 (worse). Diagnosis: too aggressive, underfitting on small folds.
  Attempt 2: reg_alpha=0.1 -> Brier -0.001 (marginal). Diagnosis: right direction, fold 5 still degraded.
  Attempt 3: reg_alpha=0.1 + reg_lambda=0.5 -> Brier -0.004 (improvement). Diagnosis: L1+L2 combo works.
  Attempt 4: reg_alpha=0.05 + reg_lambda=1.0 -> Brier -0.002 (less improvement). Diagnosis: L2 dominant helps less.
  Verdict: KEEP attempt 3 config. L1+L2 regularization with moderate alpha is the sweet spot.
```

NOT this:
```
Experiment: Add regularization to XGBoost
  Attempt 1: reg_alpha=1.0 -> Brier +0.003 (worse).
  Verdict: REVERT. Regularization doesn't help.
```

### Rule 4: Single-Variable Discipline

Change ONE thing per experiment. If you change learning_rate AND add features AND switch calibration, you cannot attribute the result to any single change.

Exceptions: when two changes are mechanically coupled (e.g., lowering learning_rate requires raising n_estimators to compensate). Document the coupling in the hypothesis.

### Rule 5: Diagnose Between Attempts

After each attempt that does not meet success criteria, analyze WHY before the next attempt:

- **Per-fold breakdown**: Which folds improved? Which degraded? Why?
- **Feature importance shifts**: Did the change alter which features the model relies on?
- **Calibration**: Did predicted probabilities shift? Check ECE, reliability curves.
- **Train-test divergence**: Is the model overfitting or underfitting?

Use `pipeline(action="diagnostics")` and `experiments(action="compare")` for this analysis. Each diagnosis should directly inform the next attempt's configuration.

### Rule 6: Log Everything

Every experiment result goes in EXPERIMENT_LOG.md regardless of outcome. Failed experiments are as valuable as successful ones — they narrow the search space.

### Rule 7: Compare to Baseline

Every result is compared to the current baseline. Use `experiments(action="run")` which automatically compares against production config. Report absolute and relative metric changes.

---

## Step-by-Step Workflow

### Step 1: Define the Experiment

Before touching any tool, write down:

- **Variable being changed**: The single thing you are modifying
- **Hypothesis**: What you expect and why (see Rule 1)
- **Success criteria**: Quantified threshold (e.g., "Brier improves by at least 0.003")
- **Iteration plan**: What you will try if the first config does not work

### Step 2: Create the Experiment

```
experiments(action="create",
  description="Test L2 regularization on gradient boosted model",
  hypothesis="Adding reg_lambda=0.5 should reduce overfitting on folds 2,6 where train-test gap is >0.02. Expect overall Brier improvement of 0.003-0.008."
)
```

The `hypothesis` parameter is REQUIRED. The experiment will not be created without it.

### Step 3: Write the Overlay

```
experiments(action="write_overlay",
  experiment_id="exp-001",
  overlay={"models.xgb_main.params.reg_lambda": 0.5}
)
```

Overlays modify config in isolation. Production config is never touched.

### Step 4: Run the Experiment

```
experiments(action="run",
  experiment_id="exp-001",
  primary_metric="brier"
)
```

This runs the backtest with the overlay applied and compares to baseline automatically.

### Step 5: Analyze Results

Do NOT skip this step. Check:

1. **Aggregate metrics**: Did the primary metric improve?
2. **Per-fold deltas**: Which folds moved and by how much?
3. **Secondary metrics**: Did accuracy, ECE, or log_loss change in unexpected ways?
4. **Diagnostics**: `pipeline(action="diagnostics")` for calibration and feature importance

If the result does not meet success criteria, diagnose WHY (Rule 5) and return to Step 3 with a new overlay. Repeat for 3-5 attempts minimum (Rule 3).

### Step 6: Decide — After Exhaustive Iteration

Only after multiple attempts with diagnosis between each one:

- **KEEP**: Primary metric improves consistently. Promote to production.
  ```
  experiments(action="promote", experiment_id="exp-001", primary_metric="brier")
  ```

- **PARTIAL**: Directional improvement but below threshold. Keep the experiment logged — it may combine with future changes.

- **REVERT**: After 3+ meaningfully different attempts, the entire strategy is fundamentally flawed. Document WHY the strategy fails, not just that individual configs failed.

### Step 7: Log with Conclusion

```
experiments(action="log_result",
  experiment_id="exp-001",
  conclusion="L2 regularization at reg_lambda=0.5 improved Brier by 0.004. The overfitting hypothesis was confirmed — folds 2 and 6 improved by 0.008 and 0.011 respectively. Remaining folds were neutral. The mechanism is reduced leaf weight variance in low-sample folds.",
  verdict="KEEP"
)
```

The `conclusion` must explain what was LEARNED, not just restate the numbers. See Rule 2.

---

## Verdict Definitions

| Verdict | Meaning | Action |
|---------|---------|--------|
| **KEEP** | Primary metric improves. Gains are consistent across folds. | Promote to production config. |
| **PARTIAL** | Directional improvement, or improvement on subset of metrics/folds. Below success threshold. | Log for potential combination with future experiments. Do not promote yet. |
| **REVERT** | After exhaustive attempts (3+ meaningfully different configs), the strategy is fundamentally flawed. | Document why the entire approach fails. Revert all changes. Move on. |

---

## Exhausting a Strategy: Detailed Guidance

When a strategy shows initial promise or ambiguous results, the following pattern applies:

1. **First attempt fails**: Diagnose. Was it too aggressive? Too conservative? Wrong axis entirely?
2. **Second attempt**: Adjust based on diagnosis. Change the magnitude, not the strategy.
3. **Third attempt**: If still failing, try a fundamentally different parameterization within the same strategy.
4. **Fourth/fifth attempts**: If partial improvements appeared, try combining the best elements.
5. **Only then**: If all attempts fail AND you can explain the structural reason, declare REVERT.

**Questions to ask before reverting:**
- Did I try at least 3 meaningfully different configurations?
- Can I explain WHY the strategy fails structurally (not just "the numbers went up")?
- Did I check per-fold breakdowns to see if the strategy helps some subsets?
- Could this strategy work in combination with something else (PARTIAL instead of REVERT)?

---

## Quick Reference: MCP Tool Calls for Experiments

| Action | Tool Call |
|--------|-----------|
| Create experiment | `experiments(action="create", description="...", hypothesis="...")` |
| Write config overlay | `experiments(action="write_overlay", experiment_id="...", overlay={...})` |
| Run with baseline comparison | `experiments(action="run", experiment_id="...", primary_metric="...")` |
| One-shot experiment | `experiments(action="quick_run", description="...", hypothesis="...", overlay={...})` |
| Compare two experiments | `experiments(action="compare", experiment_ids=["exp-001", "exp-002"])` |
| Bayesian search | `experiments(action="explore", search_space={...})` |
| Promote to production | `experiments(action="promote", experiment_id="...", primary_metric="...")` |
| Log result with conclusion | `experiments(action="log_result", experiment_id="...", conclusion="...", verdict="...")` |
| Run diagnostics | `pipeline(action="diagnostics")` |

---

## Anti-Patterns (Never Do These)

**Single-shot revert**: Trying one config, seeing it fail, and declaring the strategy dead. You must exhaust 3+ attempts with diagnosis between each.

**Aggregate-only analysis**: Looking only at overall metric change without checking per-fold breakdowns. A strategy that helps 7 folds and hurts 3 is different from one that hurts everything uniformly.

**Hypothesis-free experiments**: Running experiments without stating what you expect. This turns experimentation into random search.

**Conclusion-free logging**: Recording results without explaining what was learned. Future experiments need the WHY, not just the WHAT.

**Multi-variable changes**: Changing multiple things at once and attributing results to one of them. Each experiment isolates one variable.

**Skipping diagnostics**: Moving from one attempt to the next without understanding why the previous one failed. Diagnosis drives the next attempt's design.

**Modifying production config directly**: Always use experiment overlays. The overlay system exists to keep production config safe and experiment history clean.
