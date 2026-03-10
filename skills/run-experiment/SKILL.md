# HarnessML: Running Experiments

Use when executing an experiment. Load `experiment-design` first to ensure you've thought through the hypothesis. This skill covers the mechanics and the discipline of execution.

---

## Execution Flow

### 1. Create the Experiment

```
experiments(action="create",
  description="...",
  hypothesis="..."
)
```

The hypothesis is required. It should follow the structure from `experiment-design`: what you're changing, why, what you expect, and what it would mean.

For follow-up experiments within the same strategy, link to the parent:

```
experiments(action="create",
  description="...",
  hypothesis="...",
  parent_id="exp-001",
  branching_reason="Adjusting parameterization based on fold 3 diagnosis"
)
```

### 2. Write the Overlay

```
experiments(action="write_overlay",
  experiment_id="exp-001",
  overlay={"models.xgb_main.params.reg_lambda": 0.5}
)
```

Overlays modify config in isolation. Production config is never touched until you explicitly promote.

### 3. Run

```
experiments(action="run",
  experiment_id="exp-001",
  primary_metric="brier"
)
```

Or for quick iterations within a strategy:

```
experiments(action="quick_run",
  description="...",
  hypothesis="...",
  overlay={...}
)
```

### 4. Understand the Results

**This is the most important step. Do not skip it.**

Load the `diagnosis` skill. Read the results with intention:

- Did the folds you predicted would improve actually improve?
- Did anything unexpected happen?
- Does the pattern of results match your hypothesis, or tell a different story?

```
pipeline(action="diagnostics")
pipeline(action="compare_latest")
```

**Do not move on until you can explain what happened and why.**

### 5. Log What You Learned

```
experiments(action="log_result",
  experiment_id="exp-001",
  conclusion="...",
  verdict="...",
  metrics={...},
  baseline_metrics={...}
)
```

The conclusion is about **what you learned**, not just what the numbers say. A good conclusion:

- Confirms or refutes the hypothesis with evidence
- Explains the mechanism (why it worked or didn't)
- Identifies what to investigate next

A bad conclusion restates the metrics.

Also record the learning in the project notebook:

```
notebook(action="write", type="finding", content="...", experiment_id="exp-001")
```

The experiment journal captures the structured result. The notebook captures the insight — what this means for the bigger picture. If your theory of the target changed, write a `theory` entry too.

### 6. Decide What's Next

Every experiment ends with one of:

- **The hypothesis was confirmed** → What follow-up does this suggest? Is there a related question?
- **The hypothesis was partially confirmed** → What part was right? What part was wrong? What would disambiguate?
- **The hypothesis was refuted** → Why? Is it the parameterization (try again differently) or the strategy (move on)?
- **The results were surprising** → This is the most interesting outcome. Investigate before moving on.

---

## Iterating Within a Strategy

When pursuing a strategy (e.g., "add regularization", "try neural network", "engineer temporal features"):

- Try 3-5 meaningfully different configurations before concluding the strategy doesn't work
- **Meaningfully different**: changing learning_rate from 0.01 to 0.011 is not meaningful. Changing from 0.01 to 0.05 while adjusting n_estimators is.
- Each attempt should be informed by diagnosis of the previous attempt
- Only conclude a strategy is flawed when you can explain the structural reason, not just that individual configs didn't work

The experiment journal should show a trail of attempts with diagnosis between each one, not isolated experiments.

---

## Promoting Changes

When an experiment improves the model AND you understand why:

```
experiments(action="promote", experiment_id="exp-001", primary_metric="brier")
```

Promote when:
- The improvement is consistent across folds (not driven by one fold)
- You can explain the mechanism
- Secondary metrics didn't degrade in unexpected ways

Do not promote just because the primary metric improved. An unexplained improvement is a time bomb.

---

## Verdicts

| Verdict | Meaning |
|---------|---------|
| **keep** | Improvement is understood and consistent. Promote. |
| **partial** | Directional improvement or improvement on a subset. Worth logging for future combination. |
| **revert** | After exhaustive attempts, the strategy is structurally flawed. Document why. |
| **inconclusive** | Ambiguous results. May revisit with more information. |

Verdicts are consequences of understanding, not goals. You don't run experiments to reach a verdict — you run them to learn. The verdict follows naturally from what you learned.
