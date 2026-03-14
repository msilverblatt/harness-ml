# HarnessML: Project Setup

Use when starting a new ML project or revisiting the scope of an existing one.

---

## Before You Touch Any Data

Answer these questions. Write the answers down (in the experiment journal or project notes). These answers guide every decision that follows.

### What are we predicting?

Not just "the target column" — the real-world outcome. "Whether a patient will be readmitted within 30 days." "The sale price of a house." "Which team wins." The framing determines everything: what features make sense, what metrics matter, what errors are costly.

### Why does this prediction matter?

What decisions does this model inform? A model that ranks loan applicants has different error costs than one that predicts equipment failure. Understanding the use case tells you whether false positives or false negatives are worse, whether calibration matters more than discrimination, whether you need point predictions or uncertainty intervals.

### What does success look like?

Not just a metric threshold — what would make this model *useful*? If the current process is a human guessing with 60% accuracy, 70% might be transformative. If the current model achieves 0.95 AUC, improving to 0.96 might not justify the effort. Anchor expectations before you start.

### What data do we have?

- Sources, freshness, coverage
- What's the grain? One row per what?
- What time period does it cover?
- What's missing that we wish we had?

---

## Setting Up the Project

### 1. Initialize

```
configure(action="init", project_dir="...", task_type="...", target_column="...", primary_metric="...")
```

Choose the task type and primary metric based on your answers above, not by default. If calibration matters, make a calibration metric primary. If ranking matters, use NDCG.

### 2. Ingest Data

```
data(action="ingest", path="...")
```

### 3. First Look

```
data(action="inspect")
data(action="list_features")
```

Don't rush past this. Read the output. Note:
- How many rows and features
- What types (numeric, categorical, text, datetime)
- Missing value patterns
- Target distribution (balanced? skewed? continuous range?)

This initial inspection should generate your first hypotheses about what might matter. Write them down. Then load the `eda` skill to go deeper.

---

## Common Mistakes at Setup

- **Picking a metric by default.** Brier score is not always the right primary metric. Think about what you're optimizing for.
- **Ignoring the grain.** If each row is a patient-visit but you're predicting patient-level outcomes, your CV strategy needs to account for that (group k-fold, not random k-fold).
- **Skipping the "why."** If you don't know what decisions the model informs, you can't evaluate whether it's good enough or where errors matter most.
