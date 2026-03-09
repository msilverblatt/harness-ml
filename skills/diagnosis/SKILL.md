# HarnessML: Diagnosis

Use after every experiment run. This is the core data science skill — reading results, understanding errors, and forming the next hypothesis. If you skip this, you're not doing science, you're doing random search.

---

## The Principle

A metric is a symptom. Diagnosis finds the cause. You don't treat symptoms — you understand the disease.

When you see "Brier improved by 0.004," that is not a conclusion. It's the beginning of a question: where did the improvement come from? Which predictions changed? Did calibration improve, or did discrimination improve? Is the gain robust across folds, or driven by one?

---

## Reading the Results

After every run:

```
pipeline(action="diagnostics")
pipeline(action="compare_latest")
```

### Aggregate Metrics

Start here, but don't stop here.

- **Primary metric moved in expected direction?** Good — now verify the mechanism matches your hypothesis.
- **Primary metric didn't move?** The change had no effect, or positive and negative effects canceled out. Check per-fold to see which.
- **Secondary metrics moved unexpectedly?** A Brier improvement with ECE degradation means you improved discrimination but hurt calibration — that's a specific diagnosis, not just "mixed results."

### Per-Fold Breakdown

This is where the real signal is.

- **Which folds improved?** Do they share characteristics? (Smaller sample, specific time period, certain subgroups)
- **Which folds degraded?** Same questions.
- **Is there a pattern?** If the same folds always struggle, that's a systematic model weakness, not random noise.
- **What's the fold variance?** High variance across folds means the model is unstable — it's learning different things from different data slices.

**What fold patterns tell you:**
- Consistently bad fold → something about that fold's data is different (temporal shift, subgroup behavior, sample size)
- Improving the bad fold while degrading good ones → your change helps edge cases but hurts the base case (need more targeted approach)
- All folds improve uniformly → genuine structural improvement
- Random fold-to-fold variation → noise, not signal. The change probably doesn't matter.

### Train-Test Divergence

- **Large train-test gap** → overfitting. The model memorizes training data. Solutions: regularization, simpler model, fewer features, more data per fold.
- **Small gap but poor test performance** → underfitting. The model is too simple. Solutions: more features, more complex model, less regularization.
- **Gap changed after your experiment** → your change affected model complexity. Did it move in the direction you predicted?

### Calibration

```
pipeline(action="diagnostics")  # includes calibration curves
```

- **ECE (Expected Calibration Error)**: Are predicted probabilities reliable? Low ECE means when the model says 70%, the event happens ~70% of the time.
- **Brier vs ECE**: Brier combines discrimination and calibration. You can have good Brier with poor ECE (discriminates well but probabilities are wrong) or good ECE with poor Brier (well-calibrated but doesn't separate classes).
- **Calibration curve shape**: S-shaped means the model is overconfident in both directions. Flat means the model doesn't use the full probability range.

### Feature Importances

- **What is the model relying on?** If the top features aren't what domain knowledge predicts, investigate why.
- **Did importances shift after your change?** If you added a feature and it's not in the top importance list, either it's redundant with existing features or the model doesn't find it useful.
- **Are importances concentrated?** If one feature dominates, the model is fragile. If importances are spread, the model captures diverse signals.

### Meta-Learner Coefficients

- **Positive coefficient**: Model contributes positively to the ensemble.
- **Near-zero coefficient**: Model adds nothing. The ensemble is ignoring it.
- **Negative coefficient**: Model actively hurts the ensemble. Its predictions are anti-correlated with the target *conditional on other models*. This doesn't mean the model is bad individually — it means it's providing signal the ensemble already has, but noisier.

### Model Correlation Matrix

- **High correlation (>0.95)**: Two models make nearly identical predictions. One is redundant.
- **Moderate correlation (0.7-0.9)**: Models agree on easy cases, disagree on hard cases. This is good — it's where ensemble diversity helps.
- **Low correlation (<0.7)**: Models see the problem very differently. High diversity, but check that both are individually competent.

---

## Forming the Next Hypothesis

This is the purpose of diagnosis. Every diagnostic finding should generate a question:

| Finding | Question it generates |
|---------|----------------------|
| Fold 3 always worst | What's different about fold 3's data? Is it a time period, a subgroup, a distribution shift? |
| Train-test gap on tree models only | Are trees overfitting to interactions that don't generalize? Would regularization or max_depth reduction help? |
| Feature X has high importance but low univariate correlation | The model found a non-linear or conditional relationship. What interaction or transformation makes this explicit? |
| Two models have 0.98 correlation | They're learning the same thing. Can I differentiate them with different feature sets? Or should I drop one and add a different model family? |
| ECE degraded while Brier improved | Discrimination improved but calibration suffered. Is the calibration method appropriate? Would a different calibrator help? |
| New feature has zero importance | Is it redundant with an existing feature? Or does the model need a different functional form (e.g., binned instead of continuous)? |
| All folds improved slightly | Genuine structural improvement, but small. Is there more headroom in this direction, or is this the ceiling? |

**Write down the next hypothesis before closing the diagnosis.** If you finish diagnosing and don't know what to try next, you didn't dig deep enough.

---

## Residual Analysis

When available, look at where the model's errors concentrate:

- **Are errors uniform or clustered?** Clustered errors in a specific region of feature space mean the model is missing a pattern in that region.
- **Are there systematic over/under-predictions?** A model that consistently over-predicts for a subgroup needs a feature that captures that subgroup's behavior.
- **Do errors correlate with a feature not in the model?** That feature should probably be added.

---

## The Diagnosis Template

After every experiment, write:

```
## Diagnosis: [Experiment ID]

### What happened
[Metric changes — aggregate and per-fold]

### Why it happened
[Mechanism — not just "the metric went down" but WHY]

### What was confirmed
[Parts of the hypothesis that were supported]

### What was surprising
[Things you didn't predict — these are the most valuable]

### Next hypothesis
[What to investigate based on these findings]
```
