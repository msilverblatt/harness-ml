# HarnessML: Experiment Design

Use before creating any experiment. This is the thinking step. If you skip it, you'll run experiments that don't teach you anything.

---

## The Test

Before you create an experiment, answer these five questions. If you can't answer all five, you're not ready.

### 1. What question am I trying to answer?

Not "try lower learning rate." A question:

- "Does the model overfit on low-sample folds because of insufficient regularization?"
- "Can the model learn the non-linear relationship between age and risk if I bin age into clinically meaningful groups?"
- "Is the poor calibration caused by the spline method or by genuinely miscalibrated base models?"

The question determines what you look at in the results. Without it, you'll just check whether the aggregate metric went up or down — and learn nothing.

### 2. What's the single variable?

One change per experiment. If you change learning_rate AND add features AND switch calibration, you can't attribute the result to anything.

**Exception:** Mechanically coupled changes. Lowering learning_rate requires raising n_estimators to compensate — that's one logical change, not two. Document the coupling in the hypothesis.

### 3. What would I expect to see if the hypothesis is correct?

Be specific. Not "the metric improves." Think about:

- **Which folds** should improve? (If you're addressing overfitting on high-variance folds, those specific folds should improve)
- **Which metrics** should move? (Regularization should improve calibration and reduce train-test gap, not just headline metric)
- **By how much?** (A rough range, not a precise number — but "should improve by 0.003-0.01" is very different from "should improve by 0.05+")

This is what makes the result interpretable. If you predicted fold 3 would improve and it did, that's confirmation. If folds you didn't expect to change moved instead, that's a different signal entirely.

### 4. What would I expect to see if the hypothesis is wrong?

Equally important. If the metric gets worse:
- Does that mean the hypothesis is wrong, or that the parameterization is wrong?
- What pattern of failure would tell you to try a different parameterization vs. abandon the strategy?

### 5. What will I do with the result?

- **If it works as expected:** What does that confirm about how the model works? What follow-up experiment does that suggest?
- **If it fails:** What's the next parameterization to try? At what point would you conclude the strategy itself is flawed?
- **If the results are ambiguous:** What additional information would disambiguate?

---

## The Hypothesis

Write it in this structure:

> **Changing** [single variable] **because** [mechanism/reasoning] **expecting** [specific predicted outcome with magnitude] **which would mean** [what it tells us about the model/data].

Example:

> **Changing** reg_lambda from 0 to 0.5 on xgb_main **because** folds 2 and 6 show >0.02 train-test Brier gap, suggesting the model memorizes low-sample folds. **Expecting** folds 2 and 6 to improve by 0.005-0.01, other folds neutral, overall Brier improves 0.003-0.005. **Which would mean** the ensemble is currently overfitting on edge cases and uniform regularization is sufficient to address it.

---

## Iteration Planning

Before your first attempt, sketch a rough plan for the strategy:

- **Attempt 1:** Your best guess at parameterization
- **If it's too aggressive:** What you'd dial back
- **If it's too conservative:** What you'd increase
- **If the direction is right but magnitude is wrong:** How you'd adjust
- **If it fundamentally doesn't work:** What alternative approach you'd try within the same strategic question

This prevents the most common failure: trying one config, seeing it fail, and abandoning the entire strategy. You committed to investigating the question, not to one parameterization of it.

---

## When NOT to Experiment

Sometimes the right move is not to run an experiment:

- **You don't have a hypothesis.** If you're just "trying things," stop and think. Load `domain-research` or `diagnosis` to generate a real hypothesis.
- **You can't interpret the result.** If you don't know what a positive or negative result would mean, the experiment can't teach you anything.
- **The question is already answered.** Check the experiment journal. Has a previous experiment already addressed this question?
- **The expected gain is negligible.** If your best case is 0.001 improvement and you can't articulate a learning value, spend time on something higher-leverage.
