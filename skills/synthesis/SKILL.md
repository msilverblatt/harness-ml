# HarnessML: Synthesis

Use after running several experiments, when you need to step back and connect the learnings. This is also the skill for deciding what to do next — and for recognizing when further experiments aren't adding understanding.

---

## Why Synthesis Matters

Individual experiments produce individual findings. Synthesis produces understanding. After 10 experiments, you should be able to say: "Here is what drives this target, here is where the model struggles, and here is why."

Without synthesis, you have a list of experiments. With synthesis, you have a theory.

---

## The Synthesis Questions

Periodically — every 5-10 experiments, or whenever you feel stuck — stop and answer:

### What do we know?

Summarize the confirmed findings. What features matter and why? What model types work best and why? What relationships has the data revealed?

This is not a list of experiment results. It's a narrative: "The target is primarily driven by X and Y. Feature Z captures X well, but Y is only partially captured — the model struggles when Y interacts with W."

### What have we ruled out?

Equally valuable. What hypotheses were tested and refuted? What strategies were exhausted? This prevents revisiting dead ends and focuses future work.

### Where does the model still struggle?

Look at the diagnostic patterns across experiments:
- Are there consistent weak folds? What do they share?
- Are there subgroups with high error rates?
- Is there a ceiling you keep hitting?

This is where the next breakthrough will come from — not from tuning what works, but from understanding what doesn't.

### What's the current theory?

State your best understanding of what drives the target. This theory should evolve with each experiment. If it hasn't changed in 5 experiments, either you're confirming it (good) or you're not learning (bad — try something different).

### What are the highest-value next experiments?

Based on everything above, what would teach you the most? Not what would improve the metric the most — what would improve your *understanding* the most. Often these are the same thing, but not always.

---

## The Experiment Journal Review

```
experiments(action="journal")
```

Read through the journal with fresh eyes. Look for:

- **Patterns across experiments**: Did all feature-addition experiments help more than all hyperparameter experiments? That tells you where the signal lives.
- **Contradictions**: Did experiment 3 suggest X but experiment 7 suggest the opposite? What changed? Was it a confound?
- **Diminishing returns**: Are recent experiments producing smaller gains than early ones? That's natural, but at some point it means the model has extracted most of what the data offers.
- **Unexplored directions**: What hypotheses from early experiments were noted as "follow-up" but never pursued?

---

## When to Stop

There is no "done." But there is a point of diminishing returns where additional experiments aren't adding meaningful understanding.

**Signals you're approaching diminishing returns:**
- Last 3+ experiments produced <0.001 metric change
- Your theory of what drives the target hasn't been updated in 5+ experiments
- Diagnostic patterns are stable — the same folds are weak, the same subgroups are hard
- New features are highly correlated with existing ones (the novel signal space is exhausted)
- Meta-learner coefficients are stable across experiments

**What to do at this point:**

Don't declare victory. Instead, present the state of knowledge:

1. **Current model performance** with key metrics and per-fold stability
2. **What drives the target** — your best theory, with evidence
3. **Where the model struggles** — honest about remaining weaknesses
4. **What would improve it further** — if there were more data, different data, or more time, what would you investigate?
5. **Confidence assessment** — how robust is this model? Is it overfit to the training period? Would it generalize to new data?

---

## The Synthesis Template

```
## Project Synthesis — [Date or Experiment Count]

### State of Knowledge
[Narrative of what drives the target and how the model captures it]

### Key Findings
[3-5 most important confirmed learnings, with evidence]

### Ruled Out
[Strategies and hypotheses that were tested and refuted]

### Remaining Weaknesses
[Where the model struggles and current best understanding of why]

### Model Composition
[What's in the ensemble, why each model is there, diversity assessment]

### Recommended Next Steps
[Ranked by expected learning value, not expected metric gain]

### Open Questions
[What you'd investigate with more time/data]
```

---

## Connecting to Other Skills

Synthesis naturally triggers other skills:

- "The model struggles with segment X" → `diagnosis` to understand the errors, `domain-research` to generate hypotheses about X's behavior
- "We haven't tried model family Y" → `model-diversity` to evaluate whether it would add value
- "Feature area Z is under-explored" → `feature-engineering` with domain-driven hypotheses
- "Results are contradictory" → `eda` to re-examine the data in light of new understanding
- "I don't know what to try next" → `domain-research` to bring in external knowledge

Synthesis is the hub. Everything flows through understanding.
