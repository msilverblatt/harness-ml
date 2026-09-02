# Harness roadmap after v2.0

## North star

Harness exists to make machine-learning improvement loops reproducible,
inspectable, and useful. The long-term opportunity is to apply the same rigor to
LLM and agent workflows, eventually allowing models to propose and evaluate
bounded improvements to their own behavior.

That ambition is directional. It is not the current implementation queue.
Harness must first prove that its existing tabular workflow solves real problems.

## Current rule: evidence before infrastructure

Harness 2 has barely been used outside its own test suite. The immediate phase is
therefore dogfooding, not platform expansion.

New infrastructure requires evidence from an actual project. A proposal must
identify:

1. the concrete project and workflow that exposed the problem;
2. the observed failure, cost, or repeated manual workaround;
3. the smallest change that could address it;
4. how success will be measured;
5. what can be removed or deferred to keep the system small.

Speculative support for distributed execution, generalized scheduling, elaborate
artifact migration, remote locking, signing, broad authorization systems, or
training orchestration should not be added merely because a mature system might one
day need it.

## What v2 already provides

The present product is intentionally focused:

- declarative tabular data preparation;
- leakage-aware cross-validation;
- model training and ensembles;
- transactional experiment versions and rollback;
- persisted predictions, diagnostics, eval reports, and production bundles;
- CLI, MCP, and Studio access to the same workspace;
- release and clean-install CI coverage.

Correctness defects, security defects, data loss risks, and dependency breakage
remain valid maintenance work even before broad usage. Product and operational
abstractions require stronger evidence.

---

# Phase 1 — Prove the core loop

## Goal

Determine whether Harness materially improves real tabular ML work before
expanding its architecture.

## Dogfood projects

Run at least three representative projects:

1. **Small binary classification** — fast iteration, calibration, feature changes,
   and production prediction.
2. **Regression with temporal or grouped structure** — realistic leakage risks,
   interval behavior, and version comparisons.
3. **Messy multiclass project** — source updates, missing data, failed models,
   and nontrivial debugging.

At least one project should be maintained through repeated data refreshes rather
than ending after a single successful run.

## Evidence log

For every meaningful session, record:

- objective and dataset;
- commands or MCP operations used;
- experiment sequence;
- elapsed time and compute cost;
- failures and confusing behavior;
- manual edits or workarounds;
- features that were useful;
- features that were ignored;
- whether the final result was reproducible;
- candidate fixes ranked by frequency and impact.

Evidence belongs in `docs/dogfood/`. It should include failures, not only polished
success stories.

## Allowed work during Phase 1

Changes should normally fit one of these categories:

- fix a correctness, leakage, security, or data-loss defect;
- repair a broken documented workflow;
- remove confusing or unused surface area;
- address friction observed in a dogfood project;
- keep dependencies and supported runtimes healthy;
- improve diagnostics needed to understand a real failure;
- add a regression test for an observed defect.

## Changes requiring an explicit stop-and-review

- a new package;
- a new background service or daemon;
- a generalized scheduler or workflow engine;
- a persistent state machine beyond existing experiment versions;
- a new model category;
- a remote artifact or coordination protocol;
- more than one new public command/tool for a single observed problem;
- infrastructure whose primary justification is anticipated LLM support.

## Phase 1 success criteria

Phase 1 is complete when evidence—not implementation volume—shows that:

- three real projects completed end-to-end;
- repeated experiments were easier to understand or reproduce than the equivalent
  ad hoc workflow;
- at least one project survived a real data refresh and subsequent iteration;
- production export and prediction were actually used;
- failures did not corrupt accepted experiment history;
- the highest-frequency usability problems were fixed or explicitly accepted;
- unused features and abstractions were identified for removal;
- a concise case study explains where Harness helped and where it did not;
- there is a justified, usage-derived list of the next five improvements.

Time spent running projects is part of the phase. These criteria cannot be
satisfied by unit tests or synthetic fault injection alone.

## Phase 1 non-goals

- proving every possible filesystem or deployment environment;
- building enterprise operations machinery;
- supporting arbitrary untrusted model artifacts;
- implementing LLM fine-tuning;
- adding broad HPO, drift, notebook, or cloud-provider integrations;
- maximizing feature count or test count as a proxy for usefulness.

---

# Phase 2 — Evaluate LLM and agent workflows

Phase 2 begins only after Phase 1 produces evidence that Harness's experiment
model is useful and understandable.

The first LLM milestone should be evaluation-only. It should not train models.

A candidate may describe:

- prompts and model parameters;
- available tools;
- retrieval configuration;
- memory or stopping policy;
- an agent workflow graph.

An execution may record:

- messages and responses;
- tool calls and results;
- task output and errors;
- token usage, latency, and cost.

Evaluation should begin with a small combination of deterministic checks,
reference metrics, repeated stochastic trials, and carefully reviewed model or
human judgments. Protected evaluation data must remain outside the candidate's
mutation boundary.

## Later, evidence-dependent possibilities

If eval-only agent projects prove useful, Harness may explore:

1. bounded prompt and workflow mutations;
2. paired candidate-versus-parent evaluation;
3. external fine-tuning adapters such as provider APIs, PEFT/LoRA, TRL, Axolotl,
   or Unsloth;
4. versioned checkpoint and generated-data lineage;
5. guarded self-improvement loops.

Harness should orchestrate established trainers rather than build a distributed
LLM training runtime without a demonstrated need.

## Safety invariant for self-improvement

An optimizing model must not be able to promote itself, rewrite its evaluator,
change promotion thresholds, suppress failed evaluations, or inspect protected
answers. Any future autonomous loop must separate proposer, executor/trainer,
evaluator, policy, and promotion responsibilities.

---

# Decision discipline

At the end of each dogfood project, choose one of four outcomes for every proposed
addition:

- **build now** — repeated, material problem with a small validated solution;
- **experiment** — plausible but uncertain; test outside the core first;
- **defer** — real but not frequent or costly enough;
- **reject/remove** — complexity exceeds demonstrated value.

The roadmap should be revised from accumulated evidence. It should not become a
checklist that pressures the project into implementing hypothetical requirements.
