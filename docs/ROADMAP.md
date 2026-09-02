# Harness roadmap after v2.0

## North star

Harness is a system for running trustworthy, reproducible improvement loops.
It begins with traditional tabular machine learning and should ultimately support
LLM and agent workflows that can improve their own prompts, policies, datasets,
and model checkpoints without giving up experimental integrity.

The differentiator is not merely automation. Harness should make automated
improvement **bounded, statistically rigorous, auditable, reproducible, and
reversible**.

This roadmap describes direction rather than a promise that every item will ship
in a particular release. Each implementation phase should have its own reviewed
design and delivery plan before code is merged.

## Guiding principles

1. **Correctness before autonomy.** An automated loop may only act on evidence
   produced without leakage or evaluator contamination.
2. **Immutable evidence.** Data, configuration, code, traces, checkpoints,
   evaluations, and promotion decisions must have durable lineage.
3. **Bounded execution.** Time, cost, compute, retries, tools, and mutation scope
   must be constrained by policy.
4. **Separation of powers.** A candidate proposer must not be able to silently
   redefine its evaluator, promotion policy, or protected holdout.
5. **Reversible promotion.** Failed runs cannot corrupt active state, and every
   promoted artifact must have a tested rollback path.
6. **Protocol-driven extensibility.** Harness should integrate established
   trainers, model providers, and agent runtimes instead of rebuilding all of
   them.
7. **Depth before breadth.** Existing workflows should become operationally
   trustworthy before new model categories are advertised.

---

# Step 1 — Operational hardening and proof

The immediate goal is to turn the v2 foundation into a production-proven system
for traditional ML. This work should precede autonomous LLM training.

## 1. Versioned production artifacts

Evolve `model.bundle` from an implicit serialization contract into a documented,
versioned artifact format.

Planned capabilities:

- bundle manifest and artifact schema version;
- Harness, Python, and dependency versions;
- task, input schema, feature schema, and output schema;
- model, configuration, code, and training-data fingerprints;
- checksums for all bundle components;
- compatibility checks with actionable errors before loading;
- migration policy and compatibility fixtures for older bundles;
- explicit trusted-artifact requirements for pickle-based components;
- portable export adapters such as ONNX or `skops` where models support them;
- signing or provenance attestation where deployment requirements justify it.

## 2. Operational reliability

Harden workspaces and experiments for interruption, concurrency, and expensive
execution.

Planned capabilities:

- process-safe workspace locking;
- atomic writes for all mutable pointers and artifacts;
- crash recovery and abandoned-staging cleanup;
- idempotent mutation and MCP operations;
- cancellation, deadlines, and stage-level timeouts;
- bounded retry policies for transient failures;
- CPU, memory, GPU, wall-clock, and monetary budgets;
- deterministic resumption where supported;
- concurrent reader/writer and fault-injection tests;
- explicit run states such as queued, running, cancelling, failed, and complete.

## 3. Statistical and ML correctness

Deepen the leakage and evaluation guarantees already present in v2.

Planned capabilities:

- confidence intervals for metrics and version deltas;
- paired statistical comparisons between candidate and parent versions;
- grouped, temporal, and heteroscedastic conformal methods;
- multiclass calibration diagnostics;
- minimum sample-size and unstable-estimate warnings;
- duplicate, entity, temporal, and transform leakage detection;
- property-based tests across cross-validation strategies;
- explicit train, calibration, development, validation, and final-holdout roles;
- protected final holdouts that optimization actors cannot read directly;
- reproducibility checks across seeds and supported environments.

## 4. Provenance and reproducibility

A run should be able to answer exactly what produced every result.

Planned capabilities:

- formal run manifests;
- content-addressed datasets and artifacts;
- source, transformation, and feature lineage;
- code revision and dirty-tree state;
- environment, hardware, seed, and dependency capture;
- row filtering and exclusion records;
- parent model and checkpoint lineage;
- replay tooling and explicit reproducibility status.

## 5. Security and trust boundaries

Autonomous execution requires explicit boundaries rather than implicit trust.

Planned capabilities:

- secret storage interfaces and comprehensive redaction;
- safe handling rules for untrusted model bundles;
- request, input-size, and resource limits;
- stronger expression and tool-execution sandboxing;
- MCP authentication and authorization integration points;
- read-only, experiment-running, promotion, and administration roles;
- audit events for every state mutation;
- configurable human approval gates;
- artifact retention and deletion policies.

## 6. Observability and cost accounting

Planned capabilities:

- structured logs and stable trace/run identifiers;
- experiment lifecycle events;
- timing and resource use by stage, fold, seed, and model;
- cache hit/miss explanations;
- normalized failure categories;
- OpenTelemetry-compatible traces;
- token, API, accelerator, and monetary cost accounting;
- health, readiness, and operational diagnostic endpoints;
- corresponding operational views in Studio.

## 7. Engineering quality and compatibility

**Status:** In progress. CI now enforces a zero-warning baseline for critical
source correctness, import, modernization, and undefined-name lint rules across
all Python packages. Existing source violations were fixed rather than hidden in
a baseline file. Formatting, stricter rule families, typing, dependency warning
removal, and compatibility matrices remain open.

Planned capabilities:

- enforced formatting, linting, and type-checking baselines;
- removal of current dependency and UTC deprecation warnings;
- documented public APIs and internal APIs;
- package versions aligned with repository releases;
- a supported Python and dependency compatibility matrix;
- dependency update automation;
- performance benchmarks and regression thresholds;
- workspace and artifact upgrade/rollback tests.

## 8. Product depth and dogfooding

Before broadening the product, improve the workflows users already touch.

Planned capabilities:

- richer paired experiment comparison;
- traceable links among metrics, folds, predictions, features, and source rows;
- production-bundle inspection and compatibility reporting;
- calibration, interval, and attribution visualizations;
- cancellation and recovery controls in Studio;
- actionable failure explanations;
- repeated use on several real projects and at least one sustained deployment.

## Step 1 exit criteria

Step 1 is complete only when evidence supports the claim, not when individual
features merely exist.

Required release evidence:

- multiple real projects have completed repeated experiment and promotion cycles;
- at least one sustained internal deployment has exercised production inference;
- interrupted, concurrent, and resource-limited runs recover correctly;
- bundle compatibility is tested across released artifact versions;
- performance and scale baselines are published and enforced in CI where practical;
- no known target, entity, temporal, fold, or final-holdout leakage defect remains;
- every state mutation is auditable and recoverable;
- protected holdouts cannot be inspected by an optimization actor;
- upgrade and rollback procedures have been exercised;
- no agent-accessible execution path has unbounded cost or compute;
- operational limitations are documented from real use, not inferred only from tests.

---

# Step 2 — LLM and agent improvement loops

Step 2 generalizes Harness from tabular predictions to stochastic model and agent
behavior. It should proceed in stages so evaluation becomes trustworthy before
training becomes autonomous.

## 2A. Eval-first LLM and agent workflows

First support evaluation without model training.

A candidate may include:

- system and developer prompts;
- model/provider and sampling parameters;
- tool definitions and permissions;
- retrieval configuration;
- memory policy;
- workflow or agent graph;
- retry, routing, and stopping policies.

An execution should produce structured traces containing:

- messages and model responses;
- tool calls, arguments, results, and failures;
- intermediate workflow transitions;
- retrieved context and citations where permitted;
- token use, latency, and monetary cost;
- terminal output, errors, and policy violations;
- environment and candidate fingerprints.

Evaluation should support:

- deterministic programmatic checks;
- reference-answer and task-specific metrics;
- rubric-based model judges;
- pairwise preference evaluation;
- human review and adjudication;
- task completion and tool-use correctness;
- safety and policy checks;
- latency and cost constraints;
- repeated stochastic trials with uncertainty estimates;
- judge calibration, agreement analysis, and judge-version lineage.

## 2B. Rigorous prompt and workflow optimization

Once agent evaluation is reliable, Harness may optimize bounded candidate
parameters.

The expected loop is:

1. diagnose failures from development traces;
2. propose a typed, bounded mutation;
3. execute repeated trials on development tasks;
4. compare candidate and parent using paired statistics;
5. run invariant, regression, cost, and safety suites;
6. evaluate on a protected validation set;
7. promote only when policy and approval gates pass;
8. preserve complete lineage and a rollback path.

Optimization targets may include prompts, tool descriptions, routing, retrieval,
memory, stopping policies, and workflow graphs. Evaluators and protected
holdouts are outside the mutation boundary.

## 2C. Fine-tuning orchestration

Harness should initially orchestrate established training systems rather than
become a new distributed trainer.

Likely adapters include:

- provider-hosted fine-tuning APIs;
- Hugging Face Trainer;
- PEFT and LoRA;
- TRL;
- Axolotl;
- Unsloth;
- managed GPU training platforms.

Harness should own dataset construction and versioning, contamination checks,
training configuration, budgets, checkpoint lineage, evaluation, promotion, and
rollback. The underlying trainer should remain replaceable.

## 2D. Bounded self-improvement

The aspirational goal is an LLM operating a constrained and audited optimization
loop over its prompts, workflow, training data, and model checkpoints.

This is not unrestricted self-modification. The system must maintain separation
between:

- **proposer** — diagnoses results and suggests typed changes;
- **executor/trainer** — produces candidate behavior or checkpoints;
- **evaluator** — scores candidates using versioned evaluation suites;
- **policy layer** — enforces budgets, permissions, invariants, and safety rules;
- **promoter** — applies statistical and human approval requirements;
- **protected holdout** — remains inaccessible to the optimizing model.

A candidate cannot promote itself, rewrite its promotion criteria, suppress
failed evaluations, or inspect protected answers. All generated data must retain
source-model, prompt, filtering, licensing, and review provenance.

---

# Unifying architecture

Harness should not grow an unrelated LLM platform beside `harness-ml`. The core
abstractions should be generalized so tabular systems and agent systems use the
same experiment discipline.

Candidate concepts:

- **EvaluationDataset** — versioned examples, splits, access policy, and lineage;
- **Candidate** — a model, prompt, workflow, policy, or checkpoint under test;
- **Executor** — runs a candidate against an example under resource policy;
- **Trace** — immutable structured evidence from an execution;
- **Evaluator** — produces versioned scores and findings from traces;
- **Experiment** — a typed mutation from a parent candidate;
- **Artifact** — a content-addressed product with a compatibility manifest;
- **PromotionPolicy** — gates deployment using metrics, uncertainty, safety,
  budgets, and approvals.

Tabular prediction can then be treated as one deterministic or seeded execution
kind, while an agent workflow is a stochastic, trace-producing execution kind.

## Architectural prerequisites for Step 2

Before implementing public LLM features, write and review designs for:

- candidate/executor/evaluator protocols;
- immutable trace schema and OpenTelemetry relationship;
- stochastic repetition and paired-comparison semantics;
- protected dataset access controls;
- evaluator and model-judge versioning;
- tool sandbox and secret boundaries;
- token and monetary budget enforcement;
- checkpoint and generated-data provenance.

---

# Proposed sequence

1. Artifact manifests, compatibility, and trusted loading.
2. Workspace locking, crash recovery, cancellation, and budgets.
3. Protected holdouts and stronger statistical comparisons.
4. Provenance, security, observability, and engineering quality gates.
5. Real-world dogfooding, scale benchmarks, upgrades, and rollback exercises.
6. Candidate/executor/evaluator protocol design.
7. LLM and agent trace capture plus eval-only workflows.
8. Bounded prompt, tool, retrieval, and workflow optimization.
9. External fine-tuning adapters and checkpoint evaluation.
10. Guarded autonomous dataset generation and checkpoint improvement.

## Explicit non-goals for the first Step 2 release

- building a new distributed LLM training runtime;
- allowing arbitrary unreviewed code or tool execution;
- treating a single model judge score as ground truth;
- exposing final-holdout contents to the optimizing model;
- allowing candidates to modify evaluators or promotion policy;
- claiming general autonomous self-improvement from prompt search alone.

## Success measures

The roadmap succeeds if Harness can demonstrate that it:

- finds improvements that replicate on protected data;
- detects regressions that naive aggregate scoring misses;
- produces complete evidence for every promotion decision;
- controls cost and failure blast radius during autonomous runs;
- reproduces or explains results across supported environments;
- rolls back any promoted candidate cleanly;
- improves agent task success without silently degrading safety, cost, or latency.
