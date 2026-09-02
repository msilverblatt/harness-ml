# Dogfood report: image-segment multiclass through MCP

## Context

- Date: 2026-09-02
- Task: seven-class image-segment classification
- Dataset: OpenML `segment`, version 1; 2,310 rows and 19 numeric predictors
- Clean data fingerprint: `d74f23d61724839bed7b34c7c4ea340c651ba7174afea3d5ce0730ca1e85f7ce`
- Interface: registered MCP tool handlers, matching the agent-facing contract

## Session record

| Action | Version | Duration | Log loss | Accuracy |
|---|---|---:|---:|---:|
| Ingest and inspect original string-label data | discarded | — | — | — |
| Attempt default multiclass baseline | none | — | — | — |
| Encode labels and rerun pipeline | — | 0.01 s | — | — |
| Random-forest baseline | v001 | 1.79 s | 0.09305 | 0.9701 |
| Add histogram gradient boosting | v002 | 4.47 s | 0.06357 | 0.9835 |
| Compare, conclude, and retain v002 | v002 | — | — | — |

## What helped

- The MCP tools covered initialization, ingestion, experiments, comparison,
  conclusion, and version selection without direct workspace-file manipulation.
- `experiment.propose` returned parent metrics and deltas directly. This resolved
  much of the comparison friction seen when using `WorkspaceManager` directly.
- The prediction cache reused all five random-forest fold outputs in v002 while
  training only the added model.
- The invalid string target failed before a version was created, preserving clean
  history.
- The explicit compare/conclude/switch workflow was sufficient when used; no new
  automatic promotion state machine was necessary.

## Failures and confusion

| Problem | Frequency | Impact | Evidence |
|---|---:|---:|---|
| Ordinary string class labels are rejected | First baseline attempt | Medium | `Invalid target: Target contains non-numeric values` |
| Label encoding had to happen outside Harness | Once | Medium | Deterministic seven-label mapping in session script |
| Direct registered handlers are less representative than an actual MCP client session | Entire run | Low | Transport was not the source of friction under test |
| The original OpenML fetch warns when version is omitted | Once | Low | Version 1 was then pinned explicitly |

## Manual workarounds

- Converted sorted string labels to integers before ingestion.
- Reinitialized the workspace after changing the target representation.
- Selected the better child using explicit comparison logic, then called
  `experiment.conclude` and `versions.switch`.

## Unused or distracting surface area

- Studio, SHAP, conformal intervals, feature discovery, and most mutation types
  were unnecessary.
- No additional locking, artifact manifests, schedulers, budgets, or observability
  infrastructure would have improved this session.

## Reproducibility and refresh result

- OpenML dataset name and version, label mapping, model parameters, metrics, and
  data fingerprint are recorded.
- The MCP workflow was repeatable after deterministic target conversion.
- A later source refresh was not performed.

## Proposed changes

| Proposal | Evidence | Smallest intervention | Success measure | Disposition |
|---|---|---|---|---|
| Support string multiclass labels | One common real dataset failed | First evaluate one more string-label dataset and define production output-label semantics | No external encoding while preserving class mapping in artifacts | Experiment |
| Document compare/conclude/switch as the acceptance workflow | It solved the repeated degraded-current concern without code | Add a short workflow example rather than new state | Users retain the intended version without inventing promotion infrastructure | Build documentation only |
| Test through a real MCP client | Handler workflow worked | Repeat one project over `pmcp`, without adding tools | Same result and JSON artifacts over transport | Next dogfood session |

## Overall assessment

This was the strongest session for Harness's intended agent-first interaction. The
existing MCP result shape and explicit version tools were more useful than the raw
Python API and made an additional promotion subsystem unnecessary. The only
material model-workflow issue was string-label handling, which needs one more real
example before implementation.
