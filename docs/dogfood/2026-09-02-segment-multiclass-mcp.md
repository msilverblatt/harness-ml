# Dogfood report: image-segment multiclass through MCP

## Context

- Date: 2026-09-02
- Task: seven-class image-segment classification
- Dataset: OpenML `segment`, version 1; 2,310 rows and 19 numeric predictors
- Clean data fingerprint: `d74f23d61724839bed7b34c7c4ea340c651ba7174afea3d5ce0730ca1e85f7ce`
- Interface: registered MCP tool handlers, matching the agent-facing contract
- Workspace (not committed): `/Users/msilverblatt/Projects/harness-dogfood/segment-multiclass`

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

### Follow-up source-refresh replay

A follow-up session exercised actual MCP stdio transport through `pmcp`, using the
standard Python MCP client rather than calling registered handlers directly. It
replayed a realistic append refresh: the source initially exposed 2,000 rows of the
pinned OpenML dataset and was then replaced by its complete 2,310-row snapshot.
This was a controlled replay performed in one session, not evidence of an
unattended deployment surviving the passage of time.

| Snapshot/action | Version | Log loss | Accuracy | Cache result |
|---|---|---:|---:|---|
| Initial 2,000-row random forest | v001 | 0.10253 | 0.9705 | 5 trained |
| `data_refresh` on 2,310 rows | v002 | 0.09305 | 0.9701 | 5 trained |
| Add histogram boosting on refreshed data | v003 | 0.06357 | 0.9835 | 5 cached, 5 trained |

The initial and refreshed fingerprints were respectively
`0a6b1fc83dc2cb6b1b7bb0419cd0c440c0f7687d2123e10d36ce601fb97445a1`
and `d74f23d61724839bed7b34c7c4ea340c651ba7174afea3d5ce0730ca1e85f7ce`.
The refresh baseline correctly returned no deltas against stale v001 metrics.
The v003 candidate then compared normally against v002 because both used the same
refreshed snapshot.

This walkthrough exposed and drove three focused repairs:

- PR #59 prevents explicit or implicit comparisons across dataset fingerprints;
- PR #60 adds a no-config-change `data_refresh` experiment so the accepted config
  can be re-established on new data before testing mutations;
- PR #62 includes underlying model errors when every model fails. This came from a
  failed transport setup whose Python environment had incompatible NumPy binaries;
  the improved message immediately identified the environment problem.

The standalone `pmcp test` command returned exit status zero even when an MCP tool
result had `isError: true`. That belongs to the ProtoMCP test runner rather than
Harness and was avoided by checking `CallToolResult.is_error` in the persistent
MCP client. It should be investigated in the user-owned ProtoMCP repository before
relying on the command in shell automation.

## Proposed changes

| Proposal | Evidence | Smallest intervention | Success measure | Disposition |
|---|---|---|---|---|
| Support string multiclass labels | One common real dataset failed | First evaluate one more string-label dataset and define production output-label semantics | No external encoding while preserving class mapping in artifacts | Experiment |
| Document compare/conclude/switch as the acceptance workflow | It solved the repeated degraded-current concern without code | Add a short workflow example rather than new state | Users retain the intended version without inventing promotion infrastructure | Build documentation only |
| Test through a real MCP client | Handler workflow worked | Repeat one project over `pmcp`, without adding tools | Same result and JSON artifacts over transport | Completed in refresh replay |
| Make refreshed evaluations comparable without stale deltas | Replay required an honest baseline on the new fingerprint | Add a no-mutation refresh experiment | Candidate compares only with a same-snapshot parent | Built: PRs #59–#60 |
| Preserve all-model failure causes | Broken transport environment initially emitted only an aggregate error | Include deduplicated underlying failures | Binary incompatibility is visible in the MCP error | Built: PR #62 |

## Overall assessment

This was the strongest session for Harness's intended agent-first interaction. The
existing MCP result shape and explicit version tools were more useful than the raw
Python API and made an additional promotion subsystem unnecessary. The only
material model-workflow issue was string-label handling, which needs one more real
example before implementation.
