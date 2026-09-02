# Dogfood report: Ames temporal regression

## Context

- Date: 2026-09-02
- Task: regression with expanding-window evaluation
- Dataset: OpenML `house_prices` / Ames Housing, 1,460 rows
- Selected predictors: overall quality, living area, garage capacity, basement
  area, bathrooms, and build year
- Fold metadata: build-year decade (`era`), excluded from model inputs
- Clean data fingerprint: `1033915b9393cd8a7ea31f9779ddc2d6df229e6449dc930ff6eceeac5395d891`

## Session record

| Action | Version | Duration | RMSE | MAE | R² |
|---|---|---:|---:|---:|---:|
| Ingest selected OpenML columns | — | 0.05 s | — | — | — |
| Temporal random-forest baseline | v001 | 2.59 s | 40,604 | 24,054 | 0.7410 |
| Add histogram gradient boosting | v002 | 1.72 s | 41,352 | 24,439 | 0.7314 |
| Add 90% conformal intervals | v003 | 0.74 s | 41,352 | 24,439 | 0.7314 |
| Score 20 target-free rows | v003 | — | — | — | — |

Production scoring emitted `prediction`, `lower`, and `upper` columns as expected.

## What helped

- Expanding-window CV used the era column for splitting while excluding it from
  model features.
- The second model's degradation was clear from all three metrics.
- Conformal interval metadata survived full-data fitting and appeared directly in
  CLI prediction output.
- Missing numeric values were handled within model folds rather than requiring a
  global median-fill transform.

## Observed correctness defect and repair

The first attempt used `WorkspaceManager.init(..., task_type="regression")` with
no manual metric configuration. It inherited binary defaults (`brier`,
`accuracy`), silently published `v001`, and wrote `{}` as its metrics.

This was a broken documented workflow rather than speculative product demand. It
was fixed immediately in PR #57:

- `ProjectConfig` now chooses default metrics by task;
- backtests reject metrics unsupported by the selected task instead of silently
  returning an empty result;
- regression and multiclass defaults have regression tests.

The successful session was rerun from a clean workspace after applying the manual
equivalent of that repair.

## Failures and confusion

| Problem | Frequency | Impact | Evidence |
|---|---:|---:|---|
| Default regression metrics were binary and silently produced no metrics | First run | High | Initial discarded v001; fixed by PR #57 |
| Three early temporal holdouts contained fewer than two rows, making R² undefined | 3 folds | Medium | `UndefinedMetricWarning` during each experiment |
| A global pipeline median-fill would use future/holdout data | Considered once | High if used | Avoided; model-local imputation was used instead |
| Worse v002 and unchanged v003 each became current | 2 child runs | Medium | Current pointer ended at v003 rather than best v001 |
| Experiment execution again required custom Python | Entire session | Medium | CLI only supported prediction/export |

## Manual workarounds

- Selected a small numeric subset because categorical preprocessing is not part of
  the obvious model workflow.
- Reinitialized the workspace after the invalid default-metric run.
- Configured regression metrics and expanding-window CV through Python.
- Manually retained the knowledge that v001 was best even though v003 remained
  current.

## Unused or distracting surface area

- Calibration, SHAP, Studio, MCP, and most experiment mutations were unnecessary.
- The broad transform catalog was not useful because preprocessing before temporal
  splitting could introduce leakage.

## Reproducibility and refresh result

- Source selection, external workspace, metrics, and timings were retained.
- Target-free production prediction and conformal bounds worked.
- This was not yet a repeated data refresh.

## Proposed changes

| Proposal | Evidence | Smallest intervention | Success measure | Disposition |
|---|---|---|---|---|
| Task-specific metric defaults and validation | Silent empty regression metrics | Set defaults by task and reject unsupported names | Default regression run emits RMSE/MAE/R² | Built: PR #57 |
| Warn before evaluating metrics on tiny holdouts | Three undefined R² warnings | Preflight fold sizes against selected metrics | One actionable warning before training | Defer until repeated |
| Distinguish latest from accepted/best | Both dogfood projects ended on a degraded child | First test whether an explicit switch/conclude workflow is sufficient | User can identify accepted version without manual metric archaeology | Evidence now repeated; investigate minimal fix |
| Leakage-safe preprocessing | Categorical features omitted; global fill avoided | Audit existing preprocessing path before exposing anything new | Scaling/imputation fit only on training folds and reused in production | Investigate, not yet build |

## Overall assessment

Harness added real value around temporal splitting, version comparison, and
production intervals. It also exposed a serious default-configuration bug that its
existing tests had missed. The session reinforces that actual usage is producing
better priorities than speculative operational infrastructure: task defaults,
accepted-version semantics, preprocessing clarity, and everyday experiment entry
points.
