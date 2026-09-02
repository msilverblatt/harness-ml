# Dogfood report: breast-cancer binary classification

## Context

- Date: 2026-09-02
- Task: binary classification
- Dataset: scikit-learn Wisconsin Diagnostic Breast Cancer, 569 rows and 30 predictors
- Clean data fingerprint: `c27acd8c48c0f13c0d03209c4446f3b7eee63a9c2cbd51180777cd68424510e5`
- Objective: exercise ingestion, repeated experiments, calibration, export, and prediction
- Workspace (not committed): `/Users/msilverblatt/Projects/harness-dogfood/breast-cancer`

## Session record

| Action | Version | Duration | Brier | Accuracy |
|---|---|---:|---:|---:|
| Ingest CSV and run pipeline | — | 0.07 s | — | — |
| Logistic baseline | v001 | 1.03 s | 0.03824 | 0.9508 |
| Add random forest | v002 | 1.37 s | 0.03059 | 0.9596 |
| Add isotonic calibration | v003 | 0.66 s | 0.03169 | 0.9543 |
| Export v003 bundle | v003 | — | — | — |
| Score 25 target-free rows | v003 | — | — | — |

The exported bundle was 464 KiB and produced 25 predictions through the documented
CLI workflow.

## What helped

- Workspace initialization, ingestion, version creation, and artifact persistence
  worked without manual file repair.
- The model experiment produced a directly comparable child version.
- Production export and target-free scoring worked on the first attempt.
- The metrics made it immediately clear that the random forest improved the
  baseline and isotonic calibration did not improve it further.

## Failures and confusion

| Problem | Frequency | Impact | Evidence |
|---|---:|---:|---|
| Logistic convergence warnings were repeated for folds and full-data fitting | 8 warnings | Medium | v001–v003 console output |
| Harness has preprocessing code but no obvious experiment/config path for scaling numeric features | Entire baseline | Medium | Logistic needed `max_iter=2000` and still did not converge |
| The worse calibration candidate automatically became `current` | 1/1 degraded child | Medium | v003 became current despite worse Brier and accuracy than v002 |
| There is no CLI command to run an experiment | Entire session | Medium | Training required a custom Python script; CLI covered only export/predict |
| Eval results are persisted but not summarized by `status` | Every comparison | Low | Status showed only current metrics |

## Manual workarounds

- Used the Python `WorkspaceManager` API to run experiments.
- Increased logistic `max_iter` to 2000 instead of applying a documented scaling
  transform.
- Manually compared metric JSON output to determine that v002 was preferable to
  current v003.

## Unused or distracting surface area

- Studio was not needed for this small session.
- MCP was not used because invoking a local Python script was faster for the first
  controlled run.
- SHAP, conformal intervals, pairwise discovery, and most experiment types were
  irrelevant to the task.

## Reproducibility and refresh result

- All commands and the untouched source CSV remain in the external workspace.
- Export and production prediction worked in actual use.
- A data refresh has not yet been tested; this project therefore does not satisfy
  the sustained-use Phase 1 criterion.

## Proposed changes

| Proposal | Evidence | Smallest intervention | Success measure | Disposition |
|---|---|---|---|---|
| Make numeric scaling reachable and documented | Repeated convergence warnings | First determine whether existing preprocessing can be wired through one current experiment type without a new subsystem | Logistic converges without manual parameter inflation | Experiment |
| Show candidate-vs-parent result after a run | v003 degraded but became current | Print/persist a concise comparison and make switching back obvious; do not build automatic promotion policy yet | User identifies and restores v002 without opening artifacts manually | Build if repeated once |
| Add CLI experiment execution | Custom script required | Consider one JSON/YAML-backed command reusing `WorkspaceManager`; first test whether MCP is the intended usable path | Second non-Python user workflow completes without custom code | Experiment |
| Capture model warnings in diagnostics | Eight noisy warnings | Record unique warning and count per model/run | Convergence issue appears once with model context | Build if repeated |

## Overall assessment

Harness helped preserve and compare the experiment sequence, and its export path
worked. It was less convenient than a short scikit-learn notebook for actually
starting experiments. The strongest observed issue is not missing infrastructure;
it is the gap between existing internals and a clear everyday workflow. This is
one project, so no product addition is justified yet without confirming the same
friction elsewhere.
