# Harness 2 Productionization and Replacement Plan

**Date:** 2026-03-25
**Status:** Proposed
**Target repository:** `https://github.com/msilverblatt/harness-ml`
**Goal:** Make Harness 2 correct, complete, installable, and supportable, then replace v1 in the existing GitHub repository without losing either codebase's history.

---

## Release standard

Harness 2 is ready to replace v1 only when all of the following are true:

1. Default workflows cannot leak the target or reuse stale predictions.
2. Every advertised experiment type either works end-to-end or has been removed from the public interface.
3. A real, unmocked agent workflow succeeds from project creation through data ingestion, baseline training, a child experiment, comparison, conclusion, and version switching.
4. The MCP server exposes the documented 17 tools and 5 resources with integration tests.
5. Fresh-clone installation works from built artifacts, without relying on editable installs or local filesystem dependencies.
6. Python tests, TypeScript tests, frontend build, lint, type checking, package builds, and smoke tests run in CI.
7. Important v1 capabilities have either been ported or explicitly documented as intentional removals.
8. v1 remains recoverable through a permanent branch and signed/annotated release tag.

---

## Phase 0 — Preserve work and establish repository hygiene

### 0.1 Preserve all current work

- Create a bundle or filesystem backup of:
  - `/Users/msilverblatt/Projects/harness2`
  - `/Users/msilverblatt/Projects/harness2/packages/research-loop`
  - `/Users/msilverblatt/harness-ml/harness-ml` (`umbrella-overhaul` and its dirty worktree)
  - the four dirty standalone package repositories under `/Users/msilverblatt/harness-ml/`
- Record commit IDs and `git status` output in a migration note.
- Do not reset or clean any existing worktree.

### 0.2 Clean Harness 2 repository boundaries

- Remove the nested old-repository clone at `harness2/harness-ml` from the Harness 2 working tree or add it to `.gitignore` temporarily.
- Decide how `research-loop` is consumed:
  - preferred: publish its Harness 2 extension commit to `msilverblatt/research-loop`, then depend on a released npm version;
  - acceptable during development: Git submodule pinned to a commit;
  - do not leave an untracked nested Git repository in the release tree.
- Commit the `protomcp` dependency change only after confirming compatibility with the registry release.
- Add a root `.editorconfig` and consistent generated-artifact ignores.

### Exit criteria

- Parent `git status` is clean.
- No accidental nested repositories exist.
- Every source dependency is represented by tracked files, a package version, or an intentional submodule.

---

## Phase 1 — Fix ML correctness blockers

This phase blocks all other release work.

### 1.1 Eliminate target leakage

Affected code:

- `packages/harness-ml/src/harness/ml/runners/backtest.py`
- `packages/harness-ml/src/harness/ml/runners/training.py`
- model/config validation code

Changes:

- Remove the target column from the feature frame before fold generation and training.
- Treat an empty model feature list as "all eligible non-target feature columns," not all dataframe columns.
- Reject explicit feature lists containing the target.
- Reject missing feature names before training starts.
- Define handling for fold columns, IDs, group columns, timestamps, and other non-feature metadata.
- Keep target extraction aligned after feature resolution and filtering.

Required tests:

- Random target + random features must remain near chance under the default feature path.
- Explicit target inclusion must fail validation.
- Default features must never contain target, fold, or configured metadata columns.
- Feature resolution must preserve target/index alignment.

### 1.2 Make cache fingerprints data-safe

Affected code:

- `packages/harness-ml/src/harness/ml/runners/prediction_cache.py`
- `packages/harness-ml/src/harness/ml/runners/backtest.py`

Fingerprint must include:

- data/content hash or immutable dataset version hash;
- target values/hash;
- train/test indices or fold-definition hash;
- resolved feature names, dtypes, and ordering;
- model type and full normalized parameters;
- task type;
- preprocessing/training-filter configuration;
- seed configuration;
- upstream provider fingerprints;
- relevant package/cache schema version.

Required tests:

- Same inputs hit cache.
- Changed data, target, folds, features, preprocessing, parameters, or seed miss cache.
- Reordered equivalent config is deterministic.
- Corrupt/incompatible cache entries fail closed and retrain.

### 1.3 Harden fold and model failure behavior

- Validate target type and prediction shape through `TaskType` hooks.
- Fail clearly if every model fails or no ensemble columns remain.
- Preserve row identity and original fold identity in prediction artifacts.
- Confirm multiclass prediction shape throughout cache, ensemble, and metric paths.
- Make fold ordering numeric/stable rather than string-dependent.
- Define deterministic random-state propagation.

### Exit criteria

- Correctness regression suite passes.
- A real random-label smoke test no longer reports implausible performance.
- Cache invalidation tests cover all data/config inputs.

---

## Phase 2 — Complete workspace and experiment semantics

Affected code:

- `packages/harness-app/src/harness/app/workspace/manager.py`
- `packages/harness-app/src/harness/app/workspace/versions.py`
- config schemas and tests

### 2.1 Implement all advertised experiment types

Implement and validate:

- `baseline`
- `feature`
- `model`
- `hyperparameter`
- `ensemble`
- `calibration`
- `cv_strategy`
- `feature_selection`

Requirements:

- Typed parameter models per experiment type.
- Unknown experiment types and invalid/no-op changes raise errors.
- Input dictionaries are never mutated.
- Every experiment produces a normalized, inspectable diff.
- No version is created if validation or training fails before meaningful output.

### 2.2 Make branching transactional and immutable

- Resolve a non-current parent in memory.
- Do not switch or overwrite the current working config while evaluating a branch.
- Write a new version into a temporary directory and atomically promote it when complete.
- Preserve the prior current pointer/config on failure.
- Decide and document whether a successful experiment becomes current; make behavior explicit and tested.
- Detect ancestry cycles and invalid parent IDs.

### 2.3 Complete version artifacts

Every version must include:

- `meta.yaml` with parent, hypothesis, type, verdict, timestamps, data hash, and metrics;
- `diff.yaml`;
- complete resolved config snapshot;
- per-fold and pooled metrics;
- predictions with row/fold identity;
- diagnostics;
- model/cache summary;
- failure/warning information.

Use timezone-aware UTC timestamps.

### 2.4 Add real application E2E tests

Do not mock `run_backtest` in the primary integration suite.

Scenarios:

- binary baseline and child feature experiment;
- regression baseline;
- multiclass baseline;
- branch from a non-current version;
- failed experiment rollback;
- data change and cache invalidation;
- conclude and compare;
- process restart and workspace reload.

### Exit criteria

- All eight experiment types run through `WorkspaceManager` with real training.
- Version tree behavior matches the specification or the specification is deliberately updated.
- Failure paths leave the workspace consistent.

---

## Phase 3 — Wire evaluations, diagnostics, and artifact behavior

### 3.1 Integrate the eval framework

- Add `evals.yaml` to config management and snapshots.
- Run `EvalRunner` after every experiment.
- Compare against the actual selected parent version.
- Store the complete eval report with the version.
- Surface checks, comparisons, warnings, and judgment prompts to MCP and Studio.

### 3.2 Complete diagnostics needed for a credible v2 release

Must-have for replacement:

- per-fold metric breakdown;
- calibration metrics/curve for classification;
- residual diagnostics for regression;
- feature/model importance where supported;
- model failure details;
- data-staleness detection;
- ensemble coefficients and constituent metrics.

### 3.3 Decide v1 feature parity explicitly

Create a checked-in matrix with one of `ported`, `replaced`, `deferred`, or `removed` for:

- explainability/SHAP;
- conformal intervals;
- drift analysis;
- HPO and sweeps;
- automatic feature search/selection;
- Kaggle adapter;
- notebook/journal;
- reporting and visualization;
- source adapters;
- competition/sports functionality;
- export/production artifacts;
- guardrails and audit logging.

Release blocker rule: no capability may disappear accidentally. Intentional removals must be documented in migration notes.

### Exit criteria

- Eval results are generated by real experiments and visible through APIs.
- The parity matrix is complete and approved.
- All capabilities marked `ported` have integration coverage.

---

## Phase 4 — Implement the Harness 2 MCP server

Create `packages/harness-server/` based on the existing plan, but reuse lessons and infrastructure from the published v1 protomcp server.

### 4.1 Server foundation

- Pin a compatible `protomcp` release.
- Add server context and workspace lifecycle.
- Add structured errors and JSON-serializable responses.
- Add request/event telemetry consumed by Studio.
- Avoid global mutable state where concurrent sessions can collide.

### 4.2 Implement the documented interface

Tools:

- `project.init`
- `data.add_source`
- `data.transform`
- `data.run`
- `data.profile`
- `data.inspect`
- `experiment.propose`
- `experiment.conclude`
- `analyze.diagnostics`
- `analyze.explain`
- `analyze.compare`
- `analyze.discover`
- `versions.list`
- `versions.show`
- `versions.switch`
- `versions.ancestry`
- `workspace.open`

Resources:

- `harness://data/schema`
- `harness://versions/tree`
- `harness://versions/current`
- `harness://models/available`
- `harness://tasks/supported`

### 4.3 Research-loop integration

- Publish or pin the Harness 2 research-loop extension.
- Enforce propose → run → conclude state transitions.
- Support parent selection and all four verdicts.
- Confirm dynamic tool visibility works with the selected protomcp version.

### 4.4 MCP E2E tests

Drive the actual server protocol through:

1. project initialization;
2. source registration and data pipeline execution;
3. baseline proposal;
4. conclusion;
5. child experiment;
6. analysis comparison;
7. version ancestry and switching;
8. resource reads;
9. restart/reconnection.

Do not call handler functions directly in the main E2E test.

### Exit criteria

- 17 tools and 5 resources are discoverable and functional.
- A protocol-level E2E session completes without mocks.
- Multi-workspace and restart behavior are covered.

---

## Phase 5 — Finish Studio as an operational UI

- Replace raw JSON dumps with stable tables/charts for primary workflows.
- Show version tree, parent relationships, metric directionality, and verdicts.
- Display fold metrics, calibration/residual diagnostics, failures, and cache status.
- Connect MCP event telemetry and reconnect behavior.
- Add empty, loading, stale-data, disconnected, and error states.
- Add frontend component/API tests for critical views.
- Confirm static asset packaging into the Python wheel.

### Exit criteria

- Studio can inspect every artifact produced by the E2E workflow.
- Production frontend build is included and served from an installed wheel.
- No primary view is merely an unformatted JSON placeholder.

---

## Phase 6 — Packaging, CLI, and fresh-install validation

### 6.1 Define distribution model

Recommended layout:

- `harness-data` — independent Python library;
- `harness-ml` — independent Python library;
- `harness-studio` — Python backend + built frontend;
- `harness-app` or root `harness` distribution — user-facing CLI and dependency composition;
- `harness-server` — included in the user-facing distribution or installed as a pinned dependency;
- `research-loop` — published npm dependency or bundled verified artifact.

### 6.2 Complete CLI

Required commands:

- `harness init`
- `harness status`
- `harness doctor`
- `harness serve`
- version inspection/switching as appropriate

`doctor` must verify Python, Node/protomcp/research-loop availability, optional model backends, writable workspace, and Studio assets.

### 6.3 Root project files

Add:

- README with tested quickstart;
- LICENSE;
- architecture overview;
- migration guide from v1;
- contributing guide;
- security policy as appropriate;
- root task runner or documented commands;
- pinned lockfiles and supported Python/Node versions.

### 6.4 Clean-environment tests

In temporary environments/containers:

- build all wheels/sdists;
- install only from built artifacts;
- run CLI help/doctor/init;
- run a minimal real experiment;
- start MCP server and enumerate tools/resources;
- start Studio and hit health/API/static routes.

### Exit criteria

- No local path dependencies.
- Fresh-clone and built-artifact smoke tests pass.
- The README quickstart is executed verbatim in CI.

---

## Phase 7 — CI and release gates

Add GitHub Actions jobs for:

- Python unit tests per supported Python version;
- real cross-package integration tests;
- MCP protocol E2E;
- research-loop TypeScript tests;
- frontend typecheck/test/build;
- Ruff formatting/linting;
- Python type checking on public/core boundaries;
- package build and wheel-content verification;
- dependency/secret scanning;
- clean-install smoke test.

Optional model backends should be split into:

- required lightweight model matrix on every PR;
- full model contract suite on scheduled/release workflows.

Release is blocked on all required jobs.

---

## Phase 8 — Safe replacement in the existing GitHub repository

Do not force-push Harness 2 directly over `main`.

### 8.1 Preserve v1 permanently

In `msilverblatt/harness-ml`:

- create annotated tag `v1-final` at the current v1 main commit;
- create branch `v1-maintenance` at the same commit;
- verify both exist remotely before replacement.

### 8.2 Preserve full Harness 2 development history

- Add the existing GitHub repository as a remote to the Harness 2 repository.
- Push the unrelated Harness 2 history to `harness2-development-history`.
- This branch is archival/development history and is not the PR branch.

### 8.3 Create a reviewable replacement branch

- Start `harness-v2` from the current GitHub `main` commit.
- Replace the v1 tree with the productionized Harness 2 tree in one explicit migration commit.
- Retain any repository-level assets intentionally carried forward: license, issue templates, relevant docs/history references.
- Add a migration document linking v1 paths/packages to v2 equivalents.
- Open a pull request from `harness-v2` to `main`.

This approach preserves normal repository ancestry and makes the replacement diff reviewable while retaining the complete Harness 2 commit history on its archival branch.

### 8.4 Pre-merge release candidate

- Publish prerelease artifacts from `harness-v2`.
- Install and test them in a clean environment.
- Run at least one realistic project workflow, not only synthetic fixtures.
- Confirm documentation links, package names, MCP configuration, and Studio assets.

### 8.5 Merge and release

- Merge the replacement PR only after all gates pass.
- Tag the first replacement release as `v2.0.0` or the chosen semantic version.
- Update GitHub description/topics and release notes.
- Keep `v1-maintenance` protected and document rollback instructions.

---

## Recommended implementation order

1. Repository hygiene and backups.
2. Target leakage fix.
3. Cache correctness fix.
4. Real WorkspaceManager E2E test.
5. Complete experiment types and transactional version behavior.
6. Eval/diagnostic integration and parity decisions.
7. MCP server and protocol E2E.
8. Packaging/CLI/fresh install.
9. Studio completion.
10. CI.
11. Same-repository replacement branch and release candidate.

---

## Final go/no-go checklist

- [ ] No target leakage under implicit or explicit feature selection
- [ ] Cache invalidates on all data/config/fold changes
- [ ] Binary, multiclass, and regression real E2E pass
- [ ] All advertised experiment types work or are removed
- [ ] Failed experiments are transactional
- [ ] Parent branching is immutable and correct
- [ ] Eval reports and diagnostics are persisted
- [ ] v1 parity matrix is approved
- [ ] 17 MCP tools and 5 resources pass protocol E2E
- [ ] research-loop extension is published/pinned and clean
- [ ] Studio renders primary artifacts without raw-placeholder UX
- [ ] Wheels/sdists install in clean environments
- [ ] README quickstart passes verbatim
- [ ] CI is green
- [ ] v1-final tag and v1-maintenance branch exist remotely
- [ ] Harness 2 full history is preserved remotely
- [ ] Replacement PR is reviewed and release candidate validated
