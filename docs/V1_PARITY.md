# Harness v1 → Harness 2 capability matrix

This matrix prevents capabilities from disappearing accidentally during the replacement.

| Capability | Harness 2 status | Release disposition |
|---|---|---|
| Binary, multiclass, regression tasks | Implemented | Ported |
| 14 model implementations / 5 families | Implemented | Ported |
| Cross-validation strategies | Implemented | Ported |
| Nested ensemble/meta-learner | Implemented | Ported |
| Calibration and post-processing | Implemented for evaluation and fitted production bundles | Ported |
| Fingerprint prediction cache | Implemented with data-safe v2 keys | Replaced |
| Declarative data transforms | Implemented as independent `harness-data` package | Replaced/improved |
| Data profiling and source adapters | Implemented | Ported |
| Experiment tracking | Implemented as transactional version tree | Replaced/improved |
| MCP interface | 17 tools and 5 resources implemented with strict transport validation | Replaced |
| Studio | API, eight initial views, diagnostics, and production inference endpoint implemented | Replaced |
| Eval checks/comparisons | Implemented and persisted per experiment | Replaced/improved |
| SHAP/model explainability | Native fitted-model importance is persisted; on-demand SHAP is available through `ProductionBundle.explain(data)` and the `explain` extra | Ported with explicit optional dependency |
| Conformal prediction intervals | OOF cross-conformal regression radius, evaluation bounds, and production intervals implemented | Ported |
| Drift analysis | Not yet ported | Deferred with explicit migration note unless completed |
| HPO and sweeps | Not yet ported | Deferred; typed experiments are primary replacement |
| Automatic feature search | Existing features plus pairwise product/difference candidates are ranked and emitted as declarative expressions | Replaced with bounded, auditable search |
| Kaggle/Drive adapters | Not yet ported | Optional integration; document as deferred |
| Notebook/journal | Version metadata replaces journal; notebook UI not ported | Intentional replacement/deferred UI |
| Reporting/visualization export | Not yet ported | Deferred unless required by release candidate |
| Sports competition package | Remains in v1/private standalone repository | Separate follow-up migration |
| Production model export | Full-data seed models, provider DAG, ensemble, calibration, feature resolution, explanations, and conformal metadata are serialized atomically; CLI export/predict and Studio inference are implemented | Ported |
| Guardrail inventory/audit suite | Typed config validation, target/exclusion enforcement, data-safe cache fingerprints, transactional rollback, eval reports, and strict MCP validation implemented | Replaced/improved |

## Explicit v2 deferrals

The following v1 integrations are not part of the v2.0 release contract and remain available on `v1-maintenance` / `v1-final`:

- drift dashboards (versioned data hashes and metrics remain available for external monitors),
- general-purpose HPO/sweep orchestration (typed transactional experiments are the supported primitive),
- Kaggle and Google Drive convenience adapters,
- notebook/journal UI and standalone report export,
- the sports competition package, which remains a separate project.

These are documented scope changes, not silent compatibility claims. They may return in later minor releases based on demand.
