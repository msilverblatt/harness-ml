# Harness v1 → Harness 2 capability matrix

This matrix prevents capabilities from disappearing accidentally during the replacement.

| Capability | Harness 2 status | Release disposition |
|---|---|---|
| Binary, multiclass, regression tasks | Implemented | Ported |
| 14 model implementations / 5 families | Implemented | Ported |
| Cross-validation strategies | Implemented | Ported |
| Nested ensemble/meta-learner | Implemented | Ported |
| Calibration and post-processing | Implemented; integration hardening ongoing | Ported |
| Fingerprint prediction cache | Implemented with data-safe v2 keys | Replaced |
| Declarative data transforms | Implemented as independent `harness-data` package | Replaced/improved |
| Data profiling and source adapters | Implemented | Ported |
| Experiment tracking | Implemented as transactional version tree | Replaced/improved |
| MCP interface | 17 tools and 5 resources implemented; transport/restart hardening ongoing | Replaced |
| Studio | API and eight initial views implemented; UX hardening ongoing | Replaced |
| Eval checks/comparisons | Implemented and persisted per experiment | Replaced/improved |
| SHAP/model explainability | Not yet ported; ensemble coefficients available | Release blocker |
| Conformal prediction intervals | Not yet ported | Release blocker for regression parity |
| Drift analysis | Not yet ported | Deferred with explicit migration note unless completed |
| HPO and sweeps | Not yet ported | Deferred; typed experiments are primary replacement |
| Automatic feature search | Basic correlation discovery only | Release blocker for claimed discovery parity |
| Kaggle/Drive adapters | Not yet ported | Optional integration; document as deferred |
| Notebook/journal | Version metadata replaces journal; notebook UI not ported | Intentional replacement/deferred UI |
| Reporting/visualization export | Not yet ported | Deferred unless required by release candidate |
| Sports competition package | Remains in v1/private standalone repository | Separate follow-up migration |
| Production model export | Not yet implemented | Release blocker for export claims |
| Guardrail inventory/audit suite | Partially replaced by typed validation and evals | Audit before release |
