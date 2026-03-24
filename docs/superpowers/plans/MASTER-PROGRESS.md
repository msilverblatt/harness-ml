# Harness 2 — Implementation Master Progress

**Spec:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md)

## Build Order

| # | Package | Status | Plan | Dependencies |
|---|---------|--------|------|-------------|
| 1 | harness-data | **Complete** (253 tests, 24 steps, expression engine) | [Plan](./2026-03-23-harness-data.md) | None |
| 2a | harness-ml: Task Types + Models | **Complete** (3 task types, 14 models, 5 families) | [Plan](./2026-03-23-harness-ml-2a-types-models.md) | harness-data |
| 2b | harness-ml: Features + Evals | **Complete** (4 feature types, pairwise derivatives, eval framework with presets) | [Plan](./2026-03-23-harness-ml-2b-features-evals.md) | harness-data |
| 2c | harness-ml: Training Pipeline | **Complete** (4-phase backtest, 8 CV strategies, DAG, cache, meta-learner, post-processing) | [Plan](./2026-03-23-harness-ml-2c-training-pipeline.md) | 2a + 2b |
| 3 | research-loop extensions | Pending | — | None (TypeScript) |
| 3 | research-loop extensions | **Complete** (new verdicts, parent selection, conclude-as-terminal, 79 tests) | [Plan](./2026-03-23-research-loop-extensions.md) | None (TypeScript) |
| 4 | Harness app (workspace, CLI, experiments) | **Complete** (version tree, workspace manager, 8 experiment types, CLI) | [Plan](./2026-03-23-harness-app.md) | 1 + 2a + 2b + 2c |
| 5 | harness-studio (dashboard) | Pending | — | 4 |

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-23 | Version tree instead of linear promote/discard | Avoid local maxima traps, enable branching exploration |
| 2026-03-23 | Typed experiments as primary tool | Collapse 80 tools to 17, agent thinks in hypotheses |
| 2026-03-23 | harness-data / harness-ml split | Separate data eng lifecycle from ML, harness-data also serves as feature computation engine |
| 2026-03-23 | TS/Python bridge via JSON-RPC | Use research-loop as-is (TypeScript), Python port is fallback if bridge is painful |
| 2026-03-23 | No "production" model concept | Version tree with current pointer; `harness export` deferred |
| 2026-03-23 | First-class expression engine | Formula eval is the agent's language, not a utility — needs registry, validation, discoverability |
| 2026-03-23 | Three-layer eval system | Generic framework: threshold checks + comparative evals + LLM judgments. User-defined dimensions in evals.yaml, not hardcoded |

## Notes

- Each plan is self-contained — produces working, testable software independently
- Plans are written one at a time and reviewed before moving to the next
- All plans follow TDD: write test → verify fail → implement → verify pass → commit
