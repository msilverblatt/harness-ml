# Phase 1 case study: first three Harness projects

## Scope

Three real public tabular datasets were taken through repeated experiments:
breast-cancer binary classification, Ames temporal regression, and seven-class
image segmentation. Production prediction was exercised on the first two. The
multiclass project was repeated over real MCP stdio transport and through a
controlled 2,000-to-2,310-row source-refresh replay.

## Where Harness helped

- Fold-aware training avoided target, era, and obvious preprocessing leakage.
- Immutable version directories retained hypotheses, configs, metrics,
  predictions, diagnostics, and production models together.
- Parent deltas in the MCP response made useful changes immediately legible.
- Prediction caching made adding a model reuse unchanged fold work.
- Production scoring reproduced feature handling and emitted calibrated
  probabilities or conformal regression intervals.
- Failed target validation and failed training did not publish versions.
- Compare, conclude, and switch were enough to retain a selected candidate; a new
  promotion subsystem was not needed.

## Where Harness did not help

- The Python API required custom scripts for ordinary experiments. The MCP path was
  substantially more coherent, but its accepted workflow needs a concise example.
- String multiclass labels required external integer encoding.
- Categorical and general preprocessing did not have an obvious fold-safe path, so
  useful Ames columns were omitted.
- Tiny temporal holdouts produced repeated undefined-R² warnings instead of one
  useful preflight diagnostic.
- Studio, SHAP, broad transform coverage, and most mutation types provided no value
  in these sessions.
- A controlled refresh replay is not evidence of a long-running deployment.

## Evidence-driven repairs

Dogfooding found defects that existing tests had missed:

1. regression projects inherited binary metrics and silently stored empty results;
2. versions on different data fingerprints could be presented as directly
   comparable;
3. there was no honest way to re-evaluate the accepted config after a refresh;
4. an all-model failure hid the actionable underlying exceptions.

PRs #57, #59, #60, and #62 repaired those issues with focused regression tests.
No scheduler, daemon, lock manager, artifact protocol, or promotion state machine
was added.

## Next five improvements, ranked

1. **String-label multiclass experiment.** Reproduce on one more dataset and define
   how original labels are retained in production outputs before implementing.
2. **Fold-safe preprocessing audit.** Determine whether existing preprocessing is
   actually fit on training folds and reused by production; add only the smallest
   path needed for categorical Ames features.
3. **Document the agent acceptance loop.** Show propose, inspect deltas, conclude,
   and switch. This addresses repeated confusion without new state.
4. **Temporal-fold preflight diagnostics.** Warn once when a selected metric is
   undefined for a generated holdout, if the Ames behavior recurs.
5. **Sustained refresh observation.** Maintain one workspace across a genuinely
   later source update and verify refresh, iteration, export, and rollback before
   declaring Phase 1 complete.

## Conclusion

Harness improved experimental integrity and artifact reproducibility, especially
through MCP, but did not yet prove broad operational maturity. The strongest next
work is model-workflow depth and sustained use—not additional infrastructure or an
LLM training layer.
