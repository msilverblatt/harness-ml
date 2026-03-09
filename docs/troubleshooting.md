# Troubleshooting

Common issues and solutions when working with HarnessML.

---

## "Model training failed"

**Symptom:** Backtest crashes during model fitting with a KeyError or ValueError.

**Likely causes:**
- Feature columns referenced in the model config do not exist in the data
- NaN values in the target column
- Mismatched column names between feature formulas and actual data

**Fix:**
```
data(action="inspect")
data(action="inspect", column="your_target_column")
```

Check that every column referenced in your feature formulas appears in the inspect output. If the target column has NaN values, drop or impute them:

```
data(action="drop_rows", column="target_col", condition="null")
```

---

## "Fold column not found"

**Symptom:** `KeyError` referencing the fold column during backtest.

**Likely cause:** The `fold_column` in your backtest config does not match an actual column name in your data.

**Fix:**

Verify the column name matches exactly (case-sensitive):
```
data(action="inspect", column="Season")
configure(action="backtest", fold_column="Season", fold_values=[2020, 2021, 2022, 2023, 2024])
```

If using a view as your features source, make sure the fold column survives the view pipeline (it is not filtered or dropped).

---

## "Formula syntax error"

**Symptom:** Feature formula fails with a `SyntaxError` or `UndefinedVariableError`.

**Likely causes:**
- Column names contain spaces or special characters
- Invalid pandas `eval()` syntax
- Referencing a column that does not exist

**Fix:**

Column names with spaces must be wrapped in backticks inside formulas:
```
features(action="add", name="score_diff", formula="`Home Score` - `Away Score`")
```

For safer handling, rename columns to remove spaces first:
```
data(action="rename", mapping={"Home Score": "home_score", "Away Score": "away_score"})
features(action="add", name="score_diff", formula="home_score - away_score")
```

Use standard pandas eval operators: `+`, `-`, `*`, `/`, `**`, `>`, `<`, `==`, `&`, `|`, `~`.

---

## "ImportError" or "ModuleNotFoundError"

**Symptom:** Python cannot find `harnessml` or its submodules.

**Likely cause:** Running with bare `python` instead of `uv run`.

**Fix:**

Always use `uv run` to execute commands. The workspace packages are installed in editable mode within the uv-managed environment:

```bash
# Wrong
python -c "from harnessml.core.runner import pipeline"

# Correct
uv run python -c "from harnessml.core.runner import pipeline"
```

If you see import errors even with `uv run`, ensure dependencies are installed:
```bash
uv sync
```

---

## "Hot reload not working"

**Symptom:** Changes to MCP handler code require a server restart to take effect.

**Likely cause:** The `HARNESS_DEV` environment variable is not set.

**Fix:**

Set the environment variable before starting the MCP server:
```bash
export HARNESS_DEV=1
uv run harness-plugin
```

Note: hot reload applies to handler business logic only. Changes to tool signatures or docstrings in `mcp_server.py` always require a server restart.

---

## "Calibration failed"

**Symptom:** Calibration step errors out or produces degenerate probabilities (all 0.5, all 0.0, etc.).

**Likely causes:**
- Too few samples in one or more folds for the calibrator to fit
- All predictions in a fold are identical (no variance for the calibrator to learn from)
- Using spline calibration with fewer than ~30 samples per fold

**Fix:**

Switch to a simpler calibrator or disable calibration:
```
configure(action="ensemble", calibration="isotonic")
# or
configure(action="ensemble", calibration="platt")
# or
configure(action="ensemble", calibration="none")
```

If you have very small folds, consider merging folds or using a CV strategy with larger partitions.

---

## "Meta-learner failed"

**Symptom:** The stacked ensemble meta-learner throws an error during fitting.

**Likely causes:**
- Fewer than 2 models have predictions (at least 2 are required for stacking)
- All models produce identical predictions (no variance for the meta-learner)
- One or more models failed silently, producing no output

**Fix:**

Check which models are active and producing predictions:
```
models(action="list")
pipeline(action="diagnostics")
```

Ensure at least 2 models have `active: true` and `include_in_ensemble: true`. If a model is failing silently, check its individual output:
```
models(action="show", name="the_model_name")
```

---

## GPU not detected

**Symptom:** MLP or TabNet models train on CPU even though a GPU is available.

**Likely causes:**
- The `HARNESS_DEVICE` environment variable is not set
- PyTorch is installed without CUDA support
- CUDA toolkit version does not match the PyTorch build

**Fix:**

Set the device environment variable:
```bash
export HARNESS_DEVICE=cuda
```

Verify PyTorch can see the GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

If `torch.cuda.is_available()` returns `False`, reinstall PyTorch with CUDA support following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

---

## "Guardrail blocked my action"

**Symptom:** An operation is rejected with a guardrail warning.

**Context:** HarnessML has 3 non-overridable guardrails (feature leakage, temporal ordering, critical path) and 9 overridable ones.

**Fix:**

Check which guardrails are active:
```
configure(action="check_guardrails")
```

Non-overridable guardrails cannot be bypassed — they exist to prevent data leakage and other critical errors. If a leakage guardrail fires, your feature formula likely uses information from the target or future data. Redesign the feature.

Overridable guardrails can be acknowledged in the experiment config if you understand the risk.

---

## "Experiment has no hypothesis"

**Symptom:** Experiment creation fails with a validation error about missing hypothesis.

**Context:** Hypotheses are required for all experiments.

**Fix:**

Always include a hypothesis when creating an experiment:
```
experiments(action="create",
  description="Test higher learning rate",
  hypothesis="Increasing learning_rate from 0.05 to 0.1 should improve convergence speed without overfitting, expecting Brier improvement of 0.002-0.005."
)
```

See the [experiment discipline skill](skills/harness-run-experiment.md) for guidance on writing good hypotheses.
