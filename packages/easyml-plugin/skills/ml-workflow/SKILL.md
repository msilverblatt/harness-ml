---
name: ml-workflow
description: |
  Use when working on ML experimentation tasks — adding data, discovering
  features, building models, running experiments, or evaluating results
  in an easyml project. Guides the full experimentation workflow.
---

# ML Experimentation Workflow

You have access to easyml MCP tools that handle all pipeline mechanics
automatically. Your job is to think about ML hypotheses, not plumbing.

## Available Tools

### Data
- **add_dataset** — Merge a new CSV/parquet/Excel into the feature store. Auto-detects join keys, reports correlations.
- **profile_data** — Column stats, types, null rates for the feature dataset.
- **available_features** — List all feature columns (filter by prefix).

### Feature Engineering
- **add_feature** — Create a feature from a formula (e.g., `diff_adj_em * diff_barthag`). Supports @-references to other created features.
- **add_features_batch** — Create multiple features with dependency resolution.
- **test_transformations** — Automatically test log, sqrt, cbrt, squared, rank, z-score, and interactions. Returns which transformation improves correlation most.
- **discover_features** — Correlation analysis, importance ranking (XGBoost/MI), redundancy detection, suggested groupings.

### Models
- **add_model** — Add a model by type or preset (e.g., `xgboost_classifier`). Specify features and optional param overrides.
- **remove_model** — Remove a model.
- **show_models** — List all models with status.
- **show_presets** — Available model presets with sensible defaults.

### Ensemble & Backtest
- **configure_ensemble** — Set method (stacked/average), temperature, excluded models.
- **configure_backtest** — Set CV strategy, seasons, metrics.

### Experiments
- **experiment_create** — Create experiment with auto-numbered ID and hypothesis.
- **write_overlay** — Write config overlay to experiment directory.

### Inspection
- **show_config** — Full resolved project config.
- **list_runs** — All pipeline runs.

## Workflow Pattern

### Starting a New Project
1. Scaffold: `easyml init` via CLI (or create config/ directory manually)
2. Add data: `add_dataset` for each data source
3. Explore: `discover_features` to see what's useful
4. Baseline: `add_model` with a simple preset, `configure_backtest`

### Feature Engineering Cycle
1. `test_transformations` on promising features
2. Review results — which transforms improve correlation?
3. `add_feature` for the best transformations
4. Add new features to models

### Experiment Cycle
1. `experiment_create` with clear hypothesis
2. `write_overlay` with one variable changed
3. Run backtest via CLI (`uv run python -m easyml.runner run backtest`)
4. Compare results to baseline
5. Promote winners or iterate

## Key Principles

- **One variable per experiment.** Change one thing, measure the impact.
- **Use presets.** Don't manually configure hyperparameters — start from presets and override only what you're testing.
- **Formula features are cheap.** Creating and testing features costs nothing. Be aggressive about exploring.
- **Trust the tools.** Every tool returns markdown with all the info you need. Don't re-implement analysis.
- **project_dir defaults to cwd.** All tools auto-detect the project from the current directory.
