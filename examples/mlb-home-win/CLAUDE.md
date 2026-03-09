# mlb-home-win

## Quick Start

- Validate config: `harnessml validate`
- Inspect models: `harnessml inspect models`
- Run training: `harnessml run train`
- Run backtest: `harnessml run backtest`

## Project Structure

- `config/` — YAML configuration (pipeline, models, ensemble, server)
- `data/` — Raw, processed, and feature data
- `features/` — Feature computation modules
- `experiments/` — Experiment overlays and results
- `models/` — Trained model artifacts

## Rules

- ALWAYS use `uv run` — never bare `python`
- Run `harnessml validate` after any config change
- One variable per experiment
- Log every experiment in EXPERIMENT_LOG.md
