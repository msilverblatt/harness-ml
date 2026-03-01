# Womens Tournament

## Quick Start

- Validate config: `easyml validate`
- Inspect models: `easyml inspect models`
- Run training: `easyml run train`
- Run backtest: `easyml run backtest`

## Project Structure

- `config/` — YAML configuration (pipeline, models, ensemble, server)
- `data/` — Raw, processed, and feature data
- `features/` — Feature computation modules
- `experiments/` — Experiment overlays and results
- `models/` — Trained model artifacts

## Rules

- ALWAYS use `uv run` — never bare `python`
- Run `easyml validate` after any config change
- One variable per experiment
- Log every experiment in EXPERIMENT_LOG.md
