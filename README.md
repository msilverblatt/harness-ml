# Harness 2

Harness is an agent-first platform for building, evaluating, and iterating on tabular machine-learning systems through typed experiments.

Harness 2 replaces the monolithic v1 runner with five focused Python packages:

- **harness-data** — source ingestion, declarative transforms, and profiling
- **harness-ml** — task types, model families, cross-validation, caching, ensembles, and evaluation
- **harness-app** — workspaces, immutable experiment versions, and CLI
- **harness-server** — MCP interface with 17 tools and 5 resources
- **harness-studio** — FastAPI and React experiment dashboard

> Harness 2 is the current release. The stable v1 code remains permanently available on `v1-maintenance` and at the `v1-final` tag; full incremental v2 development history is preserved on `harness2-development-history`.

## Development quickstart

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node.js 20+.

```bash
git clone https://github.com/msilverblatt/harness-ml.git
cd harness-ml
git switch harness2-development-history
uv sync --all-packages

uv run --package harness-app harness init my-project
cd my-project
uv run --package harness-app harness doctor
uv run --package harness-app harness serve --studio
# MCP clients launch packages/harness-server/src/harness/server/main.py via pmcp.
```

Successful experiments persist a fitted `model.bundle` containing full-data seed models,
the provider DAG, ensemble, calibration, and optional conformal interval metadata. Use it
from the CLI:

```bash
harness export ./model.bundle --version v003
harness predict ./scoring.csv ./predictions.parquet --version v003
```

Native feature importance is persisted with each version. Install `harness-ml[explain]`
and call `ProductionBundle.explain(frame)` for on-demand SHAP attribution.

Run the test suites:

```bash
for package in harness-data harness-ml harness-app harness-server harness-studio; do
  uv run --package "$package" --with pytest pytest "packages/$package/tests"
done

cd packages/harness-studio/frontend
npm ci
npm run build
```

## Safety and correctness

Harness excludes target and configured metadata columns from implicit model features. Prediction cache keys include dataset, target, fold, feature schema, task, and model configuration fingerprints. Experiments are staged transactionally and only update the current workspace after successful training and artifact creation.

## Documentation

- [Post-v2 roadmap: production hardening, LLM evals, and bounded self-improvement](docs/ROADMAP.md)
- [Harness 2 design](docs/superpowers/specs/2026-03-23-harness2-design.md)
- [Productionization and replacement plan](docs/superpowers/plans/2026-03-25-harness2-productionization-and-release.md)
- [Implementation progress](docs/superpowers/plans/MASTER-PROGRESS.md)

## License

MIT
