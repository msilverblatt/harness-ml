"""Click CLI for easyml-runner.

Provides validate, inspect, experiment, run, serve, and init commands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from easyml.runner.validator import validate_project


# -----------------------------------------------------------------------
# Helper: load and validate config, exit on failure
# -----------------------------------------------------------------------

def _load_config(config_dir: str, gender: str | None = None):
    """Load and validate project config, exiting on failure."""
    variant = None
    if gender and gender.upper() == "W":
        variant = "w"
    result = validate_project(config_dir, variant=variant)
    if not result.valid:
        click.echo(result.format(), err=True)
        raise SystemExit(1)
    return result


# -----------------------------------------------------------------------
# Main group
# -----------------------------------------------------------------------

@click.group()
@click.option(
    "--config-dir",
    default="config",
    help="Path to the project config directory.",
    type=click.Path(),
)
@click.option(
    "--gender",
    default="M",
    help="Gender variant (M or W).",
)
@click.pass_context
def main(ctx: click.Context, config_dir: str, gender: str) -> None:
    """easyml — YAML-driven ML pipeline runner."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["gender"] = gender


# -----------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------

@main.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate the project configuration."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    variant = "w" if gender.upper() == "W" else None

    result = validate_project(config_dir, variant=variant)
    if result.valid:
        click.echo("Config is valid. OK")
    else:
        click.echo(result.format(), err=True)
        ctx.exit(1)


# -----------------------------------------------------------------------
# inspect subgroup
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def inspect(ctx: click.Context) -> None:
    """Inspect project configuration."""
    pass


@inspect.command("config")
@click.option("--section", default=None, help="Show only this section of the config.")
@click.pass_context
def inspect_config(ctx: click.Context, section: str | None) -> None:
    """Dump resolved config as JSON."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    data = result.config.model_dump()
    if section is not None:
        if section in data:
            data = data[section]
        else:
            click.echo(f"Unknown section: {section!r}", err=True)
            ctx.exit(1)
            return

    click.echo(json.dumps(data, indent=2, default=str))


@inspect.command("models")
@click.pass_context
def inspect_models(ctx: click.Context) -> None:
    """List models with type, active status, and feature count."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    models = result.config.models
    for name, model_def in sorted(models.items()):
        status = "active" if model_def.active else "inactive"
        n_features = len(model_def.features)
        click.echo(f"  {name}: type={model_def.type}, {status}, {n_features} features")


@inspect.command("features")
@click.pass_context
def inspect_features(ctx: click.Context) -> None:
    """List declared features."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    features = result.config.features
    if not features:
        click.echo("No features declared in config.")
        return

    for name, feat in sorted(features.items()):
        click.echo(
            f"  {name}: category={feat.category}, level={feat.level}, "
            f"columns={feat.columns}"
        )


@inspect.command("data")
@click.option(
    "--columns", default=None,
    type=click.Choice(["diff", "non_diff", "high_null", "all"]),
    help="Show detailed column profiles for a category.",
)
@click.option(
    "--nulls", is_flag=True,
    help="Show columns grouped by null percentage tier.",
)
@click.pass_context
def inspect_data(ctx: click.Context, columns: str | None, nulls: bool) -> None:
    """Profile the matchup features dataset.

    Shows shape, seasons, null rates, and summary statistics.
    """
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    features_dir = result.config.data.features_dir
    parquet_path = Path(features_dir) / "matchup_features.parquet"

    if not parquet_path.exists():
        click.echo(f"Data file not found: {parquet_path}", err=True)
        raise SystemExit(1)

    from easyml.runner.data_profiler import profile_dataset

    profile = profile_dataset(parquet_path)

    if columns:
        click.echo(profile.format_columns(category=columns))
    elif nulls:
        click.echo(profile.format_null_tiers())
    else:
        click.echo(profile.format_summary())


# -----------------------------------------------------------------------
# experiment subgroup
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def experiment(ctx: click.Context) -> None:
    """Experiment management."""
    pass


def _get_experiment_manager(config):
    """Build ExperimentManager from ProjectConfig."""
    from easyml.experiments.manager import ExperimentManager

    exp_cfg = config.experiments
    if exp_cfg is None:
        click.echo("No experiments section in config.", err=True)
        raise SystemExit(1)

    return ExperimentManager(
        experiments_dir=exp_cfg.experiments_dir or "experiments",
        naming_pattern=exp_cfg.naming_pattern,
        log_path=exp_cfg.log_path,
        do_not_retry_path=exp_cfg.do_not_retry_path,
    )


@experiment.command("create")
@click.argument("experiment_id")
@click.pass_context
def experiment_create(ctx: click.Context, experiment_id: str) -> None:
    """Create a new experiment directory with an empty overlay."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    manager = _get_experiment_manager(result.config)
    try:
        exp_dir = manager.create(experiment_id)
        click.echo(f"Created experiment: {exp_dir}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@experiment.command("log")
@click.argument("experiment_id")
@click.option("--hypothesis", required=True, help="What the experiment tests.")
@click.option("--changes", required=True, help="What was changed.")
@click.option("--verdict", required=True, help="keep, revert, or partial.")
@click.option("--notes", default="", help="Additional notes.")
@click.pass_context
def experiment_log(
    ctx: click.Context,
    experiment_id: str,
    hypothesis: str,
    changes: str,
    verdict: str,
    notes: str,
) -> None:
    """Log an experiment with hypothesis, changes, and verdict."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    manager = _get_experiment_manager(result.config)
    try:
        manager.log(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            changes=changes,
            verdict=verdict,
            notes=notes,
        )
        click.echo(f"Logged experiment: {experiment_id}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@experiment.command("list")
@click.pass_context
def experiment_list(ctx: click.Context) -> None:
    """List experiment directories."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    exp_cfg = result.config.experiments
    if exp_cfg is None:
        click.echo("No experiments section in config.")
        return

    exp_dir = Path(exp_cfg.experiments_dir or "experiments")
    if not exp_dir.exists():
        click.echo("No experiments directory found.")
        return

    experiments = sorted(
        p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not experiments:
        click.echo("No experiments found.")
        return

    for name in experiments:
        click.echo(f"  {name}")


@experiment.command("run")
@click.argument("experiment_id")
@click.option("--ensemble-only", is_flag=True, help="Only re-ensemble, skip model training.")
@click.option("--primary-metric", default="brier", help="Metric for sweep ranking.")
@click.pass_context
def experiment_run(
    ctx: click.Context,
    experiment_id: str,
    ensemble_only: bool,
    primary_metric: str,
) -> None:
    """Run an experiment: load overlay, train, backtest, compare to baseline.

    If the overlay contains a sweep key, automatically runs a sweep
    across all variant configurations and ranks by primary metric.
    """
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    import yaml as _yaml

    from easyml.runner.experiment import (
        compute_deltas,
        detect_experiment_changes,
        format_change_summary,
        format_delta_table,
        format_sweep_summary,
        load_baseline_metrics,
        run_sweep,
    )
    from easyml.runner.pipeline import PipelineRunner

    result = _load_config(config_dir, gender)
    exp_cfg = result.config.experiments
    if exp_cfg is None:
        click.echo("No experiments section in config.", err=True)
        raise SystemExit(1)

    exp_dir = Path(exp_cfg.experiments_dir or "experiments") / experiment_id
    if not exp_dir.exists():
        click.echo(f"Experiment directory not found: {exp_dir}", err=True)
        raise SystemExit(1)

    overlay_path = exp_dir / "overlay.yaml"
    overlay = {}
    if overlay_path.exists():
        with open(overlay_path) as f:
            overlay = _yaml.safe_load(f) or {}

    # Route to sweep runner if overlay has a sweep key
    if "sweep" in overlay:
        click.echo(f"Sweep detected in overlay. Running sweep for {experiment_id}...")
        try:
            variant = "w" if gender.upper() == "W" else None
            experiments_dir = Path(exp_cfg.experiments_dir or "experiments")
            sweep_result = run_sweep(
                overlay_path=overlay_path,
                config_dir=Path(config_dir),
                project_dir=Path("."),
                experiments_dir=experiments_dir,
                experiment_id=experiment_id,
                variant=variant,
                primary_metric=primary_metric,
            )
            click.echo(format_sweep_summary(sweep_result))
        except Exception as exc:
            click.echo(f"Sweep {experiment_id} failed: {exc}", err=True)
            raise SystemExit(1)
        return

    # Detect changes between production config and overlay
    production_dict = result.config.model_dump()
    change_set = detect_experiment_changes(production_dict, overlay)
    click.echo(format_change_summary(change_set))

    if ensemble_only or change_set.ensemble_only:
        click.echo("Ensemble-only mode: skipping model training.")

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
        overlay=overlay,
    )

    try:
        runner.load()
        bt_result = runner.backtest()

        # Print markdown report if available, else raw JSON
        report = bt_result.get("report") if isinstance(bt_result, dict) else None
        if report:
            click.echo(report)
        else:
            click.echo(json.dumps(bt_result, indent=2, default=str))

        # Try to compare against baseline metrics
        data_cfg = result.config.data
        outputs_dir = data_cfg.outputs_dir
        if outputs_dir:
            baseline = load_baseline_metrics(Path(outputs_dir))
            if baseline and isinstance(bt_result, dict):
                deltas = compute_deltas(bt_result, baseline)
                if deltas:
                    click.echo("\n" + format_delta_table(bt_result, baseline, deltas))
    except Exception as exc:
        click.echo(f"Experiment {experiment_id} failed: {exc}", err=True)
        raise SystemExit(1)


# -----------------------------------------------------------------------
# run subgroup (stubs that will use PipelineRunner)
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run pipeline stages."""
    pass


@run.command("train")
@click.option("--run-id", default=None, help="Optional run identifier.")
@click.pass_context
def run_train(ctx: click.Context, run_id: str | None) -> None:
    """Train all active models."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    runner.load()
    result = runner.train(run_id=run_id)
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("backtest")
@click.option("--run-dir", default=None, help="Save artifacts to this run directory.")
@click.option("--json-output", is_flag=True, help="Output raw JSON instead of markdown.")
@click.pass_context
def run_backtest(ctx: click.Context, run_dir: str | None, json_output: bool) -> None:
    """Run backtesting across all configured seasons."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
        run_dir=run_dir,
    )
    runner.load()
    result = runner.backtest()

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        report = result.get("report")
        if report:
            click.echo(report)
        else:
            click.echo(json.dumps(result, indent=2, default=str))


@run.command("predict")
@click.option("--season", required=True, type=int, help="Target season to predict.")
@click.option("--run-id", default=None, help="Optional run identifier.")
@click.pass_context
def run_predict(ctx: click.Context, season: int, run_id: str | None) -> None:
    """Generate predictions for a target season."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    runner.load()
    preds = runner.predict(season, run_id=run_id)
    click.echo(f"Predicted {len(preds)} matchups for season {season}")


@run.command("pipeline")
@click.pass_context
def run_pipeline(ctx: click.Context) -> None:
    """Run the full pipeline: load, train, backtest."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    result = runner.run_full()
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("list")
@click.pass_context
def run_list(ctx: click.Context) -> None:
    """List all pipeline runs."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    from easyml.runner.run_manager import RunManager

    outputs_dir = result.config.data.outputs_dir
    if not outputs_dir:
        click.echo("No outputs_dir configured in pipeline.yaml.", err=True)
        raise SystemExit(1)

    mgr = RunManager(Path(outputs_dir))
    runs = mgr.list_runs()
    if not runs:
        click.echo("No runs found.")
        return

    for r in runs:
        marker = " (current)" if r["is_current"] else ""
        click.echo(f"  {r['run_id']}{marker}")


@run.command("promote")
@click.argument("run_id")
@click.pass_context
def run_promote(ctx: click.Context, run_id: str) -> None:
    """Promote a run to 'current' via symlink."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    from easyml.runner.run_manager import RunManager

    outputs_dir = result.config.data.outputs_dir
    if not outputs_dir:
        click.echo("No outputs_dir configured in pipeline.yaml.", err=True)
        raise SystemExit(1)

    mgr = RunManager(Path(outputs_dir))
    try:
        mgr.promote(run_id)
        click.echo(f"Promoted run: {run_id}")
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1)


# -----------------------------------------------------------------------
# serve (stub)
# -----------------------------------------------------------------------

@main.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    if result.config.server is None:
        click.echo("No server configuration found in server.yaml", err=True)
        raise SystemExit(1)

    from easyml.runner.server_gen import generate_server

    server = generate_server(
        result.config.server,
        Path(config_dir),
        guardrails=result.config.guardrails,
    )
    mcp = server.to_fastmcp()
    mcp.run()


# -----------------------------------------------------------------------
# init (stub)
# -----------------------------------------------------------------------

@main.command()
@click.argument("project_dir")
@click.option("--name", default=None, help="Project name (defaults to directory name).")
def init(project_dir: str, name: str | None) -> None:
    """Initialize a new easyml project."""
    from easyml.runner.scaffold import scaffold_project

    path = Path(project_dir)
    try:
        scaffold_project(path, project_name=name)
        click.echo(f"Project initialized at {path}")
        click.echo("Next steps:")
        click.echo(f"  1. cd {path}")
        click.echo("  2. Edit config/models.yaml to define your models")
        click.echo("  3. Add feature modules to features/")
        click.echo("  4. Run: easyml validate")
    except FileExistsError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
