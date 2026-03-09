"""Experiment operations: create, overlay, run, log, promote."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from harnessml.core.runner.config_writer._helpers import (
    _expand_dot_keys,
    _get_config_dir,
    _load_yaml,
)


def experiment_create(
    project_dir: Path,
    description: str,
    *,
    hypothesis: str = "",
    parent_id: str | None = None,
    branching_reason: str = "",
    phase: str = "",
) -> str:
    """Create a new experiment directory with auto-generated ID."""
    experiments_dir = Path(project_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    from harnessml.core.runner.experiment import auto_next_id

    exp_id = auto_next_id(experiments_dir)

    if not hypothesis or not hypothesis.strip():
        return "**Error**: 'hypothesis' is required when creating an experiment. State what you expect and why."

    exp_dir = experiments_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write empty overlay
    overlay_path = exp_dir / "overlay.yaml"
    overlay_path.write_text(
        yaml.dump({"description": description}, default_flow_style=False)
    )

    # Write hypothesis
    (exp_dir / "hypothesis.txt").write_text(hypothesis)

    # Write to JSONL journal
    journal_path = experiments_dir / "journal.jsonl"
    try:
        from harnessml.core.runner.experiment_journal import ExperimentJournal
        from harnessml.core.runner.experiment_schema import (
            ExperimentRecord,
            ExperimentStatus,
        )
        journal = ExperimentJournal(journal_path)
        record = ExperimentRecord(
            experiment_id=exp_id,
            hypothesis=hypothesis,
            status=ExperimentStatus.CREATED,
            parent_id=parent_id,
            branching_reason=branching_reason,
            phase=phase,
        )
        journal.append(record)
    except Exception:
        pass  # Fail-safe: don't block experiment creation if journal write fails

    lines = [
        f"**Created experiment**: `{exp_id}`",
        f"- Directory: `{exp_dir}`",
        f"- Overlay: `{overlay_path}`",
        f"- Description: {description}",
    ]
    if parent_id:
        lines.append(f"- Parent: `{parent_id}`")
    if phase:
        lines.append(f"- Phase: {phase}")

    return "\n".join(lines)


def write_overlay(
    project_dir: Path,
    experiment_id: str,
    overlay: dict,
) -> str:
    """Write an overlay YAML to an experiment directory.

    Dot-notation keys (e.g. ``models.xgb_core.features``) are expanded
    into nested dicts before writing so that ``deep_merge`` can apply them.
    """
    experiments_dir = Path(project_dir) / "experiments"
    exp_dir = experiments_dir / experiment_id

    if not exp_dir.exists():
        return f"**Error**: Experiment directory not found: {exp_dir}"

    nested_overlay = _expand_dot_keys(overlay)
    overlay_path = exp_dir / "overlay.yaml"
    overlay_path.write_text(
        yaml.dump(nested_overlay, default_flow_style=False, sort_keys=False)
    )

    return (
        f"**Overlay written**: `{overlay_path}`\n"
        f"- Keys: {list(overlay.keys())}"
    )


def run_experiment(
    project_dir: Path,
    experiment_id: str,
    *,
    primary_metric: str = "brier",
    variant: str | None = None,
    on_progress=None,
) -> str:
    """Run a full experiment: backtest with overlay, compare to baseline.

    Steps:
    1. Load experiment overlay and run backtest
    2. Load baseline results (from last run or re-run without overlay)
    3. Compute deltas
    4. Auto-log results
    5. Return comprehensive comparison

    Returns
    -------
    str
        Markdown-formatted experiment results with comparison.
    """
    from harnessml.core.runner.config_writer.pipeline import _format_backtest_result

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    # Load experiment overlay
    exp_dir = project_dir / "experiments" / experiment_id
    if not exp_dir.exists():
        return f"**Error**: Experiment '{experiment_id}' not found."

    overlay_path = exp_dir / "overlay.yaml"
    overlay = _load_yaml(overlay_path) if overlay_path.exists() else {}

    # Load hypothesis
    hypothesis = ""
    hyp_path = exp_dir / "hypothesis.txt"
    if hyp_path.exists():
        hypothesis = hyp_path.read_text().strip()

    try:
        from harnessml.core.runner.pipeline import PipelineRunner

        # Run experiment backtest (with overlay)
        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
        )
        runner.load()
        exp_result = runner.backtest(on_progress=on_progress)

        # Run baseline backtest (without overlay)
        baseline_runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        baseline_runner.load()
        baseline_result = baseline_runner.backtest(on_progress=on_progress)

        # Build comparison
        exp_metrics = exp_result.get("metrics", {})
        base_metrics = baseline_result.get("metrics", {})

        lines = [f"## Experiment: `{experiment_id}`\n"]
        if hypothesis:
            lines.append(f"**Hypothesis**: {hypothesis}\n")

        # Overlay changes
        if overlay:
            lines.append("### Changes Applied\n")
            lines.append(f"```yaml\n{yaml.dump(overlay, default_flow_style=False)}```\n")

        # Delta table
        lines.append("### Results Comparison\n")
        lines.append("| Metric | Baseline | Experiment | Delta |")
        lines.append("|--------|----------|------------|-------|")

        _LOWER_IS_BETTER = {"brier", "brier_score", "ece", "log_loss"}

        overall_verdict = "neutral"
        primary_delta = 0.0

        for metric in sorted(set(list(exp_metrics.keys()) + list(base_metrics.keys()))):
            base_val = base_metrics.get(metric, float("nan"))
            exp_val = exp_metrics.get(metric, float("nan"))
            delta = exp_val - base_val

            base_str = "N/A" if isinstance(base_val, float) and math.isnan(base_val) else f"{base_val:.4f}"
            exp_str = "N/A" if isinstance(exp_val, float) and math.isnan(exp_val) else f"{exp_val:.4f}"
            delta_str = "N/A" if isinstance(delta, float) and math.isnan(delta) else f"{delta:+.4f}"

            lines.append(f"| {metric} | {base_str} | {exp_str} | {delta_str} |")

            if metric == primary_metric or metric == f"{primary_metric}_score":
                primary_delta = delta
                if metric in _LOWER_IS_BETTER:
                    overall_verdict = "improved" if delta < 0 else ("regressed" if delta > 0 else "neutral")
                else:
                    overall_verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "neutral")

        lines.append(f"\n**Verdict**: {overall_verdict} (primary metric: {primary_metric}, delta: {primary_delta:+.4f})")

        # Per-fold deltas
        exp_per_fold = exp_result.get("per_fold", {})
        base_per_fold = baseline_result.get("per_fold", {})
        if exp_per_fold and base_per_fold:
            common_folds = sorted(
                set(exp_per_fold.keys()) & set(base_per_fold.keys())
            )
            if common_folds:
                lines.append("\n### Per-Fold Deltas\n")
                lines.append("| Fold | Brier (base) | Brier (exp) | Delta | Acc (base) | Acc (exp) | Delta |")
                lines.append("|------|-------------|-------------|-------|------------|-----------|-------|")
                for s in common_folds:
                    eb = exp_per_fold[s].get("brier", exp_per_fold[s].get("brier_score"))
                    bb = base_per_fold[s].get("brier", base_per_fold[s].get("brier_score"))
                    ea = exp_per_fold[s].get("accuracy")
                    ba = base_per_fold[s].get("accuracy")
                    bd = (eb - bb) if eb is not None and bb is not None else None
                    ad = (ea - ba) if ea is not None and ba is not None else None
                    bb_s = f"{bb:.4f}" if bb is not None else "-"
                    eb_s = f"{eb:.4f}" if eb is not None else "-"
                    bd_s = f"{bd:+.4f}" if bd is not None else "-"
                    ba_s = f"{ba:.4f}" if ba is not None else "-"
                    ea_s = f"{ea:.4f}" if ea is not None else "-"
                    ad_s = f"{ad:+.4f}" if ad is not None else "-"
                    lines.append(f"| {s} | {bb_s} | {eb_s} | {bd_s} | {ba_s} | {ea_s} | {ad_s} |")

        # Experiment backtest details
        lines.append("\n---\n")
        lines.append(_format_backtest_result(exp_result))

        # Save results to experiment dir
        try:
            results = {
                "experiment_id": experiment_id,
                "metrics": exp_metrics,
                "baseline_metrics": base_metrics,
                "verdict": overall_verdict,
                "primary_metric": primary_metric,
                "primary_delta": primary_delta,
            }
            (exp_dir / "results.json").write_text(json.dumps(results, indent=2))

            # Auto-log
            from harnessml.core.runner.experiment import auto_log_result
            auto_log_result(
                log_path=project_dir / "EXPERIMENT_LOG.md",
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                changes=yaml.dump(overlay, default_flow_style=True) if overlay else "",
                metrics=exp_metrics,
                baseline_metrics=base_metrics,
                verdict=overall_verdict,
            )
        except Exception as log_exc:
            lines.append(f"\n*Warning: Failed to log results: {log_exc}*")

        return "\n".join(lines)

    except Exception as exc:
        return f"**Experiment failed**: {exc}"


def quick_run_experiment(
    project_dir: Path,
    description: str,
    overlay: str | dict,
    *,
    hypothesis: str = "",
    primary_metric: str = "brier",
    on_progress=None,
) -> str:
    """Create, configure, and run an experiment in a single call.

    Combines experiment_create + write_overlay + run_experiment.
    Returns the combined results or error at any step.
    """
    if not description:
        return "**Error**: 'description' is required for quick_run."
    if not hypothesis or not hypothesis.strip():
        return "**Error**: 'hypothesis' is required for quick_run. State what you expect and why."

    project_dir = Path(project_dir)

    # Step 1: Create experiment
    create_result = experiment_create(project_dir, description, hypothesis=hypothesis)
    if "Error" in create_result:
        return create_result

    # Extract experiment ID from create result
    import re
    id_match = re.search(r'(exp-\d+)', create_result, re.IGNORECASE)
    if not id_match:
        return f"**Error**: Could not extract experiment ID from creation result.\n\n{create_result}"
    experiment_id = id_match.group(1)

    # Step 2: Write overlay
    try:
        parsed_overlay = json.loads(overlay) if isinstance(overlay, str) else overlay
    except json.JSONDecodeError as e:
        return f"**Error**: Invalid overlay JSON: {e}"

    overlay_result = write_overlay(project_dir, experiment_id, parsed_overlay)
    if "Error" in overlay_result:
        return f"**Error** writing overlay:\n\n{overlay_result}"

    # Step 3: Run experiment
    run_result = run_experiment(
        project_dir,
        experiment_id,
        primary_metric=primary_metric,
        on_progress=on_progress,
    )

    # Combine results
    lines = [
        f"## Quick Run: {experiment_id}\n",
        f"**Description:** {description}",
    ]
    if hypothesis:
        lines.append(f"**Hypothesis:** {hypothesis}")
    lines.append(f"\n### Overlay\n\n{overlay_result}")
    lines.append(f"\n### Results\n\n{run_result}")

    return "\n".join(lines)


def log_experiment_result(
    project_dir: Path,
    experiment_id: str,
    *,
    description: str = "",
    hypothesis: str = "",
    conclusion: str = "",
    metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    overlay: dict | None = None,
    verdict: str = "",
) -> str:
    """Append an experiment result to the journal (JSONL format).

    Uses the structured ExperimentJournal when possible, falling back
    to raw JSONL append for backward compatibility.
    """
    project_dir = Path(project_dir)

    # Save conclusion.txt alongside hypothesis.txt
    if conclusion:
        exp_dir = project_dir / "experiments" / experiment_id
        if exp_dir.exists():
            (exp_dir / "conclusion.txt").write_text(conclusion)

    journal_path = project_dir / "experiments" / "journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from harnessml.core.runner.experiment_journal import ExperimentJournal
        from harnessml.core.runner.experiment_manager import ExperimentManager
        from harnessml.core.runner.experiment_schema import ExperimentStatus

        journal = ExperimentJournal(journal_path)
        existing = journal.get(experiment_id)

        if existing is not None:
            # Update existing record with conclusion
            mgr = ExperimentManager(
                experiments_dir=project_dir / "experiments",
                journal_path=journal_path,
            )
            if verdict:
                struct_conclusion = mgr._build_conclusion(
                    verdict=verdict,
                    learnings=conclusion,
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                )
                journal.update(
                    experiment_id,
                    status=ExperimentStatus.COMPLETED,
                    conclusion=struct_conclusion,
                )
            else:
                journal.update(
                    experiment_id,
                    status=ExperimentStatus.COMPLETED,
                )

            # Auto-regenerate markdown
            log_path = project_dir / "EXPERIMENT_LOG.md"
            log_path.write_text(journal.generate_markdown())

            return f"Logged experiment `{experiment_id}` to journal."
        else:
            # No existing record; create one from scratch
            from harnessml.core.runner.experiment_schema import ExperimentRecord
            hyp = hypothesis or description or "No hypothesis provided"
            record = ExperimentRecord(
                experiment_id=experiment_id,
                hypothesis=hyp,
                status=ExperimentStatus.COMPLETED,
            )
            journal.append(record)

            if verdict:
                mgr = ExperimentManager(
                    experiments_dir=project_dir / "experiments",
                    journal_path=journal_path,
                )
                struct_conclusion = mgr._build_conclusion(
                    verdict=verdict,
                    learnings=conclusion,
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                )
                journal.update(
                    experiment_id,
                    conclusion=struct_conclusion,
                )

            # Auto-regenerate markdown
            log_path = project_dir / "EXPERIMENT_LOG.md"
            log_path.write_text(journal.generate_markdown())

            return f"Logged experiment `{experiment_id}` to journal."

    except Exception:
        # Fallback to raw JSONL append
        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "description": description,
            "hypothesis": hypothesis,
            "conclusion": conclusion,
            "metrics": metrics or {},
            "verdict": verdict,
        }
        if overlay:
            entry["overlay_summary"] = str(overlay)[:200]

        with open(journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return f"Logged experiment `{experiment_id}` to journal."


def show_journal(project_dir: Path, *, last_n: int = 20) -> str:
    """Show the experiment journal as a markdown table."""
    project_dir = Path(project_dir)
    journal_path = project_dir / "experiments" / "journal.jsonl"

    if not journal_path.exists():
        return "No experiment journal found. Run experiments to build history."

    entries = []
    for line in journal_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))

    if not entries:
        return "Journal is empty."

    entries = entries[-last_n:]

    # Detect all metric keys across entries
    metric_keys = []
    for e in entries:
        for k in e.get("metrics", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    lines = ["## Experiment Journal\n"]
    header = "| # | ID | Description | " + " | ".join(metric_keys) + " | Verdict |"
    sep = "|---|----|-----------| " + " | ".join("------" for _ in metric_keys) + " |---------|"
    lines.extend([header, sep])

    _dash = "\u2014"
    for i, e in enumerate(entries, 1):
        metrics = e.get("metrics", {})
        vals = " | ".join(
            f"{metrics.get(k, _dash):.4f}" if isinstance(metrics.get(k), (int, float)) else str(metrics.get(k, _dash))
            for k in metric_keys
        )
        verdict = e.get("verdict", "")
        desc = e.get("description", "")[:40]
        lines.append(f"| {i} | {e['experiment_id']} | {desc} | {vals} | {verdict} |")

    return "\n".join(lines)


def compare_experiments(
    project_dir: Path,
    experiment_ids: list[str],
) -> str:
    """Compare two or more experiments using the JSONL journal.

    Returns a markdown table with side-by-side metrics comparison.
    """
    project_dir = Path(project_dir)
    journal_path = project_dir / "experiments" / "journal.jsonl"

    if not journal_path.exists():
        return "No experiment journal found. Run experiments first."

    try:
        from harnessml.core.runner.experiment_manager import ExperimentManager

        mgr = ExperimentManager(
            experiments_dir=project_dir / "experiments",
            journal_path=journal_path,
        )
        return mgr.compare(experiment_ids)
    except Exception as exc:
        return f"**Error** comparing experiments: {exc}"


def promote_experiment(
    project_dir: Path,
    experiment_id: str,
    *,
    primary_metric: str = "brier_score",
) -> str:
    """Promote a successful experiment's config changes to production.

    Returns markdown confirmation or rejection reason.
    """
    project_dir = Path(project_dir)

    try:
        from harnessml.core.runner.experiment import promote_experiment as _promote

        result = _promote(
            experiment_id=experiment_id,
            experiments_dir=project_dir / "experiments",
            config_dir=project_dir / "config",
            primary_metric=primary_metric,
        )

        if result.get("promoted"):
            lines = [f"## Promoted: `{experiment_id}`\n"]
            improvement = result.get("improvement", {})
            for metric, val in improvement.items():
                lines.append(f"- **{metric}**: improved by {val:+.4f}")
            changes = result.get("changes", [])
            if changes:
                lines.append("\n### Changes Applied\n")
                for c in changes:
                    lines.append(f"- {c}")
            warning = result.get("warning")
            if warning:
                lines.append(f"\n**Warning**: {warning}")
            return "\n".join(lines)
        else:
            reason = result.get("reason", "Unknown reason")
            return f"**Not promoted**: {reason}"

    except Exception as exc:
        return f"**Promotion failed**: {exc}"


def promote_exploration_trial(
    project_dir: Path,
    exploration_id: str,
    *,
    trial: int | None = None,
    primary_metric: str = "brier",
    hypothesis: str = "",
) -> str:
    """Promote a specific trial (or best trial) from an exploration run.

    Reads the trial overlay, creates a new exp-NNN experiment, runs it,
    and returns the results. No re-running of the backtest from scratch --
    the trial's overlay is simply applied as a new experiment.

    Parameters
    ----------
    exploration_id : str
        e.g. 'expl-002'
    trial : int | None
        Trial number to promote. If None, uses best_overlay.yaml (the
        best trial according to the exploration's primary metric).
    """
    project_dir = Path(project_dir)
    expl_dir = project_dir / "experiments" / exploration_id

    if not expl_dir.exists():
        return f"**Error**: Exploration '{exploration_id}' not found."

    if trial is not None:
        overlay_path = expl_dir / "trials" / f"trial-{trial:03d}" / "overlay.yaml"
        trial_label = f"trial {trial}"
    else:
        overlay_path = expl_dir / "best_overlay.yaml"
        trial_label = "best trial"

    if not overlay_path.exists():
        return f"**Error**: Overlay not found for {trial_label} in {exploration_id} ({overlay_path})."

    overlay = _load_yaml(overlay_path)
    if not overlay:
        return f"**Error**: Empty overlay for {trial_label} in {exploration_id}."

    description = f"Promote {trial_label} from {exploration_id}"
    if not hypothesis:
        hypothesis = f"Applying overlay from {exploration_id} {trial_label}."

    create_result = experiment_create(project_dir, description, hypothesis=hypothesis)
    if "Error" in create_result:
        return create_result

    import re
    id_match = re.search(r"(exp-\d+)", create_result, re.IGNORECASE)
    if not id_match:
        return f"**Error**: Could not extract experiment ID.\n\n{create_result}"
    experiment_id = id_match.group(1)

    overlay_result = write_overlay(project_dir, experiment_id, overlay)
    if "Error" in overlay_result:
        return overlay_result

    run_result = run_experiment(project_dir, experiment_id, primary_metric=primary_metric)

    return "\n".join([
        f"## Promote: `{exploration_id}` {trial_label} -> `{experiment_id}`\n",
        f"### Overlay\n\n{overlay_result}",
        f"\n### Results\n\n{run_result}",
    ])
