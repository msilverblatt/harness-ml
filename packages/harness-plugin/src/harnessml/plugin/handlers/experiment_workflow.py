"""Experiment lifecycle workflow — enforces discipline via tool visibility.

The agent cannot start a new experiment without completing log_result on the
current one, because experiment.create is only visible when no workflow is active.

Flow: create → [write_overlay] → run → log_result → [promote | done]
"""
from __future__ import annotations

import re

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from protomcp.context import ToolContext
from protomcp.workflow import StepResult, step, workflow


@workflow(
    "experiment",
    description="Run a single ML experiment with discipline enforcement.",
    allow_during=[
        "notebook.*", "configure.*", "data.*", "features.*",
        "models.*", "pipeline.*",
    ],
)
class ExperimentWorkflow:
    def __init__(self):
        self.experiment_id = None
        self.project_dir = None

    @step(
        initial=True,
        next=["write_overlay", "run"],
        description="Create a new experiment. Requires description and hypothesis.",
    )
    def create(
        self,
        description: str,
        hypothesis: str,
        project_dir=None,
        parent_id: str = None,
        phase: str = "",
    ) -> StepResult:
        from harnessml.core.runner.config_writer.experiments import (
            _check_discipline,
            experiment_create,
        )

        self.project_dir = resolve_project_dir(project_dir)

        gate_err = _check_discipline(self.project_dir)
        if gate_err:
            raise ValueError(gate_err)

        result = experiment_create(
            self.project_dir,
            description,
            hypothesis=hypothesis,
            parent_id=parent_id,
            phase=phase or "",
        )

        match = re.search(r"(exp-\d+)", result)
        if match:
            self.experiment_id = match.group(1)

        return StepResult(result=result)

    @step(
        next=["run"],
        description="Write a YAML config overlay for the experiment.",
    )
    def write_overlay(self, overlay: str | dict, project_dir=None) -> StepResult:
        from harnessml.core.runner.config_writer.experiments import write_overlay

        proj = self.project_dir or resolve_project_dir(project_dir)
        parsed = parse_json_param(overlay)
        result = write_overlay(proj, self.experiment_id, parsed)
        return StepResult(result=result)

    @step(
        next=["log_result"],
        no_cancel=True,
        description="Run the experiment backtest.",
    )
    def run(
        self,
        primary_metric: str = "rmse",
        variant: str = None,
        baseline_run_id: str = None,
        project_dir=None,
        ctx: ToolContext = None,
    ) -> StepResult:
        from harnessml.core.runner.config_writer.experiments import run_experiment

        proj = self.project_dir or resolve_project_dir(project_dir)

        def _progress_callback(current, total, message):
            if ctx is not None:
                ctx.report_progress(current, total, message)

        if ctx is not None:
            ctx.report_progress(0, 1, f"Running experiment {self.experiment_id}...")

        result = run_experiment(
            proj,
            self.experiment_id,
            primary_metric=primary_metric,
            variant=variant,
            baseline_run_id=baseline_run_id,
            on_progress=_progress_callback,
        )

        if ctx is not None:
            ctx.report_progress(1, 1, "Experiment complete.")

        return StepResult(result=result)

    @step(
        next=["promote", "done"],
        description="Log experiment results. Requires conclusion and verdict.",
    )
    def log_result(
        self,
        conclusion: str,
        verdict: str = "inconclusive",
        description: str = "",
        hypothesis: str = "",
        project_dir=None,
    ) -> StepResult:
        from harnessml.core.runner.config_writer.experiments import log_experiment_result

        proj = self.project_dir or resolve_project_dir(project_dir)
        result = log_experiment_result(
            proj,
            self.experiment_id,
            conclusion=conclusion,
            verdict=verdict,
            description=description,
            hypothesis=hypothesis,
        )
        return StepResult(result=result)

    @step(terminal=True, description="Promote experiment config to production.")
    def promote(self, primary_metric: str = "rmse", project_dir=None) -> StepResult:
        from harnessml.core.runner.config_writer.experiments import promote_experiment

        proj = self.project_dir or resolve_project_dir(project_dir)
        result = promote_experiment(
            proj, self.experiment_id, primary_metric=primary_metric,
        )
        return StepResult(result=result)

    @step(terminal=True, description="Finish experiment without promoting.")
    def done(self) -> StepResult:
        return StepResult(
            result=f"Experiment `{self.experiment_id}` complete. Ready for next experiment.",
        )

    def on_cancel(self):
        return f"Experiment `{self.experiment_id}` cancelled."

    def on_complete(self):
        if self.project_dir:
            try:
                from harnessml.core.runner.experiments.journal import ExperimentJournal

                journal_path = self.project_dir / "experiments" / "journal.jsonl"
                if journal_path.exists():
                    j = ExperimentJournal(journal_path)
                    j.export_markdown(self.project_dir / "EXPERIMENT_LOG.md")
            except Exception:
                pass
