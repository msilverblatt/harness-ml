"""JSONL-based experiment journal — the single source of truth.

Append-only log where each line is a full ExperimentRecord snapshot.
When a record is updated, a new line is appended with the updated fields.
Reading always returns the latest snapshot per experiment_id.

EXPERIMENT_LOG.md is generated from this file, never hand-edited.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from harnessml.core.runner.experiments.schema import (
    ExperimentRecord,
    ExperimentStatus,
)


class ExperimentJournal:
    """Read/write interface for the JSONL experiment journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> None:
        """Append a record to the journal."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[ExperimentRecord]:
        """Read all records from the journal (raw, including duplicates)."""
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(ExperimentRecord.model_validate_json(line))
        return records

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """Get the latest snapshot of an experiment by ID."""
        latest = None
        for record in self.read_all():
            if record.experiment_id == experiment_id:
                latest = record
        return latest

    def update(self, experiment_id: str, **kwargs) -> ExperimentRecord | None:
        """Update an experiment by appending a new snapshot with changed fields.

        Reads the latest version, applies kwargs, and appends the updated record.
        """
        current = self.get(experiment_id)
        if current is None:
            return None
        data = current.model_dump()
        data.update(kwargs)
        data["updated_at"] = datetime.now(timezone.utc)
        updated = ExperimentRecord.model_validate(data)
        self.append(updated)
        return updated

    def list_by_status(self, status: ExperimentStatus) -> list[ExperimentRecord]:
        """List experiments with a given status (latest snapshot only)."""
        latest = self._latest_snapshots()
        return [r for r in latest.values() if r.status == status]

    def get_children(self, parent_id: str) -> list[ExperimentRecord]:
        """Get all experiments that have the given parent_id."""
        latest = self._latest_snapshots()
        return [r for r in latest.values() if r.parent_id == parent_id]

    def _latest_snapshots(self) -> dict[str, ExperimentRecord]:
        """Return {experiment_id: latest_record} for all experiments."""
        snapshots: dict[str, ExperimentRecord] = {}
        for record in self.read_all():
            snapshots[record.experiment_id] = record
        return snapshots

    def generate_markdown(self) -> str:
        """Generate EXPERIMENT_LOG.md content from the journal."""
        snapshots = self._latest_snapshots()
        if not snapshots:
            return "# Experiment Log\n\nNo experiments recorded.\n"

        lines = ["# Experiment Log\n"]
        lines.append("| ID | Hypothesis | Verdict | Primary Metric | Baseline | Result | Improvement | Phase |")
        lines.append("|---|---|---|---|---|---|---|---|")

        for exp_id in sorted(snapshots.keys()):
            r = snapshots[exp_id]
            c = r.conclusion
            if c:
                verdict = c.verdict.value
                metric = c.primary_metric
                baseline = f"{c.baseline_value:.4f}" if c.baseline_value is not None else "-"
                result = f"{c.result_value:.4f}" if c.result_value is not None else "-"
                improvement = f"{c.improvement_pct:+.2f}%" if c.improvement_pct is not None else "-"
            else:
                verdict = r.status.value
                metric = baseline = result = improvement = "-"

            lines.append(
                f"| {exp_id} | {r.hypothesis[:60]}{'...' if len(r.hypothesis) > 60 else ''} "
                f"| {verdict} | {metric} | {baseline} | {result} | {improvement} | {r.phase or '-'} |"
            )

        lines.append("")

        # Detailed entries
        for exp_id in sorted(snapshots.keys()):
            r = snapshots[exp_id]
            lines.append(f"\n## {exp_id}")
            lines.append(f"**Status:** {r.status.value}")
            lines.append(f"**Hypothesis:** {r.hypothesis}")
            if r.parent_id:
                lines.append(f"**Parent:** {r.parent_id}")
            if r.config_changes:
                lines.append("**Config Changes:**")
                for cc in r.config_changes:
                    lines.append(f"  - `{cc.path}`: {cc.old_value} -> {cc.new_value}")
            if r.trials:
                lines.append("**Trials:**")
                for t in r.trials:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in t.metrics.items())
                    lines.append(f"  - {t.trial_id}: {metrics_str}")
            if r.conclusion:
                c = r.conclusion
                lines.append(f"**Verdict:** {c.verdict.value}")
                if c.learnings:
                    lines.append(f"**Learnings:** {c.learnings}")
                if c.next_steps:
                    lines.append("**Next Steps:**")
                    for ns in c.next_steps:
                        lines.append(f"  - {ns}")
            lines.append("")

        return "\n".join(lines)
