"""Experiment management — creation, change detection, do-not-retry, logging, promotion."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from harnessml.core.config.merge import deep_merge, resolve_feature_mutations
from harnessml.core.runner.experiment_schema import StructuredConclusion
from harnessml.core.schemas.contracts import GuardrailViolation

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ExperimentError(Exception):
    """Exception raised by experiment management operations.

    Wraps a GuardrailViolation model with the violation details.
    """

    def __init__(self, violation: GuardrailViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


# ---------------------------------------------------------------------------
# Change report
# ---------------------------------------------------------------------------

@dataclass
class ChangeReport:
    """Summary of what an overlay changes relative to production config."""

    changed_models: list[str] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    removed_models: list[str] = field(default_factory=list)
    ensemble_changes: list[str] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return (
            len(self.changed_models)
            + len(self.new_models)
            + len(self.removed_models)
            + len(self.ensemble_changes)
        )


# ---------------------------------------------------------------------------
# Do-not-retry entry
# ---------------------------------------------------------------------------

@dataclass
class DoNotRetryEntry:
    """One pattern that should block future experiments."""

    pattern: str
    reference: str
    reason: str


# ---------------------------------------------------------------------------
# Experiment manager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """Manages the experiment lifecycle: create, detect changes, log, promote.

    Parameters
    ----------
    experiments_dir:
        Root directory where experiment subdirectories are created.
    naming_pattern:
        Optional regex that experiment IDs must match.
    log_path:
        Path to the experiment log markdown file.  When set, mandatory
        logging enforcement is enabled.
    journal_path:
        Path to the JSONL journal file.  When set, experiment records
        are written to/read from this journal.  The markdown log_path
        is auto-generated from it.
    baseline_metrics:
        Optional dict of baseline metric values for comparison.
    do_not_retry_path:
        Path to a JSON file storing do-not-retry patterns.  Loaded on
        init if it exists; written on every ``add_do_not_retry`` call.
    verdict_thresholds:
        Optional dict of metric_name -> minimum improvement threshold.
    """

    def __init__(
        self,
        experiments_dir: str | Path,
        naming_pattern: str | None = None,
        log_path: str | Path | None = None,
        journal_path: str | Path | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        do_not_retry_path: str | Path | None = None,
        verdict_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.naming_pattern = re.compile(naming_pattern) if naming_pattern else None
        self.log_path = Path(log_path) if log_path else None
        self.baseline_metrics = baseline_metrics or {}
        self.do_not_retry_path = Path(do_not_retry_path) if do_not_retry_path else None
        self.verdict_thresholds = verdict_thresholds or {}

        # Set up JSONL journal if path is provided
        self.journal = None
        if journal_path:
            from harnessml.core.runner.experiment_journal import ExperimentJournal
            self.journal = ExperimentJournal(journal_path)

        # In-memory do-not-retry list, loaded from disk if available
        self._do_not_retry: list[DoNotRetryEntry] = []
        if self.do_not_retry_path and self.do_not_retry_path.exists():
            self._load_do_not_retry()

    # ------------------------------------------------------------------
    # 6.1  Experiment creation + naming validation
    # ------------------------------------------------------------------

    def create(
        self,
        experiment_id: str,
        *,
        hypothesis: str = "",
        parent_id: str | None = None,
        branching_reason: str = "",
        phase: str = "",
    ) -> Path:
        """Create a new experiment directory with an empty overlay.

        Validates the naming pattern, checks for duplicates, and enforces
        mandatory logging of prior experiments.

        Returns the experiment directory path.
        """
        # Enforce mandatory logging before allowing a new experiment
        if (self.log_path is not None or self.journal is not None) and self.has_unlogged():
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="mandatory_logging",
                    message=(
                        "Previous experiment(s) have not been logged. "
                        "Log all experiments before creating a new one."
                    ),
                    source="ExperimentManager.create",
                )
            )

        # Validate naming pattern
        if self.naming_pattern and not self.naming_pattern.match(experiment_id):
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="naming",
                    message=(
                        f"Experiment ID '{experiment_id}' does not match "
                        f"naming pattern: {self.naming_pattern.pattern}"
                    ),
                    source="ExperimentManager.create",
                )
            )

        # Check for duplicates
        exp_dir = self.experiments_dir / experiment_id
        if exp_dir.exists():
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="duplicate",
                    message=f"Experiment '{experiment_id}' already exists at {exp_dir}",
                    source="ExperimentManager.create",
                )
            )

        # Create directory and empty overlay
        exp_dir.mkdir(parents=True)
        overlay_path = exp_dir / "overlay.yaml"
        overlay_path.write_text("# Overlay config — only what differs from production\n")

        # Write hypothesis file if provided
        if hypothesis:
            (exp_dir / "hypothesis.txt").write_text(hypothesis)

        # Write to JSONL journal if configured
        if self.journal is not None and hypothesis:
            from harnessml.core.runner.experiment_schema import (
                ExperimentRecord,
                ExperimentStatus,
            )
            record = ExperimentRecord(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                status=ExperimentStatus.CREATED,
                parent_id=parent_id,
                branching_reason=branching_reason,
                phase=phase,
            )
            self.journal.append(record)

        return exp_dir

    # ------------------------------------------------------------------
    # 6.2  Change detection + single-variable enforcement
    # ------------------------------------------------------------------

    def detect_changes(
        self,
        production: dict[str, Any],
        overlay: dict[str, Any],
    ) -> ChangeReport:
        """Compare an overlay against production config to detect changes.

        Parameters
        ----------
        production:
            The current production configuration dict.
        overlay:
            The experiment overlay dict (same shape, only overrides).

        Returns
        -------
        ChangeReport
            Summary of changed, new, and removed models + ensemble changes.
        """
        report = ChangeReport()

        prod_models = production.get("models", {})
        overlay_models = overlay.get("models", {})

        # Detect changed models
        for name in overlay_models:
            if name in prod_models:
                if overlay_models[name] != prod_models[name]:
                    report.changed_models.append(name)
            else:
                report.new_models.append(name)

        # Detect removed models
        for name in prod_models:
            if name not in overlay_models and overlay_models:
                # Only count as removed if the overlay specifies models
                # (an empty overlay models section means no model changes)
                pass
        # Explicitly check for removals: models in production but not in
        # overlay, when the overlay has a "models" key
        if "models" in overlay:
            for name in prod_models:
                if name not in overlay_models:
                    report.removed_models.append(name)

        # Detect ensemble changes
        prod_ensemble = production.get("ensemble", {})
        overlay_ensemble = overlay.get("ensemble", {})
        if overlay_ensemble:
            for key in overlay_ensemble:
                if key in prod_ensemble:
                    if overlay_ensemble[key] != prod_ensemble[key]:
                        report.ensemble_changes.append(key)
                else:
                    report.ensemble_changes.append(key)

        return report

    # ------------------------------------------------------------------
    # 6.3  Do-not-retry registry
    # ------------------------------------------------------------------

    def add_do_not_retry(
        self,
        pattern: str,
        reference: str,
        reason: str,
    ) -> None:
        """Register a pattern that should block future experiments.

        Parameters
        ----------
        pattern:
            Substring pattern to match against experiment descriptions.
        reference:
            Experiment ID that established this pattern (e.g. "EXP-002").
        reason:
            Why this pattern should not be retried.
        """
        entry = DoNotRetryEntry(pattern=pattern, reference=reference, reason=reason)
        self._do_not_retry.append(entry)
        if self.do_not_retry_path:
            self._save_do_not_retry()

    def check_do_not_retry(self, description: str) -> None:
        """Check a description against all do-not-retry patterns.

        Raises ExperimentError if any pattern matches (case-insensitive
        substring match).
        """
        desc_lower = description.lower()
        for entry in self._do_not_retry:
            if entry.pattern.lower() in desc_lower:
                raise ExperimentError(
                    GuardrailViolation(
                        blocked=True,
                        rule="do_not_retry",
                        message=(
                            f"Blocked by do-not-retry pattern: '{entry.pattern}' "
                            f"(ref: {entry.reference}, reason: {entry.reason})"
                        ),
                        source="ExperimentManager.check_do_not_retry",
                    )
                )

    def _save_do_not_retry(self) -> None:
        """Persist do-not-retry entries to JSON."""
        data = [
            {"pattern": e.pattern, "reference": e.reference, "reason": e.reason}
            for e in self._do_not_retry
        ]
        self.do_not_retry_path.write_text(json.dumps(data, indent=2))

    def _load_do_not_retry(self) -> None:
        """Load do-not-retry entries from JSON."""
        data = json.loads(self.do_not_retry_path.read_text())
        self._do_not_retry = [
            DoNotRetryEntry(
                pattern=entry["pattern"],
                reference=entry["reference"],
                reason=entry["reason"],
            )
            for entry in data
        ]

    # ------------------------------------------------------------------
    # 6.4  Experiment logging + mandatory logging enforcement
    # ------------------------------------------------------------------

    def log(
        self,
        experiment_id: str,
        hypothesis: str,
        changes: str,
        verdict: str,
        conclusion: str = "",
        notes: str = "",
        metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        """Append a log entry for an experiment to the log file.

        Parameters
        ----------
        experiment_id:
            The experiment identifier.
        hypothesis:
            What the experiment expected to show.
        changes:
            Brief description of what was changed.
        verdict:
            One of "keep", "revert", "partial", or "inconclusive".
        conclusion:
            What was learned from this experiment.
        notes:
            Optional additional observations.
        metrics:
            Experiment metrics dict (e.g. {"brier": 0.13, "accuracy": 0.82}).
        baseline_metrics:
            Baseline metrics for computing improvement.
        """
        if self.log_path is None and self.journal is None:
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="log_path",
                    message="No log_path configured — cannot log experiments.",
                    source="ExperimentManager.log",
                )
            )

        # Update JSONL journal if configured
        if self.journal is not None:
            struct_conclusion = self._build_conclusion(
                verdict=verdict,
                learnings=conclusion,
                metrics=metrics,
                baseline_metrics=baseline_metrics,
            )
            config_changes = self._extract_config_changes(experiment_id)

            from harnessml.core.runner.experiment_schema import ExperimentStatus
            self.journal.update(
                experiment_id,
                status=ExperimentStatus.COMPLETED,
                conclusion=struct_conclusion,
                config_changes=config_changes,
            )

            # Auto-regenerate markdown from journal
            if self.log_path:
                self.log_path.write_text(self.journal.generate_markdown())

            # Write conclusion.txt for backward compat
            exp_dir = self.experiments_dir / experiment_id
            if exp_dir.exists() and (conclusion or struct_conclusion.learnings):
                (exp_dir / "conclusion.txt").write_text(
                    conclusion or struct_conclusion.learnings
                )
        else:
            # Legacy markdown-only path
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            entry_lines = [
                f"## {experiment_id}",
                f"**Date:** {timestamp}",
                f"**Hypothesis:** {hypothesis}",
                f"**Changes:** {changes}",
                f"**Verdict:** {verdict}",
            ]
            if conclusion:
                entry_lines.append(f"**Conclusion:** {conclusion}")
            if notes:
                entry_lines.append(f"**Notes:** {notes}")
            entry_lines.append("")  # trailing newline

            entry_text = "\n".join(entry_lines) + "\n"

            # Append to log file (create if needed)
            with open(self.log_path, "a") as fh:
                fh.write(entry_text)

    def has_unlogged(self) -> bool:
        """Check if any created experiment lacks a log entry.

        An experiment is "created" if its directory exists under
        experiments_dir.  An experiment is "logged" if its ID appears in
        the log file or has a completed/failed status in the journal.
        """
        # Journal-based check
        if self.journal is not None:
            from harnessml.core.runner.experiment_schema import ExperimentStatus
            snapshots = self.journal._latest_snapshots()
            created_dirs = self._list_created_experiments()
            for exp_id in created_dirs:
                if exp_id in snapshots:
                    record = snapshots[exp_id]
                    if record.status in (ExperimentStatus.CREATED, ExperimentStatus.RUNNING):
                        return True
                else:
                    # Directory exists but no journal entry - unlogged
                    return True
            return False

        # Legacy markdown-based check
        if self.log_path is None:
            return False

        created = self._list_created_experiments()
        if not created:
            return False

        logged = self._list_logged_experiments()
        return bool(created - logged)

    def _list_created_experiments(self) -> set[str]:
        """Return the set of experiment IDs that have directories."""
        if not self.experiments_dir.exists():
            return set()
        return {
            p.name
            for p in self.experiments_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        }

    def _list_logged_experiments(self) -> set[str]:
        """Return the set of experiment IDs mentioned in the log file."""
        if self.log_path is None or not self.log_path.exists():
            return set()

        logged: set[str] = set()
        content = self.log_path.read_text()
        # Match markdown headings like "## exp-001-test"
        for match in re.finditer(r"^## (.+)$", content, re.MULTILINE):
            logged.add(match.group(1).strip())
        return logged

    # ------------------------------------------------------------------
    # Structured conclusion builder
    # ------------------------------------------------------------------

    # Metrics where lower is better
    _LOWER_IS_BETTER = {"brier", "brier_score", "log_loss", "ece", "mae", "mse", "rmse"}

    def _build_conclusion(
        self,
        verdict: str,
        learnings: str,
        metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> StructuredConclusion:
        """Build a StructuredConclusion from raw inputs."""
        from harnessml.core.runner.experiment_schema import (
            ExperimentVerdict,
            StructuredConclusion,
        )

        baseline = baseline_metrics or self.baseline_metrics or {}
        metrics = metrics or {}

        # Determine primary metric (first metric in baseline, or first in metrics)
        primary = ""
        baseline_val = None
        result_val = None
        improvement = None
        improvement_pct = None

        if baseline:
            primary = next(iter(baseline))
            baseline_val = baseline.get(primary)
            result_val = metrics.get(primary)
            if baseline_val is not None and result_val is not None and baseline_val != 0:
                lower_is_better = primary in self._LOWER_IS_BETTER
                if lower_is_better:
                    improvement = baseline_val - result_val
                else:
                    improvement = result_val - baseline_val
                improvement_pct = (improvement / abs(baseline_val)) * 100

        return StructuredConclusion(
            verdict=ExperimentVerdict(verdict),
            primary_metric=primary,
            baseline_value=baseline_val,
            result_value=result_val,
            improvement=improvement,
            improvement_pct=improvement_pct,
            secondary_metrics={k: v for k, v in metrics.items() if k != primary},
            learnings=learnings,
        )

    def _extract_config_changes(self, experiment_id: str) -> list:
        """Extract config changes from the experiment overlay."""

        exp_dir = self.experiments_dir / experiment_id
        overlay_path = exp_dir / "overlay.yaml"
        if not overlay_path.exists():
            return []

        try:
            overlay = yaml.safe_load(overlay_path.read_text()) or {}
        except yaml.YAMLError:
            return []

        changes = []
        _extract_paths(overlay, "", changes)
        return changes

    # ------------------------------------------------------------------
    # Experiment comparison
    # ------------------------------------------------------------------

    def compare(self, experiment_ids: list[str]) -> str:
        """Generate a side-by-side comparison of experiments as markdown."""
        if self.journal is None:
            return "Journal not configured. Cannot compare experiments."

        records = []
        for eid in experiment_ids:
            r = self.journal.get(eid)
            if r is None:
                return f"Experiment '{eid}' not found in journal."
            records.append(r)

        # Collect all metrics across experiments
        all_metrics: set[str] = set()
        for r in records:
            if r.conclusion:
                if r.conclusion.primary_metric:
                    all_metrics.add(r.conclusion.primary_metric)
                all_metrics.update(r.conclusion.secondary_metrics.keys())
            for t in r.trials:
                all_metrics.update(t.metrics.keys())

        # Build comparison table
        lines = ["## Experiment Comparison\n"]
        header = "| Metric | " + " | ".join(r.experiment_id for r in records) + " |"
        sep = "|---|" + "|".join("---" for _ in records) + "|"
        lines.extend([header, sep])

        # Hypothesis row
        lines.append("| **Hypothesis** | " + " | ".join(
            r.hypothesis[:50] + "..." if len(r.hypothesis) > 50 else r.hypothesis
            for r in records
        ) + " |")

        # Verdict row
        lines.append("| **Verdict** | " + " | ".join(
            r.conclusion.verdict.value if r.conclusion else "-"
            for r in records
        ) + " |")

        # Metric rows
        for metric in sorted(all_metrics):
            values = []
            for r in records:
                val = None
                if r.conclusion:
                    if r.conclusion.primary_metric == metric:
                        val = r.conclusion.result_value
                    else:
                        val = r.conclusion.secondary_metrics.get(metric)
                if val is None and r.trials:
                    val = r.trials[-1].metrics.get(metric)
                values.append(f"{val:.4f}" if val is not None else "-")

            lines.append(f"| {metric} | " + " | ".join(values) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        production_config_path: str | Path,
        backup_path: str | Path,
    ) -> None:
        """Restore production config from a backup file."""
        production_config_path = Path(production_config_path)
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="rollback_backup_missing",
                    message=f"Backup file not found: {backup_path}",
                    source="ExperimentManager.rollback",
                )
            )
        shutil.copy2(backup_path, production_config_path)

    # ------------------------------------------------------------------
    # 6.5  Atomic promote with rollback
    # ------------------------------------------------------------------

    def promote(
        self,
        experiment_id: str,
        production_config_path: str | Path,
    ) -> Path:
        """Promote an experiment's overlay into the production config.

        Requirements:
        1. Experiment must be logged with a "keep" or "partial" verdict.
        2. Creates a timestamped backup of the production config.
        3. Deep merges the overlay into production config.
        4. Writes atomically (write to temp, rename).

        Returns the backup path.
        """
        production_config_path = Path(production_config_path)

        # Check that the experiment has been logged
        logged_verdicts = self._get_logged_verdicts()
        if experiment_id not in logged_verdicts:
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="promote_requires_verdict",
                    message=(
                        f"Experiment '{experiment_id}' has no logged verdict. "
                        "Log the experiment before promoting."
                    ),
                    source="ExperimentManager.promote",
                )
            )

        verdict = logged_verdicts[experiment_id]
        if verdict not in ("keep", "partial"):
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="promote_requires_keep",
                    message=(
                        f"Experiment '{experiment_id}' has verdict '{verdict}'. "
                        "Only experiments with 'keep' or 'partial' verdict can "
                        "be promoted."
                    ),
                    source="ExperimentManager.promote",
                )
            )

        # Read the overlay
        overlay_path = self.experiments_dir / experiment_id / "overlay.yaml"
        if not overlay_path.exists():
            raise ExperimentError(
                GuardrailViolation(
                    blocked=True,
                    rule="promote_overlay_missing",
                    message=f"Overlay not found at {overlay_path}",
                    source="ExperimentManager.promote",
                )
            )
        overlay = yaml.safe_load(overlay_path.read_text()) or {}

        # Read the current production config
        prod_config = yaml.safe_load(production_config_path.read_text()) or {}

        # Create backup with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = production_config_path.parent / (
            f"{production_config_path.name}.bak.{timestamp}"
        )
        shutil.copy2(production_config_path, backup_path)

        # Deep merge overlay into production
        merged = deep_merge(prod_config, overlay)
        merged = resolve_feature_mutations(merged)

        # Write atomically: write to temp file in same directory, then rename
        parent_dir = production_config_path.parent
        fd, tmp_path_str = tempfile.mkstemp(
            dir=parent_dir, suffix=".yaml.tmp", prefix=".promote_"
        )
        try:
            tmp_path = Path(tmp_path_str)
            with open(fd, "w") as fh:
                yaml.dump(merged, fh, default_flow_style=False, sort_keys=False)
            tmp_path.rename(production_config_path)
        except Exception:
            # Clean up temp file on failure
            Path(tmp_path_str).unlink(missing_ok=True)
            raise

        return backup_path

    def _get_logged_verdicts(self) -> dict[str, str]:
        """Parse the log file and return {experiment_id: verdict} mapping.

        Checks both JSONL journal and markdown log file.
        """
        verdicts: dict[str, str] = {}

        # Check journal first
        if self.journal is not None:
            snapshots = self.journal._latest_snapshots()
            for exp_id, record in snapshots.items():
                if record.conclusion:
                    verdicts[exp_id] = record.conclusion.verdict.value

        # Fall back to/augment with markdown parsing
        if self.log_path is not None and self.log_path.exists():
            content = self.log_path.read_text()
            current_id: str | None = None
            for line in content.splitlines():
                heading_match = re.match(r"^## (.+)$", line)
                if heading_match:
                    current_id = heading_match.group(1).strip()
                    continue

                if current_id is not None:
                    verdict_match = re.match(r"\*\*Verdict:\*\*\s*(.+)", line)
                    if verdict_match:
                        # Don't overwrite journal verdicts
                        if current_id not in verdicts:
                            verdicts[current_id] = verdict_match.group(1).strip()
                        current_id = None

        return verdicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_paths(obj: dict, prefix: str, changes: list) -> None:
    """Recursively extract dot-notation paths from a dict into ConfigChange objects."""
    from harnessml.core.runner.experiment_schema import ConfigChange

    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _extract_paths(value, path, changes)
        else:
            changes.append(ConfigChange(path=path, new_value=value))
