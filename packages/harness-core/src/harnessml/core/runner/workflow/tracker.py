"""Workflow phase tracking for methodical ML exploration.

Analyzes experiment history and project state to determine which exploration
phases have been completed.  Supports soft warnings (default) and hard gates
(when ``enforce_phases`` is True in pipeline config).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from harnessml.core.schemas.contracts import GuardrailViolation


class WorkflowGateError(Exception):
    """Raised when a workflow phase gate blocks an action."""

    def __init__(self, violation: GuardrailViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MIN_MODEL_TYPES = 4
_DEFAULT_REQUIRE_FEATURE_DISCOVERY = True

# Model types recognized in HarnessML
_ALL_MODEL_TYPES = {
    "xgboost", "xgboost_regression", "lightgbm", "catboost",
    "random_forest", "logistic_regression", "elastic_net",
    "mlp", "tabnet",
}

# Broad categories for diversity counting
_MODEL_CATEGORIES = {
    "xgboost": "boosted_tree",
    "xgboost_regression": "boosted_tree",
    "lightgbm": "boosted_tree",
    "catboost": "boosted_tree",
    "random_forest": "bagging",
    "logistic_regression": "linear",
    "elastic_net": "linear",
    "mlp": "neural",
    "tabnet": "neural",
}


# ---------------------------------------------------------------------------
# WorkflowStatus dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkflowStatus:
    """Snapshot of exploration progress across all phases."""

    # Phase 1: EDA & Feature Discovery
    feature_discovery_run: bool = False
    auto_search_run: bool = False

    # Phase 2: Model Diversity
    model_types_tried: list[str] = field(default_factory=list)
    model_categories_tried: list[str] = field(default_factory=list)
    active_model_count: int = 0
    diversity_analysis_run: bool = False
    baseline_established: bool = False

    # Phase 3: Feature Engineering
    total_experiments_run: int = 0
    feature_experiments_run: int = 0

    # Phase 4: Tuning
    tuning_experiments_run: int = 0

    # Config thresholds
    min_model_types: int = _DEFAULT_MIN_MODEL_TYPES
    require_feature_discovery: bool = _DEFAULT_REQUIRE_FEATURE_DISCOVERY

    @property
    def phase1_complete(self) -> bool:
        if self.require_feature_discovery:
            return self.feature_discovery_run
        return True

    @property
    def phase2_complete(self) -> bool:
        return (
            len(self.model_categories_tried) >= self.min_model_types
            and self.baseline_established
        )

    @property
    def ready_for_tuning(self) -> bool:
        return self.phase1_complete and self.phase2_complete

    def warnings(self) -> list[str]:
        """Return a list of warnings about incomplete phases."""
        warns: list[str] = []
        if not self.feature_discovery_run:
            warns.append(
                "Feature discovery has not been run. "
                "Use features(action=\"discover\") to understand correlations before modeling."
            )
        if not self.auto_search_run:
            warns.append(
                "Auto-search has not been run. "
                "Use features(action=\"auto_search\") to find interaction/lag/rolling candidates."
            )
        if not self.baseline_established:
            warns.append(
                "No baseline has been established. "
                "Run a backtest with your initial model(s) to set a baseline."
            )
        n_cat = len(self.model_categories_tried)
        if n_cat < self.min_model_types:
            tried_str = ", ".join(sorted(self.model_categories_tried)) if self.model_categories_tried else "none"
            untried = sorted(
                {cat for cat in set(_MODEL_CATEGORIES.values())}
                - set(self.model_categories_tried)
            )
            untried_str = ", ".join(untried) if untried else "none"
            warns.append(
                f"Only {n_cat} model category(ies) tried ({tried_str}). "
                f"Minimum is {self.min_model_types}. "
                f"Consider exploring: {untried_str}."
            )
        if not self.diversity_analysis_run and self.active_model_count >= 2:
            warns.append(
                "Diversity analysis has not been run. "
                "Use features(action=\"diversity\") to check prediction correlations."
            )
        return warns

    def format_markdown(self) -> str:
        """Format status as a markdown progress report."""
        def _check(done: bool) -> str:
            return "[x]" if done else "[ ]"

        lines = ["## Workflow Progress\n"]

        # Phase 1
        lines.append("### Phase 1: EDA & Feature Discovery")
        lines.append(f"- {_check(self.feature_discovery_run)} Feature discovery")
        lines.append(f"- {_check(self.auto_search_run)} Auto-search")
        lines.append("")

        # Phase 2
        lines.append("### Phase 2: Model Diversity")
        lines.append(f"- {_check(self.baseline_established)} Baseline established")
        types_str = ", ".join(sorted(self.model_types_tried)) if self.model_types_tried else "none"
        cats_str = ", ".join(sorted(self.model_categories_tried)) if self.model_categories_tried else "none"
        lines.append(
            f"- {_check(len(self.model_categories_tried) >= self.min_model_types)} "
            f"Model categories tried: {cats_str} "
            f"({len(self.model_categories_tried)}/{self.min_model_types} required)"
        )
        lines.append(f"- Model types used: {types_str}")
        lines.append(f"- {_check(self.diversity_analysis_run)} Diversity analysis")
        lines.append(f"- Active models in ensemble: {self.active_model_count}")
        lines.append("")

        # Phase 3
        lines.append("### Phase 3: Feature Engineering")
        lines.append(f"- Total experiments run: {self.total_experiments_run}")
        lines.append(f"- Feature-related experiments: {self.feature_experiments_run}")
        lines.append("")

        # Phase 4
        lines.append("### Phase 4: Hyperparameter Tuning")
        ready = "Ready" if self.ready_for_tuning else "NOT ready"
        lines.append(f"- Status: **{ready}** for tuning")
        lines.append(f"- Tuning experiments run: {self.tuning_experiments_run}")
        lines.append("")

        # Warnings
        warns = self.warnings()
        if warns:
            lines.append("### Warnings\n")
            for w in warns:
                lines.append(f"- {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WorkflowTracker
# ---------------------------------------------------------------------------

class WorkflowTracker:
    """Analyzes project state to determine workflow phase completion.

    Parameters
    ----------
    project_dir:
        Root project directory containing config/, experiments/, etc.
    workflow_config:
        Optional workflow section from pipeline.yaml with settings like
        ``enforce_phases``, ``min_model_types``, ``require_feature_discovery``.
    """

    def __init__(
        self,
        project_dir: str | Path,
        workflow_config: dict[str, Any] | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.config = workflow_config or {}

    def get_status(self) -> WorkflowStatus:
        """Analyze project state and return current workflow status."""
        status = WorkflowStatus(
            min_model_types=self.config.get(
                "min_model_types", _DEFAULT_MIN_MODEL_TYPES
            ),
            require_feature_discovery=self.config.get(
                "require_feature_discovery", _DEFAULT_REQUIRE_FEATURE_DISCOVERY
            ),
        )

        self._check_feature_discovery(status)
        self._check_models(status)
        self._check_experiments(status)
        self._check_baseline(status)

        return status

    def check_ready_for_tuning(self, *, enforce: bool = False) -> str | None:
        """Check if the project is ready for hyperparameter tuning.

        Returns
        -------
        str | None
            None if ready. Warning string if not ready and enforce=False.
            Raises GuardrailViolation if not ready and enforce=True.
        """
        status = self.get_status()
        if status.ready_for_tuning:
            return None

        warns = status.warnings()
        message = (
            "Workflow phases incomplete — exploration recommended before tuning:\n"
            + "\n".join(f"  - {w}" for w in warns)
        )

        if enforce:
            raise WorkflowGateError(
                GuardrailViolation(
                    blocked=True,
                    rule="workflow_phase_gate",
                    message=message,
                    source="WorkflowTracker.check_ready_for_tuning",
                )
            )

        return f"**Warning**: {message}"

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _load_notebook_entries(self) -> list[dict]:
        """Load notebook entries from JSONL, returning latest snapshot per ID."""
        notebook_path = self.project_dir / "notebook" / "entries.jsonl"
        if not notebook_path.exists():
            return []
        snapshots: dict[str, dict] = {}
        for line in notebook_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_id = entry.get("id")
            if entry_id and not entry.get("struck"):
                snapshots[entry_id] = entry
        return sorted(snapshots.values(), key=lambda e: e.get("timestamp", ""))

    def _check_feature_discovery(self, status: WorkflowStatus) -> None:
        """Check if feature discovery and auto-search have been run.

        Checks notebook phase_transition entries first, then falls back to
        structural checks (feature cache existence).
        """
        # Check notebook for explicit phase transitions
        for entry in self._load_notebook_entries():
            content = entry.get("content", "").lower()
            entry_type = entry.get("type", "")
            if entry_type == "phase_transition":
                if any(kw in content for kw in ["feature discovery", "eda", "exploration"]):
                    status.feature_discovery_run = True
                if any(kw in content for kw in ["auto search", "auto_search", "interaction search"]):
                    status.auto_search_run = True

        # Structural fallback: feature cache existence
        features_dir = self.project_dir / "data" / "features"
        if features_dir.exists():
            cache_dir = features_dir / "cache"
            if cache_dir.exists() and any(cache_dir.iterdir()):
                status.feature_discovery_run = True

    def _check_models(self, status: WorkflowStatus) -> None:
        """Analyze which model types are configured."""
        config_dir = self.project_dir / "config"
        if not config_dir.exists():
            return

        models_config: dict[str, Any] = {}
        for yaml_file in config_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text()) or {}
            except yaml.YAMLError:
                continue
            if "models" in data and isinstance(data["models"], dict):
                models_config.update(data["models"])

        types_seen: set[str] = set()
        categories_seen: set[str] = set()
        active_count = 0

        for model_name, model_def in models_config.items():
            if not isinstance(model_def, dict):
                continue
            model_type = model_def.get("type", "")
            if model_type:
                types_seen.add(model_type)
                cat = _MODEL_CATEGORIES.get(model_type)
                if cat:
                    categories_seen.add(cat)
            if model_def.get("active", True) and model_def.get("include_in_ensemble", True):
                active_count += 1

        status.model_types_tried = sorted(types_seen)
        status.model_categories_tried = sorted(categories_seen)
        status.active_model_count = active_count

    def _check_experiments(self, status: WorkflowStatus) -> None:
        """Analyze experiment history from journal.

        Uses notebook phase_transition entries to categorize experiments
        instead of keyword-scanning descriptions.
        """
        journal_path = self.project_dir / "experiments" / "journal.jsonl"
        if not journal_path.exists():
            return

        total = 0
        for line in journal_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1

        status.total_experiments_run = total

        # Categorize experiments using notebook phase transitions
        current_phase = ""
        feature_exps = 0
        tuning_exps = 0
        for entry in self._load_notebook_entries():
            if entry.get("type") == "phase_transition":
                content = entry.get("content", "").lower()
                if "feature" in content or "engineering" in content:
                    current_phase = "feature"
                elif "tuning" in content or "hyperparameter" in content:
                    current_phase = "tuning"
                elif "model" in content or "diversity" in content:
                    current_phase = "model"
                elif "eda" in content or "exploration" in content or "discovery" in content:
                    current_phase = "eda"

        # Count experiments per phase from journal descriptions as fallback
        # (for projects that haven't adopted phase transitions yet)
        if total > 0 and current_phase == "":
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                desc = (entry.get("description", "") + " " + entry.get("hypothesis", "")).lower()
                if any(kw in desc for kw in ["feature", "add feature", "interaction", "signal"]):
                    feature_exps += 1
                if any(kw in desc for kw in ["tune", "tuning", "hyperparameter", "optuna", "exploration", "sweep"]):
                    tuning_exps += 1

        status.feature_experiments_run = feature_exps
        status.tuning_experiments_run = tuning_exps

        # Check diversity analysis
        log_path = self.project_dir / "EXPERIMENT_LOG.md"
        if log_path.exists():
            content = log_path.read_text().lower()
            if "diversity" in content:
                status.diversity_analysis_run = True

    def _check_baseline(self, status: WorkflowStatus) -> None:
        """Check if a baseline backtest has been run."""
        outputs_dir = self.project_dir / "outputs"
        if outputs_dir.exists() and any(outputs_dir.iterdir()):
            status.baseline_established = True
            return

        # Also check experiment history for baseline runs
        journal_path = self.project_dir / "experiments" / "journal.jsonl"
        if journal_path.exists():
            content = journal_path.read_text().lower()
            if "baseline" in content:
                status.baseline_established = True
