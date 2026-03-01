"""Pipeline execution planner — diff configs to determine minimum work."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in the execution plan."""
    stage: str              # "train", "ensemble", "all"
    target: str             # model name or "all"
    reason: str             # why this needs to run
    estimated_scope: str    # "1 model", "3 seasons", etc.


@dataclass
class PipelinePlan:
    """What needs to run and why."""
    steps: list[PipelineStep] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def ensemble_only(self) -> bool:
        """True if only the ensemble stage needs to run."""
        return (
            len(self.steps) > 0
            and all(s.stage == "ensemble" for s in self.steps)
        )

    @property
    def is_empty(self) -> bool:
        """True if nothing needs to run."""
        return len(self.steps) == 0

    @property
    def models_to_retrain(self) -> list[str]:
        """List of model names that need retraining."""
        return [
            s.target for s in self.steps
            if s.stage == "train" and s.target != "all"
        ]

    def format_summary(self) -> str:
        """Markdown summary of what will run."""
        if self.is_empty:
            return "## Execution Plan\n\nNo changes detected. Nothing to run."

        lines = ["## Execution Plan\n"]
        lines.append(f"**Reason**: {self.reason}\n")

        if self.ensemble_only:
            lines.append("**Mode**: Ensemble-only (no model retraining needed)\n")

        lines.append("### Steps\n")
        lines.append("| # | Stage | Target | Reason | Scope |")
        lines.append("|---|-------|--------|--------|-------|")
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"| {i} | {step.stage} | {step.target} "
                f"| {step.reason} | {step.estimated_scope} |"
            )

        if self.cache_hits:
            lines.append(f"\n### Cache Hits ({len(self.cache_hits)} models unchanged)\n")
            for name in self.cache_hits:
                lines.append(f"- {name}")

        return "\n".join(lines)


def plan_execution(
    current_config,  # ProjectConfig (use duck typing to avoid circular import)
    new_config,      # ProjectConfig
) -> PipelinePlan:
    """Diff two configs and determine minimum work needed.

    Detects:
    - New model added -> train only that model, re-ensemble
    - Model removed -> re-ensemble only
    - Model params changed -> retrain only that model, re-ensemble
    - Model features changed -> retrain only that model, re-ensemble
    - Model type changed -> retrain only that model, re-ensemble
    - Model activated/deactivated -> retrain if activated, re-ensemble
    - Ensemble config changed -> re-ensemble only (skip all training)
    - Backtest config changed -> retrain all
    - Feature config changed -> retrain all
    - No changes -> empty plan

    Parameters
    ----------
    current_config : ProjectConfig
        The current/production config.
    new_config : ProjectConfig
        The new/experiment config.

    Returns
    -------
    PipelinePlan
    """
    steps: list[PipelineStep] = []
    cache_hits: list[str] = []
    reasons: list[str] = []

    current_models = current_config.models
    new_models = new_config.models

    # Check backtest config changes — forces full retrain
    current_bt = current_config.backtest.model_dump()
    new_bt = new_config.backtest.model_dump()
    if current_bt != new_bt:
        n_seasons = len(new_config.backtest.seasons)
        n_models = len([m for m in new_models.values() if m.active])
        return PipelinePlan(
            steps=[PipelineStep(
                stage="train",
                target="all",
                reason="Backtest config changed",
                estimated_scope=f"{n_models} models x {n_seasons} seasons",
            )],
            reason="Backtest configuration changed — full retrain required.",
        )

    # Check feature_config changes — forces full retrain
    current_fc = (current_config.feature_config.model_dump()
                  if current_config.feature_config else None)
    new_fc = (new_config.feature_config.model_dump()
              if new_config.feature_config else None)
    if current_fc != new_fc:
        n_models = len([m for m in new_models.values() if m.active])
        return PipelinePlan(
            steps=[PipelineStep(
                stage="train",
                target="all",
                reason="Feature config changed",
                estimated_scope=f"{n_models} models",
            )],
            reason="Feature configuration changed — full retrain required.",
        )

    # Check per-model changes
    all_model_names = set(current_models.keys()) | set(new_models.keys())
    needs_ensemble = False

    for name in sorted(all_model_names):
        in_current = name in current_models
        in_new = name in new_models

        if in_new and not in_current:
            # New model added
            if new_models[name].active:
                steps.append(PipelineStep(
                    stage="train",
                    target=name,
                    reason="New model added",
                    estimated_scope="1 model",
                ))
                reasons.append(f"New model: {name}")
                needs_ensemble = True
            continue

        if in_current and not in_new:
            # Model removed
            reasons.append(f"Model removed: {name}")
            needs_ensemble = True
            continue

        # Model exists in both — check for changes
        current_def = current_models[name]
        new_def = new_models[name]

        # Check activation change
        if not current_def.active and new_def.active:
            steps.append(PipelineStep(
                stage="train",
                target=name,
                reason="Model activated",
                estimated_scope="1 model",
            ))
            reasons.append(f"Activated: {name}")
            needs_ensemble = True
            continue

        if current_def.active and not new_def.active:
            reasons.append(f"Deactivated: {name}")
            needs_ensemble = True
            continue

        if not new_def.active:
            # Both inactive, no change
            continue

        # Compare model config (type, features, params)
        changed = False
        change_reason = []

        if current_def.type != new_def.type:
            changed = True
            change_reason.append("type changed")

        if sorted(current_def.features) != sorted(new_def.features):
            changed = True
            added = set(new_def.features) - set(current_def.features)
            removed = set(current_def.features) - set(new_def.features)
            parts = []
            if added:
                parts.append(f"+{len(added)} features")
            if removed:
                parts.append(f"-{len(removed)} features")
            change_reason.append(", ".join(parts))

        if current_def.params != new_def.params:
            changed = True
            change_reason.append("params changed")

        if current_def.provides != new_def.provides:
            changed = True
            change_reason.append("provides changed")

        if current_def.include_in_ensemble != new_def.include_in_ensemble:
            needs_ensemble = True
            if not changed:
                # Only ensemble inclusion changed — no retraining needed
                reasons.append(f"Ensemble inclusion changed: {name}")
                continue

        if changed:
            steps.append(PipelineStep(
                stage="train",
                target=name,
                reason="; ".join(change_reason),
                estimated_scope="1 model",
            ))
            reasons.append(f"Changed: {name} ({'; '.join(change_reason)})")
            needs_ensemble = True
        else:
            cache_hits.append(name)

    # Check ensemble config changes
    current_ens = current_config.ensemble.model_dump()
    new_ens = new_config.ensemble.model_dump()
    if current_ens != new_ens:
        needs_ensemble = True
        reasons.append("Ensemble config changed")

    # Add ensemble step if needed
    if needs_ensemble:
        steps.append(PipelineStep(
            stage="ensemble",
            target="all",
            reason="Re-ensemble needed",
            estimated_scope="all active models",
        ))

    if not steps:
        return PipelinePlan(
            reason="No changes detected.",
            cache_hits=cache_hits,
        )

    return PipelinePlan(
        steps=steps,
        cache_hits=cache_hits,
        reason="; ".join(reasons) if reasons else "Changes detected.",
    )
