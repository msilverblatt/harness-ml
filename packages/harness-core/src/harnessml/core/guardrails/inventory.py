"""Full guardrail inventory — all 12 concrete guardrail implementations.

Overridable:
    SanityCheckGuardrail, NamingConventionGuardrail, DoNotRetryGuardrail,
    SingleVariableGuardrail, ConfigProtectionGuardrail, RateLimitGuardrail,
    ExperimentLoggedGuardrail, FeatureStalenessGuardrail, FeatureDiversityGuardrail

Non-overridable:
    FeatureLeakageGuardrail, CriticalPathGuardrail, TemporalOrderingGuardrail
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from harnessml.core.guardrails.base import Guardrail

_LEAKY_PATTERNS = [
    re.compile(r"^future_"),
    re.compile(r"^target_"),
    re.compile(r"^label_"),
    re.compile(r"^result_"),
    re.compile(r"^outcome_"),
]


def detect_leaky_columns(
    columns: list[str],
    target_column: str,
    *,
    df=None,
    corr_threshold: float = 0.90,
) -> list[str]:
    """Detect potentially leaky feature columns.

    1. Name-based: columns matching suspicious patterns (future_*, result_*, etc.)
    2. Correlation-based: columns with |corr| > threshold with target (if df provided)
    """
    flagged = []
    for col in columns:
        if col == target_column:
            continue
        for pat in _LEAKY_PATTERNS:
            if pat.search(col):
                flagged.append(col)
                break

    if df is not None and target_column in df.columns:
        for col in columns:
            if col in flagged or col == target_column:
                continue
            if col not in df.columns:
                continue
            if df[col].dtype.kind not in ("f", "i", "u"):
                continue
            try:
                corr = abs(df[target_column].corr(df[col]))
                if corr > corr_threshold:
                    flagged.append(col)
            except Exception:
                pass

    return flagged


# ---------------------------------------------------------------------------
# 1. SanityCheckGuardrail (overridable)
# ---------------------------------------------------------------------------

class SanityCheckGuardrail(Guardrail):
    """Runs a sanity-check script; blocks on failure.

    Context keys:
        sanity_result (dict): Must contain ``"status"`` key.
            ``"success"`` passes; anything else fails.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sanity_check",
            overridable=True,
            description="Blocks pipeline execution when sanity check fails.",
        )

    def _check(self, context: dict) -> None:
        result = context.get("sanity_result")
        if result is None:
            self._fail(
                "No sanity_result in context — run sanity check first.",
                source="SanityCheckGuardrail",
            )
        if result.get("status") != "success":
            errors = result.get("errors", [])
            summary = "; ".join(errors[:5]) if errors else result.get("stderr", "unknown error")
            self._fail(
                f"Sanity check failed: {summary}",
                source="SanityCheckGuardrail",
            )


# ---------------------------------------------------------------------------
# 2. NamingConventionGuardrail (overridable)
# ---------------------------------------------------------------------------

class NamingConventionGuardrail(Guardrail):
    """Validates an identifier against a regex pattern.

    Context keys:
        experiment_id (str): The identifier to validate.
    """

    def __init__(self, pattern: str) -> None:
        super().__init__(
            name="naming_convention",
            overridable=True,
            description=f"Validates naming against pattern: {pattern}",
        )
        self.pattern = re.compile(pattern)

    def _check(self, context: dict) -> None:
        experiment_id = context.get("experiment_id", "")
        if not self.pattern.match(experiment_id):
            self._fail(
                f"Invalid experiment ID '{experiment_id}'. "
                f"Must match pattern: {self.pattern.pattern}",
                source="NamingConventionGuardrail",
            )


# ---------------------------------------------------------------------------
# 3. DoNotRetryGuardrail (overridable)
# ---------------------------------------------------------------------------

class DoNotRetryGuardrail(Guardrail):
    """Checks a description against known failed patterns.

    Context keys:
        description (str): Text to match against patterns.
    """

    def __init__(self, patterns: list[dict[str, str]] | None = None) -> None:
        super().__init__(
            name="do_not_retry",
            overridable=True,
            description="Blocks experiments matching known failed patterns.",
        )
        # Each entry: {"pattern": "...", "reference": "EXP-xxx", "reason": "..."}
        self.patterns = patterns or []

    def _check(self, context: dict) -> None:
        description = context.get("description", "").lower()
        if not description:
            return

        for entry in self.patterns:
            pattern = entry.get("pattern", "").lower()
            if pattern and pattern in description:
                ref = entry.get("reference", "")
                reason = entry.get("reason", "")
                self._fail(
                    f"Blocked by do-not-retry pattern: '{entry.get('pattern', '')}' "
                    f"(ref: {ref}, reason: {reason})",
                    source="DoNotRetryGuardrail",
                )


# ---------------------------------------------------------------------------
# 4. SingleVariableGuardrail (overridable)
# ---------------------------------------------------------------------------

class SingleVariableGuardrail(Guardrail):
    """Blocks multi-variable experiment changes.

    Context keys:
        total_changes (int): Number of independent changes detected.
    """

    def __init__(self, max_changes: int = 1) -> None:
        super().__init__(
            name="single_variable",
            overridable=True,
            description="Enforces single-variable experiment discipline.",
        )
        self.max_changes = max_changes

    def _check(self, context: dict) -> None:
        total = context.get("total_changes", 0)
        if total > self.max_changes:
            self._fail(
                f"Multi-variable change detected: {total} changes "
                f"(max allowed: {self.max_changes}). "
                "Change only ONE thing per experiment.",
                source="SingleVariableGuardrail",
            )


# ---------------------------------------------------------------------------
# 5. FeatureLeakageGuardrail (NON-overridable)
# ---------------------------------------------------------------------------

class FeatureLeakageGuardrail(Guardrail):
    """Checks model features against a denylist of known leaky columns.

    Context keys:
        model_features (list[str]): Feature column names to check.
    """

    def __init__(self, denylist: list[str] | None = None) -> None:
        super().__init__(
            name="feature_leakage",
            overridable=False,  # NEVER bypass
            description="Blocks models that use known leaky features.",
        )
        self.denylist = set(denylist or [])

    def _check(self, context: dict) -> None:
        model_features = context.get("model_features", [])
        found = self.denylist & set(model_features)
        if found:
            self._fail(
                f"Feature leakage detected — denied columns in model features: "
                f"{sorted(found)}",
                source="FeatureLeakageGuardrail",
            )


# ---------------------------------------------------------------------------
# 6. ConfigProtectionGuardrail (overridable)
# ---------------------------------------------------------------------------

class ConfigProtectionGuardrail(Guardrail):
    """Blocks unreviewed changes to protected config files.

    Context keys:
        changed_files (list[str]): Paths of files being modified.
    """

    def __init__(self, protected_paths: list[str] | None = None) -> None:
        super().__init__(
            name="config_protection",
            overridable=True,
            description="Blocks unreviewed changes to protected config files.",
        )
        self.protected_paths = set(protected_paths or [])

    def _check(self, context: dict) -> None:
        changed = context.get("changed_files", [])
        violations = [f for f in changed if f in self.protected_paths]
        if violations:
            self._fail(
                f"Protected config files would be modified: {violations}. "
                "Review changes before proceeding.",
                source="ConfigProtectionGuardrail",
            )


# ---------------------------------------------------------------------------
# 7. CriticalPathGuardrail (NON-overridable)
# ---------------------------------------------------------------------------

class CriticalPathGuardrail(Guardrail):
    """Blocks delete/overwrite operations on protected directories.

    Context keys:
        operation (str): One of ``"delete"``, ``"overwrite"``, ``"write"``.
        target_path (str): The path being operated on.
    """

    def __init__(self, protected_dirs: list[str] | None = None) -> None:
        super().__init__(
            name="critical_path",
            overridable=False,  # NEVER bypass
            description="Blocks destructive operations on protected directories.",
        )
        self.protected_dirs = [Path(d) for d in (protected_dirs or [])]

    def _check(self, context: dict) -> None:
        operation = context.get("operation", "")
        target = context.get("target_path", "")
        if not target or operation not in ("delete", "overwrite"):
            return

        target_path = Path(target)
        for protected in self.protected_dirs:
            try:
                target_path.relative_to(protected)
                self._fail(
                    f"Cannot {operation} '{target}' — it is under protected "
                    f"directory '{protected}'.",
                    source="CriticalPathGuardrail",
                )
            except ValueError:
                continue  # not under this protected dir


# ---------------------------------------------------------------------------
# 8. RateLimitGuardrail (overridable)
# ---------------------------------------------------------------------------

class RateLimitGuardrail(Guardrail):
    """Timestamp-based cooldown between operations.

    Context keys:
        last_run_timestamp (float | None): Unix timestamp of last run.
    """

    def __init__(self, cooldown_seconds: int = 60) -> None:
        super().__init__(
            name="rate_limit",
            overridable=True,
            description=f"Enforces {cooldown_seconds}s cooldown between runs.",
        )
        self.cooldown_seconds = cooldown_seconds

    def _check(self, context: dict) -> None:
        last_ts = context.get("last_run_timestamp")
        if last_ts is None:
            return

        elapsed = time.time() - last_ts
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            self._fail(
                f"Rate limit: {remaining:.0f}s remaining in "
                f"{self.cooldown_seconds}s cooldown.",
                source="RateLimitGuardrail",
            )


# ---------------------------------------------------------------------------
# 9. ExperimentLoggedGuardrail (overridable)
# ---------------------------------------------------------------------------

class ExperimentLoggedGuardrail(Guardrail):
    """Blocks if previous experiments are unlogged.

    Context keys:
        has_unlogged (bool): Whether any experiment lacks a log entry.
    """

    def __init__(self) -> None:
        super().__init__(
            name="experiment_logged",
            overridable=True,
            description="Blocks new experiments when previous ones are unlogged.",
        )

    def _check(self, context: dict) -> None:
        if context.get("has_unlogged", False):
            self._fail(
                "Previous experiment(s) have not been logged. "
                "Log all experiments before creating a new one.",
                source="ExperimentLoggedGuardrail",
            )


# ---------------------------------------------------------------------------
# 10. FeatureStalenessGuardrail (overridable)
# ---------------------------------------------------------------------------

class FeatureStalenessGuardrail(Guardrail):
    """Compares manifest hashes to source code to detect stale features.

    Context keys:
        manifest_hashes (dict[str, str]): ``{filename: hash}`` from manifest.
        source_hashes  (dict[str, str]): ``{filename: hash}`` from current source.
    """

    def __init__(self) -> None:
        super().__init__(
            name="feature_staleness",
            overridable=True,
            description="Detects stale features via manifest hash comparison.",
        )

    def _check(self, context: dict) -> None:
        manifest = context.get("manifest_hashes", {})
        source = context.get("source_hashes", {})
        if not manifest or not source:
            return  # nothing to compare

        stale = []
        for filename, manifest_hash in manifest.items():
            source_hash = source.get(filename)
            if source_hash is not None and source_hash != manifest_hash:
                stale.append(filename)

        if stale:
            self._fail(
                f"Stale features detected — source code changed since last "
                f"featurize for: {stale}. Re-run featurize.",
                source="FeatureStalenessGuardrail",
            )


# ---------------------------------------------------------------------------
# 11. TemporalOrderingGuardrail (NON-overridable)
# ---------------------------------------------------------------------------

class TemporalOrderingGuardrail(Guardrail):
    """Validates temporal ordering to prevent data leakage.

    Context keys:
        training_folds (list[int]): Fold values used for training.
        test_fold (int): Fold value being predicted/evaluated.
    """

    def __init__(self) -> None:
        super().__init__(
            name="temporal_ordering",
            overridable=False,  # NEVER bypass
            description="Validates temporal ordering — no future data in training.",
        )

    def _check(self, context: dict) -> None:
        training_folds = context.get("training_folds", [])
        test_fold = context.get("test_fold")
        if test_fold is None or not training_folds:
            return

        future = [s for s in training_folds if s >= test_fold]
        if future:
            self._fail(
                f"Temporal leakage: training folds {future} overlap with "
                f"or follow test fold {test_fold}.",
                source="TemporalOrderingGuardrail",
            )


# ---------------------------------------------------------------------------
# 12. FeatureDiversityGuardrail (overridable)
# ---------------------------------------------------------------------------

class FeatureDiversityGuardrail(Guardrail):
    """Checks that ensemble models use sufficiently diverse feature sets.

    Context keys:
        models (dict[str, dict]): Model configs with ``"features"`` and
            ``"active"`` keys.
    """

    def __init__(self, min_diversity_score: float = 0.5) -> None:
        super().__init__(
            name="feature_diversity",
            overridable=True,
            description=(
                f"Checks that model feature diversity score >= {min_diversity_score}."
            ),
        )
        self.min_diversity_score = min_diversity_score

    def _check(self, context: dict) -> None:
        from harnessml.core.runner.feature_diversity import compute_diversity_score

        models = context.get("models", {})
        if not models:
            return

        # Filter to active models
        active = {k: v for k, v in models.items() if v.get("active", True)}
        if len(active) <= 1:
            return  # trivially diverse

        score = compute_diversity_score(active)
        if score < self.min_diversity_score:
            self._fail(
                f"Low feature diversity: score={score:.2f} "
                f"(min={self.min_diversity_score:.2f}). "
                f"Use manage_features(action='diversity') to analyze overlap "
                f"and diversify feature sets across models.",
                source="FeatureDiversityGuardrail",
            )
