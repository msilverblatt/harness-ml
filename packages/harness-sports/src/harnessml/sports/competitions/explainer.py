"""Generic explanation engine with hook-based narratives.

Provides :class:`CompetitionExplainer` for generating human-readable
pick stories, entity profiles, and feature differentials from competition
simulation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pandas as pd
from harnessml.sports.competitions.schemas import (
    CompetitionResult,
    MatchupContext,
)

if TYPE_CHECKING:
    from harnessml.sports.competitions.simulator import CompetitionSimulator


class CompetitionExplainer:
    """Generate explanations and narratives for competition picks.

    Parameters
    ----------
    entity_features:
        DataFrame with an ``entity`` column plus numeric feature columns.
    feature_display_names:
        Optional mapping from feature column name to human-readable display
        name.
    narrative_hook:
        Optional callable ``(ctx: MatchupContext, differentials: list[dict])
        -> str | None``.  If it returns a string, that string is used as the
        narrative.  If it returns ``None``, the generic narrative is used.
    """

    def __init__(
        self,
        entity_features: pd.DataFrame,
        feature_display_names: dict[str, str] | None = None,
        narrative_hook: Callable | None = None,
    ) -> None:
        self.entity_features = entity_features
        self.feature_display_names = feature_display_names or {}
        self.narrative_hook = narrative_hook

        # Pre-index by entity for fast lookups
        if not entity_features.empty and "entity" in entity_features.columns:
            self._entity_index: dict[str, int] = {
                str(row["entity"]): idx
                for idx, row in entity_features.iterrows()
            }
        else:
            self._entity_index = {}

        # Auto-discover numeric columns (exclude 'entity')
        self._numeric_cols: list[str] = [
            c
            for c in entity_features.columns
            if c != "entity" and pd.api.types.is_numeric_dtype(entity_features[c])
        ]

    # -------------------------------------------------------------------
    # Differentials
    # -------------------------------------------------------------------

    def compute_differentials(
        self,
        entity_a: str,
        entity_b: str,
        top_n: int = 5,
    ) -> list[dict]:
        """Compute feature differentials between two entities.

        Auto-discovers numeric columns and returns the top *top_n* features
        sorted by absolute magnitude of difference.

        Each returned dict contains:
        ``feature``, ``display_name``, ``entity_a_value``,
        ``entity_b_value``, ``difference``, ``favors``.
        """
        if not self._numeric_cols:
            return []

        idx_a = self._entity_index.get(entity_a)
        idx_b = self._entity_index.get(entity_b)
        if idx_a is None or idx_b is None:
            return []

        row_a = self.entity_features.iloc[idx_a]  # type: ignore[arg-type]
        row_b = self.entity_features.iloc[idx_b]  # type: ignore[arg-type]

        diffs: list[dict] = []
        for col in self._numeric_cols:
            val_a = float(row_a[col])
            val_b = float(row_b[col])
            diff = val_a - val_b
            diffs.append(
                {
                    "feature": col,
                    "display_name": self.feature_display_names.get(col, col),
                    "entity_a_value": val_a,
                    "entity_b_value": val_b,
                    "difference": diff,
                    "favors": entity_a if diff > 0 else entity_b,
                }
            )

        # Sort by absolute magnitude descending, take top N
        diffs.sort(key=lambda d: abs(d["difference"]), reverse=True)
        return diffs[:top_n]

    # -------------------------------------------------------------------
    # Narratives
    # -------------------------------------------------------------------

    def _generic_narrative(
        self,
        ctx: MatchupContext,
        differentials: list[dict],
    ) -> str:
        """Build a generic narrative string for a matchup pick."""
        opponent = ctx.entity_b if ctx.pick == ctx.entity_a else ctx.entity_a
        prob = ctx.prob_a if ctx.pick == ctx.entity_a else 1.0 - ctx.prob_a
        prob_pct = round(prob * 100)

        n_models = len(ctx.model_probs)
        if n_models > 0:
            n_favor = sum(
                1
                for p in ctx.model_probs.values()
                if (p >= 0.5) == (ctx.pick == ctx.entity_a)
            )
        else:
            n_favor = 0

        parts: list[str] = [
            f"{ctx.pick} over {opponent} at {prob_pct}% confidence.",
        ]
        if n_models > 0:
            parts.append(
                f"{n_favor} of {n_models} models favor {ctx.pick}."
            )
        if differentials:
            diff_strs = [
                f"{d['display_name']} ({d['favors']} +{abs(d['difference']):.2f})"
                for d in differentials
            ]
            parts.append(f"Key differences: {', '.join(diff_strs)}")

        return " ".join(parts)

    # -------------------------------------------------------------------
    # Pick stories
    # -------------------------------------------------------------------

    def generate_pick_stories(
        self,
        result: CompetitionResult,
    ) -> list[dict]:
        """Generate a narrative story for each pick in a competition result.

        Returns a list of dicts, one per matchup in ``result.matchups``,
        each containing: ``slot``, ``round``, ``pick``, ``opponent``,
        ``probability``, ``model_agreement``, ``upset``, ``strategy``,
        ``key_differentials``, ``narrative``, ``model_probs``.
        """
        stories: list[dict] = []

        for slot, matchup_data in result.matchups.items():
            # matchup_data can be a MatchupContext or a dict
            if isinstance(matchup_data, MatchupContext):
                ctx = matchup_data
            else:
                ctx = MatchupContext(**matchup_data)

            differentials = self.compute_differentials(
                ctx.entity_a, ctx.entity_b
            )

            # Try narrative hook first
            narrative: str | None = None
            if self.narrative_hook is not None:
                narrative = self.narrative_hook(ctx, differentials)

            # Fall back to generic narrative
            if narrative is None:
                narrative = self._generic_narrative(ctx, differentials)

            opponent = ctx.entity_b if ctx.pick == ctx.entity_a else ctx.entity_a
            prob = ctx.prob_a if ctx.pick == ctx.entity_a else 1.0 - ctx.prob_a

            stories.append(
                {
                    "slot": ctx.slot,
                    "round": ctx.round_num,
                    "pick": ctx.pick,
                    "opponent": opponent,
                    "probability": prob,
                    "model_agreement": ctx.model_agreement,
                    "upset": ctx.upset,
                    "strategy": ctx.strategy,
                    "key_differentials": differentials,
                    "narrative": narrative,
                    "model_probs": ctx.model_probs,
                }
            )

        return stories

    # -------------------------------------------------------------------
    # Entity profiles
    # -------------------------------------------------------------------

    def generate_entity_profiles(
        self,
        simulator: CompetitionSimulator,
        round_probs: pd.DataFrame,
        top_n: int = 20,
    ) -> list[dict]:
        """Generate top-N entity profiles sorted by champion probability.

        Parameters
        ----------
        simulator:
            The competition simulator (used for seed lookup).
        round_probs:
            DataFrame from ``simulator.entity_round_probabilities()``.
            Must contain an ``entity`` column and a ``champion`` column.
        top_n:
            Number of entities to include.

        Returns
        -------
        list[dict]
            Each dict: ``entity``, ``seed``, ``champion_prob``,
            ``round_probs`` (dict of round_col -> probability).
        """
        if round_probs.empty or "champion" not in round_probs.columns:
            return []

        sorted_df = round_probs.sort_values("champion", ascending=False).head(
            top_n
        )

        round_cols = [
            c for c in round_probs.columns if c not in ("entity",)
        ]

        profiles: list[dict] = []
        for _, row in sorted_df.iterrows():
            entity = str(row["entity"])
            seed = simulator.structure.entity_to_seed.get(entity, "")

            rp = {col: float(row[col]) for col in round_cols}

            profiles.append(
                {
                    "entity": entity,
                    "seed": seed,
                    "champion_prob": float(row["champion"]),
                    "round_probs": rp,
                }
            )

        return profiles
