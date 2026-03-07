"""Monte Carlo competition simulator.

Provides :class:`CompetitionSimulator` for running vectorized bracket
simulations, deterministic chalk picks, and per-entity round probability
aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from harnessml.sports.competitions.schemas import (
    CompetitionStructure,
    MatchupContext,
)

if TYPE_CHECKING:
    from harnessml.sports.competitions.schemas import CompetitionConfig


def _canon_key(a: str, b: str) -> tuple[str, str]:
    """Return (entity_a, entity_b) in lexicographic order."""
    return (a, b) if a < b else (b, a)


class CompetitionSimulator:
    """Vectorized Monte Carlo simulator for bracket competitions.

    Parameters
    ----------
    config:
        Competition configuration.
    structure:
        Pre-built bracket structure (slots, matchups, round mappings).
    probabilities:
        DataFrame with columns ``entity_a``, ``entity_b``,
        ``prob_ensemble``, and optional per-model probability columns
        (``prob_model_*``).
    """

    def __init__(
        self,
        config: CompetitionConfig,
        structure: CompetitionStructure,
        probabilities: pd.DataFrame,
    ) -> None:
        self.config = config
        self.structure = structure
        self.probabilities = probabilities

        # --- Build probability lookup dict (canonical key order) -----------
        self._prob_lookup: dict[tuple[str, str], float] = {}
        self._model_probs: dict[tuple[str, str], dict[str, float]] = {}

        model_cols = [
            c for c in probabilities.columns
            if c.startswith("prob_model_")
        ]
        self._model_columns = model_cols

        for _, row in probabilities.iterrows():
            a, b = str(row["entity_a"]), str(row["entity_b"])
            prob = float(row["prob_ensemble"])
            canon_a, canon_b = _canon_key(a, b)
            # Store prob that canon_a beats canon_b
            if canon_a == a:
                self._prob_lookup[(canon_a, canon_b)] = prob
            else:
                self._prob_lookup[(canon_a, canon_b)] = 1.0 - prob

            # Per-model probs (also canonical)
            mprobs: dict[str, float] = {}
            for col in model_cols:
                mp = float(row[col])
                if canon_a == a:
                    mprobs[col] = mp
                else:
                    mprobs[col] = 1.0 - mp
            self._model_probs[(canon_a, canon_b)] = mprobs

        # --- Sorted entity list and index mapping -------------------------
        entities = set()
        for a, b in self._prob_lookup:
            entities.add(a)
            entities.add(b)
        self._entities_sorted = sorted(entities)
        self._entity_to_idx = {e: i for i, e in enumerate(self._entities_sorted)}
        n_ent = len(self._entities_sorted)

        # --- Dense numpy probability matrix (i beats j) -------------------
        self._prob_matrix = np.full((n_ent, n_ent), 0.5)
        for (a, b), prob in self._prob_lookup.items():
            ia, ib = self._entity_to_idx[a], self._entity_to_idx[b]
            self._prob_matrix[ia, ib] = prob
            self._prob_matrix[ib, ia] = 1.0 - prob
        # Diagonal = 0.5 (self vs self, used for byes effectively)

        # --- Sorted slot list with index ----------------------------------
        self._slots_sorted = list(self.structure.slots)
        self._slot_to_idx = {s: i for i, s in enumerate(self._slots_sorted)}

    # -------------------------------------------------------------------
    # Lookups
    # -------------------------------------------------------------------

    def get_win_prob(self, entity_a: str, entity_b: str) -> float:
        """Return probability that *entity_a* beats *entity_b*.

        If the pair is unknown, returns 0.5.
        """
        if entity_a == entity_b:
            return 0.5
        canon_a, canon_b = _canon_key(entity_a, entity_b)
        prob_canon_a = self._prob_lookup.get((canon_a, canon_b), 0.5)
        if entity_a == canon_a:
            return prob_canon_a
        return 1.0 - prob_canon_a

    def get_model_agreement(self, entity_a: str, entity_b: str) -> float:
        """Return model agreement score for a matchup.

        Defined as ``1 - std(model_probs) / 0.25``, clipped to [0, 1].
        Returns 1.0 if no model columns exist or pair is unknown.
        """
        if entity_a == entity_b or not self._model_columns:
            return 1.0
        canon_a, canon_b = _canon_key(entity_a, entity_b)
        mprobs = self._model_probs.get((canon_a, canon_b))
        if mprobs is None or len(mprobs) < 2:
            return 1.0
        vals = np.array(list(mprobs.values()))
        std = float(np.std(vals))
        return float(np.clip(1.0 - std / 0.25, 0.0, 1.0))

    def get_matchup_context(
        self,
        slot: str,
        entity_a: str,
        entity_b: str,
        round_num: int | None = None,
    ) -> MatchupContext:
        """Build a rich :class:`MatchupContext` for a single matchup."""
        if round_num is None:
            round_num = self.structure.slot_to_round.get(slot, 0)

        prob_a = self.get_win_prob(entity_a, entity_b)
        agreement = self.get_model_agreement(entity_a, entity_b)

        # Per-model probs (from entity_a's perspective)
        canon_a, canon_b = _canon_key(entity_a, entity_b)
        raw = self._model_probs.get((canon_a, canon_b), {})
        model_probs: dict[str, float] = {}
        for col, val in raw.items():
            if entity_a == canon_a:
                model_probs[col] = val
            else:
                model_probs[col] = 1.0 - val

        # Pick = higher-prob entity
        pick = entity_a if prob_a >= 0.5 else entity_b
        upset = prob_a < 0.5

        return MatchupContext(
            slot=slot,
            round_num=round_num,
            entity_a=entity_a,
            entity_b=entity_b,
            prob_a=prob_a,
            model_probs=model_probs,
            model_agreement=agreement,
            pick=pick,
            upset=upset,
        )

    # -------------------------------------------------------------------
    # Simulation: scalar
    # -------------------------------------------------------------------

    def _resolve_entity(self, ref: str, results: dict[str, str]) -> str:
        """Resolve a matchup reference to an entity ID.

        *ref* is either a seed code (``S1``), a slot name whose winner is
        in *results*, or ``BYE``.
        """
        if ref == "BYE":
            return "BYE"
        if ref in self.structure.seed_to_entity:
            return self.structure.seed_to_entity[ref]
        return results.get(ref, ref)

    def simulate_once(self, rng: np.random.Generator) -> dict[str, str]:
        """Run a single stochastic simulation, returning slot -> winner."""
        results: dict[str, str] = {}

        # Process rounds in order
        max_round = max(self.structure.round_slots.keys())
        for r in range(min(self.structure.round_slots.keys()), max_round + 1):
            if r not in self.structure.round_slots:
                continue
            for slot in self.structure.round_slots[r]:
                a_ref, b_ref = self.structure.slot_matchups[slot]
                ea = self._resolve_entity(a_ref, results)
                eb = self._resolve_entity(b_ref, results)

                # Bye handling: same entity on both sides
                if ea == eb or ea == "BYE" or eb == "BYE":
                    winner = eb if ea == "BYE" else ea
                else:
                    prob_a = self.get_win_prob(ea, eb)
                    winner = ea if rng.random() < prob_a else eb

                results[slot] = winner

        return results

    # -------------------------------------------------------------------
    # Simulation: vectorized
    # -------------------------------------------------------------------

    def simulate_many(self, n: int, seed: int = 42) -> list[dict[str, str]]:
        """Run *n* vectorized simulations.

        Pre-generates a random matrix and resolves entity indices for
        vectorized probability lookups from the dense matrix.
        """
        rng = np.random.default_rng(seed)
        n_slots = len(self._slots_sorted)

        # Pre-generate all random draws: shape (n, n_slots)
        rand_matrix = rng.random((n, n_slots))

        # Each sim tracks entity per slot: shape (n,) per slot but we build
        # results iteratively by round.
        # slot_winners[sim_idx][slot] = entity_id
        # For vectorization we track entity *index* arrays per slot.
        slot_entity_idx: dict[str, np.ndarray] = {}  # slot -> array of entity indices (n,)

        # Map seed refs to entity indices
        seed_entity_idx: dict[str, int] = {}
        for seed_code, entity in self.structure.seed_to_entity.items():
            if entity in self._entity_to_idx:
                seed_entity_idx[seed_code] = self._entity_to_idx[entity]

        max_round = max(self.structure.round_slots.keys())
        min_round = min(self.structure.round_slots.keys())

        for r in range(min_round, max_round + 1):
            if r not in self.structure.round_slots:
                continue
            for slot in self.structure.round_slots[r]:
                slot_idx = self._slot_to_idx[slot]
                a_ref, b_ref = self.structure.slot_matchups[slot]

                # Resolve entity index arrays
                idx_a = self._resolve_entity_idx_array(a_ref, seed_entity_idx, slot_entity_idx, n)
                idx_b = self._resolve_entity_idx_array(b_ref, seed_entity_idx, slot_entity_idx, n)

                # Detect byes: where both indices are the same
                is_bye_a = (idx_a == -1)
                is_bye_b = (idx_b == -1)
                is_same = (idx_a == idx_b)

                # Vectorized prob lookup from dense matrix
                probs = self._prob_matrix[idx_a, idx_b]

                # Determine winners
                a_wins = rand_matrix[:, slot_idx] < probs

                winners = np.where(a_wins, idx_a, idx_b)
                # Handle byes
                winners = np.where(is_bye_a, idx_b, winners)
                winners = np.where(is_bye_b, idx_a, winners)
                winners = np.where(is_same, idx_a, winners)

                slot_entity_idx[slot] = winners

        # Convert index arrays back to entity IDs
        idx_to_entity = self._entities_sorted
        all_results: list[dict[str, str]] = []
        for sim in range(n):
            result: dict[str, str] = {}
            for slot in self._slots_sorted:
                eidx = int(slot_entity_idx[slot][sim])
                result[slot] = idx_to_entity[eidx]
            all_results.append(result)

        return all_results

    def _resolve_entity_idx_array(
        self,
        ref: str,
        seed_entity_idx: dict[str, int],
        slot_entity_idx: dict[str, np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Resolve a ref to an array of entity indices across sims."""
        if ref == "BYE":
            return np.full(n, -1, dtype=np.intp)
        if ref in seed_entity_idx:
            return np.full(n, seed_entity_idx[ref], dtype=np.intp)
        if ref in slot_entity_idx:
            return slot_entity_idx[ref]
        # Unknown ref — return -1
        return np.full(n, -1, dtype=np.intp)

    # -------------------------------------------------------------------
    # Deterministic chalk
    # -------------------------------------------------------------------

    def pick_most_likely(self) -> dict[str, str]:
        """Return deterministic chalk bracket (always pick the favorite)."""
        results: dict[str, str] = {}

        max_round = max(self.structure.round_slots.keys())
        min_round = min(self.structure.round_slots.keys())

        for r in range(min_round, max_round + 1):
            if r not in self.structure.round_slots:
                continue
            for slot in self.structure.round_slots[r]:
                a_ref, b_ref = self.structure.slot_matchups[slot]
                ea = self._resolve_entity(a_ref, results)
                eb = self._resolve_entity(b_ref, results)

                if ea == eb or ea == "BYE" or eb == "BYE":
                    winner = eb if ea == "BYE" else ea
                else:
                    prob_a = self.get_win_prob(ea, eb)
                    winner = ea if prob_a >= 0.5 else eb

                results[slot] = winner

        return results

    # -------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------

    def entity_round_probabilities(
        self,
        n_sims: int = 10_000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Estimate per-entity, per-round win probabilities via simulation.

        Returns a DataFrame with columns ``entity``, ``round_1``, ``round_2``,
        etc., plus ``champion`` for the final round.
        """
        results = self.simulate_many(n_sims, seed=seed)

        # Determine round labels
        rounds_sorted = sorted(self.structure.round_slots.keys())
        max_round = rounds_sorted[-1]

        # Count wins per entity per round
        entity_round_wins: dict[str, dict[str, int]] = {}
        for entity in self._entities_sorted:
            entity_round_wins[entity] = {}
            for r in rounds_sorted:
                col = f"round_{r}" if r != max_round else "champion"
                entity_round_wins[entity][col] = 0

        for sim_result in results:
            for slot, winner in sim_result.items():
                r = self.structure.slot_to_round[slot]
                col = f"round_{r}" if r != max_round else "champion"
                if winner in entity_round_wins:
                    entity_round_wins[winner][col] += 1

        # Build DataFrame
        rows = []
        for entity in self._entities_sorted:
            row = {"entity": entity}
            for col, count in entity_round_wins[entity].items():
                row[col] = count / n_sims
            rows.append(row)

        return pd.DataFrame(rows)
