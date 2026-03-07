"""Pre-competition confidence diagnostics.

Provides tools for assessing prediction confidence before running a
competition simulation: feature outlier detection and model disagreement
ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from harnessml.sports.competitions.simulator import CompetitionSimulator


def compute_feature_outliers(
    entity_features: pd.DataFrame,
    target_entities: list[str],
    training_features: pd.DataFrame,
    z_threshold: float = 2.0,
) -> list[dict]:
    """Compare target entities' features against training distribution.

    Auto-discovers numeric columns (no hardcoded feature list).
    Flags features > *z_threshold* standard deviations from the training mean.

    Parameters
    ----------
    entity_features:
        DataFrame with an ``entity`` column and numeric feature columns.
    target_entities:
        Entity IDs to check for outliers.
    training_features:
        DataFrame with the same numeric columns, representing the training
        distribution.
    z_threshold:
        Number of standard deviations beyond which a value is flagged.

    Returns
    -------
    list[dict]
        Each dict has keys: ``entity``, ``feature``, ``value``, ``z_score``,
        ``training_mean``, ``training_std``.  Sorted by ``|z_score|``
        descending.
    """
    # Auto-discover numeric columns (exclude 'entity' and any other non-numeric)
    numeric_cols = training_features.select_dtypes(include=[np.number]).columns.tolist()
    # Only use columns present in both DataFrames
    entity_numeric = entity_features.select_dtypes(include=[np.number]).columns.tolist()
    shared_cols = [c for c in numeric_cols if c in entity_numeric]

    if not shared_cols:
        return []

    # Compute training stats
    training_means = training_features[shared_cols].mean()
    training_stds = training_features[shared_cols].std()

    # Filter to target entities
    target_rows = entity_features[entity_features["entity"].isin(target_entities)]

    outliers: list[dict] = []
    for _, row in target_rows.iterrows():
        entity = str(row["entity"])
        for col in shared_cols:
            value = float(row[col])
            std = float(training_stds[col])
            mean = float(training_means[col])
            if std == 0.0:
                continue
            z_score = (value - mean) / std
            if abs(z_score) > z_threshold:
                outliers.append({
                    "entity": entity,
                    "feature": col,
                    "value": value,
                    "z_score": z_score,
                    "training_mean": mean,
                    "training_std": std,
                })

    # Sort by |z_score| descending
    outliers.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return outliers


def compute_model_disagreement(
    simulator: CompetitionSimulator,
    top_n: int = 15,
) -> list[dict]:
    """Rank first-round matchups by model agreement (lowest first).

    Only includes matchups where both refs are seed codes (not slot refs).

    Parameters
    ----------
    simulator:
        A fully constructed :class:`CompetitionSimulator`.
    top_n:
        Maximum number of matchups to return.

    Returns
    -------
    list[dict]
        Each dict has keys: ``slot``, ``round``, ``entity_a``, ``entity_b``,
        ``agreement``, ``prob_ensemble``.  Sorted by ``agreement`` ascending.
    """
    structure = simulator.structure
    seed_codes = set(structure.seed_to_entity.keys())

    results: list[dict] = []
    for slot in structure.slots:
        a_ref, b_ref = structure.slot_matchups[slot]
        # Only include matchups where both refs are seed codes
        if a_ref not in seed_codes or b_ref not in seed_codes:
            continue

        entity_a = structure.seed_to_entity[a_ref]
        entity_b = structure.seed_to_entity[b_ref]
        round_num = structure.slot_to_round.get(slot, 0)

        agreement = simulator.get_model_agreement(entity_a, entity_b)
        prob_ensemble = simulator.get_win_prob(entity_a, entity_b)

        results.append({
            "slot": slot,
            "round": round_num,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "agreement": agreement,
            "prob_ensemble": prob_ensemble,
        })

    # Sort by agreement ascending (most disagreement first)
    results.sort(key=lambda x: x["agreement"])
    return results[:top_n]


def generate_confidence_report(
    simulator: CompetitionSimulator,
    entity_features: pd.DataFrame,
    training_features: pd.DataFrame,
    z_threshold: float = 2.0,
    disagreement_top_n: int = 15,
) -> dict:
    """Assemble feature outliers and model disagreement into a single report.

    Parameters
    ----------
    simulator:
        A fully constructed :class:`CompetitionSimulator`.
    entity_features:
        DataFrame with an ``entity`` column and numeric feature columns.
    training_features:
        DataFrame with the same numeric columns (training distribution).
    z_threshold:
        Z-score threshold for flagging outliers.
    disagreement_top_n:
        Max number of disagreement matchups to include.

    Returns
    -------
    dict
        Keys: ``feature_outliers``, ``high_disagreement``.
    """
    # Target entities = all entities the simulator knows about
    target_entities = list(simulator._entities_sorted)

    feature_outliers = compute_feature_outliers(
        entity_features=entity_features,
        target_entities=target_entities,
        training_features=training_features,
        z_threshold=z_threshold,
    )

    high_disagreement = compute_model_disagreement(
        simulator=simulator,
        top_n=disagreement_top_n,
    )

    return {
        "feature_outliers": feature_outliers,
        "high_disagreement": high_disagreement,
    }
