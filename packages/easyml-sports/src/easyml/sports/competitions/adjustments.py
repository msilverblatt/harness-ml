"""Post-model probability adjustments for competition predictions."""

from __future__ import annotations

import pandas as pd

from easyml.sports.competitions.schemas import AdjustmentConfig

# Clamp bounds
_PROB_MIN = 0.01
_PROB_MAX = 0.99


def apply_adjustments(
    probabilities: pd.DataFrame,
    adjustments: AdjustmentConfig,
    external_probabilities: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply post-model adjustments to probability predictions.

    Three adjustment types (applied in order):
    1. Entity multipliers - scale an entity's win prob (e.g., injuries)
       When entity is entity_a: prob *= multiplier
       When entity is entity_b: prob = 1 - (1-prob) * multiplier
    2. External probability blending - weighted blend with external source
       Merge on (entity_a, entity_b), blend: (1-w)*model + w*external
    3. Hard overrides - direct probability assignment
       Key format: "entityA_entityB"

    All clamped to [0.01, 0.99]. All logged.

    Parameters
    ----------
    probabilities : pd.DataFrame
        Must have columns: entity_a, entity_b, prob (at minimum).
    adjustments : AdjustmentConfig
        Adjustment configuration with multipliers, external weight, overrides.
    external_probabilities : pd.DataFrame | None
        Optional external probability source with entity_a, entity_b, prob columns.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        Adjusted probabilities DataFrame and list of adjustment log entries.
    """
    df = probabilities.copy()
    log: list[dict] = []

    # 1. Entity multipliers
    for entity, multiplier in adjustments.entity_multipliers.items():
        # When entity is entity_a, scale prob directly
        mask_a = df["entity_a"] == entity
        if mask_a.any():
            old_probs = df.loc[mask_a, "prob"].copy()
            df.loc[mask_a, "prob"] = (df.loc[mask_a, "prob"] * multiplier).clip(
                _PROB_MIN, _PROB_MAX
            )
            for idx in df.loc[mask_a].index:
                log.append(
                    {
                        "type": "entity_multiplier",
                        "entity": entity,
                        "side": "entity_a",
                        "multiplier": multiplier,
                        "old_prob": float(old_probs[idx]),
                        "new_prob": float(df.loc[idx, "prob"]),
                    }
                )

        # When entity is entity_b, scale from the B perspective
        mask_b = df["entity_b"] == entity
        if mask_b.any():
            old_probs = df.loc[mask_b, "prob"].copy()
            df.loc[mask_b, "prob"] = (
                1 - (1 - df.loc[mask_b, "prob"]) * multiplier
            ).clip(_PROB_MIN, _PROB_MAX)
            for idx in df.loc[mask_b].index:
                log.append(
                    {
                        "type": "entity_multiplier",
                        "entity": entity,
                        "side": "entity_b",
                        "multiplier": multiplier,
                        "old_prob": float(old_probs[idx]),
                        "new_prob": float(df.loc[idx, "prob"]),
                    }
                )

    # 2. External probability blending
    if (
        external_probabilities is not None
        and adjustments.external_weight > 0
    ):
        w = adjustments.external_weight
        merged = df.merge(
            external_probabilities[["entity_a", "entity_b", "prob"]].rename(
                columns={"prob": "ext_prob"}
            ),
            on=["entity_a", "entity_b"],
            how="left",
        )
        has_ext = merged["ext_prob"].notna()
        if has_ext.any():
            old_probs = df.loc[has_ext, "prob"].copy()
            blended = (
                (1 - w) * merged.loc[has_ext, "prob"]
                + w * merged.loc[has_ext, "ext_prob"]
            ).clip(_PROB_MIN, _PROB_MAX)
            df.loc[has_ext, "prob"] = blended.values
            for idx in df.loc[has_ext].index:
                log.append(
                    {
                        "type": "external_blend",
                        "entity_a": df.loc[idx, "entity_a"],
                        "entity_b": df.loc[idx, "entity_b"],
                        "weight": w,
                        "old_prob": float(old_probs[idx]),
                        "ext_prob": float(merged.loc[idx, "ext_prob"]),
                        "new_prob": float(df.loc[idx, "prob"]),
                    }
                )

    # 3. Hard overrides
    for key, override_prob in adjustments.probability_overrides.items():
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        ea, eb = parts
        mask = (df["entity_a"] == ea) & (df["entity_b"] == eb)
        if mask.any():
            old_probs = df.loc[mask, "prob"].copy()
            clamped = max(_PROB_MIN, min(_PROB_MAX, override_prob))
            df.loc[mask, "prob"] = clamped
            for idx in df.loc[mask].index:
                log.append(
                    {
                        "type": "override",
                        "entity_a": ea,
                        "entity_b": eb,
                        "old_prob": float(old_probs[idx]),
                        "new_prob": float(clamped),
                    }
                )

    return df, log
