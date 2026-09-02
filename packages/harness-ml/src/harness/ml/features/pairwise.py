import numpy as np
import pandas as pd


def generate_pairwise_derivatives(
    df: pd.DataFrame,
    feature_name: str,
    methods: list[str],
    entity_a_prefix: str = "entity_a_",
    entity_b_prefix: str = "entity_b_",
) -> pd.DataFrame:
    result = df.copy()
    col_a = f"{entity_a_prefix}{feature_name}"
    col_b = f"{entity_b_prefix}{feature_name}"

    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"Entity columns not found: '{col_a}' and/or '{col_b}'")

    a = df[col_a].astype(float)
    b = df[col_b].astype(float)

    for method in methods:
        if method == "diff":
            result[f"diff_{feature_name}"] = a - b
        elif method == "ratio":
            result[f"ratio_{feature_name}"] = np.where(b != 0, a / b, 0.0)
        else:
            raise ValueError(f"Unknown pairwise method: {method}")

    return result
