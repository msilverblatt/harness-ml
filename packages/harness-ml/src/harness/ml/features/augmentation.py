import numpy as np
import pandas as pd


def augment_symmetric(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary",
    diff_prefix: str = "diff_",
    ratio_prefix: str = "ratio_",
) -> pd.DataFrame:
    reversed_df = df.copy()

    for col in reversed_df.columns:
        if col.startswith(diff_prefix):
            reversed_df[col] = -reversed_df[col]

    for col in reversed_df.columns:
        if col.startswith(ratio_prefix):
            reversed_df[col] = np.where(reversed_df[col] != 0, 1.0 / reversed_df[col], 0.0)

    if task_type == "binary":
        reversed_df[target_col] = 1 - reversed_df[target_col]
    elif task_type == "regression":
        reversed_df[target_col] = -reversed_df[target_col]

    return pd.concat([df, reversed_df], ignore_index=True)
