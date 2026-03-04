"""Backward-compat shim — re-exports from easyml.core.schemas.metrics."""

from easyml.core.schemas.metrics import (  # noqa: F401
    accuracy,
    auc_roc,
    brier_score,
    calibration_table,
    ece,
    f1,
    log_loss,
    mae,
    model_audit,
    model_correlations,
    r_squared,
    rmse,
)
