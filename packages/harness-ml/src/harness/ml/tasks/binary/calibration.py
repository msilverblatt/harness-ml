"""Calibration method definitions for binary classification."""

from __future__ import annotations

from harness.ml.tasks.protocol import CalibrationType

CALIBRATION_METHODS: list[CalibrationType] = [
    CalibrationType(
        name="isotonic",
        description="Isotonic regression calibration (non-parametric, monotonic)",
    ),
    CalibrationType(
        name="platt",
        description="Platt scaling (logistic sigmoid fit)",
    ),
    CalibrationType(
        name="spline",
        description="Cubic spline calibration",
    ),
    CalibrationType(
        name="beta",
        description="Beta calibration (parametric, flexible)",
    ),
]
