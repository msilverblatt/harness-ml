from enum import StrEnum


class ExperimentType(StrEnum):
    BASELINE = "baseline"
    FEATURE = "feature"
    MODEL = "model"
    HYPERPARAMETER = "hyperparameter"
    ENSEMBLE = "ensemble"
    CALIBRATION = "calibration"
    CV_STRATEGY = "cv_strategy"
    FEATURE_SELECTION = "feature_selection"


# Maps experiment type → which config files it can modify
EXPERIMENT_CONFIG_MAP = {
    ExperimentType.BASELINE: ["models", "features"],
    ExperimentType.FEATURE: ["features"],
    ExperimentType.MODEL: ["models"],
    ExperimentType.HYPERPARAMETER: ["models"],
    ExperimentType.ENSEMBLE: ["ensemble"],
    ExperimentType.CALIBRATION: ["ensemble"],
    ExperimentType.CV_STRATEGY: ["project"],
    ExperimentType.FEATURE_SELECTION: ["models"],
}
