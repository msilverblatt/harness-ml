import numpy as np

from harness.ml.config.ensemble import EnsembleConfig


def apply_postprocessing(
    predictions: np.ndarray,
    ensemble_config: EnsembleConfig,
) -> np.ndarray:
    """Apply post-processing steps in order:
    1. Temperature scaling
    2. Probability clipping
    3. Logit adjustments (future)
    4. Prior-proximity compression (future)
    """
    result = predictions.copy()

    # Temperature scaling
    if ensemble_config.temperature != 1.0:
        # Apply temperature to logits
        logits = np.log(
            np.clip(result, 1e-15, 1 - 1e-15) / (1 - np.clip(result, 1e-15, 1 - 1e-15))
        )
        scaled_logits = logits / ensemble_config.temperature
        result = 1.0 / (1.0 + np.exp(-scaled_logits))

    # Probability clipping
    if ensemble_config.clip_floor is not None:
        result = np.clip(result, ensemble_config.clip_floor, 1.0 - ensemble_config.clip_floor)

    # Always clip to [0, 1]
    result = np.clip(result, 0.0, 1.0)

    return result
