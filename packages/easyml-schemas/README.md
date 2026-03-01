# easyml-schemas

Shared Pydantic contracts and metrics for the EasyML framework. This is the core
dependency -- all other packages import schemas from here.

## Installation

```bash
pip install easyml-schemas
```

## Quick Start

```python
from easyml.schemas import ModelConfig, EnsembleConfig, brier_score, accuracy, calibration_table
import numpy as np

# Define a model config
model = ModelConfig(
    name="xgb_core", type="xgboost", mode="classifier",
    features=["win_rate", "seed_num"], params={"max_depth": 3},
)

# Compute metrics
y_true = np.array([1, 0, 1, 1, 0])
y_prob = np.array([0.8, 0.3, 0.7, 0.9, 0.2])
print(f"Brier: {brier_score(y_true, y_prob):.4f}")
print(f"Accuracy: {accuracy(y_true, y_prob):.2%}")
print(calibration_table(y_true, y_prob, n_bins=5))
```

## Key APIs

**Schemas:** `ModelConfig`, `EnsembleConfig`, `PipelineConfig`, `StageConfig`,
`ArtifactDecl`, `FeatureMeta`, `SourceMeta`, `GuardrailViolation`,
`ExperimentResult`, `RunManifest`, `Fold`, `TemporalFilter`

**Probability metrics:** `brier_score`, `log_loss`, `accuracy`, `ece`,
`calibration_table`, `auc_roc`, `f1`

**Regression metrics:** `rmse`, `mae`, `r_squared`

**Ensemble diagnostics:** `model_correlations`, `model_audit`
