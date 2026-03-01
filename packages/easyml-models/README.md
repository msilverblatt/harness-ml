# easyml-models

Model training, ensembling, and calibration for EasyML -- 8 model wrappers,
5 CV strategies, fingerprint caching, 3 calibrators, and stacked ensembles.

## Installation

```bash
pip install easyml-models              # sklearn-only (logistic, elastic net, random forest)
pip install easyml-models[xgboost]     # + XGBoost
pip install easyml-models[all]         # + XGBoost, CatBoost, LightGBM, PyTorch (MLP, TabNet)
```

## Quick Start

```python
from easyml.models import (
    ModelRegistry, TrainOrchestrator, StackedEnsemble, LeaveOneSeasonOut,
)
import numpy as np

# Registry with all built-in wrappers
registry = ModelRegistry.with_defaults()

# Train models from config
orchestrator = TrainOrchestrator(
    model_registry=registry,
    model_configs=config["models"],
    output_dir="models/",
)
trained = orchestrator.train_all(X, y, feature_columns=feature_cols)

# Stacked ensemble with LOSO CV
ensemble = StackedEnsemble(method="stacked", meta_learner_params={"C": 2.5})
cv = LeaveOneSeasonOut(min_train_folds=1)
predictions = {name: model.predict_proba(X) for name, model in trained.items()}
ensemble.fit(predictions, y, cv=cv, fold_ids=seasons)
ensemble_probs = ensemble.predict(predictions)
```

## Key APIs

- `ModelRegistry` -- Maps type strings to model classes; `with_defaults()` registers all built-ins
- `TrainOrchestrator` -- Trains multiple models from config with fingerprint caching
- `BacktestRunner` -- Leave-one-season-out backtesting across historical seasons
- `StackedEnsemble` -- Stacked or averaging ensemble with per-fold pre-calibration
- `LeaveOneSeasonOut`, `ExpandingWindow`, `SlidingWindow`, `PurgedKFold`, `NestedCV` -- CV strategies
- `SplineCalibrator`, `PlattCalibrator`, `IsotonicCalibrator` -- Probability calibration
- `EnsemblePostprocessor` -- Temperature scaling + probability clipping
- `RunManager` -- Versioned output directory management
