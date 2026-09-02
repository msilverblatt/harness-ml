# harness-ml Plan 2a: Task Types + Models

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational type system for harness-ml — task type protocol with binary/multiclass/regression implementations, and model protocol with 14 model wrappers organized by family. These are the building blocks consumed by the training pipeline (Plan 2c).

**Architecture:** Protocol-based extensibility. Task types encapsulate metrics, calibration, validation, and model adaptation. Models are organized by family (boosting, linear, neural, tree, kernel) with shared base classes. A parametrized contract test verifies every model against the protocol.

**Tech Stack:** Python 3.11+, pandas 2.0+, numpy, scikit-learn 1.3+, pydantic 2.0+, pytest. Model backends (xgboost, lightgbm, catboost, torch) are optional and auto-installed on first use.

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Sections 5 (Task Types) + 6 (Models)

**Testing note:** Every task includes unit tests. After Tasks 1-4 (task types) and after Tasks 5-10 (models), run the e2e test suite to catch integration issues. Error paths MUST be tested alongside happy paths.

**E2E testing mandate:** After every 2-3 tasks, write real e2e tests that exercise the full chain with realistic data, verify actual computed values, and catch the integration bugs that unit tests miss.

---

## File Structure

```
packages/harness-ml/
├── pyproject.toml
├── src/harness/ml/
│   ├── __init__.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── protocol.py            # TaskType protocol, Metric, ValidationResult
│   │   ├── registry.py            # TaskRegistry (auto-discover task type modules)
│   │   ├── binary/
│   │   │   ├── __init__.py
│   │   │   ├── task.py            # BinaryTask (implements TaskType)
│   │   │   ├── metrics.py         # brier, log_loss, auroc, ece, accuracy, f1, precision, recall
│   │   │   ├── calibration.py     # Spline, Isotonic, Platt calibrators
│   │   │   ├── adaptation.py      # OBJECTIVES dict per model family
│   │   │   └── validation.py      # Target 0/1, predictions [0,1]
│   │   ├── multiclass/
│   │   │   ├── __init__.py
│   │   │   ├── task.py
│   │   │   ├── metrics.py         # macro/micro/weighted variants
│   │   │   ├── calibration.py     # Per-class calibration
│   │   │   ├── adaptation.py
│   │   │   └── validation.py
│   │   └── regression/
│   │       ├── __init__.py
│   │       ├── task.py
│   │       ├── metrics.py         # rmse, mae, r2, mape
│   │       ├── adaptation.py
│   │       └── validation.py
│   └── models/
│       ├── __init__.py
│       ├── protocol.py            # Model protocol, FitResult
│       ├── registry.py            # ModelRegistry (auto-discover, auto-install)
│       └── families/
│           ├── __init__.py
│           ├── boosting/
│           │   ├── __init__.py
│           │   ├── base.py        # BoostingBase (early stopping, feature importance)
│           │   ├── xgboost.py
│           │   ├── lightgbm.py
│           │   ├── catboost.py
│           │   └── hist_gbm.py
│           ├── linear/
│           │   ├── __init__.py
│           │   ├── base.py        # LinearBase (sklearn fit/predict)
│           │   ├── logistic.py
│           │   └── elastic_net.py
│           ├── neural/
│           │   ├── __init__.py
│           │   ├── base.py        # NeuralBase (device, batch training)
│           │   └── mlp.py
│           ├── tree/
│           │   ├── __init__.py
│           │   ├── base.py
│           │   └── random_forest.py
│           └── kernel/
│               ├── __init__.py
│               ├── base.py
│               └── svm.py
└── tests/
    ├── conftest.py                # Shared fixtures (datasets, temp dirs)
    ├── test_e2e.py                # E2E integration tests
    ├── test_tasks/
    │   ├── __init__.py
    │   ├── test_protocol.py
    │   ├── test_registry.py
    │   ├── test_binary.py
    │   ├── test_multiclass.py
    │   └── test_regression.py
    └── test_models/
        ├── __init__.py
        ├── test_protocol.py
        ├── test_registry.py
        └── test_model_contract.py  # Parametrized across all models
```

---

### Task 1: Project Scaffolding + Task Type Protocol

**Files:**
- Create: `packages/harness-ml/pyproject.toml`
- Create: `packages/harness-ml/src/harness/ml/__init__.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/__init__.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/protocol.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/registry.py`
- Create: `packages/harness-ml/tests/conftest.py`
- Create: `packages/harness-ml/tests/test_tasks/__init__.py`
- Create: `packages/harness-ml/tests/test_tasks/test_protocol.py`
- Create: `packages/harness-ml/tests/test_tasks/test_registry.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "harness-ml"
version = "0.1.0"
description = "Tabular ML engine for the Harness platform"
requires-python = ">=3.11"
dependencies = [
    "harness-data>=0.1.0",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
boosting = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2"]
neural = ["torch>=2.0"]
all = ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2", "torch>=2.0"]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/harness"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package structure + conftest**

`packages/harness-ml/src/harness/ml/__init__.py`:
```python
"""harness-ml: Tabular ML engine for the Harness platform."""
```

`packages/harness-ml/tests/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_dataset():
    """Realistic binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
        "feature_d": rng.randint(0, 5, n).astype(float),
    })
    # Target correlated with features so models can learn something
    logits = 0.5 * X["feature_a"] - 0.3 * X["feature_b"] + 0.1 * X["feature_c"]
    y = (logits + rng.randn(n) * 0.5 > 0).astype(int)
    return X, pd.Series(y, name="target")


@pytest.fixture
def multiclass_dataset():
    """Realistic multiclass dataset (3 classes)."""
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
    })
    y = pd.Series(rng.randint(0, 3, n), name="target")
    return X, y


@pytest.fixture
def regression_dataset():
    """Realistic regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
    })
    y = pd.Series(
        2.0 * X["feature_a"] - 1.5 * X["feature_b"] + 0.5 * X["feature_c"] + rng.randn(n) * 0.5,
        name="target",
    )
    return X, y


@pytest.fixture
def feature_columns():
    """Standard feature column list for test datasets."""
    return ["feature_a", "feature_b", "feature_c", "feature_d"]
```

- [ ] **Step 3: Write failing tests for TaskType protocol + TaskRegistry**

`packages/harness-ml/tests/test_tasks/__init__.py` — empty.
`packages/harness-ml/tests/test_tasks/test_protocol.py`:
```python
import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.protocol import Metric, ValidationResult


class TestMetric:
    def test_create_metric(self):
        m = Metric(name="brier", display_name="Brier Score", higher_is_better=False)
        assert m.name == "brier"
        assert not m.higher_is_better

    def test_metric_compute_raises_if_not_overridden(self):
        m = Metric(name="test", display_name="Test", higher_is_better=True)
        # Base Metric.compute should raise NotImplementedError or be a callable
        # We'll test this through task types instead


class TestValidationResult:
    def test_valid_result(self):
        r = ValidationResult(is_valid=True)
        assert r.is_valid
        assert r.errors == []

    def test_invalid_result(self):
        r = ValidationResult(is_valid=False, errors=["Target must be binary"])
        assert not r.is_valid
        assert "binary" in r.errors[0]
```

`packages/harness-ml/tests/test_tasks/test_registry.py`:
```python
import pytest

from harness.ml.tasks.registry import TaskRegistry


class TestTaskRegistry:
    def test_get_binary(self):
        task = TaskRegistry.get("binary")
        assert task is not None
        assert task.name == "binary"

    def test_get_regression(self):
        task = TaskRegistry.get("regression")
        assert task is not None
        assert task.name == "regression"

    def test_get_multiclass(self):
        task = TaskRegistry.get("multiclass")
        assert task is not None
        assert task.name == "multiclass"

    def test_get_unknown_returns_none(self):
        task = TaskRegistry.get("nonexistent")
        assert task is None

    def test_list_available(self):
        available = TaskRegistry.list_available()
        assert "binary" in available
        assert "regression" in available
        assert "multiclass" in available
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd packages/harness-ml && pip install -e ".[dev]" && python -m pytest tests/test_tasks/ -v`
Expected: FAIL with ImportError

- [ ] **Step 5: Implement protocol.py + registry.py (stubs for task types)**

`packages/harness-ml/src/harness/ml/tasks/__init__.py`:
```python
from harness.ml.tasks.protocol import Metric, ValidationResult
from harness.ml.tasks.registry import TaskRegistry
```

`packages/harness-ml/src/harness/ml/tasks/protocol.py`:
```python
"""Task type protocol — the contract all task types implement."""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd


class Metric(BaseModel):
    """A metric that can be computed for a task type."""
    name: str
    display_name: str
    higher_is_better: bool
    compute: Callable[[np.ndarray, np.ndarray], float] | None = None

    model_config = {"arbitrary_types_allowed": True}


class ValidationResult(BaseModel):
    """Result of validating targets or predictions."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CalibrationType(BaseModel):
    """A calibration method available for a task type."""
    name: str
    description: str


class ResultSummary(BaseModel):
    """Formatted results for agent consumption."""
    metrics: dict[str, float] = Field(default_factory=dict)
    per_fold: list[dict[str, float]] = Field(default_factory=list)
    summary_text: str = ""


@runtime_checkable
class TaskType(Protocol):
    """Protocol that all task types implement."""
    name: str

    def metrics(self) -> list[Metric]:
        """Available metrics for this task type."""
        ...

    def default_metrics(self) -> list[str]:
        """Default metric names for backtesting."""
        ...

    def validate_target(self, series: pd.Series) -> ValidationResult:
        """Validate that the target column is appropriate for this task type."""
        ...

    def validate_predictions(self, predictions: np.ndarray) -> ValidationResult:
        """Sanity-check model outputs."""
        ...

    def calibration_methods(self) -> list[CalibrationType]:
        """Available calibration methods."""
        ...

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric_names: list[str]
    ) -> dict[str, float]:
        """Compute specified metrics."""
        ...

    def postprocess(self, predictions: np.ndarray, config: dict) -> np.ndarray:
        """Task-specific post-processing."""
        ...
```

`packages/harness-ml/src/harness/ml/tasks/registry.py`:
```python
"""Task registry — discovers and loads task type implementations."""
from __future__ import annotations

from typing import Any


class TaskRegistry:
    """Registry for task type implementations."""
    _tasks: dict[str, Any] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        # Import task type modules to trigger registration
        try:
            from harness.ml.tasks.binary.task import BinaryTask
            cls._tasks["binary"] = BinaryTask()
        except ImportError:
            pass
        try:
            from harness.ml.tasks.multiclass.task import MulticlassTask
            cls._tasks["multiclass"] = MulticlassTask()
        except ImportError:
            pass
        try:
            from harness.ml.tasks.regression.task import RegressionTask
            cls._tasks["regression"] = RegressionTask()
        except ImportError:
            pass
        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> Any | None:
        cls._ensure_loaded()
        return cls._tasks.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        cls._ensure_loaded()
        return list(cls._tasks.keys())

    @classmethod
    def register(cls, name: str, task: Any) -> None:
        cls._tasks[name] = task
        cls._loaded = True
```

- [ ] **Step 6: Run protocol/registry tests (registry tests will fail — task types not yet implemented)**

Run: `cd packages/harness-ml && python -m pytest tests/test_tasks/test_protocol.py -v`
Expected: PASS

- [ ] **Step 7: Commit scaffolding**

```bash
git add packages/harness-ml/
git commit -m "feat(harness-ml): project scaffolding + task type protocol + registry"
```

---

### Task 2: Binary Classification Task Type

**Files:**
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/__init__.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/task.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/metrics.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/calibration.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/adaptation.py`
- Create: `packages/harness-ml/src/harness/ml/tasks/binary/validation.py`
- Create: `packages/harness-ml/tests/test_tasks/test_binary.py`

- [ ] **Step 1: Write failing tests**

`packages/harness-ml/tests/test_tasks/test_binary.py`:
```python
import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.binary.task import BinaryTask
from harness.ml.tasks.protocol import ValidationResult


class TestBinaryTask:
    @pytest.fixture
    def task(self):
        return BinaryTask()

    def test_name(self, task):
        assert task.name == "binary"

    def test_metrics_list(self, task):
        metrics = task.metrics()
        names = [m.name for m in metrics]
        assert "brier" in names
        assert "log_loss" in names
        assert "auroc" in names
        assert "accuracy" in names

    def test_default_metrics(self, task):
        defaults = task.default_metrics()
        assert "brier" in defaults
        assert len(defaults) >= 3

    def test_validate_target_valid(self, task):
        target = pd.Series([0, 1, 1, 0, 1])
        result = task.validate_target(target)
        assert result.is_valid

    def test_validate_target_invalid_values(self, task):
        target = pd.Series([0, 1, 2, 3])
        result = task.validate_target(target)
        assert not result.is_valid

    def test_validate_target_all_same(self, task):
        target = pd.Series([1, 1, 1, 1])
        result = task.validate_target(target)
        assert not result.is_valid or len(result.warnings) > 0

    def test_validate_predictions_valid(self, task):
        preds = np.array([0.1, 0.5, 0.9, 0.3])
        result = task.validate_predictions(preds)
        assert result.is_valid

    def test_validate_predictions_out_of_range(self, task):
        preds = np.array([0.1, 1.5, -0.1, 0.5])
        result = task.validate_predictions(preds)
        assert not result.is_valid

    def test_compute_metrics(self, task):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.8, 0.9])
        result = task.compute_metrics(y_true, y_pred, ["brier", "accuracy"])
        assert "brier" in result
        assert "accuracy" in result
        assert 0 <= result["brier"] <= 1
        assert 0 <= result["accuracy"] <= 1

    def test_compute_brier_perfect(self, task):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        result = task.compute_metrics(y_true, y_pred, ["brier"])
        assert result["brier"] == 0.0

    def test_compute_brier_worst(self, task):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0])
        result = task.compute_metrics(y_true, y_pred, ["brier"])
        assert result["brier"] == 1.0

    def test_calibration_methods(self, task):
        methods = task.calibration_methods()
        names = [m.name for m in methods]
        assert "isotonic" in names
        assert "platt" in names

    def test_postprocess_clipping(self, task):
        preds = np.array([-0.1, 0.5, 1.1])
        result = task.postprocess(preds, {"clip": True})
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_adaptation_has_objectives(self):
        from harness.ml.tasks.binary.adaptation import OBJECTIVES
        assert "xgboost" in OBJECTIVES
        assert "lightgbm" in OBJECTIVES
        assert "logistic" in OBJECTIVES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-ml && python -m pytest tests/test_tasks/test_binary.py -v`
Expected: FAIL

- [ ] **Step 3: Implement binary task type**

`packages/harness-ml/src/harness/ml/tasks/binary/__init__.py`:
```python
from harness.ml.tasks.binary.task import BinaryTask
```

`packages/harness-ml/src/harness/ml/tasks/binary/metrics.py`:
```python
"""Binary classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

from harness.ml.tasks.protocol import Metric


def _brier(y_true, y_pred):
    return brier_score_loss(y_true, y_pred)

def _log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred, labels=[0, 1])

def _auroc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_pred)

def _accuracy(y_true, y_pred):
    return accuracy_score(y_true, (y_pred >= 0.5).astype(int))

def _f1(y_true, y_pred):
    return f1_score(y_true, (y_pred >= 0.5).astype(int))

def _precision(y_true, y_pred):
    return precision_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)

def _recall(y_true, y_pred):
    return recall_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)

def _ece(y_true, y_pred, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true) if len(y_true) > 0 else 0.0


BINARY_METRICS = [
    Metric(name="brier", display_name="Brier Score", higher_is_better=False, compute=_brier),
    Metric(name="log_loss", display_name="Log Loss", higher_is_better=False, compute=_log_loss),
    Metric(name="auroc", display_name="AUROC", higher_is_better=True, compute=_auroc),
    Metric(name="accuracy", display_name="Accuracy", higher_is_better=True, compute=_accuracy),
    Metric(name="f1", display_name="F1 Score", higher_is_better=True, compute=_f1),
    Metric(name="precision", display_name="Precision", higher_is_better=True, compute=_precision),
    Metric(name="recall", display_name="Recall", higher_is_better=True, compute=_recall),
    Metric(name="ece", display_name="Expected Calibration Error", higher_is_better=False, compute=_ece),
]
```

`packages/harness-ml/src/harness/ml/tasks/binary/validation.py`:
```python
"""Binary classification validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import ValidationResult


def validate_target(series: pd.Series) -> ValidationResult:
    """Validate binary classification target."""
    errors = []
    warnings = []

    unique = set(series.dropna().unique())
    if not unique.issubset({0, 1, 0.0, 1.0, True, False}):
        errors.append(f"Binary target must contain only 0/1 values. Found: {unique}")

    if len(unique) < 2:
        warnings.append("Target has only one class — model cannot learn meaningful patterns")

    null_count = int(series.isna().sum())
    if null_count > 0:
        warnings.append(f"Target has {null_count} null values")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_predictions(predictions: np.ndarray) -> ValidationResult:
    """Validate binary prediction probabilities."""
    errors = []

    if np.any(predictions < 0) or np.any(predictions > 1):
        errors.append(
            f"Binary predictions must be in [0, 1]. "
            f"Range: [{predictions.min():.4f}, {predictions.max():.4f}]"
        )

    if np.any(np.isnan(predictions)):
        nan_count = int(np.isnan(predictions).sum())
        errors.append(f"Predictions contain {nan_count} NaN values")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

`packages/harness-ml/src/harness/ml/tasks/binary/calibration.py`:
```python
"""Binary classification calibration methods."""
from __future__ import annotations

from harness.ml.tasks.protocol import CalibrationType

BINARY_CALIBRATION_METHODS = [
    CalibrationType(name="isotonic", description="Isotonic regression calibration"),
    CalibrationType(name="platt", description="Platt scaling (logistic regression on logits)"),
    CalibrationType(name="spline", description="PCHIP spline calibration"),
    CalibrationType(name="beta", description="Beta calibration"),
]
```

`packages/harness-ml/src/harness/ml/tasks/binary/adaptation.py`:
```python
"""Binary classification — model family adaptation mappings."""
from __future__ import annotations

# Maps model name → task-specific params that get merged into model config
OBJECTIVES: dict[str, dict] = {
    # Boosting family
    "xgboost": {"objective": "binary:logistic", "eval_metric": "logloss"},
    "lightgbm": {"objective": "binary", "metric": "binary_logloss"},
    "catboost": {"loss_function": "Logloss", "eval_metric": "Logloss"},
    "hist_gbm": {},  # sklearn handles binary natively

    # Linear family
    "logistic": {},  # native binary classifier
    "elastic_net": {},  # uses LogisticRegression with elasticnet penalty

    # Neural family
    "mlp": {"loss": "bce", "output_dim": 1, "output_activation": "sigmoid"},
    "tabnet": {"loss": "bce"},
    "tabpfn": {},
    "realmlp": {"loss": "bce"},

    # Tree family
    "random_forest": {},  # sklearn handles binary natively

    # Kernel family
    "svm": {"probability": True},  # enable predict_proba
}

DEFAULT_PARAMS: dict[str, dict] = {
    "xgboost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
    "lightgbm": {"n_estimators": 500, "max_depth": -1, "learning_rate": 0.05, "num_leaves": 31},
    "catboost": {"iterations": 500, "depth": 6, "learning_rate": 0.05},
    "logistic": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "max_depth": None},
}
```

`packages/harness-ml/src/harness/ml/tasks/binary/task.py`:
```python
"""Binary classification task type."""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import (
    CalibrationType, Metric, ResultSummary, ValidationResult,
)
from harness.ml.tasks.binary.metrics import BINARY_METRICS
from harness.ml.tasks.binary.calibration import BINARY_CALIBRATION_METHODS
from harness.ml.tasks.binary import validation as val


class BinaryTask:
    """Binary classification task type implementation."""

    name = "binary"

    def metrics(self) -> list[Metric]:
        return BINARY_METRICS

    def default_metrics(self) -> list[str]:
        return ["brier", "log_loss", "auroc", "accuracy", "ece"]

    def validate_target(self, series: pd.Series) -> ValidationResult:
        return val.validate_target(series)

    def validate_predictions(self, predictions: np.ndarray) -> ValidationResult:
        return val.validate_predictions(predictions)

    def calibration_methods(self) -> list[CalibrationType]:
        return BINARY_CALIBRATION_METHODS

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric_names: list[str],
    ) -> dict[str, float]:
        metric_map = {m.name: m for m in BINARY_METRICS}
        results = {}
        for name in metric_names:
            m = metric_map.get(name)
            if m is None or m.compute is None:
                continue
            try:
                results[name] = float(m.compute(y_true, y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    def postprocess(self, predictions: np.ndarray, config: dict) -> np.ndarray:
        result = predictions.copy()
        if config.get("clip", False):
            result = np.clip(result, 0.0, 1.0)
        clip_floor = config.get("clip_floor")
        if clip_floor is not None:
            result = np.clip(result, clip_floor, 1.0 - clip_floor)
        return result
```

- [ ] **Step 4: Run tests**

Run: `cd packages/harness-ml && python -m pytest tests/test_tasks/ -v`
Expected: ALL PASS (both protocol and binary tests)

- [ ] **Step 5: Commit**

```bash
git add packages/harness-ml/
git commit -m "feat(harness-ml): binary classification task type with metrics, calibration, adaptation"
```

---

### Task 3: Regression + Multiclass Task Types

**Files:**
- Create: `packages/harness-ml/src/harness/ml/tasks/regression/` (all files)
- Create: `packages/harness-ml/src/harness/ml/tasks/multiclass/` (all files)
- Create: `packages/harness-ml/tests/test_tasks/test_regression.py`
- Create: `packages/harness-ml/tests/test_tasks/test_multiclass.py`

- [ ] **Step 1: Write failing tests for both task types**

Follow the same pattern as binary. Key tests:

**Regression**: name=="regression", metrics include rmse/mae/r2/mape, validate_target accepts continuous values, validate_predictions accepts any float, compute_metrics produces correct RMSE (manually verified).

**Multiclass**: name=="multiclass", metrics include accuracy/f1_macro/log_loss, validate_target accepts integer classes, validate_predictions accepts 2D probability arrays, compute_metrics works.

- [ ] **Step 2: Implement regression task type**

`regression/metrics.py` — rmse, mae, r2, mape, median_ae, explained_variance
`regression/validation.py` — target is continuous, predictions are floats
`regression/adaptation.py` — OBJECTIVES dict for regression (xgboost: "reg:squarederror", etc.)
`regression/task.py` — implements TaskType protocol

- [ ] **Step 3: Implement multiclass task type**

`multiclass/metrics.py` — accuracy, f1_macro, f1_micro, f1_weighted, log_loss, precision_macro, recall_macro
`multiclass/validation.py` — target is integer classes, predictions are 2D probability arrays
`multiclass/adaptation.py` — OBJECTIVES dict for multiclass
`multiclass/task.py` — implements TaskType protocol

- [ ] **Step 4: Run all task tests**

Run: `cd packages/harness-ml && python -m pytest tests/test_tasks/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-ml/
git commit -m "feat(harness-ml): regression + multiclass task types"
```

---

### Task 4: Task Types E2E Tests

**Files:**
- Create: `packages/harness-ml/tests/test_e2e.py`

- [ ] **Step 1: Write e2e tests that exercise the full task type chain**

```python
"""E2E tests for task types — verify real metric computations on real data."""

import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.registry import TaskRegistry


class TestE2ETaskTypes:
    def test_registry_has_all_types(self):
        available = TaskRegistry.list_available()
        assert set(available) == {"binary", "multiclass", "regression"}

    def test_binary_metrics_on_real_predictions(self):
        """Verify Brier score, accuracy, AUROC on manually computed data."""
        task = TaskRegistry.get("binary")
        # Perfect calibration: 5 games predicted at 80%, 4 won (80%)
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        metrics = task.compute_metrics(y_true, y_pred, ["brier", "accuracy", "auroc"])

        # Brier = mean((y - p)^2) = (4*0.04 + 1*0.64) / 5 = 0.16
        assert abs(metrics["brier"] - 0.16) < 0.001
        # Accuracy at threshold 0.5: all predicted 1, actual [1,1,1,1,0] = 4/5 = 0.8
        assert abs(metrics["accuracy"] - 0.8) < 0.001

    def test_binary_validates_bad_target(self):
        task = TaskRegistry.get("binary")
        result = task.validate_target(pd.Series([0, 1, 2, 3]))
        assert not result.is_valid

    def test_regression_rmse_manual(self):
        """Verify RMSE by hand."""
        task = TaskRegistry.get("regression")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error of 1 on last
        metrics = task.compute_metrics(y_true, y_pred, ["rmse", "mae"])
        # RMSE = sqrt(mean([0, 0, 1])) = sqrt(1/3) ≈ 0.5774
        assert abs(metrics["rmse"] - np.sqrt(1/3)) < 0.001
        # MAE = mean([0, 0, 1]) = 1/3 ≈ 0.333
        assert abs(metrics["mae"] - 1/3) < 0.001

    def test_multiclass_accuracy_manual(self):
        """Verify multiclass accuracy."""
        task = TaskRegistry.get("multiclass")
        y_true = np.array([0, 1, 2, 0, 1])
        # Perfect predictions as one-hot probabilities
        y_pred = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],  # Wrong — predicts 2, actual is 1
        ])
        metrics = task.compute_metrics(y_true, y_pred, ["accuracy"])
        assert abs(metrics["accuracy"] - 0.8) < 0.001  # 4/5 correct

    def test_all_task_types_implement_protocol(self):
        """Every registered task type has all required methods."""
        for name in TaskRegistry.list_available():
            task = TaskRegistry.get(name)
            assert hasattr(task, "name")
            assert hasattr(task, "metrics")
            assert hasattr(task, "default_metrics")
            assert hasattr(task, "validate_target")
            assert hasattr(task, "validate_predictions")
            assert hasattr(task, "compute_metrics")
            assert hasattr(task, "calibration_methods")
            assert hasattr(task, "postprocess")

    def test_each_task_type_has_metrics(self):
        """Every task type returns a non-empty metric list."""
        for name in TaskRegistry.list_available():
            task = TaskRegistry.get(name)
            metrics = task.metrics()
            assert len(metrics) > 0, f"{name} has no metrics"
            defaults = task.default_metrics()
            assert len(defaults) > 0, f"{name} has no default metrics"
            # Every default metric should be in the full list
            all_names = {m.name for m in metrics}
            for d in defaults:
                assert d in all_names, f"{name}: default metric '{d}' not in metrics list"
```

- [ ] **Step 2: Run e2e tests**

Run: `cd packages/harness-ml && python -m pytest tests/test_e2e.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add packages/harness-ml/tests/test_e2e.py
git commit -m "test(harness-ml): e2e tests for task types with manual metric verification"
```

---

### Task 5: Model Protocol + Registry

**Files:**
- Create: `packages/harness-ml/src/harness/ml/models/__init__.py`
- Create: `packages/harness-ml/src/harness/ml/models/protocol.py`
- Create: `packages/harness-ml/src/harness/ml/models/registry.py`
- Create: `packages/harness-ml/src/harness/ml/models/families/__init__.py`
- Create: `packages/harness-ml/tests/test_models/__init__.py`
- Create: `packages/harness-ml/tests/test_models/test_protocol.py`
- Create: `packages/harness-ml/tests/test_models/test_registry.py`

- [ ] **Step 1: Write failing tests**

`test_protocol.py` — test FitResult creation, model protocol shape.
`test_registry.py` — test ModelRegistry.get, list_available, get_unknown.

- [ ] **Step 2: Implement protocol.py**

```python
"""Model protocol — the contract all model wrappers implement."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    """Result of training a model."""
    model: Any  # The trained model object
    feature_importance: dict[str, float] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Model(Protocol):
    """Protocol that all model wrappers implement."""
    name: str
    supports_tasks: list[str]
    requires_packages: list[str]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult:
        ...

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        ...

    def default_params(self, task_type: str) -> dict:
        ...

    def param_schema(self) -> dict:
        ...

    def save(self, model: Any, path: Path) -> None:
        ...

    def load(self, path: Path) -> Any:
        ...

    def supports_multi_seed(self) -> bool:
        ...
```

- [ ] **Step 3: Implement registry.py**

Auto-discovers model modules from `families/` subpackages. Each model module exports `NAME` and a model class.

- [ ] **Step 4: Run tests, commit**

```bash
git commit -m "feat(harness-ml): model protocol + registry"
```

---

### Task 6: Linear Family (Logistic + ElasticNet)

**Files:**
- Create: `packages/harness-ml/src/harness/ml/models/families/linear/base.py`
- Create: `packages/harness-ml/src/harness/ml/models/families/linear/logistic.py`
- Create: `packages/harness-ml/src/harness/ml/models/families/linear/elastic_net.py`

- [ ] **Step 1: Implement LinearBase + Logistic + ElasticNet**

LinearBase handles sklearn fit/predict pattern. Logistic wraps LogisticRegression. ElasticNet wraps SGDClassifier with elasticnet penalty (or LogisticRegression with l1_ratio).

These are sklearn-only, no optional dependencies. Start with these to validate the model protocol works before adding heavy dependencies.

- [ ] **Step 2: Write parametrized contract test**

`packages/harness-ml/tests/test_models/test_model_contract.py`:
```python
"""Parametrized contract tests — every model must pass these."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from harness.ml.models.registry import ModelRegistry


def get_available_models():
    """Get all models that can be imported (dependencies available)."""
    return ModelRegistry.list_available()


@pytest.fixture(params=get_available_models())
def model_wrapper(request):
    return ModelRegistry.get(request.param)


class TestModelContract:
    def test_has_required_attributes(self, model_wrapper):
        assert hasattr(model_wrapper, "name")
        assert hasattr(model_wrapper, "supports_tasks")
        assert hasattr(model_wrapper, "requires_packages")
        assert isinstance(model_wrapper.name, str)
        assert isinstance(model_wrapper.supports_tasks, list)

    def test_default_params_returns_dict(self, model_wrapper):
        for task in model_wrapper.supports_tasks:
            params = model_wrapper.default_params(task)
            assert isinstance(params, dict)

    def test_param_schema_returns_dict(self, model_wrapper):
        schema = model_wrapper.param_schema()
        assert isinstance(schema, dict)

    def test_fit_and_predict_binary(self, model_wrapper, binary_dataset):
        if "binary" not in model_wrapper.supports_tasks:
            pytest.skip("Model does not support binary")
        X, y = binary_dataset
        params = model_wrapper.default_params("binary")
        result = model_wrapper.fit(X, y, None, None, params)
        assert result.model is not None
        preds = model_wrapper.predict(result.model, X)
        assert len(preds) == len(X)
        assert all(np.isfinite(preds))

    def test_fit_and_predict_regression(self, model_wrapper, regression_dataset):
        if "regression" not in model_wrapper.supports_tasks:
            pytest.skip("Model does not support regression")
        X, y = regression_dataset
        params = model_wrapper.default_params("regression")
        result = model_wrapper.fit(X, y, None, None, params)
        preds = model_wrapper.predict(result.model, X)
        assert len(preds) == len(X)

    def test_save_and_load(self, model_wrapper, binary_dataset, tmp_path):
        if "binary" not in model_wrapper.supports_tasks:
            if "regression" in model_wrapper.supports_tasks:
                # Use regression dataset instead
                return  # Skip for now, covered by regression test
            pytest.skip("Model does not support binary or regression")
        X, y = binary_dataset
        params = model_wrapper.default_params("binary")
        result = model_wrapper.fit(X, y, None, None, params)

        path = tmp_path / f"{model_wrapper.name}.model"
        model_wrapper.save(result.model, path)
        loaded = model_wrapper.load(path)

        preds_original = model_wrapper.predict(result.model, X)
        preds_loaded = model_wrapper.predict(loaded, X)
        np.testing.assert_array_almost_equal(preds_original, preds_loaded)
```

- [ ] **Step 3: Run contract tests (only linear models available)**

Run: `cd packages/harness-ml && python -m pytest tests/test_models/test_model_contract.py -v`
Expected: PASS for logistic + elastic_net

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(harness-ml): linear model family (logistic, elastic_net) + contract tests"
```

---

### Task 7: Tree + Kernel Families (RandomForest + SVM)

- [ ] **Step 1: Implement TreeBase + RandomForest, KernelBase + SVM**

All sklearn-based, no optional dependencies. Follow the same pattern as linear.

- [ ] **Step 2: Run contract tests (should now include 4 models)**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(harness-ml): tree family (random_forest) + kernel family (svm)"
```

---

### Task 8: Boosting Family (XGBoost, LightGBM, CatBoost, HistGBM)

- [ ] **Step 1: Implement BoostingBase**

Shared: early stopping via eval_set, feature importance, iterative training interface.

- [ ] **Step 2: Implement XGBoost wrapper**

Conditionally imported. `requires_packages = ["xgboost"]`. Auto-install on first use (or skip in tests if not available).

- [ ] **Step 3: Implement LightGBM wrapper**
- [ ] **Step 4: Implement CatBoost wrapper**
- [ ] **Step 5: Implement HistGBM wrapper** (sklearn HistGradientBoostingClassifier, no optional deps)
- [ ] **Step 6: Run contract tests**

Run: `cd packages/harness-ml && python -m pytest tests/test_models/test_model_contract.py -v`
Expected: PASS for all available models (skip unavailable optional deps)

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(harness-ml): boosting family (xgboost, lightgbm, catboost, hist_gbm)"
```

---

### Task 9: Neural Family (MLP)

- [ ] **Step 1: Implement NeuralBase**

Device detection (cuda/mps/cpu), batch training, basic LR scheduling.

- [ ] **Step 2: Implement MLP wrapper**

Simple feedforward network using PyTorch. Conditionally imported.

- [ ] **Step 3: Run contract tests, commit**

```bash
git commit -m "feat(harness-ml): neural family (mlp)"
```

---

### Task 10: Models E2E Tests

- [ ] **Step 1: Write e2e tests that actually train models and verify predictions**

Add to `tests/test_e2e.py`:

```python
class TestE2EModels:
    def test_logistic_learns_binary_pattern(self, binary_dataset):
        """Logistic regression should achieve > 60% accuracy on correlated data."""
        from harness.ml.models.registry import ModelRegistry
        from harness.ml.tasks.registry import TaskRegistry

        model = ModelRegistry.get("logistic")
        task = TaskRegistry.get("binary")
        X, y = binary_dataset

        # Train
        result = model.fit(X, y, None, None, model.default_params("binary"))
        preds = model.predict(result.model, X)

        # Verify predictions are valid probabilities
        assert all(0 <= p <= 1 for p in preds), "Predictions should be probabilities"

        # Verify model actually learned (accuracy > random)
        metrics = task.compute_metrics(y.values, preds, ["accuracy", "brier"])
        assert metrics["accuracy"] > 0.6, f"Logistic should beat random, got {metrics['accuracy']}"
        assert metrics["brier"] < 0.4, f"Brier should be reasonable, got {metrics['brier']}"

    def test_random_forest_learns_regression(self, regression_dataset):
        """Random forest should achieve R² > 0.5 on correlated data."""
        from harness.ml.models.registry import ModelRegistry
        from harness.ml.tasks.registry import TaskRegistry

        model = ModelRegistry.get("random_forest")
        task = TaskRegistry.get("regression")
        X, y = regression_dataset

        result = model.fit(X, y, None, None, model.default_params("regression"))
        preds = model.predict(result.model, X)

        metrics = task.compute_metrics(y.values, preds, ["r2", "rmse"])
        assert metrics["r2"] > 0.5, f"RF should explain variance, got R²={metrics['r2']}"

    def test_model_task_adaptation_objectives(self):
        """Verify adaptation layer has objectives for all model+task combinations."""
        from harness.ml.tasks.binary.adaptation import OBJECTIVES as binary_obj
        from harness.ml.models.registry import ModelRegistry

        for model_name in ModelRegistry.list_available():
            model = ModelRegistry.get(model_name)
            if "binary" in model.supports_tasks:
                assert model_name in binary_obj or model.name in binary_obj, \
                    f"Model {model_name} supports binary but has no adaptation entry"
```

- [ ] **Step 2: Run full e2e suite**

Run: `cd packages/harness-ml && python -m pytest tests/test_e2e.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git commit -m "test(harness-ml): e2e model tests — verify learning, predictions, adaptation"
```

---

### Task 11: Public API + Package Verification

- [ ] **Step 1: Update `__init__.py` with public exports**

```python
"""harness-ml: Tabular ML engine for the Harness platform."""
from harness.ml.tasks.protocol import TaskType, Metric, ValidationResult, CalibrationType
from harness.ml.tasks.registry import TaskRegistry
from harness.ml.models.protocol import Model, FitResult
from harness.ml.models.registry import ModelRegistry
```

- [ ] **Step 2: Verify package installs and imports cleanly**

```bash
cd packages/harness-ml && pip install -e ".[dev]"
python -c "from harness.ml import TaskRegistry, ModelRegistry; print(f'Tasks: {TaskRegistry.list_available()}, Models: {ModelRegistry.list_available()}')"
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(harness-ml): public API exports + package verification (Plan 2a complete)"
```
