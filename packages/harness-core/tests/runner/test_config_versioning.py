"""Tests for config_version and compute_config_hash on ProjectConfig."""
from harnessml.core.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ModelDef,
    ProjectConfig,
)


def _minimal_config(**overrides):
    defaults = dict(
        data=DataConfig(),
        models={"xgb": ModelDef(type="xgboost")},
        ensemble=EnsembleDef(method="stacked"),
        backtest=BacktestConfig(cv_strategy="leave_one_out"),
    )
    defaults.update(overrides)
    return ProjectConfig(**defaults)


class TestConfigVersion:
    def test_default_version(self):
        pc = _minimal_config()
        assert pc.config_version == "1.0"

    def test_custom_version(self):
        pc = _minimal_config(config_version="2.0")
        assert pc.config_version == "2.0"


class TestComputeConfigHash:
    def test_hash_is_hex_string(self):
        pc = _minimal_config()
        h = pc.compute_config_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        pc1 = _minimal_config()
        pc2 = _minimal_config()
        assert pc1.compute_config_hash() == pc2.compute_config_hash()

    def test_version_excluded(self):
        """Changing config_version should not change the hash."""
        pc1 = _minimal_config(config_version="1.0")
        pc2 = _minimal_config(config_version="2.0")
        assert pc1.compute_config_hash() == pc2.compute_config_hash()

    def test_content_change_changes_hash(self):
        pc1 = _minimal_config()
        pc2 = _minimal_config(
            models={"lr": ModelDef(type="logistic_regression")},
        )
        assert pc1.compute_config_hash() != pc2.compute_config_hash()
