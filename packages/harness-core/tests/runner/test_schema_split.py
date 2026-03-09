"""Tests for DataConfig sub-config decomposition with backward compatibility."""
from harnessml.core.runner.schema import (
    DataCleaningConfig,
    DataConfig,
    DataPathsConfig,
    MLProblemConfig,
)


class TestDataConfigFlatKwargs:
    """Flat kwargs should be routed to the correct sub-config."""

    def test_flat_paths(self):
        dc = DataConfig(raw_dir="mydata/raw", outputs_dir="/tmp/out")
        assert dc.paths.raw_dir == "mydata/raw"
        assert dc.paths.outputs_dir == "/tmp/out"
        # Property accessors
        assert dc.raw_dir == "mydata/raw"
        assert dc.outputs_dir == "/tmp/out"

    def test_flat_ml_problem(self):
        dc = DataConfig(task="regression", target_column="price", key_columns=["id"])
        assert dc.ml_problem.task == "regression"
        assert dc.ml_problem.target_column == "price"
        assert dc.ml_problem.key_columns == ["id"]
        # Property accessors
        assert dc.task == "regression"
        assert dc.target_column == "price"
        assert dc.key_columns == ["id"]

    def test_flat_cleaning(self):
        dc = DataConfig(column_renames={"old": "new"})
        assert dc.cleaning.column_renames == {"old": "new"}
        assert dc.column_renames == {"old": "new"}

    def test_defaults(self):
        dc = DataConfig()
        assert dc.raw_dir == "data/raw"
        assert dc.task == "classification"
        assert dc.target_column == "result"
        assert dc.column_renames == {}

    def test_mixed_flat_and_nested(self):
        dc = DataConfig(
            raw_dir="custom/raw",
            ml_problem={"task": "ranking", "target_column": "rank"},
        )
        assert dc.raw_dir == "custom/raw"
        assert dc.task == "ranking"
        assert dc.target_column == "rank"


class TestDataConfigPropertySetters:
    """Property setters should mutate the sub-config."""

    def test_set_raw_dir(self):
        dc = DataConfig()
        dc.raw_dir = "new/raw"
        assert dc.paths.raw_dir == "new/raw"
        assert dc.raw_dir == "new/raw"

    def test_set_target_column(self):
        dc = DataConfig()
        dc.target_column = "label"
        assert dc.ml_problem.target_column == "label"

    def test_set_column_renames(self):
        dc = DataConfig()
        dc.column_renames = {"a": "b"}
        assert dc.cleaning.column_renames == {"a": "b"}


class TestSubConfigsStandalone:
    """Sub-configs should work independently."""

    def test_data_paths_config(self):
        p = DataPathsConfig(raw_dir="x", features_file="f.parquet")
        assert p.raw_dir == "x"
        assert p.features_file == "f.parquet"
        assert p.outputs_dir is None

    def test_ml_problem_config(self):
        m = MLProblemConfig(task="regression")
        assert m.task == "regression"
        assert m.target_column == "result"

    def test_data_cleaning_config(self):
        c = DataCleaningConfig()
        assert c.column_renames == {}


class TestResolveTargetStillWorks:
    """resolve_target should work with the new structure."""

    def test_default_target(self):
        dc = DataConfig(task="binary", target_column="win")
        col, task, metrics = dc.resolve_target()
        assert col == "win"
        assert task == "binary"
        assert metrics == []
