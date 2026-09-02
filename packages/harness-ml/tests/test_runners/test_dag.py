import pytest

from harness.ml.config.models import SingleModelConfig
from harness.ml.runners.dag import ModelDAG


def make_config(name: str, depends_on: list[str] = None) -> SingleModelConfig:
    return SingleModelConfig(
        name=name,
        model_type="xgb",
        depends_on=depends_on or [],
    )


class TestTopologicalWaves:
    def test_no_deps_single_wave(self):
        models = {
            "a": make_config("a"),
            "b": make_config("b"),
            "c": make_config("c"),
        }
        dag = ModelDAG(models)
        waves = dag.topological_waves()
        assert len(waves) == 1
        assert sorted(waves[0]) == ["a", "b", "c"]

    def test_linear_chain_three_waves(self):
        models = {
            "a": make_config("a"),
            "b": make_config("b", depends_on=["a"]),
            "c": make_config("c", depends_on=["b"]),
        }
        dag = ModelDAG(models)
        waves = dag.topological_waves()
        assert len(waves) == 3
        assert waves[0] == ["a"]
        assert waves[1] == ["b"]
        assert waves[2] == ["c"]

    def test_parallel_models_same_wave(self):
        models = {
            "a": make_config("a"),
            "b": make_config("b"),
        }
        dag = ModelDAG(models)
        waves = dag.topological_waves()
        assert len(waves) == 1
        assert sorted(waves[0]) == ["a", "b"]

    def test_diamond_two_waves(self):
        # A and B no deps; C depends on both A and B
        models = {
            "a": make_config("a"),
            "b": make_config("b"),
            "c": make_config("c", depends_on=["a", "b"]),
        }
        dag = ModelDAG(models)
        waves = dag.topological_waves()
        assert len(waves) == 2
        assert sorted(waves[0]) == ["a", "b"]
        assert waves[1] == ["c"]

    def test_cycle_raises_value_error(self):
        models = {
            "a": make_config("a", depends_on=["b"]),
            "b": make_config("b", depends_on=["a"]),
        }
        dag = ModelDAG(models)
        with pytest.raises(ValueError, match="Circular dependency"):
            dag.topological_waves()


class TestValidate:
    def test_missing_dep_detected(self):
        models = {
            "a": make_config("a", depends_on=["nonexistent"]),
        }
        dag = ModelDAG(models)
        errors = dag.validate()
        assert any("nonexistent" in e for e in errors)

    def test_valid_dag_no_errors(self):
        models = {
            "a": make_config("a"),
            "b": make_config("b", depends_on=["a"]),
        }
        dag = ModelDAG(models)
        assert dag.validate() == []

    def test_cycle_reported_in_validate(self):
        models = {
            "x": make_config("x", depends_on=["y"]),
            "y": make_config("y", depends_on=["x"]),
        }
        dag = ModelDAG(models)
        errors = dag.validate()
        assert len(errors) > 0


class TestDependencies:
    def test_returns_correct_set(self):
        models = {
            "a": make_config("a"),
            "b": make_config("b", depends_on=["a"]),
        }
        dag = ModelDAG(models)
        assert dag.dependencies("b") == {"a"}
        assert dag.dependencies("a") == set()

    def test_returns_empty_set_for_unknown(self):
        models = {"a": make_config("a")}
        dag = ModelDAG(models)
        assert dag.dependencies("missing") == set()
