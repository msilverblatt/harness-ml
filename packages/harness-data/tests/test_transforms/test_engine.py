import pandas as pd
import pytest
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


class TestTransformEngine:
    def test_register_and_list_steps(self):
        engine = TransformEngine()
        assert "filter" in engine.available_steps()

    def test_apply_unknown_step_raises(self, sample_df):
        engine = TransformEngine()
        config = StepConfig(op="nonexistent_step")
        with pytest.raises(ValueError, match="Unknown"):
            engine.apply_step(sample_df, config)

    def test_run_pipeline_empty_steps(self, sample_df):
        engine = TransformEngine()
        result = engine.run_pipeline(sample_df, [])
        assert len(result) == len(sample_df)

    def test_run_pipeline_chained_steps(self, sample_df):
        engine = TransformEngine()
        steps = [
            StepConfig(op="filter", params={"expr": "score > 80"}),
            StepConfig(op="select", params={"columns": ["name", "score"]}),
        ]
        result = engine.run_pipeline(sample_df, steps)
        assert "name" in result.columns
        assert "grade" not in result.columns
        assert all(result["score"] > 80)
