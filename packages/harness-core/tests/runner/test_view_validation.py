"""Tests for view DAG validation in the validator."""
from __future__ import annotations

from harnessml.core.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    FilterStep,
    GroupByStep,
    JoinStep,
    ProjectConfig,
    SourceConfig,
    ViewDef,
)
from harnessml.core.runner.validator import _validate_views


def _make_config(**data_overrides) -> ProjectConfig:
    data = DataConfig(
        sources={"raw": SourceConfig(name="raw", path="data/raw/test.csv")},
        **data_overrides,
    )
    return ProjectConfig(
        data=data,
        models={},
        ensemble=EnsembleDef(method="stacked"),
        backtest=BacktestConfig(cv_strategy="leave_one_out"),
    )


class TestViewValidation:
    """Tests for _validate_views."""

    def test_empty_views_valid(self):
        config = _make_config()
        errors = _validate_views(config, [])
        assert errors == []

    def test_features_view_null_valid(self):
        config = _make_config(features_view=None)
        errors = _validate_views(config, [])
        assert errors == []

    def test_valid_views_pass(self):
        config = _make_config(
            views={
                "clean": ViewDef(
                    source="raw",
                    steps=[FilterStep(expr="status == 'active'")],
                ),
            },
        )
        errors = _validate_views(config, [])
        assert errors == []

    def test_unknown_source_fails(self):
        config = _make_config(
            views={
                "bad_view": ViewDef(source="nonexistent"),
            },
        )
        errors = _validate_views(config, [])
        assert len(errors) == 1
        assert "unknown source 'nonexistent'" in errors[0]

    def test_unknown_join_other_fails(self):
        config = _make_config(
            views={
                "joined": ViewDef(
                    source="raw",
                    steps=[JoinStep(other="missing_table", on=["id"])],
                ),
            },
        )
        errors = _validate_views(config, [])
        assert len(errors) == 1
        assert "missing_table" in errors[0]
        assert "join" in errors[0]

    def test_cycle_detected(self):
        config = _make_config(
            views={
                "a": ViewDef(source="b"),
                "b": ViewDef(source="a"),
            },
        )
        errors = _validate_views(config, [])
        assert any("Cycle" in e or "cycle" in e.lower() for e in errors)

    def test_features_view_must_exist(self):
        config = _make_config(
            views={
                "clean": ViewDef(source="raw"),
            },
            features_view="nonexistent_view",
        )
        errors = _validate_views(config, [])
        assert len(errors) == 1
        assert "nonexistent_view" in errors[0]
        assert "features_view" in errors[0]

    def test_unsupported_agg_fails(self):
        config = _make_config(
            views={
                "agg_view": ViewDef(
                    source="raw",
                    steps=[
                        GroupByStep(
                            keys=["team_id"],
                            aggs={"score": "bogus_agg"},
                        ),
                    ],
                ),
            },
        )
        errors = _validate_views(config, [])
        assert len(errors) == 1
        assert "bogus_agg" in errors[0]
        assert "unsupported aggregation" in errors[0]

    def test_valid_view_chain(self):
        """Views can reference other views as sources."""
        config = _make_config(
            views={
                "clean": ViewDef(source="raw", steps=[FilterStep(expr="x > 0")]),
                "enriched": ViewDef(source="clean"),
            },
            features_view="enriched",
        )
        errors = _validate_views(config, [])
        assert errors == []

    def test_valid_join_other_is_view(self):
        """Join step can reference another view."""
        config = _make_config(
            views={
                "lookup": ViewDef(source="raw"),
                "main": ViewDef(
                    source="raw",
                    steps=[JoinStep(other="lookup", on=["id"])],
                ),
            },
        )
        errors = _validate_views(config, [])
        assert errors == []
