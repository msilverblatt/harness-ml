"""Tests for model dependency graph resolution."""
from __future__ import annotations

import pytest

from easyml.core.runner.dag import (
    build_provider_map,
    detect_cycle,
    infer_dependencies,
    topological_waves,
)
from easyml.core.runner.schema import ModelDef


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _model(features=None, provides=None, **kwargs):
    """Shorthand to create a ModelDef."""
    return ModelDef(
        type="xgboost",
        features=features or [],
        provides=provides or [],
        **kwargs,
    )


# -----------------------------------------------------------------------
# Tests: build_provider_map
# -----------------------------------------------------------------------

class TestBuildProviderMap:

    def test_no_providers(self):
        models = {"m1": _model(), "m2": _model()}
        assert build_provider_map(models) == {}

    def test_single_provider(self):
        models = {
            "survival": _model(provides=["surv_e8", "surv_f4"]),
            "xgb_core": _model(),
        }
        pmap = build_provider_map(models)
        assert pmap == {"surv_e8": "survival", "surv_f4": "survival"}

    def test_multiple_providers(self):
        models = {
            "provider_a": _model(provides=["feat_a"]),
            "provider_b": _model(provides=["feat_b", "feat_c"]),
            "consumer": _model(),
        }
        pmap = build_provider_map(models)
        assert pmap == {
            "feat_a": "provider_a",
            "feat_b": "provider_b",
            "feat_c": "provider_b",
        }

    def test_duplicate_provides_raises(self):
        models = {
            "provider_a": _model(provides=["shared_feat"]),
            "provider_b": _model(provides=["shared_feat"]),
        }
        with pytest.raises(ValueError, match="provided by both"):
            build_provider_map(models)


# -----------------------------------------------------------------------
# Tests: infer_dependencies
# -----------------------------------------------------------------------

class TestInferDependencies:

    def test_no_dependencies(self):
        models = {
            "m1": _model(features=["diff_prior"]),
            "m2": _model(features=["diff_win_pct"]),
        }
        deps = infer_dependencies(models)
        assert deps == {"m1": set(), "m2": set()}

    def test_diff_prefix_dependency(self):
        models = {
            "survival": _model(provides=["surv_e8"]),
            "trajectory": _model(features=["diff_prior", "diff_surv_e8"]),
        }
        deps = infer_dependencies(models)
        assert deps["trajectory"] == {"survival"}
        assert deps["survival"] == set()

    def test_raw_name_dependency(self):
        """Consumer can reference provider output without diff_ prefix."""
        models = {
            "provider": _model(provides=["predicted_margin"]),
            "consumer": _model(features=["predicted_margin", "diff_prior"]),
        }
        deps = infer_dependencies(models)
        assert deps["consumer"] == {"provider"}

    def test_no_self_dependency(self):
        """A model should not depend on itself."""
        models = {
            "m1": _model(
                features=["diff_feat_a"],
                provides=["feat_a"],
            ),
        }
        deps = infer_dependencies(models)
        assert deps["m1"] == set()

    def test_diamond_dependencies(self):
        """A -> B, A -> C, B -> D, C -> D."""
        models = {
            "base": _model(provides=["feat_base"]),
            "mid_a": _model(features=["diff_feat_base"], provides=["feat_a"]),
            "mid_b": _model(features=["diff_feat_base"], provides=["feat_b"]),
            "top": _model(features=["diff_feat_a", "diff_feat_b"]),
        }
        deps = infer_dependencies(models)
        assert deps["base"] == set()
        assert deps["mid_a"] == {"base"}
        assert deps["mid_b"] == {"base"}
        assert deps["top"] == {"mid_a", "mid_b"}

    def test_precomputed_provider_map(self):
        """Can pass a pre-built provider map."""
        models = {
            "survival": _model(provides=["surv_e8"]),
            "trajectory": _model(features=["diff_surv_e8"]),
        }
        pmap = build_provider_map(models)
        deps = infer_dependencies(models, provider_map=pmap)
        assert deps["trajectory"] == {"survival"}


# -----------------------------------------------------------------------
# Tests: topological_waves
# -----------------------------------------------------------------------

class TestTopologicalWaves:

    def test_empty_graph(self):
        assert topological_waves({}) == []

    def test_single_model(self):
        waves = topological_waves({"m1": set()})
        assert waves == [["m1"]]

    def test_all_independent(self):
        waves = topological_waves({
            "m1": set(),
            "m2": set(),
            "m3": set(),
        })
        assert len(waves) == 1
        assert sorted(waves[0]) == ["m1", "m2", "m3"]

    def test_linear_chain(self):
        waves = topological_waves({
            "a": set(),
            "b": {"a"},
            "c": {"b"},
        })
        assert waves == [["a"], ["b"], ["c"]]

    def test_diamond(self):
        waves = topological_waves({
            "base": set(),
            "mid_a": {"base"},
            "mid_b": {"base"},
            "top": {"mid_a", "mid_b"},
        })
        assert waves[0] == ["base"]
        assert sorted(waves[1]) == ["mid_a", "mid_b"]
        assert waves[2] == ["top"]

    def test_mixed_independent_and_dependent(self):
        waves = topological_waves({
            "independent": set(),
            "provider": set(),
            "consumer": {"provider"},
        })
        assert len(waves) == 2
        assert sorted(waves[0]) == ["independent", "provider"]
        assert waves[1] == ["consumer"]

    def test_cycle_raises(self):
        with pytest.raises(ValueError, match="cycle detected"):
            topological_waves({
                "a": {"b"},
                "b": {"a"},
            })

    def test_three_node_cycle_raises(self):
        with pytest.raises(ValueError, match="cycle detected"):
            topological_waves({
                "a": {"c"},
                "b": {"a"},
                "c": {"b"},
            })

    def test_partial_cycle_raises(self):
        """Graph with one acyclic node and a 2-node cycle."""
        with pytest.raises(ValueError, match="cycle detected"):
            topological_waves({
                "ok": set(),
                "bad_a": {"bad_b"},
                "bad_b": {"bad_a"},
            })

    def test_waves_are_sorted(self):
        """Models within each wave are sorted alphabetically."""
        waves = topological_waves({
            "z_model": set(),
            "a_model": set(),
            "m_model": set(),
        })
        assert waves[0] == ["a_model", "m_model", "z_model"]


# -----------------------------------------------------------------------
# Tests: detect_cycle
# -----------------------------------------------------------------------

class TestDetectCycle:

    def test_no_cycle(self):
        assert detect_cycle({"a": set(), "b": {"a"}}) is None

    def test_simple_cycle(self):
        result = detect_cycle({"a": {"b"}, "b": {"a"}})
        assert result is not None
        assert set(result) == {"a", "b"}

    def test_cycle_with_acyclic_nodes(self):
        result = detect_cycle({
            "ok": set(),
            "bad_a": {"bad_b"},
            "bad_b": {"bad_a"},
        })
        assert result is not None
        assert "ok" not in result
        assert set(result) == {"bad_a", "bad_b"}


# -----------------------------------------------------------------------
# Tests: end-to-end (models → waves)
# -----------------------------------------------------------------------

class TestEndToEnd:

    def test_survival_trajectory_pattern(self):
        """The mm survival → trajectory pattern."""
        models = {
            "logreg_seed": _model(features=["diff_prior"]),
            "xgb_core": _model(features=["diff_prior", "diff_sr_srs"]),
            "survival": _model(
                features=["seed_num", "win_pct"],
                provides=["surv_e8", "surv_f4"],
                provides_level="entity",
                include_in_ensemble=False,
            ),
            "xgb_trajectory": _model(
                features=["diff_prior", "diff_surv_e8", "diff_surv_f4"],
            ),
        }
        pmap = build_provider_map(models)
        deps = infer_dependencies(models, pmap)
        waves = topological_waves(deps)

        # Wave 0: logreg_seed, survival, xgb_core (all independent)
        # Wave 1: xgb_trajectory (depends on survival)
        assert len(waves) == 2
        assert "survival" in waves[0]
        assert "logreg_seed" in waves[0]
        assert "xgb_core" in waves[0]
        assert waves[1] == ["xgb_trajectory"]

    def test_chained_providers(self):
        """Provider A → provider B → consumer C."""
        models = {
            "base": _model(provides=["feat_base"]),
            "mid": _model(
                features=["diff_feat_base"],
                provides=["feat_mid"],
            ),
            "top": _model(features=["diff_feat_mid"]),
        }
        pmap = build_provider_map(models)
        deps = infer_dependencies(models, pmap)
        waves = topological_waves(deps)

        assert waves == [["base"], ["mid"], ["top"]]
