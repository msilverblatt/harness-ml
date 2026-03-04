"""Tests for feature discovery tools."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.feature_discovery import (
    compute_feature_correlations,
    compute_feature_importance,
    detect_redundant_features,
    format_discovery_report,
    suggest_feature_groups,
    suggest_features,
)
from easyml.core.runner.schema import FeatureDef, FeatureType


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """Dataset with known feature properties."""
    rng = np.random.default_rng(42)
    n = 200

    seed = rng.normal(0, 3, n)
    result = (seed + rng.normal(0, 1, n) > 0).astype(float)

    return pd.DataFrame({
        "result": result,
        "Season": np.repeat([2020, 2021, 2022, 2023], 50),
        "diff_prior": seed,
        "diff_strong_signal": seed * 2 + rng.normal(0, 0.1, n),
        "diff_weak_signal": rng.normal(0, 10, n),
        "diff_redundant_a": seed * 1.5,
        "diff_redundant_b": seed * 1.5 + rng.normal(0, 0.01, n),  # near-perfect correlation
        "diff_noise": rng.normal(0, 1, n),
        "diff_bt_barthag": rng.normal(0, 1, n),
        "diff_bt_adj_o": rng.normal(0, 1, n),
        "diff_sr_srs": rng.normal(0, 1, n),
        "diff_sr_sos": rng.normal(0, 1, n),
        "non_diff_col": rng.normal(0, 1, n),  # should be excluded by prefix filter
    })


# -----------------------------------------------------------------------
# compute_feature_correlations
# -----------------------------------------------------------------------

class TestCorrelations:
    def test_returns_sorted_by_abs_correlation(self, sample_df):
        result = compute_feature_correlations(sample_df)
        assert list(result.columns) == ["feature", "correlation", "abs_correlation"]
        # Should be sorted descending
        assert result["abs_correlation"].is_monotonic_decreasing

    def test_includes_all_numeric_except_target(self, sample_df):
        result = compute_feature_correlations(sample_df)
        # All numeric columns except "result" should be included
        assert "non_diff_col" in result["feature"].values
        assert "result" not in result["feature"].values

    def test_strong_signal_ranks_high(self, sample_df):
        result = compute_feature_correlations(sample_df, top_n=3)
        top_features = result["feature"].tolist()
        # diff_strong_signal and diff_prior should be near the top
        assert "diff_strong_signal" in top_features or "diff_prior" in top_features

    def test_top_n_limits_output(self, sample_df):
        result = compute_feature_correlations(sample_df, top_n=3)
        assert len(result) == 3

    def test_top_n_zero_returns_all(self, sample_df):
        result = compute_feature_correlations(sample_df, top_n=0)
        numeric_non_target = [
            c for c in sample_df.select_dtypes(include=[np.number]).columns
            if c != "result"
        ]
        assert len(result) == len(numeric_non_target)

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"result": [1, 0]})
        result = compute_feature_correlations(df)
        assert len(result) == 0


# -----------------------------------------------------------------------
# compute_feature_importance
# -----------------------------------------------------------------------

class TestImportance:
    def test_xgboost_method_returns_sorted(self, sample_df):
        result = compute_feature_importance(sample_df, method="xgboost")
        assert list(result.columns) == ["feature", "importance"]
        assert result["importance"].is_monotonic_decreasing

    def test_mutual_info_method(self, sample_df):
        result = compute_feature_importance(sample_df, method="mutual_info")
        assert len(result) > 0
        assert all(result["importance"] >= 0)

    def test_importances_sum_to_one(self, sample_df):
        result = compute_feature_importance(sample_df, method="xgboost", top_n=0)
        assert result["importance"].sum() == pytest.approx(1.0, abs=0.01)

    def test_top_n_limits(self, sample_df):
        result = compute_feature_importance(sample_df, top_n=3)
        assert len(result) == 3

    def test_invalid_method_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unknown importance method"):
            compute_feature_importance(sample_df, method="invalid")

    def test_includes_all_numeric_except_target(self, sample_df):
        result = compute_feature_importance(sample_df, top_n=0)
        assert "result" not in result["feature"].values
        # Should include all numeric non-target columns
        assert "non_diff_col" in result["feature"].values


# -----------------------------------------------------------------------
# detect_redundant_features
# -----------------------------------------------------------------------

class TestRedundancy:
    def test_finds_near_perfect_correlation(self, sample_df):
        pairs = detect_redundant_features(sample_df, threshold=0.95)
        pair_features = {(a, b) for a, b, _ in pairs}
        # redundant_a and redundant_b should be flagged
        assert any(
            ("diff_redundant_a" in {a, b} and "diff_redundant_b" in {a, b})
            for a, b in pair_features
        )

    def test_high_threshold_finds_fewer(self, sample_df):
        pairs_low = detect_redundant_features(sample_df, threshold=0.8)
        pairs_high = detect_redundant_features(sample_df, threshold=0.99)
        assert len(pairs_high) <= len(pairs_low)

    def test_sorted_by_abs_correlation(self, sample_df):
        pairs = detect_redundant_features(sample_df, threshold=0.5)
        if len(pairs) > 1:
            abs_corrs = [abs(r) for _, _, r in pairs]
            assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_no_duplicates(self, sample_df):
        pairs = detect_redundant_features(sample_df, threshold=0.5)
        # Each pair should appear only once (upper triangle)
        seen = set()
        for a, b, _ in pairs:
            key = tuple(sorted([a, b]))
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

    def test_empty_with_single_feature(self):
        df = pd.DataFrame({"feat_a": [1.0, 2.0], "result": [0, 1]})
        # Only one feature column → can't form a pair
        pairs = detect_redundant_features(df, feature_columns=["feat_a"])
        assert pairs == []


# -----------------------------------------------------------------------
# suggest_feature_groups
# -----------------------------------------------------------------------

class TestFeatureGroups:
    def test_groups_by_first_segment(self, sample_df):
        groups = suggest_feature_groups(sample_df)
        # All diff_ columns share first segment "diff"
        assert "diff" in groups
        # non_diff_col shares first segment "non"
        assert "non" in groups

    def test_diff_group_has_all_diff_cols(self, sample_df):
        groups = suggest_feature_groups(sample_df)
        diff_group = groups["diff"]
        assert "diff_bt_barthag" in diff_group
        assert "diff_bt_adj_o" in diff_group
        assert "diff_prior" in diff_group

    def test_includes_all_numeric_columns(self, sample_df):
        groups = suggest_feature_groups(sample_df)
        all_features = [f for feats in groups.values() for f in feats]
        assert "non_diff_col" in all_features


# -----------------------------------------------------------------------
# suggest_features
# -----------------------------------------------------------------------

class TestSuggestFeatures:
    def test_returns_requested_count(self, sample_df):
        features = suggest_features(sample_df, count=5)
        assert len(features) == 5

    def test_excludes_specified(self, sample_df):
        features = suggest_features(
            sample_df, count=5, exclude=["diff_prior"]
        )
        assert "diff_prior" not in features

    def test_filters_redundant(self, sample_df):
        features = suggest_features(sample_df, count=10)
        # Should not include both redundant_a and redundant_b
        has_a = "diff_redundant_a" in features
        has_b = "diff_redundant_b" in features
        assert not (has_a and has_b), "Both redundant features should not be suggested"


# -----------------------------------------------------------------------
# format_discovery_report
# -----------------------------------------------------------------------

class TestFormatReport:
    def test_produces_markdown(self, sample_df):
        corr = compute_feature_correlations(sample_df)
        imp = compute_feature_importance(sample_df, method="mutual_info")
        red = detect_redundant_features(sample_df, threshold=0.95)
        groups = suggest_feature_groups(sample_df)

        report = format_discovery_report(corr, imp, red, groups)
        assert "## Feature Discovery Report" in report
        assert "### Target Correlations" in report
        assert "### Feature Importance" in report
        assert "### Feature Groups" in report

    def test_includes_redundancy_section_when_present(self, sample_df):
        corr = compute_feature_correlations(sample_df)
        imp = compute_feature_importance(sample_df, method="mutual_info")
        red = detect_redundant_features(sample_df, threshold=0.95)
        groups = suggest_feature_groups(sample_df)

        report = format_discovery_report(corr, imp, red, groups)
        assert "Redundant Pairs" in report

    def test_no_redundancy_section_when_empty(self, sample_df):
        corr = compute_feature_correlations(sample_df)
        imp = compute_feature_importance(sample_df, method="mutual_info")
        groups = suggest_feature_groups(sample_df)

        report = format_discovery_report(corr, imp, [], groups)
        assert "Redundant Pairs" not in report


# -----------------------------------------------------------------------
# Store-aware tests (feature_defs parameter)
# -----------------------------------------------------------------------

@pytest.fixture()
def sample_feature_defs():
    """Feature definitions matching sample_df columns."""
    return {
        "diff_prior": FeatureDef(
            name="diff_prior", type=FeatureType.PAIRWISE,
            formula="seed_num_a - seed_num_b", category="seeding",
        ),
        "diff_strong_signal": FeatureDef(
            name="diff_strong_signal", type=FeatureType.PAIRWISE,
            formula="strong_a - strong_b", category="efficiency",
        ),
        "diff_weak_signal": FeatureDef(
            name="diff_weak_signal", type=FeatureType.PAIRWISE,
            formula="weak_a - weak_b", category="general",
        ),
        "diff_redundant_a": FeatureDef(
            name="diff_redundant_a", type=FeatureType.PAIRWISE,
            formula="red_a_1 - red_a_2", category="general",
        ),
        "diff_redundant_b": FeatureDef(
            name="diff_redundant_b", type=FeatureType.PAIRWISE,
            formula="red_b_1 - red_b_2", category="general",
        ),
        "diff_noise": FeatureDef(
            name="diff_noise", type=FeatureType.PAIRWISE,
            formula="noise_a - noise_b", category="general",
        ),
        "diff_bt_barthag": FeatureDef(
            name="diff_bt_barthag", type=FeatureType.PAIRWISE,
            formula="bt_barthag_a - bt_barthag_b", category="barttorvik",
        ),
        "diff_bt_adj_o": FeatureDef(
            name="diff_bt_adj_o", type=FeatureType.PAIRWISE,
            formula="bt_adj_o_a - bt_adj_o_b", category="barttorvik",
        ),
        "diff_sr_srs": FeatureDef(
            name="diff_sr_srs", type=FeatureType.PAIRWISE,
            formula="sr_srs_a - sr_srs_b", category="sports_ref",
        ),
        "diff_sr_sos": FeatureDef(
            name="diff_sr_sos", type=FeatureType.PAIRWISE,
            formula="sr_sos_a - sr_sos_b", category="sports_ref",
        ),
    }


class TestStoreAwareCorrelations:
    def test_type_column_added(self, sample_df, sample_feature_defs):
        result = compute_feature_correlations(
            sample_df, feature_defs=sample_feature_defs,
        )
        assert "type" in result.columns
        # Known features should have type "pairwise"
        seed_row = result[result["feature"] == "diff_prior"]
        assert seed_row["type"].iloc[0] == "pairwise"

    def test_unregistered_features_have_empty_type(self, sample_df, sample_feature_defs):
        result = compute_feature_correlations(
            sample_df, feature_defs=sample_feature_defs,
        )
        # non_diff_col is not in feature_defs
        non_diff = result[result["feature"] == "non_diff_col"]
        if not non_diff.empty:
            assert non_diff["type"].iloc[0] == ""

    def test_no_type_column_without_defs(self, sample_df):
        result = compute_feature_correlations(sample_df)
        assert "type" not in result.columns


class TestStoreAwareImportance:
    def test_type_column_added(self, sample_df, sample_feature_defs):
        result = compute_feature_importance(
            sample_df, feature_defs=sample_feature_defs, method="mutual_info",
        )
        assert "type" in result.columns

    def test_no_type_column_without_defs(self, sample_df):
        result = compute_feature_importance(sample_df, method="mutual_info")
        assert "type" not in result.columns


class TestStoreAwareGroups:
    def test_groups_by_type_category(self, sample_df, sample_feature_defs):
        groups = suggest_feature_groups(
            sample_df, feature_defs=sample_feature_defs,
        )
        # Features registered as pairwise/barttorvik should be grouped together
        assert "pairwise/barttorvik" in groups
        assert "diff_bt_barthag" in groups["pairwise/barttorvik"]
        assert "diff_bt_adj_o" in groups["pairwise/barttorvik"]

    def test_unregistered_features_in_other_group(self, sample_df, sample_feature_defs):
        groups = suggest_feature_groups(
            sample_df, feature_defs=sample_feature_defs,
        )
        # non_diff_col is not registered, should be in other/non
        assert "other/non" in groups
        assert "non_diff_col" in groups["other/non"]

    def test_without_defs_uses_prefix(self, sample_df):
        groups = suggest_feature_groups(sample_df)
        # Without defs, should group by prefix as before
        assert "diff" in groups
        assert "non" in groups


class TestStoreAwareReport:
    def test_report_includes_type_column(self, sample_df, sample_feature_defs):
        corr = compute_feature_correlations(
            sample_df, feature_defs=sample_feature_defs,
        )
        imp = compute_feature_importance(
            sample_df, method="mutual_info", feature_defs=sample_feature_defs,
        )
        red = detect_redundant_features(sample_df, threshold=0.95)
        groups = suggest_feature_groups(
            sample_df, feature_defs=sample_feature_defs,
        )

        report = format_discovery_report(corr, imp, red, groups)
        assert "| Type |" in report
        assert "pairwise/barttorvik" in report
