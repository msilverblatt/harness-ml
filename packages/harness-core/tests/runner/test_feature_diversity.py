"""Tests for feature diversity analysis module."""

import pytest
from harnessml.core.runner.feature_diversity import (
    compute_diversity_score,
    compute_overlap_matrix,
    find_redundant_features,
    format_diversity_report,
    suggest_removal,
)


class TestOverlapMatrix:
    """Tests for compute_overlap_matrix."""

    def test_disjoint(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["d", "e", "f"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        assert matrix[("m1", "m2")] == 0.0

    def test_identical(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["a", "b", "c"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        assert matrix[("m1", "m2")] == 1.0

    def test_partial(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d"], "active": True},
            "m2": {"features": ["c", "d", "e", "f"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        # Jaccard: |intersection|/|union| = 2/6
        assert matrix[("m1", "m2")] == pytest.approx(2.0 / 6.0)

    def test_symmetry(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["b", "c", "d"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        assert matrix[("m1", "m2")] == matrix[("m2", "m1")]

    def test_diagonal(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        assert matrix[("m1", "m1")] == 1.0
        assert matrix[("m2", "m2")] == 1.0

    def test_inactive_filtered(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["a", "b", "c"], "active": False},
            "m3": {"features": ["d", "e", "f"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        # m2 should be excluded
        assert ("m1", "m2") not in matrix
        assert ("m2", "m1") not in matrix
        assert ("m2", "m2") not in matrix
        assert ("m1", "m3") in matrix

    def test_single_model(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
        }
        matrix = compute_overlap_matrix(models)
        assert matrix[("m1", "m1")] == 1.0
        assert len(matrix) == 1


class TestDiversityScore:
    """Tests for compute_diversity_score."""

    def test_perfect(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        assert compute_diversity_score(models) == 1.0

    def test_zero(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["a", "b"], "active": True},
        }
        assert compute_diversity_score(models) == 0.0

    def test_partial(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["b", "c", "d"], "active": True},
        }
        score = compute_diversity_score(models)
        assert 0.0 < score < 1.0

    def test_single_model(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
        }
        assert compute_diversity_score(models) == 1.0

    def test_empty(self):
        models = {}
        assert compute_diversity_score(models) == 1.0

    def test_inactive(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["a", "b"], "active": False},
        }
        # Only m1 active -> single model -> 1.0
        assert compute_diversity_score(models) == 1.0


class TestFindRedundant:
    """Tests for find_redundant_features."""

    def test_above_threshold(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d", "e"], "active": True},
            "m2": {"features": ["a", "b", "c", "d", "f"], "active": True},
        }
        # Jaccard: 4/6 ~ 0.667
        results = find_redundant_features(models, threshold=0.5)
        assert len(results) == 1
        assert results[0]["model_a"] == "m1"
        assert results[0]["model_b"] == "m2"

    def test_below_threshold(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        results = find_redundant_features(models, threshold=0.5)
        assert len(results) == 0

    def test_sorted(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["a", "b", "d"], "active": True},
            "m3": {"features": ["a", "b", "c"], "active": True},
        }
        results = find_redundant_features(models, threshold=0.0)
        # m1-m3 overlap=1.0 should be first
        assert results[0]["overlap"] >= results[1]["overlap"]
        for i in range(len(results) - 1):
            assert results[i]["overlap"] >= results[i + 1]["overlap"]

    def test_shared_features(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["b", "c", "d"], "active": True},
        }
        results = find_redundant_features(models, threshold=0.0)
        assert len(results) == 1
        assert set(results[0]["shared_features"]) == {"b", "c"}

    def test_default_threshold(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d", "e"], "active": True},
            "m2": {"features": ["a", "b", "c", "d", "f"], "active": True},
        }
        # Jaccard 4/6 ~ 0.667 < 0.8 default
        results = find_redundant_features(models)
        assert len(results) == 0

    def test_no_redundant(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
            "m3": {"features": ["e", "f"], "active": True},
        }
        results = find_redundant_features(models, threshold=0.5)
        assert results == []


class TestSuggestRemoval:
    """Tests for suggest_removal."""

    def test_already_diverse(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        suggestions = suggest_removal(models, target_score=0.7)
        assert suggestions == []

    def test_suggests_features(self):
        models = {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["a", "b", "c"], "active": True},
        }
        suggestions = suggest_removal(models, target_score=0.5)
        assert len(suggestions) > 0
        # Each suggestion should have model, feature keys
        for s in suggestions:
            assert "model" in s
            assert "feature" in s

    def test_drops_from_larger(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d", "e"], "active": True},
            "m2": {"features": ["a", "b", "c"], "active": True},
        }
        suggestions = suggest_removal(models, target_score=0.5)
        if suggestions:
            # Should prefer removing from m1 (larger)
            assert suggestions[0]["model"] == "m1"

    def test_target_score(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d"], "active": True},
            "m2": {"features": ["a", "b", "c", "e"], "active": True},
        }
        # With target_score=0.0, everything is already diverse enough
        suggestions_low = suggest_removal(models, target_score=0.0)
        assert suggestions_low == []


class TestFormatReport:
    """Tests for format_diversity_report."""

    def test_contains_score(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        report = format_diversity_report(models)
        assert "1.0" in report or "1.00" in report

    def test_contains_matrix(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        report = format_diversity_report(models)
        assert "m1" in report
        assert "m2" in report

    def test_contains_status(self):
        models = {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
        report = format_diversity_report(models)
        # Should contain pass or fail indicator
        report_lower = report.lower()
        assert "pass" in report_lower or "fail" in report_lower

    def test_redundant_shown(self):
        models = {
            "m1": {"features": ["a", "b", "c", "d", "e"], "active": True},
            "m2": {"features": ["a", "b", "c", "d", "e"], "active": True},
        }
        report = format_diversity_report(models)
        # Should mention redundant pairs
        assert "redundant" in report.lower() or "overlap" in report.lower()
