import pytest
from harness.ml.evals.schema import (
    CheckResult,
    ComparisonResult,
    EvalCheck,
    EvalComparison,
    EvalDimension,
    EvalReport,
)


class TestEvalCheck:
    def test_less_than_passes(self):
        check = EvalCheck(metric="ece", op="<", value=0.05)
        assert check.evaluate(0.03) is True

    def test_less_than_fails(self):
        check = EvalCheck(metric="ece", op="<", value=0.05)
        assert check.evaluate(0.07) is False

    def test_less_than_equal_fails(self):
        check = EvalCheck(metric="ece", op="<", value=0.05)
        assert check.evaluate(0.05) is False

    def test_greater_than_passes(self):
        check = EvalCheck(metric="auroc", op=">", value=0.55)
        assert check.evaluate(0.75) is True

    def test_greater_than_fails(self):
        check = EvalCheck(metric="auroc", op=">", value=0.55)
        assert check.evaluate(0.4) is False

    def test_greater_than_equal_fails(self):
        check = EvalCheck(metric="auroc", op=">", value=0.55)
        assert check.evaluate(0.55) is False

    def test_between_passes(self):
        check = EvalCheck(metric="score", op="between", value=[0.2, 0.8])
        assert check.evaluate(0.5) is True

    def test_between_at_lower_bound_passes(self):
        check = EvalCheck(metric="score", op="between", value=[0.2, 0.8])
        assert check.evaluate(0.2) is True

    def test_between_at_upper_bound_passes(self):
        check = EvalCheck(metric="score", op="between", value=[0.2, 0.8])
        assert check.evaluate(0.8) is True

    def test_between_below_lower_fails(self):
        check = EvalCheck(metric="score", op="between", value=[0.2, 0.8])
        assert check.evaluate(0.1) is False

    def test_between_above_upper_fails(self):
        check = EvalCheck(metric="score", op="between", value=[0.2, 0.8])
        assert check.evaluate(0.9) is False

    def test_not_equal_passes(self):
        check = EvalCheck(metric="flag", op="!=", value=0.0)
        assert check.evaluate(1.0) is True

    def test_not_equal_fails(self):
        check = EvalCheck(metric="flag", op="!=", value=0.0)
        assert check.evaluate(0.0) is False

    def test_unknown_op_raises(self):
        check = EvalCheck(metric="x", op="==", value=1.0)
        with pytest.raises(ValueError, match="Unknown op"):
            check.evaluate(1.0)

    def test_default_severity_is_warning(self):
        check = EvalCheck(metric="x", op=">", value=0.5)
        assert check.severity == "warning"

    def test_error_severity(self):
        check = EvalCheck(metric="ece", op="<", value=0.05, severity="error")
        assert check.severity == "error"


class TestEvalComparison:
    def test_lower_improved(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.2, baseline=0.3)
        assert result.improved is True

    def test_lower_regression(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.35, baseline=0.3)
        assert result.improved is False

    def test_lower_equal_is_regression(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.3, baseline=0.3)
        assert result.improved is False

    def test_higher_improved(self):
        comp = EvalComparison(vs="parent", metric="auroc", expect="higher")
        result = comp.evaluate(current=0.8, baseline=0.7)
        assert result.improved is True

    def test_higher_regression(self):
        comp = EvalComparison(vs="parent", metric="auroc", expect="higher")
        result = comp.evaluate(current=0.65, baseline=0.7)
        assert result.improved is False

    def test_delta_calculation(self):
        comp = EvalComparison(vs="parent", metric="brier", expect="lower")
        result = comp.evaluate(current=0.2, baseline=0.3)
        assert abs(result.delta - (0.2 - 0.3)) < 1e-9

    def test_result_fields(self):
        comp = EvalComparison(vs="baseline", metric="rmse", expect="lower")
        result = comp.evaluate(current=5.0, baseline=8.0)
        assert result.vs == "baseline"
        assert result.metric == "rmse"
        assert result.current == 5.0
        assert result.baseline == 8.0
        assert result.improved is True

    def test_unknown_expect_not_improved(self):
        comp = EvalComparison(vs="parent", metric="x", expect="unknown")
        result = comp.evaluate(current=1.0, baseline=0.5)
        assert result.improved is False


class TestEvalDimension:
    def test_from_yaml_dict_basic(self):
        data = {
            "description": "Test dimension",
            "checks": [
                {"metric": "ece", "op": "<", "value": 0.05, "severity": "error"},
            ],
            "comparisons": [
                {"vs": "parent", "metric": "brier", "expect": "lower"},
            ],
            "judgment": "Check calibration.",
        }
        dim = EvalDimension.from_yaml_dict("calibration", data)
        assert dim.name == "calibration"
        assert dim.description == "Test dimension"
        assert len(dim.checks) == 1
        assert dim.checks[0].metric == "ece"
        assert dim.checks[0].severity == "error"
        assert len(dim.comparisons) == 1
        assert dim.comparisons[0].vs == "parent"
        assert dim.judgment == "Check calibration."

    def test_from_yaml_dict_defaults(self):
        dim = EvalDimension.from_yaml_dict("empty", {})
        assert dim.name == "empty"
        assert dim.description == ""
        assert dim.checks == []
        assert dim.comparisons == []
        assert dim.judgment == ""


class TestEvalReport:
    def _make_check(self, passed, severity="warning"):
        return CheckResult(
            metric="x", value=0.5, op=">", threshold=0.4,
            passed=passed, severity=severity,
        )

    def _make_comp(self, improved):
        return ComparisonResult(
            vs="parent", metric="x", current=0.5, baseline=0.6,
            delta=-0.1, improved=improved,
        )

    def test_summary_all_passing(self):
        report = EvalReport(dimensions={
            "dim1": {
                "checks": [self._make_check(True), self._make_check(True)],
                "comparisons": [],
            }
        })
        s = report.summary()
        assert s["checks_passed"] == 2
        assert s["checks_total"] == 2
        assert s["checks_failed_error"] == 0
        assert s["checks_failed_warning"] == 0
        assert s["improvements"] == 0
        assert s["regressions"] == 0

    def test_summary_failures(self):
        report = EvalReport(dimensions={
            "dim1": {
                "checks": [
                    self._make_check(False, severity="error"),
                    self._make_check(False, severity="warning"),
                    self._make_check(True),
                ],
                "comparisons": [],
            }
        })
        s = report.summary()
        assert s["checks_passed"] == 1
        assert s["checks_total"] == 3
        assert s["checks_failed_error"] == 1
        assert s["checks_failed_warning"] == 1

    def test_summary_comparisons(self):
        report = EvalReport(dimensions={
            "dim1": {
                "checks": [],
                "comparisons": [
                    self._make_comp(True),
                    self._make_comp(True),
                    self._make_comp(False),
                ],
            }
        })
        s = report.summary()
        assert s["improvements"] == 2
        assert s["regressions"] == 1

    def test_summary_empty(self):
        report = EvalReport()
        s = report.summary()
        assert s == {
            "checks_passed": 0,
            "checks_total": 0,
            "checks_failed_error": 0,
            "checks_failed_warning": 0,
            "improvements": 0,
            "regressions": 0,
        }

    def test_summary_multiple_dimensions(self):
        report = EvalReport(dimensions={
            "dim1": {
                "checks": [self._make_check(True)],
                "comparisons": [self._make_comp(True)],
            },
            "dim2": {
                "checks": [self._make_check(False, severity="error")],
                "comparisons": [self._make_comp(False)],
            },
        })
        s = report.summary()
        assert s["checks_passed"] == 1
        assert s["checks_total"] == 2
        assert s["checks_failed_error"] == 1
        assert s["improvements"] == 1
        assert s["regressions"] == 1
