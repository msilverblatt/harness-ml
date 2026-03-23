import pytest
import tempfile
from pathlib import Path
import yaml

from harness.ml.evals.schema import EvalDimension, EvalCheck, EvalComparison
from harness.ml.evals.runner import EvalRunner


def _make_runner_with_dims(*dims: EvalDimension) -> EvalRunner:
    return EvalRunner({d.name: d for d in dims})


class TestEvalRunnerRun:
    def test_run_all_passing(self):
        dim = EvalDimension(
            name="accuracy",
            checks=[
                EvalCheck(metric="auroc", op=">", value=0.55, severity="error"),
                EvalCheck(metric="brier", op="<", value=0.25, severity="warning"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run({"auroc": 0.75, "brier": 0.15})
        s = report.summary()
        assert s["checks_passed"] == 2
        assert s["checks_total"] == 2
        assert s["checks_failed_error"] == 0
        assert s["checks_failed_warning"] == 0

    def test_run_with_failures(self):
        dim = EvalDimension(
            name="quality",
            checks=[
                EvalCheck(metric="auroc", op=">", value=0.8, severity="error"),
                EvalCheck(metric="brier", op="<", value=0.1, severity="warning"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run({"auroc": 0.6, "brier": 0.2})
        s = report.summary()
        assert s["checks_passed"] == 0
        assert s["checks_total"] == 2
        assert s["checks_failed_error"] == 1
        assert s["checks_failed_warning"] == 1

    def test_run_with_comparisons_improvements(self):
        dim = EvalDimension(
            name="perf",
            comparisons=[
                EvalComparison(vs="parent", metric="auroc", expect="higher"),
                EvalComparison(vs="parent", metric="brier", expect="lower"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run(
            metrics={"auroc": 0.8, "brier": 0.15},
            parent_metrics={"auroc": 0.7, "brier": 0.25},
        )
        s = report.summary()
        assert s["improvements"] == 2
        assert s["regressions"] == 0

    def test_run_with_comparisons_regressions(self):
        dim = EvalDimension(
            name="perf",
            comparisons=[
                EvalComparison(vs="parent", metric="auroc", expect="higher"),
                EvalComparison(vs="parent", metric="brier", expect="lower"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run(
            metrics={"auroc": 0.6, "brier": 0.3},
            parent_metrics={"auroc": 0.7, "brier": 0.25},
        )
        s = report.summary()
        assert s["improvements"] == 0
        assert s["regressions"] == 2

    def test_run_missing_metric_skipped(self):
        dim = EvalDimension(
            name="quality",
            checks=[
                EvalCheck(metric="auroc", op=">", value=0.55),
                EvalCheck(metric="missing_metric", op=">", value=0.5),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run({"auroc": 0.7})
        s = report.summary()
        # Only the auroc check should run
        assert s["checks_total"] == 1
        assert s["checks_passed"] == 1

    def test_run_no_parent_metrics_skips_comparisons(self):
        dim = EvalDimension(
            name="perf",
            comparisons=[
                EvalComparison(vs="parent", metric="auroc", expect="higher"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run({"auroc": 0.8}, parent_metrics=None)
        s = report.summary()
        assert s["improvements"] == 0
        assert s["regressions"] == 0

    def test_run_parent_missing_metric_skips_comparison(self):
        dim = EvalDimension(
            name="perf",
            comparisons=[
                EvalComparison(vs="parent", metric="auroc", expect="higher"),
            ],
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run(
            metrics={"auroc": 0.8},
            parent_metrics={"brier": 0.2},
        )
        s = report.summary()
        assert s["improvements"] == 0
        assert s["regressions"] == 0

    def test_run_report_has_judgment_prompt(self):
        dim = EvalDimension(
            name="calib",
            judgment="Review the calibration curve.",
        )
        runner = _make_runner_with_dims(dim)
        report = runner.run({})
        assert report.dimensions["calib"]["judgment_prompt"] == "Review the calibration curve."

    def test_run_multiple_dimensions(self):
        dim1 = EvalDimension(
            name="d1",
            checks=[EvalCheck(metric="auroc", op=">", value=0.55, severity="error")],
        )
        dim2 = EvalDimension(
            name="d2",
            checks=[EvalCheck(metric="brier", op="<", value=0.25, severity="warning")],
        )
        runner = _make_runner_with_dims(dim1, dim2)
        report = runner.run({"auroc": 0.7, "brier": 0.1})
        assert "d1" in report.dimensions
        assert "d2" in report.dimensions
        s = report.summary()
        assert s["checks_passed"] == 2
        assert s["checks_total"] == 2


class TestEvalRunnerFromYaml:
    def test_load_from_yaml(self, tmp_path):
        config = {
            "evals": {
                "accuracy": {
                    "description": "Accuracy checks",
                    "checks": [
                        {"metric": "auroc", "op": ">", "value": 0.55, "severity": "error"},
                    ],
                    "comparisons": [
                        {"vs": "parent", "metric": "auroc", "expect": "higher"},
                    ],
                    "judgment": "Check ROC curve.",
                }
            }
        }
        yaml_path = tmp_path / "eval.yaml"
        yaml_path.write_text(yaml.dump(config))

        runner = EvalRunner.from_yaml(yaml_path)
        assert "accuracy" in runner._dimensions
        dim = runner._dimensions["accuracy"]
        assert dim.description == "Accuracy checks"
        assert len(dim.checks) == 1
        assert dim.checks[0].metric == "auroc"
        assert len(dim.comparisons) == 1
        assert dim.judgment == "Check ROC curve."

    def test_load_from_yaml_and_run(self, tmp_path):
        config = {
            "evals": {
                "perf": {
                    "checks": [
                        {"metric": "auroc", "op": ">", "value": 0.55, "severity": "error"},
                    ],
                }
            }
        }
        yaml_path = tmp_path / "eval.yaml"
        yaml_path.write_text(yaml.dump(config))

        runner = EvalRunner.from_yaml(yaml_path)
        report = runner.run({"auroc": 0.75})
        s = report.summary()
        assert s["checks_passed"] == 1

    def test_load_binary_preset(self):
        preset_path = (
            Path(__file__).parent.parent.parent
            / "src" / "harness" / "ml" / "evals" / "presets" / "binary.yaml"
        )
        runner = EvalRunner.from_yaml(preset_path)
        assert "probability_accuracy" in runner._dimensions
        assert "discrimination" in runner._dimensions
        assert "stability" in runner._dimensions

    def test_load_regression_preset(self):
        preset_path = (
            Path(__file__).parent.parent.parent
            / "src" / "harness" / "ml" / "evals" / "presets" / "regression.yaml"
        )
        runner = EvalRunner.from_yaml(preset_path)
        assert "accuracy" in runner._dimensions

    def test_load_multiclass_preset(self):
        preset_path = (
            Path(__file__).parent.parent.parent
            / "src" / "harness" / "ml" / "evals" / "presets" / "multiclass.yaml"
        )
        runner = EvalRunner.from_yaml(preset_path)
        assert "classification" in runner._dimensions

    def test_binary_preset_run_passing(self):
        preset_path = (
            Path(__file__).parent.parent.parent
            / "src" / "harness" / "ml" / "evals" / "presets" / "binary.yaml"
        )
        runner = EvalRunner.from_yaml(preset_path)
        metrics = {
            "ece": 0.02,
            "brier": 0.15,
            "auroc": 0.80,
            "accuracy": 0.75,
            "fold_std_brier": 0.01,
        }
        report = runner.run(metrics)
        s = report.summary()
        assert s["checks_failed_error"] == 0
