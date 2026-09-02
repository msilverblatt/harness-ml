from pathlib import Path

import yaml
from harness.ml.evals.schema import (
    CheckResult,
    EvalDimension,
    EvalReport,
)


class EvalRunner:
    def __init__(self, dimensions: dict[str, EvalDimension]):
        self._dimensions = dimensions

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalRunner":
        content = yaml.safe_load(Path(path).read_text())
        evals = content.get("evals", {})
        dimensions = {
            name: EvalDimension.from_yaml_dict(name, data)
            for name, data in evals.items()
        }
        return cls(dimensions)

    def run(
        self,
        metrics: dict[str, float],
        parent_metrics: dict[str, float] | None = None,
    ) -> EvalReport:
        report_dims = {}
        for dim_name, dim in self._dimensions.items():
            check_results = []
            for check in dim.checks:
                if check.metric not in metrics:
                    continue
                actual = metrics[check.metric]
                passed = check.evaluate(actual)
                check_results.append(CheckResult(
                    metric=check.metric,
                    value=actual,
                    op=check.op,
                    threshold=check.value,
                    passed=passed,
                    severity=check.severity,
                ))

            comp_results = []
            for comp in dim.comparisons:
                if comp.metric not in metrics:
                    continue
                if parent_metrics is None or comp.metric not in parent_metrics:
                    continue
                result = comp.evaluate(metrics[comp.metric], parent_metrics[comp.metric])
                comp_results.append(result)

            report_dims[dim_name] = {
                "checks": check_results,
                "comparisons": comp_results,
                "judgment_prompt": dim.judgment,
            }

        return EvalReport(dimensions=report_dims)
