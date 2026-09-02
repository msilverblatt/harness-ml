from pydantic import BaseModel, Field
from typing import Any


class CheckResult(BaseModel):
    metric: str
    value: float
    op: str
    threshold: float | list[float]
    passed: bool
    severity: str = "warning"


class ComparisonResult(BaseModel):
    vs: str
    metric: str
    current: float
    baseline: float
    delta: float
    improved: bool


class EvalCheck(BaseModel):
    metric: str
    op: str  # "<", ">", "between", "!="
    value: float | list[float]
    severity: str = "warning"  # "error" or "warning"

    def evaluate(self, actual: float) -> bool:
        if self.op == "<":
            return actual < self.value
        elif self.op == ">":
            return actual > self.value
        elif self.op == "between":
            return self.value[0] <= actual <= self.value[1]
        elif self.op == "!=":
            return actual != self.value
        raise ValueError(f"Unknown op: {self.op}")


class EvalComparison(BaseModel):
    vs: str       # "parent", "baseline"
    metric: str
    expect: str   # "lower", "higher"

    def evaluate(self, current: float, baseline: float) -> ComparisonResult:
        delta = current - baseline
        if self.expect == "lower":
            improved = current < baseline
        elif self.expect == "higher":
            improved = current > baseline
        else:
            improved = False
        return ComparisonResult(
            vs=self.vs,
            metric=self.metric,
            current=current,
            baseline=baseline,
            delta=delta,
            improved=improved,
        )


class EvalDimension(BaseModel):
    name: str
    description: str = ""
    checks: list[EvalCheck] = Field(default_factory=list)
    comparisons: list[EvalComparison] = Field(default_factory=list)
    judgment: str = ""

    @classmethod
    def from_yaml_dict(cls, name: str, data: dict) -> "EvalDimension":
        checks = [EvalCheck(**c) for c in data.get("checks", [])]
        comparisons = [EvalComparison(**c) for c in data.get("comparisons", [])]
        return cls(
            name=name,
            description=data.get("description", ""),
            checks=checks,
            comparisons=comparisons,
            judgment=data.get("judgment", ""),
        )


class EvalReport(BaseModel):
    dimensions: dict[str, dict] = Field(default_factory=dict)
    # Each dimension value has: "checks" (list[CheckResult]), "comparisons" (list[ComparisonResult]), "judgment_prompt" (str)

    def summary(self) -> dict:
        checks_passed = 0
        checks_total = 0
        checks_failed_error = 0
        checks_failed_warning = 0
        improvements = 0
        regressions = 0

        for dim_name, dim_data in self.dimensions.items():
            for check in dim_data.get("checks", []):
                checks_total += 1
                if check.passed:
                    checks_passed += 1
                elif check.severity == "error":
                    checks_failed_error += 1
                else:
                    checks_failed_warning += 1
            for comp in dim_data.get("comparisons", []):
                if comp.improved:
                    improvements += 1
                else:
                    regressions += 1

        return {
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "checks_failed_error": checks_failed_error,
            "checks_failed_warning": checks_failed_warning,
            "improvements": improvements,
            "regressions": regressions,
        }
