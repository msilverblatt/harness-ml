# easyml-guardrails

AI guardrails and MCP server for EasyML -- 11 guardrails (3 non-overridable),
pipeline command execution, MCP tool server, and structured audit logging.

## Installation

```bash
pip install easyml-guardrails
```

## Quick Start

```python
from easyml.guardrails import (
    NamingConventionGuardrail, FeatureLeakageGuardrail,
    TemporalOrderingGuardrail, AuditLogger, GuardrailError,
)

# Overridable guardrail
naming = NamingConventionGuardrail(pattern=r"^exp-\d{3}-.+$")
naming.check({"experiment_id": "exp-001-test"})  # passes
naming.check({"experiment_id": "bad"}, human_override=True)  # overridden

# Non-overridable guardrail (human_override has no effect)
leakage = FeatureLeakageGuardrail(denylist=["kp_adj_o", "post_tourney"])
leakage.check({"model_features": ["win_rate", "seed_num"]})  # passes
# leakage.check({"model_features": ["kp_adj_o"]})  # raises GuardrailError

# Audit logging (append-only JSONL)
logger = AuditLogger(log_path="logs/audit.jsonl")
logger.log_invocation(
    tool="train_models", args={"gender": "M"},
    guardrails_passed=True, result_status="success", duration_s=45.2,
)
entries = logger.query(tool="train_models", status="success")
```

## Key APIs

**Overridable guardrails:** `SanityCheckGuardrail`, `NamingConventionGuardrail`,
`DoNotRetryGuardrail`, `SingleVariableGuardrail`, `ConfigProtectionGuardrail`,
`RateLimitGuardrail`, `ExperimentLoggedGuardrail`, `FeatureStalenessGuardrail`

**Non-overridable guardrails:** `FeatureLeakageGuardrail`, `CriticalPathGuardrail`,
`TemporalOrderingGuardrail`

**Infrastructure:** `Guardrail` (base class), `GuardrailError`, `run_pipeline_command`,
`PipelineServer`, `ToolDef`, `AuditLogger`
