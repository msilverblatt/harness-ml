"""Transform protocol — step configuration model."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    """Configuration for a single transform step."""
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
