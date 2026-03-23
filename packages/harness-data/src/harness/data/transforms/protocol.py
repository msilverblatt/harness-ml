"""Transform protocol — step configuration model."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class StepConfig(BaseModel):
    """Configuration for a single transform step."""
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
