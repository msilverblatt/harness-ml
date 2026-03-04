"""Shared validation utilities for MCP handlers."""
from __future__ import annotations

from difflib import get_close_matches


def validate_enum(value: str, valid: set[str], param_name: str) -> str | None:
    """Validate value against allowed set, return error message or None."""
    if value in valid:
        return None
    closest = get_close_matches(value, sorted(valid), n=1, cutoff=0.6)
    msg = f"**Error**: Invalid `{param_name}` '{value}'. Valid: {', '.join(sorted(valid))}"
    if closest:
        msg += f"\n\nDid you mean **{closest[0]}**?"
    return msg


def validate_required(value, param_name: str) -> str | None:
    """Return error message if value is None/empty, else None."""
    if value is None or value == "":
        return f"**Error**: `{param_name}` is required."
    return None
