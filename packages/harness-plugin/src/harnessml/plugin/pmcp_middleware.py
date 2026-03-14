"""Local middleware for harness-ml: error formatting and auto-install."""
from __future__ import annotations

import json
import subprocess
import sys
import traceback

from protomcp import ToolResult, local_middleware

_IMPORT_TO_PACKAGE = {
    "pytorch_tabnet": "pytorch-tabnet",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "google": "google-api-python-client",
    "googleapiclient": "google-api-python-client",
    "google_auth_oauthlib": "google-auth-oauthlib",
}


def _auto_install(module_name: str) -> bool:
    package = _IMPORT_TO_PACKAGE.get(module_name, module_name)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, timeout=120, check=True,
        )
        return True
    except Exception:
        return False


@local_middleware(priority=10)
def auto_install_middleware(ctx, tool_name, args, next_handler):
    """Catch ModuleNotFoundError, auto-install, retry once."""
    try:
        return next_handler(ctx, args)
    except ModuleNotFoundError as e:
        package = _IMPORT_TO_PACKAGE.get(e.name, e.name)
        if _auto_install(e.name):
            try:
                return next_handler(ctx, args)
            except ModuleNotFoundError as retry_err:
                return ToolResult(
                    result=f"**Error**: {retry_err} (after installing {package})",
                    is_error=True, error_code="MISSING_PACKAGE",
                )
        return ToolResult(
            result=f"**Error**: Missing package `{package}`. Auto-install failed.",
            is_error=True, error_code="MISSING_PACKAGE",
        )


@local_middleware(priority=90)
def error_format_middleware(ctx, tool_name, args, next_handler):
    """Convert unhandled exceptions to markdown error strings."""
    try:
        return next_handler(ctx, args)
    except json.JSONDecodeError as e:
        return ToolResult(
            result=f"**Error**: Invalid JSON input: {e}",
            is_error=True, error_code="INVALID_JSON",
        )
    except ValueError as e:
        return ToolResult(
            result=f"**Error**: {e}",
            is_error=True, error_code="VALIDATION_ERROR",
        )
    except Exception as e:
        tb_lines = traceback.format_exception(e)
        tb_str = "".join(tb_lines)[-2000:]
        return ToolResult(
            result=f"**Error**: Unexpected error in `{tool_name}`: {e}\n\n```\n{tb_str}\n```",
            is_error=True, error_code="INTERNAL_ERROR",
        )
