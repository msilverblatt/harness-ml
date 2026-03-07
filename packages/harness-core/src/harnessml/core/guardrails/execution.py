"""Pipeline command execution with logging.

Runs subprocesses with stdout/stderr capture, timeout handling, and
automatic log file creation.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_pipeline_command(
    cmd: list[str],
    tool_name: str,
    timeout: int = 600,
    log_dir: Path | None = None,
    cwd: Path | None = None,
) -> dict:
    """Run a subprocess with logging.

    Parameters
    ----------
    cmd:
        Command and arguments (e.g. ``["uv", "run", "python", "train.py"]``).
    tool_name:
        Tool name used for log file naming.
    timeout:
        Maximum seconds to wait before killing the process.
    log_dir:
        Directory for log files. Defaults to ``/tmp``.
    cwd:
        Working directory for the subprocess.

    Returns
    -------
    dict
        Keys: ``status`` (``"success"``, ``"error"``, ``"timeout"``),
        ``log_path`` (:class:`Path`), ``stdout``, ``stderr``,
        ``duration_s``, ``returncode``.
    """
    if log_dir is None:
        log_dir = Path("/tmp")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"harnessml_{tool_name}_{timestamp}.log"

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        duration_s = round(time.monotonic() - start, 1)

        # Write full output to log file
        with open(log_path, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Duration: {duration_s}s\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"\n{'=' * 60} STDOUT {'=' * 60}\n")
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\n{'=' * 60} STDERR {'=' * 60}\n")
                f.write(result.stderr)

        status = "success" if result.returncode == 0 else "error"
        return {
            "status": status,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": duration_s,
            "log_path": log_path,
        }
    except subprocess.TimeoutExpired:
        duration_s = round(time.monotonic() - start, 1)

        # Write timeout info to log
        with open(log_path, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Duration: {duration_s}s\n")
            f.write(f"Status: TIMEOUT after {timeout}s\n")

        return {
            "status": "timeout",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "duration_s": duration_s,
            "log_path": log_path,
        }
