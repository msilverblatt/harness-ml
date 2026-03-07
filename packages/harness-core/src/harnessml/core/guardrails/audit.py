"""Structured audit log — append-only JSONL for all tool invocations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only JSONL audit log for all tool invocations.

    Each :meth:`log_invocation` call appends one JSON line to the log file.
    The file is created on first write.

    Parameters
    ----------
    log_path:
        Path to the JSONL log file.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)

    def log_invocation(
        self,
        tool: str,
        args: dict,
        guardrails_passed: bool,
        result_status: str,
        duration_s: float,
        human_override: bool = False,
        error: str | None = None,
    ) -> None:
        """Append a log entry with timestamp.

        Parameters
        ----------
        tool:
            Name of the tool invoked.
        args:
            Arguments passed to the tool.
        guardrails_passed:
            Whether all guardrails passed before execution.
        result_status:
            Final status (e.g. ``"success"``, ``"error"``, ``"timeout"``).
        duration_s:
            Wall-clock duration in seconds.
        human_override:
            Whether human_override was used.
        error:
            Error message, if any.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "args": args,
            "guardrails_passed": guardrails_passed,
            "result_status": result_status,
            "duration_s": duration_s,
            "human_override": human_override,
        }
        if error is not None:
            entry["error"] = error

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def query(
        self,
        tool: str | None = None,
        since: datetime | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Query log entries with optional filters.

        Parameters
        ----------
        tool:
            Filter by tool name.
        since:
            Only return entries after this datetime.
        status:
            Filter by ``result_status``.

        Returns
        -------
        list[dict]
            Matching log entries in chronological order.
        """
        if not self.log_path.exists():
            return []

        results: list[dict] = []
        for line in self.log_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)

            if tool is not None and entry.get("tool") != tool:
                continue
            if status is not None and entry.get("result_status") != status:
                continue
            if since is not None:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < since:
                    continue

            results.append(entry)

        return results
