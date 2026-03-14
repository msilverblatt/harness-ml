"""Thread-safety tests for MCP server global state."""
import threading
from unittest.mock import MagicMock, patch


def test_get_emitter_thread_safe():
    """Verify telemetry emitter doesn't initialize twice under concurrent calls."""
    import harnessml.plugin.pmcp_telemetry as tel

    original_emitter = tel._emitter
    tel._emitter = None  # Reset
    call_count = 0

    def counting_create(*a, **kw):
        nonlocal call_count
        call_count += 1
        import time
        time.sleep(0.01)
        m = MagicMock()
        m.enabled = False
        return m

    try:
        with patch(
            "harnessml.plugin.event_emitter.create_emitter",
            side_effect=counting_create,
        ):
            threads = [threading.Thread(target=tel._get_emitter) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert call_count == 1
    finally:
        tel._emitter = original_emitter


def test_set_active_emitter_thread_safe():
    """Verify set_active_emitter uses a lock (no interleaving)."""
    from harnessml.plugin.handlers._common import (
        get_active_emitter,
        set_active_emitter,
    )

    results = []
    barrier = threading.Barrier(10)

    def _set_and_read(value):
        barrier.wait()
        set_active_emitter(value)
        results.append(get_active_emitter())

    sentinels = [object() for _ in range(10)]
    threads = [
        threading.Thread(target=_set_and_read, args=(s,)) for s in sentinels
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each result should be one of the sentinels (no None or corrupt state)
    for r in results:
        assert r in sentinels


def test_telemetry_sink_thread_safe():
    """Verify telemetry sink handles concurrent events without crashing."""
    from protomcp import ToolCallEvent, emit_telemetry

    errors = []

    def _emit_event(i):
        try:
            emit_telemetry(ToolCallEvent(
                tool_name="test",
                phase="start",
                action=f"action_{i}",
                args={"i": i},
            ))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_emit_event, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
