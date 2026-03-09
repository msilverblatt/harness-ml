"""Thread-safety tests for MCP server global state."""
import threading
from unittest.mock import MagicMock, patch


def test_get_emitter_thread_safe():
    """Verify _get_emitter doesn't initialize twice under concurrent calls."""
    import harnessml.plugin.mcp_server as srv

    original_emitter = srv._emitter
    srv._emitter = None  # Reset
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
            threads = [threading.Thread(target=srv._get_emitter) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert call_count == 1
    finally:
        srv._emitter = original_emitter


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


def test_studio_url_logged_thread_safe():
    """Verify _studio_url_logged flag is only set once under concurrency."""
    import harnessml.plugin.mcp_server as srv

    original_flag = srv._studio_url_logged
    srv._studio_url_logged = False
    log_count = 0
    lock = threading.Lock()

    original_fn = srv._check_studio_url_once

    def counting_check():
        nonlocal log_count
        logged = original_fn()
        if logged:
            with lock:
                log_count += 1

    try:
        threads = [threading.Thread(target=counting_check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert log_count == 1
    finally:
        srv._studio_url_logged = original_flag
