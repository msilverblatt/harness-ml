import structlog
from harnessml.core.logging import configure_logging, get_logger


def test_get_logger_returns_bound_logger():
    logger = get_logger("test.module")
    assert logger is not None


def test_logging_captures_events(capsys):
    configure_logging(json_output=False)
    logger = get_logger("test")
    with structlog.testing.capture_logs() as logs:
        logger.info("test_event", key="value")
    assert len(logs) == 1
    assert logs[0]["key"] == "value"
    assert logs[0]["event"] == "test_event"
