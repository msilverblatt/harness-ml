import pytest

from harness.server.context import clear_workspace


@pytest.fixture(autouse=True)
def reset_workspace_context():
    clear_workspace()
    yield
    clear_workspace()
