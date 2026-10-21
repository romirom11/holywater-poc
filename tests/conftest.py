import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")


@pytest.fixture(scope="session", autouse=True)
def _prefect_harness() -> Iterator[None]:
    """One temporary Prefect server for the whole session (only started when a flow runs)."""
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=60):
        yield
