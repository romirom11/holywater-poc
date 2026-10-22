from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _prefect_harness() -> Iterator[None]:
    """One temporary Prefect server (a local uvicorn subprocess) for the integration session.

    Only `tests/integration` gets it: unit tests never run a flow, so they start no server.
    """
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=60):
        yield
