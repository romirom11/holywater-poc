import os

# Before anything imports Prefect: quiet logs and no telemetry from the test server or the SDK.
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")
