"""Required O1 walking skeleton: Parquet → XGBoost → Bundle → Serving."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.tributo_walking_skeleton,
    # Ray 2.55.1 leaks /dev/null file handles from its internal process
    # spawn (gcs/raylet/monitor); the garbage-collected handles surface as
    # unraisable ResourceWarnings that the project-wide
    # filterwarnings=error would turn into failures. This test cannot fix
    # Ray internals, so suppress that warning class here only.
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


def test_walking_skeleton() -> None:
    """Run the required golden path through the Docker cluster's Jobs API."""
    from tests.integration.test_model_export_ray_cluster import (
        _run_model_export_job,
    )

    payload = _run_model_export_job()
    assert payload["status"] == "succeeded"
    assert payload["formats"] == {"native": "ubj", "onnx-model": "onnx"}
    assert payload["batch_rows"] == 2
    assert payload["http_rows"] == 2
    assert payload["mlflow_runs"] == 1
    assert payload["model_versions_created"] == 0
