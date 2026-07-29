"""MinIO S3 connectivity tests — requires ``@pytest.mark.s3``.

Environment variables expected: ``S3_ENDPOINT``, ``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``.  Compatible with the MinIO service container
in the nightly CI workflow.

Tests skip gracefully when credentials are missing so that the default
``uv run pytest`` invocation does not break local development.
"""

from __future__ import annotations

import pyarrow.fs
import pyarrow.parquet as pq
import pytest

from tributo.data._s3 import (
    resolve_access_key_id,
    resolve_endpoint,
    resolve_secret_access_key,
)

pytestmark = pytest.mark.s3


def _require_s3_env() -> tuple[str | None, str | None, str | None]:
    """Return (endpoint, access_key, secret_key), skipping if unset."""
    endpoint = resolve_endpoint()
    access_key = resolve_access_key_id()
    secret_key = resolve_secret_access_key()

    if not endpoint or not access_key or not secret_key:
        pytest.skip(
            "S3 credentials not set — set S3_ENDPOINT, AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY to run this test."
        )
    return endpoint, access_key, secret_key


def test_s3_env_resolution() -> None:
    """Environment variables are read by the S3 resolution helpers."""
    endpoint, access_key, secret_key = _require_s3_env()
    assert endpoint
    assert access_key
    assert secret_key


def test_s3_connectivity() -> None:
    """PyArrow can connect to the MinIO service and perform basic I/O."""
    endpoint, access_key, secret_key = _require_s3_env()

    raw = endpoint.replace("http://", "").replace("https://", "")
    scheme = "http" if endpoint.startswith("http://") else "https"

    fs = pyarrow.fs.S3FileSystem(
        access_key=access_key,
        secret_key=secret_key,
        endpoint_override=raw,
        scheme=scheme,
        region="us-east-1",
    )

    bucket = "tributo-test"
    path = f"{bucket}/s3-smoke-test.parquet"

    table = pyarrow.table({"x": [1, 2, 3]})
    pq.write_table(table, path, filesystem=fs)

    result = pq.read_table(path, filesystem=fs)
    assert result.num_rows == 3

    fs.delete_file(path)
