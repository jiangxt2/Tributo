"""Real MinIO compatibility smoke tests.

The full S3 contract suite is also executed against the same MinIO endpoint
in CI. These tests keep the compatibility-specific requirements explicit:
path-style addressing, object round-trips, and conditional writes.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from botocore.exceptions import ClientError

from tributo._common.storage import get_boto3_client

pytestmark = [
    pytest.mark.minio_compat,
    pytest.mark.usefixtures("s3_environment"),
]


@pytest.fixture()
def minio_bucket() -> Generator[str, None, None]:
    """Create and remove an isolated MinIO bucket."""
    client = get_boto3_client(path_style=True)
    bucket = f"tributo-minio-compat-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    try:
        yield bucket
    finally:
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
            keys = [item["Key"] for item in page.get("Contents", [])]
            if keys:
                client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": key} for key in keys]},
                )
        client.delete_bucket(Bucket=bucket)


def test_minio_path_style_round_trip(minio_bucket: str) -> None:
    """The storage client can write, read, and inspect a MinIO object."""
    client = get_boto3_client(path_style=True)
    assert client.meta.config.s3.get("addressing_style") == "path"
    key = "compatibility/round-trip.bin"
    payload = b"tributo-minio-compatibility"

    client.put_object(Bucket=minio_bucket, Key=key, Body=payload)
    head = client.head_object(Bucket=minio_bucket, Key=key)
    body = client.get_object(Bucket=minio_bucket, Key=key)["Body"].read()

    assert head["ContentLength"] == len(payload)
    assert body == payload


def test_minio_conditional_create_is_enforced(minio_bucket: str) -> None:
    """MinIO enforces the conditional-create primitive used by Publisher."""
    client = get_boto3_client(path_style=True)
    key = "compatibility/conditional.bin"
    client.put_object(Bucket=minio_bucket, Key=key, Body=b"first")

    with pytest.raises(ClientError) as raised:
        client.put_object(
            Bucket=minio_bucket,
            Key=key,
            Body=b"second",
            IfNoneMatch="*",
        )

    assert raised.value.response["Error"]["Code"] in {
        "412",
        "PreconditionFailed",
    }
