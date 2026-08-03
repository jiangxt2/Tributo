"""S3-compatible bundle publishing contract tests.

The default backend is an ephemeral Moto server. The same suite runs against
real MinIO in the explicit compatibility gate.

Covers the plan's PR3 acceptance items: publish round-trip, lease +
conditional writes, checksum metadata, alias CAS conflicts, and orphan
GC protection.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tributo._common.storage import get_boto3_client
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.gc import BundleGarbageCollector
from tributo.exporting.manifest import ManifestSourceInfo
from tributo.exporting.models import (
    AliasConfig,
    ArtifactFile,
    ArtifactRef,
    ExportExecutionResult,
    LogicalArtifact,
    NodeResult,
    ProducerInfo,
)
from tributo.exporting.publisher import Publisher

pytestmark = [
    pytest.mark.s3_contract,
    pytest.mark.usefixtures("s3_environment"),
]


def _env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point credentials at MinIO and enable path-style addressing."""
    monkeypatch.setenv("S3_ENDPOINT", os.environ["S3_ENDPOINT"])
    monkeypatch.setenv(
        "AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    )
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY",
        os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    )
    monkeypatch.setenv(
        "AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    # MinIO requires path-style addressing; the "test" profile enables it.
    monkeypatch.setenv(
        "TRIBUTO_STORAGE_PROFILE_TEST",
        json.dumps({"path_style": True}),
    )


@pytest.fixture()
def s3_bucket(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    """Create a unique bucket, point credentials at MinIO, clean up after."""
    _env_overrides(monkeypatch)
    client = get_boto3_client(path_style=True)
    bucket = f"tributo-export-s3-test-{uuid.uuid4().hex[:12]}"
    client.create_bucket(Bucket=bucket)
    try:
        yield bucket
    finally:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            keys = [o["Key"] for o in page.get("Contents", [])]
            if keys:
                client.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in keys]},
                )
        client.delete_bucket(Bucket=bucket)


def _logical_artifact(name: str = "fp32") -> LogicalArtifact:
    files = (
        ArtifactFile(
            relative_path="model.onnx", sha256="a" * 64, size_bytes=64, role="model"
        ),
    )
    return LogicalArtifact(
        name=name,
        format="onnx",
        flavor_id="onnx-runtime-v1",
        files=files,
        entrypoint="model.onnx",
        tree_digest=LogicalArtifact.compute_tree_digest(files),
        producer=ProducerInfo(exporter_id="test-v1"),
    )


def _publish(
    bucket: str,
    tmp_path: Path,
    *,
    bundle_id: str | None = None,
    execution_id: str = "exec-1",
    alias: AliasConfig | None = None,
) -> Any:
    """Publish one artifact to ``s3://<bucket>/models`` via the real Publisher."""
    artifact = _logical_artifact()
    staging = tmp_path / f"staging-{uuid.uuid4().hex[:8]}"
    for af in artifact.files:
        fp = staging / "nodes" / artifact.name / "artifact" / af.relative_path
        fp.parent.mkdir(parents=True)
        fp.write_bytes(b"x" * af.size_bytes)
    ref = ArtifactRef(
        node_id="fp32",
        artifact_name="fp32",
        tree_digest=artifact.tree_digest,
    )
    execution = ExportExecutionResult(
        execution_id=execution_id,
        status="succeeded",
        node_results=(
            NodeResult(
                node_id="fp32",
                target_name="fp32",
                status="succeeded",
                required=True,
                publish=True,
                exporter_id="test-v1",
                artifact_ref=ref,
            ),
        ),
        roles={},
        staged_artifacts={"fp32": artifact},
    )
    publisher = Publisher()
    return publisher.publish(
        execution=execution,
        staging_root=staging,
        bundle_uri=f"s3://{bucket}/models",
        bundle_id=bundle_id or f"bundle-{uuid.uuid4().hex[:32]}",
        execution_id=execution_id,
        tributo_version="0.1.0",
        source_info=ManifestSourceInfo(
            source_kind="pytorch_result",
            framework="pytorch",
            framework_version="2.5.0",
            task_type="classification",
        ),
        storage_profile="test",
        alias_config=alias,
    )


class TestS3Publish:
    def test_publish_roundtrip_and_checksum_metadata(
        self, s3_bucket: str, tmp_path: Path
    ) -> None:
        """Publish, read the manifest back, verify checksum metadata."""
        published = _publish(s3_bucket, tmp_path)
        assert published.result.status == "succeeded"
        assert published.result.manifest_uri.startswith(f"s3://{s3_bucket}/models/")

        manifest = BundleReader().read_manifest(
            published.result.manifest_uri, storage_profile="test"
        )
        assert manifest.bundle_id == published.result.bundle_id
        assert len(manifest.artifacts) == 1
        assert manifest.artifacts[0].name == "fp32"

        # Checksum metadata on the manifest object (plan: "write
        # tributo-sha256 metadata as fallback").
        client = get_boto3_client(path_style=True)
        head = client.head_object(
            Bucket=s3_bucket,
            Key=published.result.manifest_uri[len(f"s3://{s3_bucket}/") :],
        )
        # botocore Title-Cases metadata header keys — compare case-insensitively.
        meta = {k.lower(): v for k, v in head["Metadata"].items()}
        assert meta["tributo-sha256"] == published.result.manifest_sha256

    def test_idempotent_retry_same_bundle(self, s3_bucket: str, tmp_path: Path) -> None:
        """Retrying the same bundle id is idempotent with a stable sha."""
        bundle_id = f"bundle-{uuid.uuid4().hex[:32]}"
        first = _publish(s3_bucket, tmp_path, bundle_id=bundle_id)
        second = _publish(s3_bucket, tmp_path, bundle_id=bundle_id)

        assert second.result.status == "succeeded"
        assert second.result.manifest_sha256 == first.result.manifest_sha256
        # Manifest-last: only one manifest object exists.
        client = get_boto3_client(path_style=True)
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=s3_bucket, Prefix=f"models/{bundle_id}/"
        )
        manifests = [
            o["Key"]
            for page in pages
            for o in page.get("Contents", [])
            if o["Key"].endswith("manifest.json")
        ]
        assert len(manifests) == 1

    def test_alias_cas_flow(self, s3_bucket: str, tmp_path: Path) -> None:
        """CAS: create-only, then digest-guarded update, then mismatch."""
        # 1. Create-only CAS (expected_sha None) on a fresh alias.
        alias = AliasConfig(name="latest", policy="compare_and_swap")
        first = _publish(s3_bucket, tmp_path, alias=alias)
        assert first.result.alias_status == "updated"
        assert first.result.alias_uri == f"s3://{s3_bucket}/models/aliases/latest.json"

        # 2. Guarded update: expected digest matches the current alias.
        alias = AliasConfig(
            name="latest",
            policy="compare_and_swap",
            expected_manifest_sha256=first.result.manifest_sha256,
        )
        second = _publish(s3_bucket, tmp_path, alias=alias)
        assert second.result.alias_status == "updated"

        # 3. Mismatch: expected digest does not match — alias rejected.
        alias = AliasConfig(
            name="latest",
            policy="compare_and_swap",
            expected_manifest_sha256="f" * 64,
        )
        third = _publish(s3_bucket, tmp_path, alias=alias)
        assert third.result.alias_status == "failed"
        assert third.result.alias_failure is not None
        assert third.result.alias_failure.code == "CAS_MISMATCH"


class TestS3GarbageCollection:
    def test_collects_orphans_and_protects_lease(
        self, s3_bucket: str, tmp_path: Path
    ) -> None:
        """Orphan prefixes are deleted; lease-protected ones are kept."""
        client = get_boto3_client(path_style=True)
        collector = BundleGarbageCollector()

        orphan_id = f"bundle-{uuid.uuid4().hex[:32]}"
        client.put_object(
            Bucket=s3_bucket, Key=f"models/{orphan_id}/partial.bin", Body=b"x"
        )

        protected_id = f"bundle-{uuid.uuid4().hex[:32]}"
        client.put_object(
            Bucket=s3_bucket, Key=f"models/{protected_id}/partial.bin", Body=b"x"
        )
        client.put_object(
            Bucket=s3_bucket,
            Key=f"models/.leases/{protected_id}.json",
            Body=json.dumps(
                {
                    "owner": "other-exec-12345678",
                    "created_at": 0,
                    "expires_at": 10**18,  # far future — live lease
                    "bundle_id": protected_id,
                }
            ).encode(),
        )

        # Non-bundle prefixes are never touched.
        client.put_object(Bucket=s3_bucket, Key="models/random-dir/x.bin", Body=b"x")

        result = collector.collect(
            f"s3://{s3_bucket}/models",
            storage_profile="test",
            orphan_ttl_seconds=0,
            dry_run=False,
        )
        assert result["deleted"] == 1

        remaining = {
            o["Key"]
            for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=s3_bucket, Prefix="models/"
            )
            for o in page.get("Contents", [])
        }
        # Orphan is gone; lease-protected and non-bundle prefixes remain.
        assert not any(k.startswith(f"models/{orphan_id}/") for k in remaining)
        assert any(k.startswith(f"models/{protected_id}/") for k in remaining)
        assert any(k == "models/random-dir/x.bin" for k in remaining)
