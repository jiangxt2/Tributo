"""Tests for Publisher — local atomic commit, S3 publish, alias CAS."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from tributo.exporting.manifest import ManifestSourceInfo
from tributo.exporting.models import (
    AliasConfig,
    ArtifactFile,
    ArtifactRef,
    ExportExecutionResult,
    FailureInfo,
    LogicalArtifact,
    NodeResult,
    ProducerInfo,
)
from tributo.exporting.publisher import Publisher

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_producer() -> ProducerInfo:
    return ProducerInfo(exporter_id="test-v1")


def _make_artifact_files() -> tuple[ArtifactFile, ...]:
    return (
        ArtifactFile(
            relative_path="model.onnx", sha256="a" * 64, size_bytes=100, role="model"
        ),
        ArtifactFile(
            relative_path="config.json", sha256="b" * 64, size_bytes=50, role="config"
        ),
    )


def _make_logical_artifact(
    name: str = "fp32",
    fmt: str = "onnx",
    files: tuple[ArtifactFile, ...] | None = None,
) -> LogicalArtifact:
    f = files or _make_artifact_files()
    tree_digest = LogicalArtifact.compute_tree_digest(f)
    return LogicalArtifact(
        name=name,
        format=fmt,
        flavor_id="onnx-runtime-v1",
        files=f,
        entrypoint="model.onnx",
        tree_digest=tree_digest,
        producer=_make_producer(),
    )


def _make_node_result(
    node_id: str = "fp32",
    status: str = "succeeded",
    required: bool = True,
    publish: bool = True,
    artifact_ref: ArtifactRef | None = None,
    exporter_id: str = "test-v1",
    failure: FailureInfo | None = None,
) -> NodeResult:
    return NodeResult(
        node_id=node_id,
        target_name=node_id,
        status=status,
        required=required,
        publish=publish,
        exporter_id=exporter_id,
        artifact_ref=artifact_ref,
        failure=failure,
    )


def _make_execution(
    nodes: tuple[NodeResult, ...] | None = None,
    status: str = "succeeded",
    roles: dict[str, str] | None = None,
    staged_artifacts: dict[str, LogicalArtifact] | None = None,
) -> ExportExecutionResult:
    if nodes is None:
        ref = ArtifactRef(node_id="fp32", artifact_name="fp32", tree_digest="a" * 64)
        nodes = (_make_node_result(artifact_ref=ref),)
    return ExportExecutionResult(
        execution_id="exec-1",
        status=status,
        node_results=nodes,
        roles=roles or {},
        staged_artifacts=staged_artifacts or {},
    )


def _setup_staging(
    staging_root: Path,
    artifacts: list[LogicalArtifact],
) -> dict[str, LogicalArtifact]:
    """Create staging directory with fake artifact files.

    Returns a dict suitable for ``ExportExecutionResult.staged_artifacts``.
    """
    for artifact in artifacts:
        artifact_dir = staging_root / "nodes" / artifact.name / "artifact"
        artifact_dir.mkdir(parents=True)
        for af in artifact.files:
            fp = artifact_dir / af.relative_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            # Pad to the declared size_bytes so S3 idempotency head checks
            # (length + sha256 metadata) can match.
            fp.write_bytes(
                (b"fake-content" + af.relative_path.encode()).ljust(
                    af.size_bytes, b"\0"
                )
            )
    return {a.name: a for a in artifacts}


def _make_source_info() -> ManifestSourceInfo:
    return ManifestSourceInfo(
        source_kind="pytorch_result",
        framework="pytorch",
        framework_version="2.5.0",
        task_type="classification",
    )


# ── Local publish tests ───────────────────────────────────────────────────────


class TestLocalPublish:
    def test_basic_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.status == "succeeded"
        assert published.result.bundle_id == "test-bundle-1"
        assert published.result.manifest_sha256
        assert len(published.result.manifest_sha256) == 64
        assert published.local_bundle_dir.exists()
        assert (published.local_bundle_dir / "manifest.json").is_file()
        assert (
            published.local_bundle_dir / "artifacts" / "fp32" / "model.onnx"
        ).is_file()

    def test_manifest_json_is_valid(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        manifest_path = published.local_bundle_dir / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        assert manifest_data["schema_version"] == 1
        assert manifest_data["bundle_id"] == "test-bundle-1"
        assert len(manifest_data["artifacts"]) == 1

    def test_idempotent_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        result1 = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        # Second publish with same content + same execution_id should be idempotent.
        result2 = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",  # Same execution_id → identical manifest.
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert result1.result.manifest_sha256 == result2.result.manifest_sha256

    def test_partial_bundle_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        _setup_staging(staging, [artifact])

        ref_ok = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(
                _make_node_result(node_id="fp32", artifact_ref=ref_ok),
                _make_node_result(
                    node_id="opt",
                    status="failed",
                    required=False,
                    artifact_ref=None,
                    failure=FailureInfo(
                        code="EXPORT_FAILED",
                        category="export",
                        message="optional export failed",
                    ),
                ),
            ),
            status="partial",
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.status == "partial"
        assert published.local_bundle_dir.exists()

        # The manifest records partial at the top level and keeps the
        # failed node's status and failure details.
        manifest = json.loads(
            (published.local_bundle_dir / "manifest.json").read_bytes()
        )
        assert manifest["status"] == "partial"
        opt_node = next(
            n for n in manifest["execution"]["nodes"] if n["node_id"] == "opt"
        )
        assert opt_node["status"] == "failed"
        assert opt_node["failure"]["message"] == "optional export failed"

    def test_failed_execution_rejected(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"

        execution = _make_execution(
            nodes=(_make_node_result(status="failed", artifact_ref=None),),
            status="failed",
        )

        publisher = Publisher()
        with pytest.raises(ValueError, match="Cannot publish a failed execution"):
            publisher.publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=str(dest),
                bundle_id="test-bundle-1",
                execution_id="exec-1",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_roles_in_result(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            roles={"inference": "fp32"},
        )

        publisher = Publisher()
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert published.result.roles == {"inference": "fp32"}

    def test_alias_written_for_local(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        publisher = Publisher()
        # alias with local URI should write alias file.
        published = publisher.publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-1",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
            alias_config=AliasConfig(name="latest", policy="newer"),
        )

        assert published.result.alias_status == "updated"
        assert published.result.alias_uri is not None
        # Verify alias file was written.
        alias_path = Path(published.result.alias_uri)  # type: ignore[arg-type]
        assert alias_path.is_file()


# ── S3 publish tests (fake client recording requests) ────────────────────────


class _FakeClientError(Exception):
    """Minimal botocore-style ClientError (only ``.response`` is consulted)."""

    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class _FakeS3:
    """In-memory S3 fake that records calls and honours conditional writes.

    The Publisher consults only ``client.exceptions.ClientError`` and the
    ``response["Error"]["Code"]`` attribute, so this fake is sufficient to
    assert request parameters (If-None-Match / If-Match / checksum
    metadata) without a network.  Real end-to-end behaviour against an
    S3-compatible store is covered by the MinIO integration tests.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exceptions = type("_E", (), {"ClientError": _FakeClientError})

    def _etag(self, key: str) -> str:
        # Unquoted hex digest — matches ``_s3_head`` which strips quotes.
        return hashlib.sha256(self.objects.get(key, b"")).hexdigest()

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(("put_object", kwargs))
        key = kwargs["Key"]
        body = kwargs.get("Body", b"")
        if not isinstance(body, bytes):
            body = body.encode()
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise _FakeClientError("PreconditionFailed")
        if "IfMatch" in kwargs and kwargs["IfMatch"].strip('"') != self._etag(key):
            raise _FakeClientError("PreconditionFailed")
        self.objects[key] = body
        if kwargs.get("Metadata"):
            self.metadata[key] = dict(kwargs["Metadata"])
        return {"ETag": f'"{self._etag(key)}"'}  # botocore-style quoted ETag

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_object", kwargs))
        key = kwargs["Key"]
        if key not in self.objects:
            raise _FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[key])}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_object", kwargs))
        key = kwargs["Key"]
        if key not in self.objects:
            raise _FakeClientError("404")
        return {
            "ContentLength": len(self.objects[key]),
            "ETag": f'"{self._etag(key)}"',  # botocore-style quoted ETag
            "Metadata": self.metadata.get(key, {}),
        }

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("delete_object", kwargs))
        self.objects.pop(kwargs["Key"], None)
        return {}


def _publish_s3(
    fake: _FakeS3,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    alias: AliasConfig | None = None,
    bundle_id: str = "bundle-abc123",
) -> Any:
    """Run a publish to ``s3://test-bucket/models`` against *fake*."""
    monkeypatch.setattr(
        "tributo.exporting.publisher._s3_client_from_profile",
        lambda resolver, profile: fake,
    )
    # Unique staging dir per call so idempotent-retry tests can publish
    # multiple times against the same tmp_path.
    staging = tmp_path / f"staging-{len(fake.calls)}"
    artifact = _make_logical_artifact("fp32")
    staged = _setup_staging(staging, [artifact])
    ref = ArtifactRef(
        node_id="fp32",
        artifact_name="fp32",
        tree_digest=artifact.tree_digest,
    )
    execution = _make_execution(
        nodes=(_make_node_result(artifact_ref=ref),),
        staged_artifacts=staged,
    )
    publisher = Publisher()
    return publisher.publish(
        execution=execution,
        staging_root=staging,
        bundle_uri="s3://test-bucket/models",
        bundle_id=bundle_id,
        execution_id="exec-1",
        tributo_version="0.1.0",
        source_info=_make_source_info(),
        alias_config=alias,
    )


class TestS3PublishLogic:
    """S3 request-parameter verification via a recording fake client.

    Integration tests with a real S3-compatible endpoint (MinIO) are in
    ``tests/integration/test_export_s3.py`` (pytest ``s3`` marker).
    """

    def test_s3_manifest_uses_condition_write_and_checksum_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Manifest-last: If-None-Match + tributo-sha256 metadata."""
        fake = _FakeS3()
        published = _publish_s3(fake, monkeypatch, tmp_path)

        assert (
            published.result.manifest_uri
            == "s3://test-bucket/models/bundle-abc123/manifest.json"
        )

        puts_by_key = {kw["Key"]: kw for n, kw in fake.calls if n == "put_object"}
        manifest_put = puts_by_key["models/bundle-abc123/manifest.json"]
        assert manifest_put["IfNoneMatch"] == "*"
        assert manifest_put["ContentType"] == "application/json"
        assert (
            manifest_put["Metadata"]["tributo-sha256"]
            == published.result.manifest_sha256
        )

        # Publish lease uses If-None-Match create semantics.
        lease_put = next(
            kw
            for n, kw in fake.calls
            if n == "put_object" and kw["Key"].endswith(".leases/bundle-abc123.json")
        )
        assert lease_put["IfNoneMatch"] == "*"

        # Artifact upload carries the file body under the artifact key.
        artifact_put = puts_by_key["models/bundle-abc123/artifacts/fp32/model.onnx"]
        assert artifact_put["Body"] == (b"fake-contentmodel.onnx").ljust(100, b"\0")

    def test_s3_idempotent_retry_logically_equal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrying the same bundle does not raise a collision."""
        fake = _FakeS3()
        first = _publish_s3(fake, monkeypatch, tmp_path)
        second = _publish_s3(fake, monkeypatch, tmp_path)

        # Second publish succeeded idempotently with the same manifest.
        assert second.result.status == "succeeded"
        assert second.result.manifest_sha256 == first.result.manifest_sha256

    def test_failed_manifest_publish_cleans_new_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An artifact uploaded before manifest failure must not remain orphaned."""
        fake = _FakeS3()
        original_put = __import__(
            "tributo.exporting.publisher", fromlist=["_s3_put_bytes"]
        )._s3_put_bytes

        def fail_manifest(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("key", args[2] if len(args) > 2 else "").endswith(
                "manifest.json"
            ):
                raise RuntimeError("injected manifest failure")
            return original_put(*args, **kwargs)

        monkeypatch.setattr("tributo.exporting.publisher._s3_put_bytes", fail_manifest)
        with pytest.raises(RuntimeError, match="injected manifest failure"):
            _publish_s3(fake, monkeypatch, tmp_path)

        assert not any("/artifacts/" in key for key in fake.objects)
        assert not any(key.endswith("manifest.json") for key in fake.objects)

    def test_alias_newer_policy_comparison(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """policy=newer: an older existing alias is replaced via If-Match."""
        fake = _FakeS3()
        alias = AliasConfig(name="latest", policy="newer")
        alias_key = "models/aliases/latest.json"
        fake.objects[alias_key] = json.dumps(
            {
                "manifest_uri": "s3://test-bucket/models/old-bundle/manifest.json",
                "manifest_sha256": "d" * 64,
                "bundle_id": "bundle-old",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ).encode()

        published = _publish_s3(fake, monkeypatch, tmp_path, alias=alias)

        assert published.result.alias_status == "updated"
        alias_puts = [
            kw for n, kw in fake.calls if n == "put_object" and kw["Key"] == alias_key
        ]
        assert len(alias_puts) == 1
        assert "IfMatch" in alias_puts[0] and alias_puts[0]["IfMatch"] != "*"

    def test_alias_newer_keeps_fresher_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """policy=newer: a fresher existing alias is left unchanged."""
        fake = _FakeS3()
        alias = AliasConfig(name="latest", policy="newer")
        alias_key = "models/aliases/latest.json"
        fake.objects[alias_key] = json.dumps(
            {
                "manifest_uri": "s3://test-bucket/models/newer-bundle/manifest.json",
                "manifest_sha256": "e" * 64,
                "bundle_id": "bundle-newer",
                "created_at": "2999-01-01T00:00:00+00:00",
            }
        ).encode()

        published = _publish_s3(fake, monkeypatch, tmp_path, alias=alias)

        assert published.result.alias_status == "unchanged"
        assert not any(
            n == "put_object" and kw["Key"] == alias_key for n, kw in fake.calls
        )

    def test_alias_cas_policy_rejects_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compare_and_swap with a mismatched digest fails the alias."""
        fake = _FakeS3()
        alias = AliasConfig(
            name="latest",
            policy="compare_and_swap",
            expected_manifest_sha256="f" * 64,  # differs from existing
        )
        alias_key = "models/aliases/latest.json"
        fake.objects[alias_key] = json.dumps(
            {
                "manifest_uri": "s3://test-bucket/models/old/manifest.json",
                "manifest_sha256": "d" * 64,
                "bundle_id": "bundle-old",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ).encode()

        published = _publish_s3(fake, monkeypatch, tmp_path, alias=alias)

        assert published.result.alias_status == "failed"
        assert published.result.alias_failure is not None
        assert published.result.alias_failure.code == "CAS_MISMATCH"
        assert not any(
            n == "put_object" and kw["Key"] == alias_key for n, kw in fake.calls
        )
