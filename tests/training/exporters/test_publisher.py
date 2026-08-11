"""Tests for Publisher — local atomic commit, S3 publish, alias CAS."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tributo.exceptions import BundleCommitBusyError
from tributo.exporting.assembler import BundleAssembler
from tributo.exporting.bundle_reader import BundleReader
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


def _artifact_payload(relative_path: str, size_bytes: int) -> bytes:
    return (b"fake-content" + relative_path.encode()).ljust(size_bytes, b"\0")


def _make_artifact_files() -> tuple[ArtifactFile, ...]:
    model_payload = _artifact_payload("model.onnx", 100)
    config_payload = _artifact_payload("config.json", 50)
    return (
        ArtifactFile(
            relative_path="model.onnx",
            sha256=hashlib.sha256(model_payload).hexdigest(),
            size_bytes=len(model_payload),
            role="model",
        ),
        ArtifactFile(
            relative_path="config.json",
            sha256=hashlib.sha256(config_payload).hexdigest(),
            size_bytes=len(config_payload),
            role="config",
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
    execution_id: str = "exec-1",
) -> ExportExecutionResult:
    if nodes is None:
        ref = ArtifactRef(node_id="fp32", artifact_name="fp32", tree_digest="a" * 64)
        nodes = (_make_node_result(artifact_ref=ref),)
    return ExportExecutionResult(
        execution_id=execution_id,
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
            fp.write_bytes(_artifact_payload(af.relative_path, af.size_bytes))
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
    def test_alias_decode_failure_does_not_erase_committed_bundle(
        self, tmp_path: Path
    ) -> None:
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
        alias_dir = dest / "aliases"
        alias_dir.mkdir(parents=True)
        (alias_dir / "latest.json").write_bytes(b"{not-json")

        published = Publisher().publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-alias-failure",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
            alias_config=AliasConfig(name="latest", policy="newer"),
        )

        assert published.result.status == "succeeded"
        assert published.result.alias_status == "failed"
        assert published.result.alias_failure is not None
        assert published.result.alias_failure.code == "ValueError"
        assert Path(published.result.manifest_uri).is_file()

    def test_oversized_alias_is_bounded_after_bundle_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.integrations.storage import bundle_repository

        monkeypatch.setattr(bundle_repository, "_CONTROL_DOCUMENT_MAX_BYTES", 8)
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
        alias_dir = dest / "aliases"
        alias_dir.mkdir(parents=True)
        (alias_dir / "latest.json").write_bytes(b'{"value":1}')

        published = Publisher().publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(dest),
            bundle_id="test-bundle-oversized-alias",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
            alias_config=AliasConfig(name="latest", policy="newer"),
        )

        assert published.result.alias_status == "failed"
        assert published.result.alias_failure is not None
        assert "exceeds limit" in published.result.alias_failure.message
        assert Path(published.result.manifest_uri).is_file()

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

    def test_idempotent_retry_rejects_missing_committed_artifact(
        self, tmp_path: Path
    ) -> None:
        staging = tmp_path / "staging-missing-committed"
        dest = tmp_path / "dest-missing-committed"
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
            bundle_id="test-bundle-missing-committed",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )
        (published.local_bundle_dir / "artifacts" / "fp32" / "model.onnx").unlink()

        with pytest.raises(FileNotFoundError, match="Artifact file missing"):
            publisher.publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=str(dest),
                bundle_id="test-bundle-missing-committed",
                execution_id="exec-1",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_concurrent_identical_publish_is_idempotent(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"

        def publish(index: int) -> Any:
            staging = tmp_path / f"staging-{index}"
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
                execution_id="exec-concurrent",
            )
            return Publisher().publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=str(dest),
                bundle_id="test-bundle-concurrent",
                execution_id="exec-concurrent",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(publish, range(2)))

        assert results[0].result.manifest_sha256 == results[1].result.manifest_sha256
        assert (dest / "test-bundle-concurrent" / "manifest.json").is_file()

    def test_repository_rejects_tampered_staged_manifest_digest(
        self, tmp_path: Path
    ) -> None:
        class TamperedAssembler(BundleAssembler):
            def assemble(self, **kwargs: Any) -> Any:
                staged = super().assemble(**kwargs)
                return replace(staged, manifest_sha256="f" * 64)

        staging = tmp_path / "staging-tampered"
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

        with pytest.raises(ValueError, match="digest does not match"):
            Publisher(assembler=TamperedAssembler()).publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=str(tmp_path / "dest"),
                bundle_id="test-bundle-tampered",
                execution_id="exec-1",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_local_commit_excludes_undeclared_staging_files(
        self, tmp_path: Path
    ) -> None:
        staging = tmp_path / "staging-extra"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])
        artifact_dir = staging / "nodes" / "fp32" / "artifact"
        (artifact_dir / "undeclared.secret").write_bytes(b"must-not-publish")
        ref = ArtifactRef(
            node_id="fp32",
            artifact_name="fp32",
            tree_digest=artifact.tree_digest,
        )
        execution = _make_execution(
            nodes=(_make_node_result(artifact_ref=ref),),
            staged_artifacts=staged,
        )

        published = Publisher().publish(
            execution=execution,
            staging_root=staging,
            bundle_uri=str(tmp_path / "dest-extra"),
            bundle_id="test-bundle-extra",
            execution_id="exec-1",
            tributo_version="0.1.0",
            source_info=_make_source_info(),
        )

        assert not (
            published.local_bundle_dir / "artifacts" / "fp32" / "undeclared.secret"
        ).exists()

    def test_partial_bundle_publish(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        dest = tmp_path / "dest"
        artifact = _make_logical_artifact("fp32")
        staged = _setup_staging(staging, [artifact])

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

    def test_publishable_node_requires_staged_artifact(self, tmp_path: Path) -> None:
        execution = _make_execution()

        with pytest.raises(ValueError, match="has no staged artifact"):
            Publisher().publish(
                execution=execution,
                staging_root=tmp_path / "staging",
                bundle_uri=str(tmp_path / "dest"),
                bundle_id="test-bundle-missing-artifact",
                execution_id="exec-1",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_execution_identity_must_match_publish_request(
        self, tmp_path: Path
    ) -> None:
        execution = _make_execution()

        with pytest.raises(ValueError, match="does not match publication"):
            Publisher().publish(
                execution=execution,
                staging_root=tmp_path / "staging",
                bundle_uri=str(tmp_path / "dest"),
                bundle_id="test-bundle-wrong-execution",
                execution_id="another-execution",
                tributo_version="0.1.0",
                source_info=_make_source_info(),
            )

    def test_roles_in_result(self, tmp_path: Path) -> None:
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
            roles={"inference": "fp32"},
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
        manifest = BundleReader().read_manifest(published.result.alias_uri)
        assert manifest.bundle_id == published.result.bundle_id


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
        kwargs = dict(kwargs)
        key = kwargs["Key"]
        body = kwargs.get("Body", b"")
        if hasattr(body, "read"):
            body = body.read()
        elif not isinstance(body, bytes):
            body = body.encode()
        kwargs["Body"] = body
        self.calls.append(("put_object", kwargs))
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
        return {
            "Body": io.BytesIO(self.objects[key]),
            "ETag": f'"{self._etag(key)}"',
        }

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
        if "IfMatch" in kwargs and kwargs["IfMatch"].strip('"') != self._etag(
            kwargs["Key"]
        ):
            raise _FakeClientError("PreconditionFailed")
        self.objects.pop(kwargs["Key"], None)
        return {}

    def upload_fileobj(
        self,
        stream: Any,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, Any],
    ) -> None:
        self.calls.append(
            (
                "upload_fileobj",
                {"Bucket": bucket, "Key": key, "ExtraArgs": ExtraArgs},
            )
        )
        self.objects[key] = stream.read()
        self.metadata[key] = dict(ExtraArgs.get("Metadata", {}))


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
        "tributo.integrations.storage.bundle_repository._s3_client_from_profile",
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

    def test_s3_control_document_read_is_bounded_without_content_length(
        self,
    ) -> None:
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        fake.objects["models/aliases/latest.json"] = b'{"value":"too-large"}'

        with pytest.raises(ValueError, match="exceeds limit"):
            bundle_repository._s3_get_json_with_etag(
                fake,
                "test-bucket",
                "models/aliases/latest.json",
                8,
            )

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

    def test_disappearing_lease_conflict_is_not_treated_as_acquired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raced-away lease must make the caller retry, never upload unlocked."""
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        monkeypatch.setattr(
            bundle_repository, "_s3_get_json_with_etag", lambda *args: None
        )

        def conflict(*args: Any, **kwargs: Any) -> None:
            raise _FakeClientError("PreconditionFailed")

        monkeypatch.setattr(bundle_repository, "_s3_put_json", conflict)

        with pytest.raises(BundleCommitBusyError, match="changed during"):
            bundle_repository._s3_lease_acquire(
                fake,
                "test-bucket",
                "models/.leases/",
                "bundle-abc123",
                "owner-1",
            )

    def test_lease_takeover_uses_etag_from_the_same_read_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raced live owner cannot be overwritten through a GET/HEAD gap."""
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        lease_key = "models/.leases/bundle-abc123.json"
        fake.objects[lease_key] = json.dumps(
            {
                "owner": "expired-owner",
                "created_at": 0,
                "expires_at": 0,
                "bundle_id": "bundle-abc123",
            }
        ).encode()
        original_put = bundle_repository._s3_put_json
        raced = False

        def replace_before_cas(*args: Any, **kwargs: Any) -> Any:
            nonlocal raced
            if kwargs.get("if_match") is not None and not raced:
                raced = True
                fake.objects[lease_key] = json.dumps(
                    {
                        "owner": "live-owner",
                        "created_at": 1,
                        "expires_at": 10**18,
                        "bundle_id": "bundle-abc123",
                    }
                ).encode()
            return original_put(*args, **kwargs)

        monkeypatch.setattr(bundle_repository, "_s3_put_json", replace_before_cas)

        with pytest.raises(BundleCommitBusyError, match="held by live-owner"):
            bundle_repository._s3_lease_acquire(
                fake,
                "test-bucket",
                "models/.leases/",
                "bundle-abc123",
                "new-owner",
            )

        assert json.loads(fake.objects[lease_key])["owner"] == "live-owner"

    def test_file_above_atomic_single_put_limit_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Oversized files fail before any non-atomic multipart upload."""
        fake = _FakeS3()
        monkeypatch.setattr(
            "tributo.integrations.storage.bundle_repository._S3_SINGLE_PUT_MAX_BYTES",
            1,
        )

        with pytest.raises(ValueError, match="supports files up to 5 GB"):
            _publish_s3(fake, monkeypatch, tmp_path)

        assert all(name != "upload_fileobj" for name, _ in fake.calls)

    def test_single_put_streams_without_path_read_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Conditional single PUT streams the file instead of loading it all."""
        fake = _FakeS3()
        original_read_bytes = Path.read_bytes

        def reject_artifact_read_bytes(path: Path) -> bytes:
            if path.name in {"model.onnx", "config.json"}:
                raise AssertionError("artifact upload must stream")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", reject_artifact_read_bytes)

        published = _publish_s3(fake, monkeypatch, tmp_path)

        assert published.result.status == "succeeded"

    def test_slow_upload_renews_lease_and_rechecks_before_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking upload is protected by heartbeat and final renewal."""
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        renewed = threading.Event()
        acquire_calls = 0
        original_acquire = bundle_repository._s3_lease_acquire
        original_put_stream = bundle_repository._s3_put_stream

        def acquire(*args: Any, **kwargs: Any) -> str:
            nonlocal acquire_calls
            lease_key = original_acquire(*args, **kwargs)
            acquire_calls += 1
            if acquire_calls >= 2:
                renewed.set()
            return lease_key

        def blocking_put_stream(*args: Any, **kwargs: Any) -> Any:
            assert renewed.wait(timeout=1)
            return original_put_stream(*args, **kwargs)

        monkeypatch.setattr(bundle_repository, "_s3_lease_acquire", acquire)
        monkeypatch.setattr(
            bundle_repository, "_LEASE_HEARTBEAT_INTERVAL_SECONDS", 0.01
        )
        monkeypatch.setattr(bundle_repository, "_s3_put_stream", blocking_put_stream)

        published = _publish_s3(fake, monkeypatch, tmp_path)

        assert published.result.status == "succeeded"
        assert acquire_calls >= 3

    def test_lost_lease_never_commits_or_deletes_unowned_objects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A superseded writer leaves its orphan for GC and never commits."""
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        acquire_calls = 0
        original_acquire = bundle_repository._s3_lease_acquire

        def lose_on_renew(*args: Any, **kwargs: Any) -> str:
            nonlocal acquire_calls
            acquire_calls += 1
            if acquire_calls > 1:
                lease_key = "models/.leases/bundle-abc123.json"
                fake.objects[lease_key] = json.dumps(
                    {
                        "owner": "new-owner",
                        "created_at": 0,
                        "expires_at": 10**18,
                        "bundle_id": "bundle-abc123",
                    }
                ).encode()
                raise BundleCommitBusyError("lease taken over")
            return original_acquire(*args, **kwargs)

        monkeypatch.setattr(bundle_repository, "_s3_lease_acquire", lose_on_renew)
        monkeypatch.setattr(bundle_repository, "_LEASE_HEARTBEAT_INTERVAL_SECONDS", 60)

        with pytest.raises(BundleCommitBusyError, match="Lost publish lease"):
            _publish_s3(fake, monkeypatch, tmp_path)

        assert not any(key.endswith("manifest.json") for key in fake.objects)
        assert any("/artifacts/" in key for key in fake.objects)
        assert (
            json.loads(fake.objects["models/.leases/bundle-abc123.json"])["owner"]
            == "new-owner"
        )

    def test_lease_heartbeat_retries_transient_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One storage error does not permanently poison a healthy lease."""
        from tributo.integrations.storage import bundle_repository

        renewed = threading.Event()
        calls = 0

        def flaky_acquire(*args: Any, **kwargs: Any) -> str:
            nonlocal calls
            del args, kwargs
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary throttle")
            renewed.set()
            return "models/.leases/bundle-abc123.json"

        monkeypatch.setattr(bundle_repository, "_s3_lease_acquire", flaky_acquire)
        heartbeat = bundle_repository._S3LeaseHeartbeat(
            object(),
            "test-bucket",
            "models/.leases/",
            "bundle-abc123",
            "owner-1",
            interval_seconds=0.01,
        )
        heartbeat.start()

        assert renewed.wait(timeout=1)
        assert heartbeat.stop()
        assert not heartbeat.failed
        assert calls >= 2

    def test_lease_heartbeat_stop_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stuck client call cannot block publisher teardown indefinitely."""
        from tributo.integrations.storage import bundle_repository

        entered = threading.Event()
        release = threading.Event()

        def blocking_acquire(*args: Any, **kwargs: Any) -> str:
            del args, kwargs
            entered.set()
            assert release.wait(timeout=2)
            return "models/.leases/bundle-abc123.json"

        monkeypatch.setattr(bundle_repository, "_s3_lease_acquire", blocking_acquire)
        monkeypatch.setattr(bundle_repository, "_LEASE_STOP_TIMEOUT_SECONDS", 0.01)
        heartbeat = bundle_repository._S3LeaseHeartbeat(
            object(),
            "test-bucket",
            "models/.leases/",
            "bundle-abc123",
            "owner-1",
            interval_seconds=0.01,
        )
        heartbeat.start()
        assert entered.wait(timeout=1)

        assert not heartbeat.stop()
        release.set()
        heartbeat._thread.join(timeout=1)
        assert heartbeat.stop()

    def test_lease_release_falls_back_to_conditional_expiry(self) -> None:
        """Stores without conditional DELETE retain owner-safe release semantics."""
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        lease_key = "models/.leases/bundle-abc123.json"
        fake.objects[lease_key] = json.dumps(
            {
                "owner": "owner-1",
                "created_at": 1,
                "expires_at": 10**18,
                "bundle_id": "bundle-abc123",
            }
        ).encode()

        def unsupported_delete(**kwargs: Any) -> dict[str, Any]:
            fake.calls.append(("delete_object", kwargs))
            raise _FakeClientError("NotImplemented")

        fake.delete_object = unsupported_delete
        bundle_repository._s3_lease_release(fake, "test-bucket", lease_key, "owner-1")

        released = json.loads(fake.objects[lease_key])
        assert released["owner"] == "owner-1"
        assert released["expires_at"] == 0
        release_put = next(
            kwargs
            for name, kwargs in reversed(fake.calls)
            if name == "put_object" and kwargs["Key"] == lease_key
        )
        assert "IfMatch" in release_put

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
        manifest_key = "models/bundle-abc123/manifest.json"
        assert second.manifest_bytes == fake.objects[manifest_key]

    def test_failed_manifest_publish_cleans_new_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An artifact uploaded before manifest failure must not remain orphaned."""
        fake = _FakeS3()
        original_put = __import__(
            "tributo.integrations.storage.bundle_repository",
            fromlist=["_s3_put_bytes"],
        )._s3_put_bytes

        def fail_manifest(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("key", args[2] if len(args) > 2 else "").endswith(
                "manifest.json"
            ):
                raise RuntimeError("injected manifest failure")
            return original_put(*args, **kwargs)

        monkeypatch.setattr(
            "tributo.integrations.storage.bundle_repository._s3_put_bytes",
            fail_manifest,
        )
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

    def test_alias_conditional_race_reports_retry_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tributo.integrations.storage import bundle_repository

        fake = _FakeS3()
        monkeypatch.setattr(
            bundle_repository, "_s3_get_json_with_etag", lambda *args: None
        )

        def conflict(*args: Any, **kwargs: Any) -> None:
            raise _FakeClientError("PreconditionFailed")

        monkeypatch.setattr(bundle_repository, "_s3_put_json", conflict)

        status, failure = bundle_repository._update_alias_s3(
            client=fake,
            bucket="test-bucket",
            alias_key_path="models/aliases/latest.json",
            alias_config=AliasConfig(name="latest", policy="newer"),
            manifest_uri="s3://test-bucket/models/bundle-abc/manifest.json",
            manifest_sha256="a" * 64,
            bundle_id="bundle-abc",
            created_at="2026-08-06T00:00:00+00:00",
        )

        assert status == "failed"
        assert failure == "RETRY_EXHAUSTED"
