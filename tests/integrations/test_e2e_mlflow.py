"""Real-server integration tests for the MLflow publication Hook."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from uuid import uuid4

import mlflow
import pytest
import requests
from pydantic import BaseModel

from tributo.exceptions import PostPublishCallbackError
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.events import OperationEvent
from tributo.exporting.manifest import ExportManifest
from tributo.exporting.models import (
    ArtifactDraft,
    BundleOutputConfig,
    DraftFile,
    ExportContext,
    ExportSource,
    ExportTarget,
    HookBinding,
    HookStatus,
    ProducerInfo,
    SupportRequest,
    SupportResult,
    ValidatorBinding,
)
from tributo.exporting.records import InMemoryOperationStore
from tributo.exporting.registries import ExportRegistry, ValidatorRegistry
from tributo.exporting.service import BundleExportService
from tributo.integrations.hooks.mlflow_hook import (
    MLflowHookOptions,
    MLflowPostPublishHook,
)

pytestmark = pytest.mark.integration

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@contextmanager
def _mlflow_tracking_uri() -> Iterator[None]:
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        yield
    finally:
        mlflow.set_tracking_uri(previous)


class _Options(BaseModel):
    pass


class _Exporter:
    api_version = 2
    exporter_id = "mlflow-it-exporter-v1"
    priority = 100
    output_format = "native"
    output_flavor_id = "test-native-v1"
    source_kinds = ("mlflow_it",)
    options_model = _Options
    validator_bindings: tuple[ValidatorBinding, ...] = ()
    mutates_source = False
    upstream_requirements: tuple[object, ...] = ()

    @classmethod
    def supports(cls, request: SupportRequest) -> SupportResult:
        return SupportResult(supported=True, code="OK")

    def export(
        self,
        context: ExportContext,
        source: ExportSource,
        upstream: dict[str, object],
        target: object,
    ) -> ArtifactDraft:
        (context.artifact_dir / "model.bin").write_bytes(b"mlflow-it-model")
        return ArtifactDraft(
            name="model",
            format="native",
            flavor_id="test-native-v1",
            files=(DraftFile(relative_path="model.bin", role="model"),),
            entrypoint="model.bin",
            producer=ProducerInfo(exporter_id=self.exporter_id),
        )


class _ManifestOnlyAccessor:
    def __init__(
        self,
        manifest: ExportManifest,
        raw_manifest: bytes,
        root: Path,
    ) -> None:
        self._manifest = manifest
        self._raw_manifest = raw_manifest
        self._root = root
        self.materialized = False
        self.manifest_materialized = False

    def read_manifest(self) -> ExportManifest:
        return self._manifest

    @contextmanager
    def materialize_manifest(self) -> Iterator[Path]:
        self.manifest_materialized = True
        self._root.mkdir(parents=True)
        manifest_path = self._root / "manifest.json"
        manifest_path.write_bytes(self._raw_manifest)
        try:
            yield manifest_path
        finally:
            manifest_path.unlink(missing_ok=True)
            self._root.rmdir()

    def materialize_bundle(self) -> AbstractContextManager[Path]:
        self.materialized = True
        raise AssertionError("remote manifest delivery must not materialize artifacts")


@pytest.fixture(scope="module", autouse=True)
def require_mlflow_server() -> None:
    try:
        response = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search",
            params={"max_results": 1},
            timeout=3,
        )
    except requests.RequestException as exc:
        pytest.fail(f"MLflow server is required at {MLFLOW_TRACKING_URI}: {exc}")
    if response.status_code != 200:
        pytest.fail(f"MLflow server health check returned HTTP {response.status_code}")


@pytest.fixture
def client() -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _model_versions(client: mlflow.MlflowClient) -> set[tuple[str, str]]:
    """Return the server-wide immutable Model Version identity snapshot."""
    return {(item.name, item.version) for item in client.search_model_versions()}


@pytest.fixture
def experiment(client: mlflow.MlflowClient) -> Iterator[str]:
    name = f"tributo-hook-it-{uuid4().hex}"
    yield name
    found = client.get_experiment_by_name(name)
    if found is not None:
        for run in client.search_runs([found.experiment_id]):
            client.delete_run(run.info.run_id)
        client.delete_experiment(found.experiment_id)


def _service(store: InMemoryOperationStore | None = None) -> BundleExportService:
    exporters = ExportRegistry()
    exporters.register(_Exporter)
    return BundleExportService(
        export_registry=exporters,
        validator_registry=ValidatorRegistry(),
        operation_store=store,
    )


def _source() -> ExportSource:
    return ExportSource(source_kind="mlflow_it", source_fingerprint="fixture-v1")


def _config(
    root: Path,
    *,
    request_id: str,
    options: dict[str, object],
    required: bool = False,
) -> BundleOutputConfig:
    return BundleOutputConfig(
        bundle_uri=str(root),
        request_id=request_id,
        targets=[
            ExportTarget(
                name="model",
                format="native",
                exporter_id=_Exporter.exporter_id,
            )
        ],
        hooks=(
            HookBinding(
                hook_id="mlflow-log-artifacts-v1",
                required=required,
                options=options,
            ),
        ),
    )


def test_real_bundle_upload_and_idempotent_replay(
    tmp_path: Path, experiment: str, client: mlflow.MlflowClient
) -> None:
    model_versions_before = _model_versions(client)
    store = InMemoryOperationStore()
    service = _service(store)
    config = _config(
        tmp_path / "bundle",
        request_id="mlflow-real-upload",
        options={
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": experiment,
            "run_name": "bundle-publication",
        },
    )

    first = service.export_bundle(source=_source(), config=config)
    replay = service.export_bundle(source=_source(), config=config)

    first_receipt = first.hook_receipts[0]
    assert first_receipt.status is HookStatus.SUCCEEDED
    assert replay.hook_receipts[0].status is HookStatus.SKIPPED
    run_id = first_receipt.external_references["mlflow_run_id"]
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["tributo.bundle_uri"] == first.canonical_uri
    assert run.data.tags["tributo.manifest_sha256"] == first.manifest_sha256
    paths = {item.path for item in client.list_artifacts(run_id, "bundle")}
    assert "bundle/manifest.json" in paths
    assert "bundle/artifacts" in paths
    found = client.get_experiment_by_name(experiment)
    assert found is not None
    assert len(client.search_runs([found.experiment_id])) == 1
    assert _model_versions(client) == model_versions_before


def test_explicit_run_is_reused(
    tmp_path: Path, experiment: str, client: mlflow.MlflowClient
) -> None:
    experiment_id = client.create_experiment(experiment)
    existing = client.create_run(experiment_id, run_name="owned-by-caller")
    result = _service().export_bundle(
        source=_source(),
        config=_config(
            tmp_path / "reuse",
            request_id="mlflow-run-reuse",
            options={
                "tracking_uri": MLFLOW_TRACKING_URI,
                "run_id": existing.info.run_id,
            },
        ),
    )
    assert result.hook_receipts[0].external_references == {
        "mlflow_run_id": existing.info.run_id
    }
    caller_run = client.get_run(existing.info.run_id)
    assert caller_run.info.status == "RUNNING"
    assert "tributo.bundle_id" not in caller_run.data.params

    conflicting = _service().export_bundle(
        source=_source(),
        config=_config(
            tmp_path / "reuse-conflict",
            request_id="mlflow-run-reuse-conflict",
            options={
                "tracking_uri": MLFLOW_TRACKING_URI,
                "run_id": existing.info.run_id,
            },
        ),
    )
    assert conflicting.hook_receipts[0].status is HookStatus.TERMINAL_FAILED
    assert conflicting.hook_receipts[0].error_code == "mlflow_run_identity_conflict"

    preserved = client.get_run(existing.info.run_id)
    assert preserved.data.tags["tributo.bundle_id"] == result.bundle_id
    download_dir = tmp_path / "caller-run-manifest"
    download_dir.mkdir()
    with _mlflow_tracking_uri():
        manifest_copy = Path(
            client.download_artifacts(
                existing.info.run_id,
                "bundle/manifest.json",
                dst_path=str(download_dir),
            )
        )
    assert (
        hashlib.sha256(manifest_copy.read_bytes()).hexdigest() == result.manifest_sha256
    )


def test_remote_bundle_logs_manifest_only(
    tmp_path: Path, experiment: str, client: mlflow.MlflowClient
) -> None:
    published = _service().export_bundle(
        source=_source(),
        config=BundleOutputConfig(
            bundle_uri=str(tmp_path / "remote-source"),
            request_id="mlflow-remote-manifest",
            targets=[
                ExportTarget(
                    name="model",
                    format="native",
                    exporter_id=_Exporter.exporter_id,
                )
            ],
        ),
    )
    manifest, raw_manifest = BundleReader().read_manifest_with_bytes(
        published.canonical_uri
    )
    event = OperationEvent.bundle_published(
        occurred_at=manifest.created_at,
        bundle_id=published.bundle_id,
        canonical_uri="s3://models/remote-bundle",
        manifest_sha256=published.manifest_sha256,
        source_kind=manifest.source_info.source_kind,
    )
    accessor = _ManifestOnlyAccessor(
        manifest,
        raw_manifest,
        tmp_path / "materialized-manifest",
    )

    outcome = MLflowPostPublishHook().deliver(
        event,
        accessor,
        MLflowHookOptions(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=experiment,
        ),
    )

    assert outcome.status is HookStatus.SUCCEEDED
    assert accessor.materialized is False
    assert accessor.manifest_materialized is True
    run_id = outcome.external_references["mlflow_run_id"]
    assert {item.path for item in client.list_artifacts(run_id, "bundle")} == {
        "bundle/manifest.json"
    }
    download_dir = tmp_path / "downloaded-manifest"
    download_dir.mkdir()
    with _mlflow_tracking_uri():
        downloaded_manifest = Path(
            client.download_artifacts(
                run_id,
                "bundle/manifest.json",
                dst_path=str(download_dir),
            )
        )
    downloaded_bytes = downloaded_manifest.read_bytes()
    assert downloaded_bytes == raw_manifest
    assert hashlib.sha256(downloaded_bytes).hexdigest() == published.manifest_sha256

    replay = MLflowPostPublishHook().deliver(
        event,
        accessor,
        MLflowHookOptions(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=experiment,
        ),
    )
    assert replay.status is HookStatus.SUCCEEDED
    assert replay.external_references == {"mlflow_run_id": run_id}
    found = client.get_experiment_by_name(experiment)
    assert found is not None
    assert len(client.search_runs([found.experiment_id])) == 1


def test_duplicate_idempotency_tags_fail_terminally(
    tmp_path: Path, experiment: str, client: mlflow.MlflowClient
) -> None:
    output = tmp_path / "ambiguous"
    request_id = "mlflow-ambiguous"
    base = BundleOutputConfig(
        bundle_uri=str(output),
        request_id=request_id,
        targets=[
            ExportTarget(
                name="model",
                format="native",
                exporter_id=_Exporter.exporter_id,
            )
        ],
    )
    published = _service().export_bundle(source=_source(), config=base)
    manifest = BundleReader().read_manifest(published.canonical_uri)
    event = OperationEvent.bundle_published(
        occurred_at=manifest.created_at,
        bundle_id=published.bundle_id,
        canonical_uri=published.canonical_uri,
        manifest_sha256=published.manifest_sha256,
        source_kind=manifest.source_info.source_kind,
        correlation_ids={
            "run_id": request_id,
            "request_id": request_id,
            "execution_id": "exec-placeholder",
        },
    )
    options = MLflowHookOptions(
        tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment
    )
    key = MLflowPostPublishHook().idempotency_key(event, options)
    experiment_id = client.create_experiment(experiment)
    for _ in range(2):
        client.create_run(experiment_id, tags={"tributo.idempotency_key": key})

    result = _service().export_bundle(
        source=_source(),
        config=base.model_copy(
            update={
                "hooks": (
                    HookBinding(
                        hook_id="mlflow-log-artifacts-v1",
                        options=options.model_dump(exclude_none=True),
                    ),
                )
            }
        ),
    )
    assert result.hook_receipts[0].status is HookStatus.TERMINAL_FAILED
    assert result.hook_receipts[0].error_code == "ambiguous_mlflow_run"


def test_optional_and_required_tracking_failures_preserve_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    unavailable = {
        "tracking_uri": "http://127.0.0.1:1",
        "experiment_name": "unreachable",
    }
    optional = _service().export_bundle(
        source=_source(),
        config=_config(
            tmp_path / "optional",
            request_id="mlflow-optional-failure",
            options=unavailable,
        ),
    )
    assert optional.hook_receipts[0].status is HookStatus.RETRYABLE_FAILED
    assert Path(optional.manifest_uri).is_file()

    with pytest.raises(PostPublishCallbackError) as exc_info:
        _service().export_bundle(
            source=_source(),
            config=_config(
                tmp_path / "required",
                request_id="mlflow-required-failure",
                options=unavailable,
                required=True,
            ),
        )
    assert Path(exc_info.value.bundle_result.manifest_uri).is_file()
    assert exc_info.value.receipts[0].status is HookStatus.RETRYABLE_FAILED
