"""Unit tests for the MLflow publication adapter contract."""

from __future__ import annotations

import builtins
import hashlib
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tributo.exporting.events import OperationEvent
from tributo.exporting.hooks import ArtifactAccessor, BundleArtifactAccessor
from tributo.exporting.manifest import ExportManifest
from tributo.exporting.models import HookStatus, LogicalArtifact
from tributo.integrations.hooks.mlflow_hook import (
    MLflowHookOptions,
    MLflowPostPublishHook,
)


def _event(
    canonical_uri: str = "file:///committed/bundle-1",
    *,
    bundle_id: str = "bundle-1",
    manifest_sha256: str = "a" * 64,
) -> OperationEvent:
    return OperationEvent.bundle_published(
        occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        bundle_id=bundle_id,
        canonical_uri=canonical_uri,
        manifest_sha256=manifest_sha256,
        source_kind="pytorch_result",
        correlation_ids={"run_id": "run-1"},
    )


class _Accessor:
    manifest_bytes = b'{"bundle_id":"bundle-1","tributo_version":"1.2.3"}'

    def __init__(self, root: Path) -> None:
        self.root = root
        self.materializations = 0
        self.manifest_materializations = 0

    def read_manifest(self) -> Any:
        return _Manifest()

    @contextmanager
    def materialize_manifest(self) -> Any:
        self.manifest_materializations += 1
        manifest_path = self.root / "manifest.json"
        manifest_path.write_bytes(self.manifest_bytes)
        try:
            yield manifest_path
        finally:
            manifest_path.unlink(missing_ok=True)

    @contextmanager
    def materialize_bundle(self) -> Any:
        self.materializations += 1
        yield self.root


class _Manifest:
    tributo_version = "1.2.3"


@pytest.mark.parametrize(
    "unsafe_name",
    [".", "..", "../escape", "/abs/path"],
)
def test_bundle_accessor_rejects_escape_before_reader_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    raw_manifest = b'{"bundle_id":"bundle-1"}'
    event = _event(
        canonical_uri=str(tmp_path / "committed"),
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
    )
    artifact = LogicalArtifact.model_construct(name=unsafe_name, files=())
    manifest = ExportManifest.model_construct(
        bundle_id=event.bundle_id,
        artifacts=(artifact,),
    )
    accessor = BundleArtifactAccessor(
        event,
        manifest=manifest,
        manifest_bytes=raw_manifest,
    )

    def unexpected_open(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("BundleReader must not access an escaping artifact path")

    monkeypatch.setattr(accessor._reader, "open_artifact", unexpected_open)

    with pytest.raises(ValueError, match="escapes materialization root"):
        with accessor.materialize_bundle():
            pass


class _FakeMlflowException(Exception):
    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class _Client:
    def __init__(self) -> None:
        self.experiment = SimpleNamespace(experiment_id="exp-1")
        self.matches: list[Any] = []
        self.created_tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str, str]] = []
        self.manifests: list[tuple[str, bytes, str]] = []
        self.params: list[tuple[str, str, str]] = []
        self.tags: list[tuple[str, str, str]] = []
        self.terminated: list[tuple[str, str]] = []
        self.run_tags: dict[str, str] = {}
        self.run_params: dict[str, str] = {}

    def get_experiment_by_name(self, name: str) -> Any:
        return self.experiment

    def create_experiment(self, name: str) -> str:
        return "exp-1"

    def search_runs(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self.matches

    def create_run(self, experiment_id: str, **kwargs: Any) -> Any:
        self.created_tags = kwargs["tags"]
        return SimpleNamespace(
            info=SimpleNamespace(run_id="run-new"),
            data=SimpleNamespace(tags=self.created_tags, params=self.run_params),
        )

    def get_run(self, run_id: str) -> Any:
        return SimpleNamespace(
            info=SimpleNamespace(run_id=run_id),
            data=SimpleNamespace(tags=self.run_tags, params=self.run_params),
        )

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None:
        self.artifacts.append((run_id, local_dir, artifact_path))

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
        self.manifests.append((run_id, Path(local_path).read_bytes(), artifact_path))

    def log_param(self, run_id: str, name: str, value: str) -> None:
        existing = self.run_params.get(name)
        if existing is not None and existing != value:
            raise _FakeMlflowException(
                "immutable parameter conflict",
                error_code="INVALID_PARAMETER_VALUE",
            )
        self.run_params[name] = value
        self.params.append((run_id, name, value))

    def set_tag(self, run_id: str, name: str, value: str) -> None:
        self.run_tags[name] = value
        self.tags.append((run_id, name, value))

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))


def _install_client(
    monkeypatch: pytest.MonkeyPatch, client: _Client
) -> tuple[list[str], list[str]]:
    fake = ModuleType("mlflow")
    fake.MlflowClient = lambda tracking_uri=None: client
    tracking_uri = ["file:///original"]
    tracking_uri_history: list[str] = []
    fake.get_tracking_uri = lambda: tracking_uri[0]

    def set_tracking_uri(value: str) -> None:
        tracking_uri_history.append(value)
        tracking_uri[0] = value

    fake.set_tracking_uri = set_tracking_uri
    exceptions = ModuleType("mlflow.exceptions")
    exceptions.MlflowException = _FakeMlflowException
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions)
    return tracking_uri, tracking_uri_history


def test_options_require_an_unambiguous_target() -> None:
    with pytest.raises(ValueError, match="experiment_name is required"):
        MLflowHookOptions()
    with pytest.raises(ValueError, match="must be omitted"):
        MLflowHookOptions(run_id="run-1", experiment_name="exp")
    with pytest.raises(ValueError, match="must not contain credentials"):
        MLflowHookOptions(
            tracking_uri="https://user:password@mlflow.example",
            experiment_name="exp",
        )
    with pytest.raises(ValueError, match="query or fragment"):
        MLflowHookOptions(
            tracking_uri="https://mlflow.example?token=secret",
            experiment_name="exp",
        )
    with pytest.raises(ValueError, match="sensitive MLflow tag"):
        MLflowHookOptions(experiment_name="exp", tags={"api-token": "value"})
    with pytest.raises(ValueError, match="sensitive MLflow tag"):
        MLflowHookOptions(experiment_name="exp", tags={"access-key": "value"})
    with pytest.raises(ValueError, match="sensitive MLflow tag"):
        MLflowHookOptions(experiment_name="exp", tags={"apiToken": "value"})

    options = MLflowHookOptions(experiment_name="exp", tags={"tokenizer_version": "v1"})
    assert options.tags == {"tokenizer_version": "v1"}


def test_new_run_is_tagged_at_creation_and_logs_verified_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    tracking_uri, tracking_uri_history = _install_client(monkeypatch, client)
    accessor: ArtifactAccessor = _Accessor(tmp_path)
    options = MLflowHookOptions(
        tracking_uri="https://mlflow.example",
        experiment_name="tributo",
        run_name="publish",
    )

    outcome = MLflowPostPublishHook().deliver(_event(), accessor, options)

    assert outcome.status is HookStatus.SUCCEEDED
    assert outcome.external_references == {"mlflow_run_id": "run-new"}
    assert "tributo.idempotency_key" in client.created_tags
    assert client.artifacts == [("run-new", str(tmp_path), "bundle")]
    assert client.terminated == [("run-new", "FINISHED")]
    assert ("run-new", "tributo.run_id", "run-1") in client.tags
    assert ("run-new", "tributo.version", "1.2.3") in client.tags
    assert tracking_uri == ["file:///original"]
    assert tracking_uri_history == ["https://mlflow.example", "file:///original"]
    assert client.run_params == {
        "tributo.bundle_id": "bundle-1",
        "tributo.manifest_sha256": "a" * 64,
    }


def test_explicit_run_is_reused_without_termination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    _install_client(monkeypatch, client)

    outcome = MLflowPostPublishHook().deliver(
        _event(), _Accessor(tmp_path), MLflowHookOptions(run_id="run-existing")
    )

    assert outcome.status is HookStatus.SUCCEEDED
    assert client.artifacts[0][0] == "run-existing"
    assert client.terminated == []
    assert client.params == []


def test_explicit_run_rejects_a_different_bundle_before_artifact_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    hook = MLflowPostPublishHook()
    options = MLflowHookOptions(run_id="run-existing")
    first = _event()
    client.run_tags = {
        "tributo.idempotency_key": hook.idempotency_key(first, options),
        "tributo.bundle_id": first.bundle_id,
        "tributo.manifest_sha256": first.manifest_sha256,
    }
    _install_client(monkeypatch, client)

    outcome = hook.deliver(
        _event(
            "file:///committed/bundle-2",
            bundle_id="bundle-2",
            manifest_sha256="b" * 64,
        ),
        _Accessor(tmp_path),
        options,
    )

    assert outcome.status is HookStatus.TERMINAL_FAILED
    assert outcome.error_code == "mlflow_run_identity_conflict"
    assert client.artifacts == []
    assert client.params == []


def test_explicit_run_with_idempotency_tag_resumes_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    hook = MLflowPostPublishHook()
    options = MLflowHookOptions(run_id="run-existing")
    client.run_tags = {
        "tributo.idempotency_key": hook.idempotency_key(_event(), options)
    }
    _install_client(monkeypatch, client)

    outcome = hook.deliver(_event(), _Accessor(tmp_path), options)

    assert outcome.status is HookStatus.SUCCEEDED
    assert client.artifacts == [("run-existing", str(tmp_path), "bundle")]
    assert ("run-existing", "tributo.bundle_id", "bundle-1") in client.tags


def test_finished_owned_run_is_recovered_by_idempotency_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    hook = MLflowPostPublishHook()
    options = MLflowHookOptions(experiment_name="tributo")
    key = hook.idempotency_key(_event(), options)
    client.matches = [
        SimpleNamespace(
            info=SimpleNamespace(run_id="run-finished", status="FINISHED"),
            data=SimpleNamespace(
                tags={
                    "tributo.idempotency_key": key,
                    "tributo.bundle_id": "bundle-1",
                    "tributo.manifest_sha256": "a" * 64,
                },
                params={
                    "tributo.bundle_id": "bundle-1",
                    "tributo.manifest_sha256": "a" * 64,
                },
            ),
        )
    ]
    _install_client(monkeypatch, client)

    outcome = hook.deliver(_event(), _Accessor(tmp_path), options)

    assert outcome.status is HookStatus.SUCCEEDED
    assert outcome.external_references == {"mlflow_run_id": "run-finished"}
    assert client.created_tags == {}


def test_remote_bundle_logs_manifest_without_materializing_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    _install_client(monkeypatch, client)
    accessor = _Accessor(tmp_path)

    outcome = MLflowPostPublishHook().deliver(
        _event("s3://models/bundle-1"),
        accessor,
        MLflowHookOptions(experiment_name="tributo"),
    )

    assert outcome.status is HookStatus.SUCCEEDED
    assert accessor.materializations == 0
    assert accessor.manifest_materializations == 1
    assert client.artifacts == []
    assert client.manifests == [("run-new", accessor.manifest_bytes, "bundle")]


def test_tracking_uri_is_restored_when_artifact_upload_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()

    def fail_upload(*args: Any, **kwargs: Any) -> None:
        raise _FakeMlflowException(
            "temporary artifact failure",
            error_code="TEMPORARILY_UNAVAILABLE",
        )

    client.log_artifacts = fail_upload
    tracking_uri, tracking_uri_history = _install_client(monkeypatch, client)

    outcome = MLflowPostPublishHook().deliver(
        _event(),
        _Accessor(tmp_path),
        MLflowHookOptions(
            tracking_uri="https://mlflow.example",
            experiment_name="tributo",
        ),
    )

    assert outcome.status is HookStatus.RETRYABLE_FAILED
    assert tracking_uri == ["file:///original"]
    assert tracking_uri_history == ["https://mlflow.example", "file:///original"]
    assert client.terminated == [("run-new", "FAILED")]


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (
            OSError("temporary filesystem failure"),
            HookStatus.RETRYABLE_FAILED,
            "bundle_materialization_io_failed",
        ),
        (
            ValueError("digest mismatch"),
            HookStatus.TERMINAL_FAILED,
            "bundle_integrity_failed",
        ),
    ],
)
def test_owned_run_is_failed_when_bundle_materialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    expected_status: HookStatus,
    expected_code: str,
) -> None:
    client = _Client()
    _install_client(monkeypatch, client)
    accessor = _Accessor(tmp_path)

    @contextmanager
    def fail_materialization() -> Any:
        raise error
        yield tmp_path

    accessor.materialize_bundle = fail_materialization
    outcome = MLflowPostPublishHook().deliver(
        _event(), accessor, MLflowHookOptions(experiment_name="tributo")
    )

    assert outcome.status is expected_status
    assert outcome.error_code == expected_code
    assert client.terminated == [("run-new", "FAILED")]


@pytest.mark.parametrize(
    ("error_code", "expected_status", "expected_code"),
    [
        (
            "INVALID_PARAMETER_VALUE",
            HookStatus.TERMINAL_FAILED,
            "mlflow_permanent_error",
        ),
        (
            "TEMPORARILY_UNAVAILABLE",
            HookStatus.RETRYABLE_FAILED,
            "mlflow_operation_failed",
        ),
    ],
)
def test_mlflow_error_codes_control_retryability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_code: str,
    expected_status: HookStatus,
    expected_code: str,
) -> None:
    client = _Client()

    def fail_create_run(*args: Any, **kwargs: Any) -> Any:
        raise _FakeMlflowException("secret-server-detail", error_code=error_code)

    client.create_run = fail_create_run
    _install_client(monkeypatch, client)

    outcome = MLflowPostPublishHook().deliver(
        _event(), _Accessor(tmp_path), MLflowHookOptions(experiment_name="tributo")
    )

    assert outcome.status is expected_status
    assert outcome.error_code == expected_code
    assert "secret-server-detail" not in (outcome.error_summary or "")


def test_multiple_idempotency_tag_matches_are_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _Client()
    client.matches = [object(), object()]
    _install_client(monkeypatch, client)

    outcome = MLflowPostPublishHook().deliver(
        _event(),
        _Accessor(tmp_path),
        MLflowHookOptions(experiment_name="tributo"),
    )

    assert outcome.status is HookStatus.TERMINAL_FAILED
    assert outcome.error_code == "ambiguous_mlflow_run"
    assert client.artifacts == []


def test_configured_hook_without_mlflow_is_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_import = builtins.__import__

    def fail_mlflow_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_mlflow_import)
    outcome = MLflowPostPublishHook().deliver(
        _event(),
        _Accessor(tmp_path),
        MLflowHookOptions(experiment_name="tributo"),
    )

    assert outcome.status is HookStatus.TERMINAL_FAILED
    assert outcome.error_code == "mlflow_not_installed"


def test_idempotency_key_includes_target_configuration() -> None:
    hook = MLflowPostPublishHook()
    first = hook.idempotency_key(
        _event(), MLflowHookOptions(experiment_name="one", tags={"env": "test"})
    )
    second = hook.idempotency_key(
        _event(), MLflowHookOptions(experiment_name="two", tags={"env": "test"})
    )
    assert first != second
