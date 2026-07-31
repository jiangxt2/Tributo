"""Tests for post-publish hooks — PublicationRunner and the MLflow hook.

Covers the review fixes: the runner injects the publisher's canonical
``_manifest_sha256`` into the manifest dict and forwards the staging-window
``local_bundle_dir``; the MLflow hook logs the full local bundle directory
via ``log_artifacts`` when one is available (dict fallback otherwise).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tributo.exporting.hooks import HookReceipt, PublicationRunner
from tributo.integrations.hooks.mlflow_hook import MLflowPostPublishHook

# ── Helpers ───────────────────────────────────────────────────────────────────


class _RecordingHook:
    """PostPublishHook that records its arguments."""

    hook_id: ClassVar[str] = "rec-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def execute(
        self,
        canonical_uri: str,
        manifest: dict[str, Any],
        options: dict[str, Any] | None = None,
        local_bundle_dir: str | None = None,
    ) -> HookReceipt:
        self.calls.append((canonical_uri, manifest, options, local_bundle_dir))
        return HookReceipt(hook_id=self.hook_id, status="success")

    def idempotency_key(
        self,
        canonical_uri: str,
        manifest_sha256: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        return "rec-key"


class _FailingHook(_RecordingHook):
    hook_id: ClassVar[str] = "fail-v1"

    def execute(
        self,
        canonical_uri: str,
        manifest: dict[str, Any],
        options: dict[str, Any] | None = None,
        local_bundle_dir: str | None = None,
    ) -> HookReceipt:
        self.calls.append((canonical_uri, manifest, options, local_bundle_dir))
        return HookReceipt(
            hook_id=self.hook_id, status="failed", error="boom", retryable=True
        )


# ── PublicationRunner ─────────────────────────────────────────────────────────


class TestPublicationRunner:
    def test_injects_manifest_sha_and_forwards_local_dir(self) -> None:
        hook = _RecordingHook()
        runner = PublicationRunner([(hook, {}, False)])
        manifest = {"bundle_id": "bundle-x"}
        sha = "d" * 64

        runner.run("s3://b/models/x/", manifest, sha, "/tmp/bundle-x")

        _, seen_manifest, _, seen_dir = hook.calls[0]
        assert seen_manifest["_manifest_sha256"] == sha
        assert seen_dir == "/tmp/bundle-x"

    def test_required_hook_failure_raises(self) -> None:
        hook = _FailingHook()
        runner = PublicationRunner([(hook, {}, True)])

        with pytest.raises(RuntimeError, match="Required hook"):
            runner.run("s3://b/models/x/", {"bundle_id": "b"}, "d" * 64)


# ── MLflowPostPublishHook ─────────────────────────────────────────────────────


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = type("_Info", (), {"run_id": run_id})()


class _FakeMlflow:
    """Minimal mlflow facade recording log_artifacts / log_dict / params."""

    def __init__(self) -> None:
        self.log_artifacts_calls: list[tuple[str, str]] = []
        self.log_dict_calls: list[tuple[Any, str]] = []
        self.log_params_calls: list[dict[str, str]] = []
        self.set_tags_calls: list[dict[str, str]] = []
        self.set_tracking_uri_calls: list[str] = []

    @contextmanager
    def start_run(self, run_id: str | None = None, run_name: str | None = None) -> Any:
        yield _FakeRun(run_id or "run-1")

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        self.log_artifacts_calls.append((local_dir, artifact_path or ""))

    def log_dict(self, dictionary: Any, artifact_file: str) -> None:
        self.log_dict_calls.append((dictionary, artifact_file))

    def log_params(self, params: dict[str, str]) -> None:
        self.log_params_calls.append(params)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.set_tags_calls.append(tags)

    def set_tracking_uri(self, uri: str) -> None:
        self.set_tracking_uri_calls.append(uri)


def _run_hook_with_fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    local_bundle_dir: str | None,
    injected_sha: str | None = None,
) -> tuple[_FakeMlflow, HookReceipt]:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    manifest: dict[str, Any] = {
        "bundle_id": "bundle-x",
        "source_info": {"source_kind": "xgboost_result"},
    }
    if injected_sha is not None:
        manifest["_manifest_sha256"] = injected_sha

    hook = MLflowPostPublishHook()
    receipt = hook.execute(
        "s3://b/models/x/",
        manifest,
        None,
        local_bundle_dir,
    )
    return fake, receipt


class TestMLflowHook:
    def test_logs_artifacts_when_bundle_dir_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A real bundle layout is logged via log_artifacts, not flattened."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "manifest.json").write_text("{}")

        fake, receipt = _run_hook_with_fake_mlflow(
            monkeypatch,
            tmp_path,
            local_bundle_dir=str(bundle_dir),
            injected_sha="e" * 64,
        )

        assert receipt.status == "success"
        assert fake.log_artifacts_calls == [(str(bundle_dir), "bundle")]
        assert fake.log_dict_calls == []
        # The sha comes from the injected canonical digest, not a fallback.
        assert fake.log_params_calls[0]["manifest_sha256"] == "e" * 64

    def test_falls_back_to_log_dict_without_bundle_layout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """S3-only publishes have no bundle-layout dir — dict fallback."""
        fake, receipt = _run_hook_with_fake_mlflow(
            monkeypatch, tmp_path, local_bundle_dir=None
        )

        assert receipt.status == "success"
        assert fake.log_artifacts_calls == []
        assert len(fake.log_dict_calls) == 1
        assert fake.log_dict_calls[0][1] == "bundle/manifest.json"

    def test_skipped_when_mlflow_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "mlflow", None)  # import fails
        hook = MLflowPostPublishHook()
        receipt = hook.execute("s3://b/models/x/", {"bundle_id": "b"}, None, None)
        assert receipt.status == "skipped"
        assert receipt.error == "mlflow not installed"
