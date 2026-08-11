"""Contract tests for the converged model-export architecture."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from tributo._bootstrap import first_party_export_plugins, first_party_model_flavors
from tributo.exporting.capabilities import get_default_capability_registry
from tributo.exporting.events import OperationEvent
from tributo.exporting.executor import _materialize_artifact
from tributo.exporting.models import (
    AliasConfig,
    ArtifactDraft,
    BundleRef,
    DraftFile,
    ProducerInfo,
    SupportRequest,
    ValidatorBinding,
)
from tributo.exporting.registries import ExportRegistry, ValidatorRegistry
from tributo.exporting.repository import (
    BundleAliasStore,
    BundleRepository,
    BundleRepositoryRouter,
    build_default_repository_router,
)
from tributo.exporting.service import BundleExportService, _load_entry_point_plugins
from tributo.integrations.storage.bundle_repository import (
    LocalBundleAliasStore,
    LocalBundleRepository,
    S3BundleAliasStore,
    S3BundleRepository,
)
from tributo.plugin import (
    _discover_storage_adapter_plugins,
    discover_bundle_alias_store_plugins,
    discover_bundle_repository_plugins,
)

DEFAULT_CAPABILITY_REGISTRY = get_default_capability_registry()


@pytest.mark.parametrize(
    "artifact_kind",
    ("model", "report", "diagnostics", "graph_snapshot"),
)
def test_artifact_kind_survives_draft_materialization(
    tmp_path: Path, artifact_kind: str
) -> None:
    artifact_dir = tmp_path / artifact_kind
    artifact_dir.mkdir()
    (artifact_dir / "payload.bin").write_bytes(b"payload")
    draft = ArtifactDraft(
        name="artifact",
        format="binary",
        flavor_id="test-v1",
        files=(DraftFile(relative_path="payload.bin", role="aux"),),
        entrypoint="payload.bin",
        producer=ProducerInfo(exporter_id="test-v1"),
        artifact_kind=artifact_kind,
    )

    materialized = _materialize_artifact(draft, artifact_dir, "node-1")

    assert materialized.artifact_kind == artifact_kind


def test_service_accepts_only_resolved_export_source() -> None:
    constructor = inspect.signature(BundleExportService)
    export_bundle = inspect.signature(BundleExportService.export_bundle)

    assert "source_provider_registry" not in constructor.parameters
    assert "provider" not in export_bundle.parameters


def test_first_party_storage_adapters_conform_to_ports() -> None:
    repositories = (LocalBundleRepository(), S3BundleRepository())
    alias_stores = (LocalBundleAliasStore(), S3BundleAliasStore())

    assert all(isinstance(adapter, BundleRepository) for adapter in repositories)
    assert all(isinstance(adapter, BundleAliasStore) for adapter in alias_stores)


def test_repository_does_not_require_alias_store_for_same_scheme() -> None:
    router = BundleRepositoryRouter((LocalBundleRepository(),), ())

    assert (
        router.resolve_alias(
            "/tmp/direct-bundle",
            storage_profile=None,
            max_alias_bytes=1024,
        )
        is None
    )
    assert isinstance(
        router.repository_for("/tmp/direct-bundle"), LocalBundleRepository
    )


def test_default_composition_root_discovers_local_and_s3() -> None:
    repositories = {cls.repository_id for cls in discover_bundle_repository_plugins()}
    alias_stores = {cls.alias_store_id for cls in discover_bundle_alias_store_plugins()}

    assert repositories == {"local-v1", "s3-v1"}
    assert alias_stores == {"local-alias-v1", "s3-alias-v1"}


def test_default_composition_root_does_not_require_entry_point_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tributo.plugin.discover_bundle_repository_plugins", lambda: [])
    monkeypatch.setattr(
        "tributo.plugin.discover_bundle_alias_store_plugins", lambda: []
    )

    router = build_default_repository_router()

    assert isinstance(router.repository_for("/tmp/bundle"), LocalBundleRepository)
    assert isinstance(router.repository_for("s3://bucket/model"), S3BundleRepository)


def test_storage_plugin_discovery_rejects_incomplete_method_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteRepository:
        api_version = 1
        repository_id = "incomplete-v1"
        schemes = ("incomplete",)

        def commit(self) -> None:
            pass

    class EntryPoint:
        name = "incomplete-v1"
        value = "test:IncompleteRepository"

        @staticmethod
        def load() -> type[IncompleteRepository]:
            return IncompleteRepository

    monkeypatch.setattr(
        "tributo.plugin._iter_entry_points", lambda group: [EntryPoint()]
    )

    discovered = _discover_storage_adapter_plugins(
        group="tributo.bundle_repositories",
        identity_attribute="repository_id",
        required_methods=("commit", "read_manifest", "materialize_artifact"),
    )

    assert discovered == []


def test_first_party_exporters_do_not_require_entry_point_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tributo.exporting import service as service_module

    monkeypatch.setattr("tributo.plugin.discover_exporter_plugins", lambda **kw: [])
    monkeypatch.setattr("tributo.plugin.discover_validator_plugins", lambda **kw: [])
    monkeypatch.setattr(service_module, "_plugins_loaded", False)
    monkeypatch.setattr(
        service_module,
        "_plugin_cache",
        {"exports": [], "validators": []},
    )
    exporters = ExportRegistry()
    validators = ValidatorRegistry()

    _load_entry_point_plugins(exporters, validators)

    assert "xgboost-onnx-v1" in exporters.list_all()
    assert "torch-onnx-v1" in exporters.list_all()
    assert "onnx-runtime-v1" in validators.list_all()


def test_first_party_flavors_declare_capability_metadata() -> None:
    for flavor in first_party_model_flavors():
        assert flavor.supported_formats
        assert all(isinstance(format_id, str) for format_id in flavor.supported_formats)
        assert isinstance(flavor.batch_supported, bool)
        assert isinstance(flavor.serveable, bool)


def test_public_exporting_import_does_not_eagerly_load_integrations() -> None:
    code = (
        "import sys; import tributo.exporting; "
        "assert 'tributo.integrations.exporters.xgboost_native' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_exporting_package_does_not_import_concrete_storage() -> None:
    import tributo.exporting

    root = Path(tributo.exporting.__file__).parent
    for source_path in root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module == "boto3" or module.startswith("tributo.integrations.storage")
            for module in imported
        ), source_path


def test_core_routing_has_no_concrete_format_branch() -> None:
    """New formats are selected by plugin metadata, never core conditionals."""
    from tributo.exporting import executor, planner, repository, service

    concrete_formats = {"onnx", "ubj", "xgboost-json", "safetensors", "pt2"}
    for module in (planner, executor, service, repository):
        source_path = Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.Match)):
                continue
            branch = node.test if isinstance(node, ast.If) else node.subject
            constants = {
                child.value
                for child in ast.walk(branch)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            assert constants.isdisjoint(concrete_formats), (
                source_path,
                node.lineno,
                constants.intersection(concrete_formats),
            )


def test_storage_plugin_rejects_constructor_without_storage_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongConstructorRepository:
        api_version = 1
        repository_id = "wrong-constructor-v1"
        schemes = ("wrong-constructor",)

        def __init__(self, resolver: object | None = None) -> None:
            self.resolver = resolver

        def commit(self) -> None:
            pass

        def read_manifest(self) -> None:
            pass

        def materialize_artifact(self) -> None:
            pass

    class EntryPoint:
        name = "wrong-constructor-v1"
        value = "test:WrongConstructorRepository"

        @staticmethod
        def load() -> type[WrongConstructorRepository]:
            return WrongConstructorRepository

    monkeypatch.setattr(
        "tributo.plugin._iter_entry_points", lambda group: [EntryPoint()]
    )

    assert discover_bundle_repository_plugins() == []


def test_every_first_party_exporter_has_explicit_capability() -> None:
    declared = {
        exporter_id
        for entry in DEFAULT_CAPABILITY_REGISTRY.entries()
        for exporter_id in entry.exporter_ids
    }
    exporters, _ = first_party_export_plugins()
    discovered = {cls.exporter_id for cls in exporters}

    assert discovered == declared


@pytest.mark.parametrize("exporter_cls", first_party_export_plugins()[0])
def test_first_party_exporter_conformance(exporter_cls: type) -> None:
    required = (
        "api_version",
        "exporter_id",
        "priority",
        "output_format",
        "output_flavor_id",
        "options_model",
        "validator_bindings",
        "mutates_source",
        "upstream_requirements",
    )
    assert all(hasattr(exporter_cls, attribute) for attribute in required)
    assert exporter_cls.api_version == 2
    assert isinstance(exporter_cls.exporter_id, str) and exporter_cls.exporter_id
    assert isinstance(exporter_cls.priority, int)
    assert isinstance(exporter_cls.validator_bindings, tuple)
    assert all(
        isinstance(binding, ValidatorBinding)
        for binding in exporter_cls.validator_bindings
    )
    unsupported = exporter_cls.supports(
        SupportRequest(source_kind="unknown-conformance-source")
    )
    assert not unsupported.supported


@pytest.mark.parametrize("validator_cls", first_party_export_plugins()[1])
def test_first_party_validator_conformance(validator_cls: type) -> None:
    assert validator_cls.api_version == 1
    assert isinstance(validator_cls.validator_id, str)
    assert validator_cls.validator_id
    assert hasattr(validator_cls, "options_model")


def test_capabilities_do_not_infer_runtime_from_onnx_format() -> None:
    onnx = DEFAULT_CAPABILITY_REGISTRY.for_flavor("onnx-runtime-v1")
    quantized = DEFAULT_CAPABILITY_REGISTRY.for_flavor("onnx-int8-v1")
    huggingface = DEFAULT_CAPABILITY_REGISTRY.for_flavor("hf-onnx-v1")

    assert onnx.exportable and onnx.readable and onnx.batch and onnx.serveable
    assert quantized.exportable and quantized.readable
    assert not quantized.batch and not quantized.serveable
    assert huggingface.exportable and huggingface.readable
    assert not huggingface.batch and not huggingface.serveable


def test_capabilities_are_derived_from_plugin_descriptors() -> None:
    from tributo.exporting.capabilities import CapabilityRegistry

    class _GGUFExporter:
        exporter_id = "third-party-gguf-v1"
        output_format = "gguf"
        output_flavor_id = "gguf-file-v1"

    registry = CapabilityRegistry.from_plugins((_GGUFExporter,))

    capability = registry.for_exporter("third-party-gguf-v1")
    assert capability.flavor_id == "gguf-file-v1"
    assert capability.format_ids == ("gguf",)
    assert capability.exportable and capability.readable
    assert not capability.batch and not capability.serveable


def test_capability_matrix_matches_runtime_serveable_matrix() -> None:
    from tributo.exporting.runtime import SERVEABLE_FLAVOR_MATRIX

    declared_serveable = {
        entry.flavor_id
        for entry in DEFAULT_CAPABILITY_REGISTRY.entries()
        if entry.serveable
    }
    runtime_serveable = {entry.flavor_id for entry in SERVEABLE_FLAVOR_MATRIX}

    assert declared_serveable == runtime_serveable


def test_operation_event_is_deterministic_committed_manifest_view() -> None:
    manifest = {
        "bundle_id": "bundle-123",
        "canonical_uri": "s3://models/bundle-123/",
        "created_at": "2026-08-06T00:00:00+00:00",
        "source_info": {"source_kind": "xgboost_result"},
    }
    first = OperationEvent.bundle_published(
        manifest=manifest,
        manifest_sha256="a" * 64,
        correlation_ids={"run_id": "run-1"},
    )
    second = OperationEvent.bundle_published(
        manifest=manifest,
        manifest_sha256="a" * 64,
        correlation_ids={"run_id": "run-2"},
    )
    other_repository = OperationEvent.bundle_published(
        manifest={**manifest, "canonical_uri": "s3://other/bundle-123/"},
        manifest_sha256="a" * 64,
    )

    assert first.event_id == second.event_id
    assert first.event_id != other_repository.event_id
    assert first.event_kind == "bundle.published"
    assert first.occurred_at == datetime(2026, 8, 6, tzinfo=timezone.utc)
    assert first.source_kind == "xgboost_result"
    assert "storage_profile" not in first.model_dump()


def test_publish_wait_window_covers_the_full_lease_ttl() -> None:
    from tributo.integrations.storage import bundle_repository

    assert (
        bundle_repository._PUBLISH_LEASE_WAIT_SECONDS
        >= bundle_repository._LEASE_TTL_SECONDS
    )


def test_operation_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        OperationEvent(
            event_kind="bundle.published",
            event_id="a" * 64,
            occurred_at=datetime(2026, 8, 6),
            bundle_id="bundle-123",
            canonical_uri="/models/bundle-123",
            manifest_sha256="b" * 64,
        )


def test_local_alias_store_reads_existing_v1_document(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases" / "latest.json"
    alias_path.parent.mkdir()
    alias_path.write_text(
        json.dumps(
            {
                "manifest_uri": str(tmp_path / "bundle-123" / "manifest.json"),
                "manifest_sha256": "c" * 64,
                "bundle_id": "bundle-123",
                "created_at": "2026-08-06T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    ref = LocalBundleAliasStore().resolve(
        str(alias_path), storage_profile=None, max_alias_bytes=1024
    )

    assert ref.bundle_id == "bundle-123"
    assert ref.manifest_sha256 == "c" * 64
    assert ref.canonical_uri == str(tmp_path / "bundle-123")


def test_local_alias_compare_and_set_serializes_concurrent_writers(
    tmp_path: Path,
) -> None:
    store = LocalBundleAliasStore()
    initial = BundleRef(
        canonical_uri=str(tmp_path / "bundle-initial"),
        bundle_id="bundle-initial",
        manifest_sha256="a" * 64,
    )
    created = store.update(
        bundle_uri=str(tmp_path),
        alias_config=AliasConfig(name="latest", policy="newer"),
        bundle_ref=initial,
        manifest_uri=f"{initial.canonical_uri}/manifest.json",
        created_at="2026-08-06T00:00:00+00:00",
        storage_profile=None,
    )
    assert created.status == "updated"

    candidates = (
        BundleRef(
            canonical_uri=str(tmp_path / "bundle-b"),
            bundle_id="bundle-b",
            manifest_sha256="b" * 64,
        ),
        BundleRef(
            canonical_uri=str(tmp_path / "bundle-c"),
            bundle_id="bundle-c",
            manifest_sha256="c" * 64,
        ),
    )

    def update(candidate: BundleRef) -> str:
        result = store.update(
            bundle_uri=str(tmp_path),
            alias_config=AliasConfig(
                name="latest",
                policy="compare_and_swap",
                expected_manifest_sha256=initial.manifest_sha256,
            ),
            bundle_ref=candidate,
            manifest_uri=f"{candidate.canonical_uri}/manifest.json",
            created_at="2026-08-06T00:00:01+00:00",
            storage_profile=None,
        )
        return result.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = list(executor.map(update, candidates))

    assert sorted(statuses) == ["failed", "updated"]


def test_gc_rechecks_manifest_after_acquiring_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError as _ClientError

    from tributo.integrations.storage import gc

    bundle_id = "bundle-" + "a" * 32
    manifest_key = f"models/{bundle_id}/manifest.json"

    class _Exceptions:
        ClientError = _ClientError

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [{"CommonPrefixes": [{"Prefix": f"models/{bundle_id}/"}]}]

    class _Client:
        exceptions = _Exceptions()

        def __init__(self) -> None:
            self.head_calls = 0

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def head_object(self, **kwargs):
            assert kwargs["Key"] == manifest_key
            self.head_calls += 1
            if self.head_calls == 1:
                raise _ClientError(
                    {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
                )
            return {"ContentLength": 1}

    client = _Client()
    deleted: list[str] = []
    released: list[str] = []
    monkeypatch.setattr(gc, "_make_client", lambda *args: client)
    monkeypatch.setattr(gc, "_check_orphan_age", lambda *args: True)
    monkeypatch.setattr(gc, "_acquire_gc_lease", lambda *args: "lease-key")
    monkeypatch.setattr(
        gc, "_delete_prefix_safely", lambda *args: deleted.append(args[-1])
    )
    monkeypatch.setattr(
        gc, "_release_gc_lease", lambda *args: released.append(args[-2])
    )

    result = gc.S3BundleGarbageCollector().collect(
        "s3://bucket/models", orphan_ttl_seconds=0, dry_run=False
    )

    assert result["deleted"] == 0
    assert deleted == []
    assert released == ["lease-key"]


def test_gc_warns_when_scanning_bucket_root(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from tributo.integrations.storage import gc

    class _Paginator:
        def paginate(self, **kwargs):
            assert kwargs == {"Bucket": "bucket", "Prefix": "", "Delimiter": "/"}
            return []

    class _Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

    monkeypatch.setattr(gc, "_make_client", lambda *args: _Client())

    with caplog.at_level("WARNING"):
        result = gc.S3BundleGarbageCollector().collect("s3://bucket")

    assert result["scanned"] == 0
    assert "exact store root used by Publisher" in caplog.text


def test_gc_skips_orphan_when_publish_lease_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError as _ClientError

    from tributo.integrations.storage import gc

    bundle_id = "bundle-" + "f" * 32

    class _Exceptions:
        ClientError = _ClientError

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [{"CommonPrefixes": [{"Prefix": f"models/{bundle_id}/"}]}]

    class _Client:
        exceptions = _Exceptions()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def head_object(self, **kwargs):
            del kwargs
            raise _ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )

    deleted: list[str] = []
    released: list[str] = []
    monkeypatch.setattr(gc, "_make_client", lambda *args: _Client())
    monkeypatch.setattr(gc, "_check_orphan_age", lambda *args: True)
    monkeypatch.setattr(gc, "_acquire_gc_lease", lambda *args: None)
    monkeypatch.setattr(
        gc, "_delete_prefix_safely", lambda *args: deleted.append(args[-1])
    )
    monkeypatch.setattr(
        gc, "_release_gc_lease", lambda *args: released.append(args[-2])
    )

    result = gc.S3BundleGarbageCollector().collect(
        "s3://bucket/models", orphan_ttl_seconds=0, dry_run=False
    )

    assert result["deleted"] == 0
    assert result["errors"] == []
    assert deleted == []
    assert released == []


def test_gc_rechecks_age_after_acquiring_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError as _ClientError

    from tributo.integrations.storage import gc

    bundle_id = "bundle-" + "e" * 32

    class _Exceptions:
        ClientError = _ClientError

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [{"CommonPrefixes": [{"Prefix": f"models/{bundle_id}/"}]}]

    class _Client:
        exceptions = _Exceptions()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def head_object(self, **kwargs):
            del kwargs
            raise _ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )

    age_checks = iter((True, False))
    deleted: list[str] = []
    released: list[str] = []
    monkeypatch.setattr(gc, "_make_client", lambda *args: _Client())
    monkeypatch.setattr(gc, "_check_orphan_age", lambda *args: next(age_checks))
    monkeypatch.setattr(gc, "_acquire_gc_lease", lambda *args: "lease-key")
    monkeypatch.setattr(
        gc, "_delete_prefix_safely", lambda *args: deleted.append(args[-1])
    )
    monkeypatch.setattr(
        gc, "_release_gc_lease", lambda *args: released.append(args[-2])
    )

    result = gc.S3BundleGarbageCollector().collect(
        "s3://bucket/models", orphan_ttl_seconds=60, dry_run=False
    )

    assert result["deleted"] == 0
    assert deleted == []
    assert released == ["lease-key"]


def test_gc_releases_lease_when_delete_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from botocore.exceptions import ClientError as _ClientError

    from tributo.integrations.storage import gc

    bundle_id = "bundle-" + "b" * 32

    class _Exceptions:
        ClientError = _ClientError

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [{"CommonPrefixes": [{"Prefix": f"models/{bundle_id}/"}]}]

    class _Client:
        exceptions = _Exceptions()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def head_object(self, **kwargs):
            del kwargs
            raise _ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )

    released: list[str] = []
    monkeypatch.setattr(gc, "_make_client", lambda *args: _Client())
    monkeypatch.setattr(gc, "_check_orphan_age", lambda *args: True)
    monkeypatch.setattr(gc, "_acquire_gc_lease", lambda *args: "lease-key")
    monkeypatch.setattr(
        gc,
        "_delete_prefix_safely",
        lambda *args: (_ for _ in ()).throw(RuntimeError("delete failed")),
    )
    monkeypatch.setattr(
        gc, "_release_gc_lease", lambda *args: released.append(args[-2])
    )

    result = gc.S3BundleGarbageCollector().collect(
        "s3://bucket/models", orphan_ttl_seconds=0, dry_run=False
    )

    assert result["deleted"] == 0
    assert result["errors"] and "delete failed" in result["errors"][0]
    assert released == ["lease-key"]


def test_gc_reports_partial_delete_response_as_failure() -> None:
    from tributo.integrations.storage import gc

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [
                {
                    "Contents": [
                        {"Key": "models/bundle-a/model.onnx"},
                        {"Key": "models/bundle-a/manifest.tmp"},
                    ]
                }
            ]

    class _Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def delete_objects(self, **kwargs):
            assert len(kwargs["Delete"]["Objects"]) == 2
            return {
                "Deleted": [{"Key": "models/bundle-a/model.onnx"}],
                "Errors": [
                    {
                        "Key": "models/bundle-a/manifest.tmp",
                        "Code": "AccessDenied",
                    }
                ],
            }

    with pytest.raises(RuntimeError, match="AccessDenied"):
        gc._delete_prefix_safely(_Client(), "bucket", "models/bundle-a/")


def test_gc_does_not_reuse_previous_lease_when_next_acquire_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError as _ClientError

    from tributo.integrations.storage import gc

    bundle_ids = ["bundle-" + "c" * 32, "bundle-" + "d" * 32]

    class _Exceptions:
        ClientError = _ClientError

    class _Paginator:
        def paginate(self, **kwargs):
            del kwargs
            return [
                {
                    "CommonPrefixes": [
                        {"Prefix": f"models/{bundle_id}/"} for bundle_id in bundle_ids
                    ]
                }
            ]

    class _Client:
        exceptions = _Exceptions()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return _Paginator()

        def head_object(self, **kwargs):
            del kwargs
            raise _ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )

    acquire_calls = 0

    def acquire(*args):
        nonlocal acquire_calls
        del args
        acquire_calls += 1
        if acquire_calls == 1:
            return "first-lease"
        raise RuntimeError("acquire failed")

    released: list[str] = []
    monkeypatch.setattr(gc, "_make_client", lambda *args: _Client())
    monkeypatch.setattr(gc, "_check_orphan_age", lambda *args: True)
    monkeypatch.setattr(gc, "_acquire_gc_lease", acquire)
    monkeypatch.setattr(gc, "_delete_prefix_safely", lambda *args: None)
    monkeypatch.setattr(
        gc, "_release_gc_lease", lambda *args: released.append(args[-2])
    )

    result = gc.S3BundleGarbageCollector().collect(
        "s3://bucket/models", orphan_ttl_seconds=0, dry_run=False
    )

    assert result["deleted"] == 1
    assert result["errors"] and "acquire failed" in result["errors"][0]
    assert released == ["first-lease"]
