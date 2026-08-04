"""Bundle export service — top-level lifecycle orchestration.

``BundleExportService`` wires together all export components:
ExportSourceProvider → StagingArea → Planner → Manager → Publisher → callback.

It is the single entry point for bundle-mode export, replacing the
legacy ``BaseTrainer.export_model()`` path when ``output.targets`` is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.exceptions import (
    BundleExportError,
    JobConfigurationError,
    PostPublishCallbackError,
)
from tributo.exporting.executor import ExportManager
from tributo.exporting.manifest import (
    ManifestExecutionNode,
    ManifestSchemaRegistry,
    ManifestSourceInfo,
)
from tributo.exporting.models import (
    BundleOutputConfig,
    BundleResult,
    ExportSource,
    PublishedBundle,
)
from tributo.exporting.planner import ExportPlanner
from tributo.exporting.protocols import ExportSourceProvider
from tributo.exporting.publisher import Publisher
from tributo.exporting.registries import (
    ExportRegistry,
    SourceProviderRegistry,
    ValidatorRegistry,
)
from tributo.util.annotations import PublicAPI

# Cache entry-point plugin classes across instances so every new
# BundleExportService gets the full set, not just the first.
_plugin_cache: dict[str, list[Any]] = {"exports": [], "providers": [], "validators": []}
_plugins_loaded = False

logger = logging.getLogger(__name__)


# ── Staging area context ───────────────────────────────────────────────────────


@contextmanager
def _staging_area() -> Generator[Path, None, None]:
    """Create a temporary staging directory that is cleaned up on exit."""
    staging = Path(tempfile.mkdtemp(prefix="tributo-export-"))
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ── BundleExportService ────────────────────────────────────────────────────────


@PublicAPI(stability="beta")
class BundleExportService:
    """Top-level service for multi-format bundle export.

    Lifecycle::

        service = BundleExportService(...)
        with provider.open_source(result, config) as source:
            with _staging_area() as staging:
                plan = planner.plan(config, source)
                execution = manager.execute(plan, source, staging, exec_id)
                published = publisher.publish(execution, staging, ...)
                callback(published)
                return published.result
    """

    def __init__(
        self,
        export_registry: ExportRegistry | None = None,
        source_provider_registry: SourceProviderRegistry | None = None,
        validator_registry: ValidatorRegistry | None = None,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        operation_store: Any | None = None,
    ) -> None:
        self._exports = export_registry or ExportRegistry()
        self._providers = source_provider_registry or SourceProviderRegistry()
        self._validators = validator_registry or ValidatorRegistry()
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        self._operation_store = operation_store
        self._hooks_runner: Any = None  # Lazily initialized in export_bundle.

        # Register built-in schema readers.
        from tributo.exporting.manifest import _read_manifest_v1

        try:
            self._manifest_registry.register(1, _read_manifest_v1)
        except ValueError:
            pass

        # Register built-in validators.
        from tributo.exporting.validators import StructureValidator

        try:
            self._validators.register(StructureValidator)
        except Exception:
            pass

        # Load entry-point plugins (cached across instances).
        _load_entry_point_plugins(self._exports, self._providers, self._validators)

    def export_bundle(
        self,
        source: ExportSource,
        config: BundleOutputConfig,
        *,
        provider: ExportSourceProvider | None = None,
        callback: Callable[[PublishedBundle], None] | None = None,
        raise_on_callback_error: bool = False,
        tributo_version: str = "0.0.0",
    ) -> BundleResult:
        """Export a bundle from a resolved source.

        Args:
            source: Resolved ``ExportSource`` (from a provider).
            config: Validated ``BundleOutputConfig`` with non-empty targets.
            provider: The ``ExportSourceProvider`` that produced *source*. Used to
                populate ``ManifestSourceInfo``.
            callback: Optional ``on_bundle_complete`` hook, called after
                publish but before staging cleanup.
            raise_on_callback_error: If ``True``, callback failures raise
                ``PostPublishCallbackError`` (bundle is still published).
            tributo_version: Version string for provenance in the manifest.

        Returns:
            ``BundleResult`` — the caller's durable reference.

        Raises:
            BundleExportError: If the execution fails (required nodes).
        """
        if config.targets is None:
            raise JobConfigurationError(
                "BundleExportService requires targets (bundle mode)"
            )

        # Generate stable IDs. bundle_id is derived solely from request_id
        # so retries with the same request_id produce the identical bundle_id
        # and execution_id (per the plan's idempotency design).
        request_id = config.request_id or uuid.uuid4().hex
        execution_id = _make_execution_id(request_id)
        bundle_id = _make_bundle_id(request_id)

        planner = ExportPlanner(self._exports, self._validators)
        manager = ExportManager(self._exports, self._validators)
        publisher = Publisher(
            storage_resolver=self._storage_resolver,
            manifest_registry=self._manifest_registry,
        )

        # Phase 1: Plan.
        plan = planner.plan(config, source)

        with _staging_area() as staging:
            # Phase 2: Execute.
            execution = manager.execute(plan, source, staging, execution_id)

            if execution.status == "failed":
                raise BundleExportError(
                    f"Bundle export failed: {execution.status}",
                    execution_result=execution,
                )

            # Phase 3: Build source info.
            source_info = ManifestSourceInfo(
                source_kind=source.source_kind,
                source_fingerprint=source.source_fingerprint,
                framework=source.metadata.get("framework"),
                framework_version=source.metadata.get("framework_version"),
                # architecture_id belongs to the Model Factory namespace;
                # never fall back to the provider ID (separate namespace).
                architecture_id=source.architecture_id,
                task_type=source.metadata.get("task_type"),
            )

            input_signature = None
            output_signature = None
            if source.checkpoint_contract is not None:
                input_signature, output_signature = (
                    source.checkpoint_contract.to_manifest_signatures()
                )

            # Phase 4: Publish.
            if config.bundle_uri is None:
                raise JobConfigurationError("bundle_uri is required for publishing")
            published = publisher.publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=config.bundle_uri,
                bundle_id=bundle_id,
                execution_id=execution_id,
                tributo_version=tributo_version,
                source_info=source_info,
                input_signature=input_signature,
                output_signature=output_signature,
                storage_profile=config.storage_profile,
                alias_config=config.alias,
                roles=config.roles,
            )

            # Phase 5: Compute bundle_digest and record execution.
            from tributo.exporting.manifest import compute_bundle_digest

            bundle_digest = compute_bundle_digest(
                artifacts=published.result.artifacts,
                roles=published.result.roles,
                exporter_options={
                    nr.node_id: {} for nr in execution.node_results if nr.exporter_id
                },
            )

            # Write execution record (when OperationStore is available).
            if self._operation_store is not None:
                from tributo.exporting.records import ExecutionRecord

                record = ExecutionRecord(
                    execution_id=execution_id,
                    bundle_id=bundle_id,
                    bundle_digest=bundle_digest,
                    status=published.result.status,
                    source_kind=source.source_kind,
                    source_fingerprint=source.source_fingerprint,
                    duration_ms=sum(nr.duration_ms for nr in execution.node_results),
                    nodes=tuple(
                        ManifestExecutionNode(
                            node_id=nr.node_id,
                            target_name=nr.target_name,
                            exporter_id=nr.exporter_id,
                            status=nr.status,
                            required=nr.required,
                            implicit=nr.node_id.startswith("_implicit__"),
                            artifact_ref=nr.artifact_ref,
                            failure=nr.failure,
                            duration_ms=nr.duration_ms,
                        )
                        for nr in execution.node_results
                    ),
                    roles=published.result.roles,
                    tributo_version=tributo_version,
                )
                self._operation_store.record_execution(record)

            # Phase 6: Post-publish hooks.  Hooks are pure functions of the
            # published bundle (canonical_uri + manifest) and always run —
            # each hook decides whether it applies (e.g. MLflow log_artifacts
            # skips when mlflow is not installed).
            from tributo.exporting.hooks import PublicationRunner

            manifest_dict = _build_manifest_dict(
                published.result,
                config.storage_profile,
                self._storage_resolver,
            )

            # Load hooks from entry points and build runner.
            hooks_entries = _discover_hook_plugins()
            if hooks_entries:
                hooks_list: list[tuple[Any, dict[str, Any], bool]] = [
                    (h(), {}, False) for h in hooks_entries
                ]
                self._hooks_runner = PublicationRunner(hooks_list)

            if self._hooks_runner is not None:
                receipts = self._hooks_runner.run(
                    canonical_uri=published.result.canonical_uri,
                    manifest=manifest_dict,
                    manifest_sha256=published.result.manifest_sha256,
                    # Valid only during the staging window — hooks run
                    # before Phase 7 (callback) and staging cleanup.
                    local_bundle_dir=str(published.local_bundle_dir),
                )

                # Record publication attempts (when OperationStore is available).
                if self._operation_store is not None:
                    from tributo.exporting.records import PublicationAttempt

                    for receipt in receipts:
                        attempt = PublicationAttempt(
                            attempt_id=uuid.uuid4().hex,
                            bundle_digest=bundle_digest,
                            hook_id=receipt.hook_id,
                            status=receipt.status,
                            retryable=receipt.retryable,
                            idempotency_key=receipt.idempotency_key,
                            error=receipt.error,
                        )
                        self._operation_store.record_publication_attempt(attempt)

            # Phase 7: Callback (before staging cleanup).
            if callback is not None:
                try:
                    callback(published)
                except Exception as exc:
                    logger.error(
                        "on_bundle_complete callback failed: %s", exc, exc_info=True
                    )
                    if raise_on_callback_error:
                        raise PostPublishCallbackError(
                            f"Callback failed after bundle publish: {exc}",
                            bundle_result=published.result,
                        ) from exc

            return published.result


# ── Plugin loading ─────────────────────────────────────────────────────────────


def _load_entry_point_plugins(
    exports: ExportRegistry,
    providers: SourceProviderRegistry,
    validators: ValidatorRegistry,
) -> None:
    """Discover and register exporter/source-provider/validator plugins."""
    global _plugins_loaded

    if not _plugins_loaded:
        from tributo.plugin import (
            discover_exporter_plugins,
            discover_source_provider_plugins,
            discover_validator_plugins,
        )

        # Collect discovery failures into registry diagnostics so they are
        # queryable via ``registry.diagnostics()``, not just logged.
        export_diags: list[Any] = []
        provider_diags: list[Any] = []
        validator_diags: list[Any] = []
        _plugin_cache["exports"] = discover_exporter_plugins(diagnostics=export_diags)
        _plugin_cache["providers"] = discover_source_provider_plugins(
            diagnostics=provider_diags
        )
        _plugin_cache["validators"] = discover_validator_plugins(
            diagnostics=validator_diags
        )
        _plugin_cache["export_diags"] = export_diags
        _plugin_cache["provider_diags"] = provider_diags
        _plugin_cache["validator_diags"] = validator_diags
        _plugins_loaded = True

    # Register cached classes + diagnostics into this instance's registries.
    for cls in _plugin_cache["exports"]:
        exports.register(cls)
    for d in _plugin_cache["export_diags"]:
        exports.record_diagnostic(d)
    for cls in _plugin_cache["providers"]:
        providers.register(cls)
    for d in _plugin_cache["provider_diags"]:
        providers.record_diagnostic(d)
    for cls in _plugin_cache["validators"]:
        validators.register(cls)
    for d in _plugin_cache["validator_diags"]:
        validators.record_diagnostic(d)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_bundle_id(request_id: str) -> str:
    """Generate a stable, deterministic bundle ID from a request_id.

    Deterministic — identical request_id always produces the identical
    bundle_id, enabling true idempotent retry.
    """
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"bundle-{digest[:32]}"


def _make_execution_id(request_id: str) -> str:
    """Generate a deterministic execution ID derived from the request_id.

    Same request_id (idempotent retry) produces the same execution_id,
    keeping the manifest stable across retries.
    """
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"exec-{digest[:12]}"


def _build_manifest_dict(
    result: BundleResult,
    storage_profile: str | None = None,
    resolver: StorageProfileResolver | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable manifest dict from a BundleResult.

    Works for both local and S3 manifest URIs — reads from the S3 manifest
    when the URI starts with ``s3://``, otherwise reads from the local path.

    Args:
        result: The published bundle result.
        storage_profile: S3 storage profile for the manifest read; the
            profile's endpoint / credentials / path-style addressing are
            used instead of the default boto3 chain (required for
            S3-compatible stores like MinIO).
        resolver: Profile resolver (defaults to ``StorageProfileResolver``).
    """
    manifest_uri = result.manifest_uri
    if manifest_uri.startswith("s3://"):
        from tributo._common.storage import get_boto3_client, parse_s3_url

        profile = (resolver or StorageProfileResolver()).resolve(storage_profile)
        client = get_boto3_client(
            endpoint=profile.endpoint,
            access_key_id=profile.access_key_id,
            secret_access_key=profile.secret_access_key,
            region=profile.region,
            use_ssl=profile.use_ssl,
            path_style=profile.path_style,
            profile_name=profile.profile_name,
        )
        bucket, key = parse_s3_url(manifest_uri)
        resp = client.get_object(Bucket=bucket, Key=key)
        raw: bytes = resp["Body"].read()
        data: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return data
    local_data: dict[str, Any] = json.loads(Path(manifest_uri).read_bytes())
    return local_data


_hook_plugins_cache: list[Any] | None = None


def _discover_hook_plugins() -> list[Any]:
    """Discover post-publish hook plugins (cached)."""
    global _hook_plugins_cache
    if _hook_plugins_cache is None:
        from tributo.plugin import _iter_entry_points

        _hook_plugins_cache = []
        for ep in _iter_entry_points("tributo.hooks"):
            try:
                cls = ep.load()
                if hasattr(cls, "hook_id"):
                    _hook_plugins_cache.append(cls)
            except Exception:
                logger.debug("Failed to load hook plugin %r", ep.name, exc_info=True)
    return _hook_plugins_cache
