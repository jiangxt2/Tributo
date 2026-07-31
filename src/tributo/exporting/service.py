"""Bundle export service — top-level lifecycle orchestration.

``BundleExportService`` wires together all export components:
SourceProvider → StagingArea → Planner → Manager → Publisher → callback.

It is the single entry point for bundle-mode export, replacing the
legacy ``BaseTrainer.export_model()`` path when ``output.targets`` is set.
"""

from __future__ import annotations

import datetime
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
from tributo.exceptions import BundleExportError, PostPublishCallbackError
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
from tributo.exporting.protocols import SourceProvider
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
        repository: Any | None = None,
        operation_store: Any | None = None,
    ) -> None:
        self._exports = export_registry or ExportRegistry()
        self._providers = source_provider_registry or SourceProviderRegistry()
        self._validators = validator_registry or ValidatorRegistry()
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        self._repository = repository
        self._operation_store = operation_store

        # Register built-in schema readers.
        from tributo.exporting.manifest import (
            _read_manifest_v1,
            _read_manifest_v2,
        )

        try:
            self._manifest_registry.register(1, _read_manifest_v1)
        except ValueError:
            pass
        try:
            self._manifest_registry.register(2, _read_manifest_v2)
        except ValueError:
            pass

        # Register built-in validators.
        from tributo.exporting.validators import StructureValidator

        try:
            self._validators.register(StructureValidator)
        except Exception:
            pass

        # Load entry-point plugins (cached across instances).
        _load_entry_point_plugins(
            self._exports, self._providers, self._validators
        )

    def export_bundle(
        self,
        source: ExportSource,
        config: BundleOutputConfig,
        *,
        provider: SourceProvider | None = None,
        callback: Callable[[PublishedBundle], None] | None = None,
        raise_on_callback_error: bool = False,
        tributo_version: str = "0.0.0",
        repository: Any | None = None,
    ) -> BundleResult:
        """Export a bundle from a resolved source.

        Args:
            source: Resolved ``ExportSource`` (from a provider).
            config: Validated ``BundleOutputConfig`` with non-empty targets.
            provider: The ``SourceProvider`` that produced *source*. Used to
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
            raise ValueError("BundleExportService requires targets (bundle mode)")

        # Generate stable IDs. started_at is captured once and used for
        # the bundle_id so that retries with the same request_id produce
        # the same bundle_id (per the plan's idempotency design).
        started_at = datetime.datetime.now(datetime.timezone.utc)
        request_id = config.request_id or uuid.uuid4().hex
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        bundle_id = _make_bundle_id(request_id, started_at)

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
                architecture_id=(
                    source.architecture_id
                    or (provider.provider_id if provider else None)
                ),
                task_type=source.metadata.get("task_type"),
            )

            # Phase 4: Publish.
            published = publisher.publish(
                execution=execution,
                staging_root=staging,
                bundle_uri=config.bundle_uri,  # type: ignore[arg-type]
                bundle_id=bundle_id,
                execution_id=execution_id,
                tributo_version=tributo_version,
                source_info=source_info,
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
                    nr.node_id: {} for nr in execution.node_results
                    if nr.exporter_id
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
                    duration_ms=sum(
                        nr.duration_ms for nr in execution.node_results
                    ),
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

            # Phase 6: Post-publish hooks.
            if self._repository is not None and hasattr(self, "_hooks_runner"):
                manifest_dict = json.loads(
                    Path(published.result.manifest_uri).read_bytes()
                )
                self._hooks_runner.run(
                    canonical_uri=published.result.canonical_uri,
                    manifest=manifest_dict,
                    manifest_sha256=published.result.manifest_sha256,
                )

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

        _plugin_cache["exports"] = discover_exporter_plugins()
        _plugin_cache["providers"] = discover_source_provider_plugins()
        _plugin_cache["validators"] = discover_validator_plugins()
        _plugins_loaded = True

    # Register cached classes into this instance's registries.
    for cls in _plugin_cache["exports"]:
        exports.register(cls)
    for cls in _plugin_cache["providers"]:
        providers.register(cls)
    for cls in _plugin_cache["validators"]:
        validators.register(cls)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_bundle_id(
    request_id: str, started_at: datetime.datetime | None = None
) -> str:
    """Generate a stable, deterministic bundle ID from a request_id.

    When *started_at* is provided, it includes sub-second resolution
    (microseconds) so that retries within the same second still produce
    the identical bundle ID.  When ``None``, uses the current time.
    """
    if started_at is not None:
        ts = started_at.strftime("%Y%m%dT%H%M%S-%f")
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S-%f"
        )
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    return f"{ts}-{digest}"
