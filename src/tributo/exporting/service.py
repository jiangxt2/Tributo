"""Bundle export service — top-level lifecycle orchestration.

``BundleExportService`` wires together all export components:
ExportSource → StagingArea → Planner → Manager → Publisher → callback.

It is the single entry point for Bundle export. First-party training
lifecycles supply their standard targets when the caller omits them.
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
from tributo.exporting.events import OperationEvent
from tributo.exporting.executor import ExportManager
from tributo.exporting.hooks import BundleArtifactAccessor
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
from tributo.exporting.planner import ExportPlanner, is_implicit_node_id
from tributo.exporting.publisher import Publisher
from tributo.exporting.registries import (
    ExportRegistry,
    ValidatorRegistry,
)
from tributo.util.annotations import PublicAPI

# Cache entry-point plugin classes across instances so every new
# BundleExportService gets the full set, not just the first.
_plugin_cache: dict[str, list[Any]] = {"exports": [], "validators": []}
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
        validator_registry: ValidatorRegistry | None = None,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        operation_store: Any | None = None,
        hook_dispatcher: Any | None = None,
    ) -> None:
        self._exports = export_registry or ExportRegistry()
        self._validators = validator_registry or ValidatorRegistry()
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        self._operation_store = operation_store
        if hook_dispatcher is None:
            from tributo.exporting.dispatch import InlineHookDispatcher

            hook_dispatcher = InlineHookDispatcher(operation_store)
        self._hook_dispatcher = hook_dispatcher
        self._last_operation_event: OperationEvent | None = None

        # Register built-in schema readers.
        from tributo.exporting.manifest import _read_manifest_v1, _read_manifest_v2

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

        if StructureValidator.validator_id not in self._validators.list_all():
            self._validators.register(StructureValidator)

        from tributo.integrations.exporters.prebuilt_onnx import (
            PrebuiltONNXExporter,
        )

        if PrebuiltONNXExporter.exporter_id not in self._exports.list_all():
            self._exports.register(PrebuiltONNXExporter)

        # Load entry-point plugins (cached across instances).
        _load_entry_point_plugins(self._exports, self._validators)

    @property
    def last_operation_event(self) -> OperationEvent | None:
        """Return the event derived by the most recent successful commit.

        This is a last-wins diagnostic property. A service instance must not
        use it to correlate concurrent calls; dispatchers consume the event
        produced inside the current export call.
        """
        return self._last_operation_event

    def export_bundle(
        self,
        source: ExportSource,
        config: BundleOutputConfig,
        *,
        callback: Callable[[PublishedBundle], None] | None = None,
        raise_on_callback_error: bool = False,
        tributo_version: str = "0.0.0",
        attempt_id: str | None = None,
    ) -> BundleResult:
        """Export a bundle from a resolved source.

        Args:
            source: Resolved ``ExportSource`` (from a provider).
            config: Validated ``BundleOutputConfig`` with non-empty targets.
            callback: Optional ``on_bundle_complete`` hook, called after
                publish but before staging cleanup.
            raise_on_callback_error: If ``True``, callback failures raise
                ``PostPublishCallbackError`` (bundle is still published).
            tributo_version: Version string for provenance in the manifest.
            attempt_id: Unique submission attempt identifier; it is recorded
                separately from the stable run and bundle identifiers.

        Returns:
            ``BundleResult`` — the caller's durable reference.

        Raises:
            BundleExportError: If the execution fails (required nodes).
        """
        if config.targets is None:
            raise JobConfigurationError(
                "BundleExportService requires targets (bundle mode)"
            )
        config = self._prepare_explainability_config(config, source)
        self._last_operation_event = None

        # Fail before planning or staging if a requested side effect is
        # unknown, disabled, unloadable, or has invalid options.  An empty
        # binding list performs no hook discovery or optional imports.
        prepared_hooks = self._hook_dispatcher.preflight(config.hooks)

        # Generate stable logical IDs.  A retry gets a fresh attempt_id, but
        # the run/request identity remains the sole input to bundle_id and
        # execution_id so publish is idempotent.
        request_id = config.request_id or config.run_id or uuid.uuid4().hex
        run_id = config.run_id or request_id
        attempt_id = attempt_id or f"attempt-{uuid.uuid4().hex}"
        execution_id = _make_execution_id(run_id)
        bundle_id = _make_bundle_id(run_id)

        planner = ExportPlanner(self._exports, self._validators)
        manager = ExportManager(self._exports, self._validators)
        publisher = Publisher(storage_resolver=self._storage_resolver)

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
                explainability=config.explainability,
            )

            # Phase 5: Compute bundle_digest and record execution.
            from tributo.exporting.manifest import (
                _read_manifest_v1,
                _read_manifest_v2,
                compute_bundle_digest,
            )

            try:
                committed_raw = json.loads(published.manifest_bytes.decode("utf-8"))
                committed_reader = (
                    _read_manifest_v2
                    if committed_raw.get("schema_version") == 2
                    else _read_manifest_v1
                )
                committed_manifest = committed_reader(
                    committed_raw, published.manifest_bytes
                )
                bundle_digest = compute_bundle_digest(
                    artifacts=published.result.artifacts,
                    roles=published.result.roles,
                    exporter_options={
                        nr.node_id: {}
                        for nr in execution.node_results
                        if nr.exporter_id
                    },
                    explainability=getattr(committed_manifest, "explainability", None),
                )
            except Exception as exc:
                logger.error(
                    "Bundle digest computation failed after bundle publish (%s)",
                    type(exc).__name__,
                )
                raise PostPublishCallbackError(
                    "Bundle digest computation failed after bundle publish",
                    bundle_result=published.result,
                ) from exc

            # Write execution record (when OperationStore is available).
            if self._operation_store is not None:
                from tributo.exporting.records import ExecutionRecord

                record = ExecutionRecord(
                    execution_id=execution_id,
                    bundle_id=bundle_id,
                    run_id=run_id,
                    request_id=request_id,
                    attempt_id=attempt_id,
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
                            implicit=is_implicit_node_id(nr.node_id),
                            artifact_ref=nr.artifact_ref,
                            failure=nr.failure,
                            duration_ms=nr.duration_ms,
                        )
                        for nr in execution.node_results
                    ),
                    roles=published.result.roles,
                    tributo_version=tributo_version,
                )
                try:
                    self._operation_store.record_execution(record)
                except Exception as exc:
                    logger.error(
                        "Execution recording failed after bundle publish (%s)",
                        type(exc).__name__,
                    )
                    raise PostPublishCallbackError(
                        "Execution recording failed after bundle publish",
                        bundle_result=published.result,
                    ) from exc

            # Phase 6: Derive the event from the exact manifest bytes that won
            # the repository commit, then run only explicitly configured hooks.
            raw_manifest = published.manifest_bytes
            actual_sha256 = hashlib.sha256(raw_manifest).hexdigest()
            if actual_sha256 != published.result.manifest_sha256:
                raise PostPublishCallbackError(
                    "Cannot derive the publication event because the committed "
                    "manifest digest differs from BundleResult",
                    bundle_result=published.result,
                )
            try:
                manifest_dict = _build_manifest_dict(published)
                event = OperationEvent.bundle_published(
                    manifest=manifest_dict,
                    manifest_sha256=published.result.manifest_sha256,
                    correlation_ids={
                        "run_id": run_id,
                        "request_id": request_id,
                        "execution_id": execution_id,
                    },
                )
                self._last_operation_event = event
                if prepared_hooks:
                    schema_version = manifest_dict.get("schema_version", 1)
                    if not isinstance(schema_version, int):
                        raise ValueError("manifest schema_version must be an integer")
                    manifest = self._manifest_registry.read(
                        schema_version,
                        manifest_dict,
                        raw_manifest,
                    )
                    artifacts = BundleArtifactAccessor(
                        event,
                        storage_profile=config.storage_profile,
                        storage_resolver=self._storage_resolver,
                        manifest_registry=self._manifest_registry,
                        manifest=manifest,
                        manifest_bytes=raw_manifest,
                    )
            except (TypeError, ValueError) as exc:
                raise PostPublishCallbackError(
                    "Cannot dispatch hooks because the committed manifest "
                    "cannot form a valid publication event",
                    bundle_result=published.result,
                ) from exc
            if prepared_hooks:
                published.result = self._hook_dispatcher.dispatch(
                    event=event,
                    bundle_result=published.result,
                    bundle_digest=bundle_digest,
                    prepared_hooks=prepared_hooks,
                    artifacts=artifacts,
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

    @staticmethod
    def _prepare_explainability_config(
        config: BundleOutputConfig, source: ExportSource
    ) -> BundleOutputConfig:
        """Delegate companion selection to the explainability boundary."""
        from tributo.explainability.export import prepare_bundle_output_config

        return prepare_bundle_output_config(config, source)


# ── Plugin loading ─────────────────────────────────────────────────────────────


def _load_entry_point_plugins(
    exports: ExportRegistry,
    validators: ValidatorRegistry,
) -> None:
    """Register first-party components and discover extension plugins."""
    global _plugins_loaded

    from tributo._bootstrap import first_party_export_plugins

    builtin_exporters, builtin_validators = first_party_export_plugins()
    for exporter_cls in builtin_exporters:
        if not exports.contains(exporter_cls.exporter_id):
            exports.register(exporter_cls)
    registered_validators = set(validators.list_all())
    for validator_cls in builtin_validators:
        if validator_cls.validator_id not in registered_validators:
            validators.register(validator_cls)
            registered_validators.add(validator_cls.validator_id)

    if not _plugins_loaded:
        from tributo.plugin import (
            discover_exporter_plugins,
            discover_validator_plugins,
        )

        # Collect discovery failures into registry diagnostics so they are
        # queryable via ``registry.diagnostics()``, not just logged.
        export_diags: list[Any] = []
        validator_diags: list[Any] = []
        _plugin_cache["exports"] = discover_exporter_plugins(diagnostics=export_diags)
        _plugin_cache["validators"] = discover_validator_plugins(
            diagnostics=validator_diags
        )
        _plugin_cache["export_diags"] = export_diags
        _plugin_cache["validator_diags"] = validator_diags
        _plugins_loaded = True

    # Register cached classes + diagnostics into this instance's registries.
    builtin_exporters_by_id = {cls.exporter_id: cls for cls in builtin_exporters}
    for exporter_cls in _plugin_cache["exports"]:
        if builtin_exporters_by_id.get(exporter_cls.exporter_id) is exporter_cls:
            continue
        exports.register(exporter_cls)
    for d in _plugin_cache["export_diags"]:
        exports.record_diagnostic(d)
    builtin_validators_by_id = {cls.validator_id: cls for cls in builtin_validators}
    for validator_cls in _plugin_cache["validators"]:
        if builtin_validators_by_id.get(validator_cls.validator_id) is validator_cls:
            continue
        validators.register(validator_cls)
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


@PublicAPI(stability="beta")
def bundle_id_for_request(request_id: str) -> str:
    """Return the stable bundle identifier for a logical run identifier."""
    if not request_id:
        raise ValueError("request_id must not be empty")
    return _make_bundle_id(request_id)


def _make_execution_id(request_id: str) -> str:
    """Generate a deterministic execution ID derived from the request_id.

    Same request_id (idempotent retry) produces the same execution_id,
    keeping the manifest stable across retries.
    """
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    return f"exec-{digest[:12]}"


def _build_manifest_dict(
    published: PublishedBundle,
) -> dict[str, Any]:
    """Build a JSON-serialisable view of the exact committed manifest.

    The repository returns the bytes that won the commit.  Consuming them here
    avoids a post-commit network read and preserves the original manifest on
    idempotent retries whose freshly assembled advisory fields may differ.

    Args:
        published: Transient publication handle containing committed bytes.
    """
    data: dict[str, Any] = json.loads(published.manifest_bytes.decode("utf-8"))
    return data
