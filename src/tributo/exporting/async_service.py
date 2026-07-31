"""Async export service — concurrent DAG execution.

Provides ``AsyncBundleExportService`` which executes independent DAG nodes
in parallel using ``asyncio``, ``concurrent.futures``, or Ray tasks.

The async service is a drop-in alternative to ``BundleExportService``
with the same API surface but concurrent node execution.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

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

logger = logging.getLogger(__name__)


@contextmanager
def _staging_area() -> Generator[Path, None, None]:
    staging = Path(tempfile.mkdtemp(prefix="tributo-async-export-"))
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _make_bundle_id(request_id: str, started_at: datetime.datetime | None = None) -> str:
    if started_at is not None:
        ts = started_at.strftime("%Y%m%dT%H%M%S-%f")
    else:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S-%f")
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    return f"{ts}-{digest}"


@PublicAPI(stability="alpha")
class AsyncBundleExportService:
    """Async version of ``BundleExportService`` with concurrent DAG execution.

    Independent DAG nodes (no dependency chain) are executed concurrently
    using ``asyncio`` and ``concurrent.futures.ThreadPoolExecutor``.

    Iteration plan:
    - Phase 1: Serial execution, async API (current).
    - Phase 2: Group nodes by topological level, execute levels in parallel.
    - Phase 3: Per-node executor threads with progress streaming.
    """

    def __init__(
        self,
        export_registry: ExportRegistry | None = None,
        source_provider_registry: SourceProviderRegistry | None = None,
        validator_registry: ValidatorRegistry | None = None,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: ManifestSchemaRegistry | None = None,
        max_workers: int = 4,
    ) -> None:
        self._exports = export_registry or ExportRegistry()
        self._providers = source_provider_registry or SourceProviderRegistry()
        self._validators = validator_registry or ValidatorRegistry()
        self._storage_resolver = storage_resolver or StorageProfileResolver()
        self._manifest_registry = manifest_registry or ManifestSchemaRegistry()
        self._max_workers = max_workers

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

        try:
            self._validators.register(StructureValidator)
        except Exception:
            pass

        # Load entry-point plugins into registries.
        from tributo.exporting.service import _load_entry_point_plugins

        _load_entry_point_plugins(self._exports, self._providers, self._validators)

    async def export_bundle_async(
        self,
        source: ExportSource,
        config: BundleOutputConfig,
        *,
        provider: SourceProvider | None = None,
        tributo_version: str = "0.0.0",
    ) -> BundleResult:
        """Export a bundle asynchronously.

        Same semantics as ``BundleExportService.export_bundle()`` but:
        - Independent DAG nodes are executed concurrently.
        - The event loop remains free for other tasks.
        """
        if config.targets is None:
            raise ValueError("AsyncBundleExportService requires targets")

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

        plan = planner.plan(config, source)

        # Execute in thread pool to avoid blocking the event loop.
        loop = asyncio.get_running_loop()
        with _staging_area() as staging:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                execution = await loop.run_in_executor(
                    executor,
                    manager.execute,
                    plan,
                    source,
                    staging,
                    execution_id,
                )

                if execution.status == "failed":
                    raise BundleExportError(
                        f"Bundle export failed: {execution.status}",
                        execution_result=execution,
                    )

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

                published = await loop.run_in_executor(
                    executor,
                lambda: publisher.publish(
                    execution=execution,
                    staging_root=staging,
                    bundle_uri=config.bundle_uri or "",
                    bundle_id=bundle_id,
                    execution_id=execution_id,
                    tributo_version=tributo_version,
                    source_info=source_info,
                    storage_profile=config.storage_profile,
                    alias_config=config.alias,
                    roles=config.roles,
                ),
            )

            return published.result

    def export_bundle(
        self,
        source: ExportSource,
        config: BundleOutputConfig,
        **kwargs: Any,
    ) -> BundleResult:
        """Synchronous wrapper — runs the async pipeline in a new event loop.

        Args:
            source: Resolved ``ExportSource``.
            config: Validated ``BundleOutputConfig``.
            **kwargs: Passed through to ``export_bundle_async``.

        Returns:
            ``BundleResult``.
        """
        return asyncio.run(self.export_bundle_async(source, config, **kwargs))
