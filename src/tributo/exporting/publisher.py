"""Storage-independent bundle publication facade."""

from __future__ import annotations

import logging
from pathlib import Path

from tributo._common.storage_profiles import StorageProfileResolver
from tributo.explainability.contracts import ExplainabilityConfig
from tributo.exporting.assembler import BundleAssembler
from tributo.exporting.errors import sanitize_error_message
from tributo.exporting.manifest import (
    ManifestSignature,
    ManifestSourceInfo,
)
from tributo.exporting.models import (
    AliasConfig,
    BundleResult,
    ExportExecutionResult,
    FailureInfo,
    PublishedBundle,
)
from tributo.exporting.repository import (
    AliasUpdate,
    BundleRepositoryRouter,
    build_default_repository_router,
)
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class Publisher:
    """Compatibility facade over assembly and repository ports.

    Storage-specific atomicity, idempotency, and alias compare-and-set
    behavior belongs to repository adapters.  This facade keeps the existing
    public method stable while composing those domain contracts.
    """

    def __init__(
        self,
        storage_resolver: StorageProfileResolver | None = None,
        manifest_registry: object | None = None,
        *,
        assembler: BundleAssembler | None = None,
        repository_router: BundleRepositoryRouter | None = None,
    ) -> None:
        # Compatibility-window argument. Manifest parsing moved to
        # BundleReader; keep accepting the old Publisher constructor surface
        # until the documented minor-version window closes.
        del manifest_registry
        self._assembler = assembler or BundleAssembler()
        self._repository_router = repository_router or build_default_repository_router(
            storage_resolver
        )

    def publish(
        self,
        execution: ExportExecutionResult,
        staging_root: Path,
        bundle_uri: str,
        bundle_id: str,
        execution_id: str,
        tributo_version: str,
        source_info: ManifestSourceInfo,
        input_signature: ManifestSignature | None = None,
        output_signature: ManifestSignature | None = None,
        storage_profile: str | None = None,
        alias_config: AliasConfig | None = None,
        roles: dict[str, str] | None = None,
        explainability: ExplainabilityConfig | None = None,
    ) -> PublishedBundle:
        """Assemble, commit, and optionally alias an immutable bundle."""
        staged_bundle = self._assembler.assemble(
            execution=execution,
            staging_root=staging_root,
            bundle_uri=bundle_uri,
            bundle_id=bundle_id,
            execution_id=execution_id,
            tributo_version=tributo_version,
            source_info=source_info,
            input_signature=input_signature,
            output_signature=output_signature,
            roles=roles,
            explainability=explainability,
        )
        repository = self._repository_router.repository_for(bundle_uri)
        commit = repository.commit(
            staged_bundle,
            bundle_uri=bundle_uri,
            storage_profile=storage_profile,
        )

        alias_update = None
        if alias_config is not None:
            try:
                alias_store = self._repository_router.alias_store_for(bundle_uri)
                alias_update = alias_store.update(
                    bundle_uri=bundle_uri,
                    alias_config=alias_config,
                    bundle_ref=commit.bundle_ref,
                    manifest_uri=commit.manifest_uri,
                    created_at=staged_bundle.manifest.created_at.isoformat(),
                    storage_profile=storage_profile,
                )
            except Exception as exc:
                # The immutable manifest is already committed. Alias updates
                # are deliberately non-transactional, so preserve that fact
                # and report the secondary failure through BundleResult.
                logger.error(
                    "Alias update failed after bundle publish (%s)",
                    type(exc).__name__,
                )
                alias_update = AliasUpdate(
                    status="failed",
                    failure=FailureInfo(
                        code=type(exc).__name__,
                        category="publish",
                        message=sanitize_error_message(str(exc))[:4096],
                    ),
                )

        effective_roles = dict(staged_bundle.manifest.roles)
        result = BundleResult(
            bundle_id=bundle_id,
            execution_id=execution_id,
            canonical_uri=commit.bundle_ref.canonical_uri,
            manifest_uri=commit.manifest_uri,
            manifest_sha256=commit.bundle_ref.manifest_sha256,
            status=staged_bundle.manifest.status,
            artifacts=staged_bundle.manifest.artifacts,
            node_results=execution.node_results,
            roles=effective_roles,
            alias_uri=alias_update.alias_uri if alias_update is not None else None,
            alias_status=(
                alias_update.status if alias_update is not None else "not_requested"
            ),
            alias_failure=(alias_update.failure if alias_update is not None else None),
        )
        return PublishedBundle(
            result=result,
            manifest_bytes=commit.manifest_bytes,
            local_bundle_dir=commit.local_bundle_dir,
            local_dir_ephemeral=commit.local_dir_ephemeral,
        )


__all__ = ["Publisher"]
