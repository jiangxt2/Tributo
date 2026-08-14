"""Deterministic support facts projected from the algorithm catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from tributo.training.algorithm_spec import AlgorithmSpec
    from tributo.training.catalog import AlgorithmCatalogRecord


@dataclass(frozen=True)
class AlgorithmSupportRecord:
    """Serializable support facts for one registered algorithm."""

    name: str
    problem_types: tuple[str, ...]
    data_modality: tuple[str, ...]
    tags: tuple[str, ...]
    execution_kind: str
    supported_tasks: tuple[str, ...]
    capabilities: tuple[str, ...]
    data_loading: str
    gpu_required: bool
    status: str
    extras_group: str | None
    implementation_ids: tuple[str, ...] = ()
    runtime_topologies: tuple[str, ...] = ()
    distribution_strategies: tuple[str, ...] = ()
    execution_profiles: tuple[str, ...] = ()
    input_views: tuple[str, ...] = ()
    stability: str = "beta"
    limitations: tuple[str, ...] = ()
    available: bool = True
    compatibility_only: bool = False
    tested: bool = False
    supported: bool = False
    validated_execution_profiles: tuple[str, ...] = ()
    native_migration_complete: bool = False

    @classmethod
    def from_spec(cls, spec: AlgorithmSpec) -> AlgorithmSupportRecord:
        """Project one immutable algorithm specification into support facts."""
        return cls(
            name=spec.name,
            problem_types=tuple(item.value for item in spec.problem_types),
            data_modality=tuple(spec.data_modality),
            tags=tuple(spec.tags),
            execution_kind=spec.execution_kind.value,
            supported_tasks=tuple(spec.supported_tasks),
            capabilities=tuple(item.value for item in spec.capabilities),
            data_loading=spec.data_loading.value,
            gpu_required=spec.resource_hints.gpu_required,
            status=spec.status.value,
            extras_group=spec.extras_group,
        )

    @classmethod
    def from_catalog_record(
        cls,
        record: AlgorithmCatalogRecord,
    ) -> AlgorithmSupportRecord:
        """Project executable and compatibility state without loading code."""
        spec = record.spec
        base = (
            cls.from_spec(spec)
            if spec is not None
            else cls(
                name=record.name,
                problem_types=(),
                data_modality=(),
                tags=(),
                execution_kind="unknown",
                supported_tasks=(),
                capabilities=(),
                data_loading="unknown",
                gpu_required=False,
                status="unknown",
                extras_group=None,
            )
        )
        return cls(
            name=base.name,
            problem_types=base.problem_types,
            data_modality=base.data_modality,
            tags=base.tags,
            execution_kind=base.execution_kind,
            supported_tasks=base.supported_tasks,
            capabilities=base.capabilities,
            data_loading=base.data_loading,
            gpu_required=base.gpu_required,
            status=base.status,
            extras_group=base.extras_group,
            implementation_ids=record.implementation_ids,
            runtime_topologies=record.runtime_topologies,
            distribution_strategies=record.distribution_strategies,
            execution_profiles=record.execution_profiles,
            input_views=record.input_views,
            stability=record.stability,
            limitations=record.limitations,
            available=record.available,
            compatibility_only=record.compatibility_only,
            tested=record.tested,
            supported=record.supported,
            validated_execution_profiles=record.validated_execution_profiles,
            native_migration_complete=record.native_migration_complete,
        )

    def to_json_object(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""
        return {
            "name": self.name,
            "problem_types": list(self.problem_types),
            "data_modality": list(self.data_modality),
            "tags": list(self.tags),
            "execution_kind": self.execution_kind,
            "supported_tasks": list(self.supported_tasks),
            "capabilities": list(self.capabilities),
            "data_loading": self.data_loading,
            "gpu_required": self.gpu_required,
            "status": self.status,
            "extras_group": self.extras_group,
            "implementation_ids": list(self.implementation_ids),
            "runtime_topologies": list(self.runtime_topologies),
            "distribution_strategies": list(self.distribution_strategies),
            "execution_profiles": list(self.execution_profiles),
            "input_views": list(self.input_views),
            "stability": self.stability,
            "limitations": list(self.limitations),
            "available": self.available,
            "compatibility_only": self.compatibility_only,
            "tested": self.tested,
            "supported": self.supported,
            "validated_execution_profiles": list(self.validated_execution_profiles),
            "native_migration_complete": self.native_migration_complete,
        }


def build_algorithm_support_snapshot(
    specs: Iterable[AlgorithmSpec | AlgorithmCatalogRecord],
) -> tuple[AlgorithmSupportRecord, ...]:
    """Build a name-sorted snapshot from one atomic Catalog read."""
    from tributo.training.catalog import AlgorithmCatalogRecord

    return tuple(
        sorted(
            (
                AlgorithmSupportRecord.from_catalog_record(item)
                if isinstance(item, AlgorithmCatalogRecord)
                else AlgorithmSupportRecord.from_spec(item)
                for item in specs
            ),
            key=lambda record: record.name,
        )
    )


def snapshot_json_objects(
    snapshot: Iterable[AlgorithmSupportRecord],
) -> list[dict[str, Any]]:
    """Serialize a snapshot produced by ``build_algorithm_support_snapshot``."""
    return [record.to_json_object() for record in snapshot]
