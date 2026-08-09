"""Deterministic support facts projected from the algorithm catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from tributo.training.algorithm_spec import AlgorithmSpec


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
        }


def build_algorithm_support_snapshot(
    specs: Iterable[AlgorithmSpec],
) -> tuple[AlgorithmSupportRecord, ...]:
    """Build a name-sorted snapshot from one atomic Catalog read."""
    return tuple(
        sorted(
            (AlgorithmSupportRecord.from_spec(spec) for spec in specs),
            key=lambda record: record.name,
        )
    )


def snapshot_json_objects(
    snapshot: Iterable[AlgorithmSupportRecord],
) -> list[dict[str, Any]]:
    """Serialize a snapshot produced by ``build_algorithm_support_snapshot``."""
    return [record.to_json_object() for record in snapshot]
