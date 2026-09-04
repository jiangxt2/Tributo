"""Versioned PyTorch Recipe and Ray Torch Adapter contracts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from tributo._common.immutable import deep_freeze
from tributo.algorithms.api.torch_runtime import (
    TorchCheckpointRef,
    TorchCompositeLossContribution,
    TorchLossContribution,
    TorchMetricContribution,
    TorchStageRunIdentity,
    TorchStepLoss,
)
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchRuntimeContext:
    """Invocation-level context supplied by the Core Torch Runtime."""

    algorithm_config: Mapping[str, Any]
    implementation_id: str
    world_size: int
    policy_digest: str
    execution_plan_digest: str
    run_identity: TorchStageRunIdentity | None = None

    input_bindings: Mapping[str, object] = field(default_factory=dict)
    output_config: Mapping[str, object] = field(default_factory=dict)
    input_binding_digest: str | None = None
    state_layout: str = "replicated"
    adapter_identity: str | None = None
    resume_supported: bool = False
    torch_runtime_api_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm_config, Mapping):
            raise ValueError("TorchRuntimeContext algorithm_config must be a mapping")
        if self.torch_runtime_api_version != 1:
            raise ValueError("TorchRuntimeContext torch_runtime_api_version must be 1")
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError("TorchRuntimeContext implementation_id is required")
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 1
        ):
            raise ValueError("TorchRuntimeContext world_size must be positive")
        if not isinstance(self.policy_digest, str) or len(self.policy_digest) != 64:
            raise ValueError("TorchRuntimeContext policy_digest is required")
        if (
            not isinstance(self.execution_plan_digest, str)
            or len(self.execution_plan_digest) != 64
        ):
            raise ValueError("TorchRuntimeContext execution_plan_digest is required")
        if self.run_identity is not None and not isinstance(
            self.run_identity, TorchStageRunIdentity
        ):
            raise ValueError("TorchRuntimeContext run_identity is invalid")
        if self.input_binding_digest is not None and (
            len(self.input_binding_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.input_binding_digest)
        ):
            raise ValueError("TorchRuntimeContext input_binding_digest is invalid")
        if self.state_layout not in {"replicated", "component", "sharded"}:
            raise ValueError("TorchRuntimeContext state_layout is invalid")
        if not isinstance(self.resume_supported, bool):
            raise ValueError("TorchRuntimeContext resume_supported must be boolean")
        if self.resume_supported:
            raise ValueError("Torch Runtime API v1 does not support cross-Run recovery")
        object.__setattr__(self, "algorithm_config", deep_freeze(self.algorithm_config))
        object.__setattr__(self, "input_bindings", deep_freeze(self.input_bindings))
        object.__setattr__(self, "output_config", deep_freeze(self.output_config))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "algorithm_config": dict(self.algorithm_config),
            "implementation_id": self.implementation_id,
            "world_size": self.world_size,
            "policy_digest": self.policy_digest,
            "execution_plan_digest": self.execution_plan_digest,
            "run_identity": self.run_identity.to_dict() if self.run_identity else None,
            "input_bindings": dict(self.input_bindings),
            "output_config": dict(self.output_config),
            "input_binding_digest": self.input_binding_digest,
            "state_layout": self.state_layout,
            "adapter_identity": self.adapter_identity,
            "resume_supported": self.resume_supported,
            "torch_runtime_api_version": self.torch_runtime_api_version,
        }
        return payload


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchStageContext:
    """Stage-aware context derived exclusively from ``TorchPolicy``."""

    runtime: TorchRuntimeContext
    stage_id: str
    stage_index: int
    is_final: bool
    input_roles: tuple[str, ...]
    predecessor_stage_id: str | None = None
    predecessor_checkpoint_descriptor: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, TorchRuntimeContext):
            raise ValueError("TorchStageContext runtime is required")
        if not isinstance(self.stage_id, str) or not self.stage_id:
            raise ValueError("TorchStageContext stage_id is required")
        if (
            not isinstance(self.stage_index, int)
            or isinstance(self.stage_index, bool)
            or self.stage_index < 0
        ):
            raise ValueError("TorchStageContext stage_index must be non-negative")
        if not isinstance(self.is_final, bool):
            raise ValueError("TorchStageContext is_final must be boolean")
        object.__setattr__(self, "input_roles", tuple(self.input_roles))
        if self.predecessor_checkpoint_descriptor is not None:
            if any(
                key in {"locator", "checkpoint_locator", "path", "credential"}
                for key in self.predecessor_checkpoint_descriptor
            ):
                raise ValueError(
                    "TorchStageContext cannot expose checkpoint locator or credentials"
                )
            object.__setattr__(
                self,
                "predecessor_checkpoint_descriptor",
                deep_freeze(self.predecessor_checkpoint_descriptor),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": self.runtime.to_dict(),
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "is_final": self.is_final,
            "input_roles": list(self.input_roles),
            "predecessor_stage_id": self.predecessor_stage_id,
            "predecessor_checkpoint_descriptor": dict(
                self.predecessor_checkpoint_descriptor
            )
            if self.predecessor_checkpoint_descriptor
            else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TorchStageContext":
        runtime_value = value.get("runtime")
        if not isinstance(runtime_value, Mapping):
            raise ValueError("TorchStageContext runtime payload is invalid")
        identity_value = runtime_value.get("run_identity")
        resume_supported = runtime_value.get("resume_supported", False)
        runtime = TorchRuntimeContext(
            algorithm_config=runtime_value.get("algorithm_config", {}),
            implementation_id=runtime_value["implementation_id"],
            world_size=runtime_value["world_size"],
            policy_digest=runtime_value["policy_digest"],
            execution_plan_digest=runtime_value["execution_plan_digest"],
            run_identity=(
                TorchStageRunIdentity.from_dict(identity_value)
                if isinstance(identity_value, Mapping)
                else None
            ),
            input_bindings=runtime_value.get("input_bindings", {}),
            output_config=runtime_value.get("output_config", {}),
            input_binding_digest=runtime_value.get("input_binding_digest"),
            state_layout=runtime_value.get("state_layout", "replicated"),
            adapter_identity=runtime_value.get("adapter_identity"),
            resume_supported=resume_supported,
            torch_runtime_api_version=runtime_value.get("torch_runtime_api_version", 1),
        )
        return cls(
            runtime=runtime,
            stage_id=cast(str, value["stage_id"]),
            stage_index=cast(int, value["stage_index"]),
            is_final=cast(bool, value["is_final"]),
            input_roles=tuple(cast(tuple[str, ...], value.get("input_roles", ()))),
            predecessor_stage_id=cast(str | None, value.get("predecessor_stage_id")),
            predecessor_checkpoint_descriptor=cast(
                Mapping[str, Any] | None,
                value.get("predecessor_checkpoint_descriptor"),
            ),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchCheckpointContext:
    """Control metadata passed beside an actual Ray Train result/checkpoint."""

    stage: TorchStageContext
    run_id: str
    invocation_id: str
    checkpoint_owner: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TorchStageContext):
            raise ValueError("TorchCheckpointContext stage is required")
        for name in ("run_id", "invocation_id", "checkpoint_owner"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"TorchCheckpointContext {name} is required")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchWorkerCheckpointContext:
    """Worker-local, non-serializable view of a selected checkpoint."""

    stage: TorchStageContext
    source: str
    checkpoint: TorchCheckpointRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TorchStageContext):
            raise ValueError("TorchWorkerCheckpointContext stage is required")
        if self.source not in {
            "none",
            "stage_dependency",
        }:
            raise ValueError("TorchWorkerCheckpointContext source is invalid")
        if self.source == "none" and self.checkpoint is not None:
            raise ValueError("empty Torch checkpoint source cannot carry a checkpoint")
        if self.checkpoint is not None and not isinstance(
            self.checkpoint, TorchCheckpointRef
        ):
            raise ValueError("TorchWorkerCheckpointContext checkpoint is invalid")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchBatch:
    """Typed algorithm batch preserving named inputs and exact row coverage."""

    positional: tuple[object, ...] = ()
    keyword: Mapping[str, object] = field(default_factory=dict)
    targets: object | None = None
    weights: object | None = None
    local_rows: int = 0
    coverage_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.local_rows < 0 or isinstance(self.local_rows, bool):
            raise ValueError("TorchBatch local_rows must be non-negative")
        for name, count in self.coverage_counts.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError(
                    "TorchBatch coverage counts must be non-negative integers"
                )
        object.__setattr__(self, "keyword", MappingProxyType(dict(self.keyword)))
        object.__setattr__(
            self,
            "coverage_counts",
            MappingProxyType(dict(self.coverage_counts)),
        )


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchStepContext:
    """Per-step context with no access to Core lifecycle internals."""

    stage: TorchStageContext
    window_index: int
    micro_batch_index: int


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchStepResult:
    """Forward outputs and an explicit ordinary/composite loss contribution."""

    outputs: Mapping[str, object]
    loss: TorchStepLoss
    coverage_counts: Mapping[str, int] = field(default_factory=dict)
    metrics: Mapping[str, TorchMetricContribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.loss, (TorchLossContribution, TorchCompositeLossContribution)
        ):
            raise ValueError("TorchStepResult requires a typed TorchStepLoss")
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        for name, count in self.coverage_counts.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError(
                    "TorchStepResult coverage counts must be non-negative integers"
                )
        if any(
            not isinstance(value, TorchMetricContribution)
            for value in self.metrics.values()
        ):
            raise ValueError(
                "TorchStepResult metrics must be TorchMetricContribution values"
            )
        object.__setattr__(
            self, "coverage_counts", MappingProxyType(dict(self.coverage_counts))
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchModuleSet:
    """Named worker modules produced by a :class:`TorchRecipe`."""

    modules: Mapping[str, object]

    def __post_init__(self) -> None:
        values = dict(self.modules)
        if "model" not in values or "loss" not in values:
            raise ValueError("TorchModuleSet requires model and loss modules")
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError("TorchModuleSet module names must be non-empty")
        object.__setattr__(self, "modules", MappingProxyType(values))

    def __getitem__(self, name: str) -> object:
        return self.modules[name]

    def get(self, name: str, default: object | None = None) -> object | None:
        return self.modules.get(name, default)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchArtifactPlan:
    """Typed export declaration consumed by Core BundleExportService."""

    source_kind: str
    input_signature: tuple[Mapping[str, object], ...]
    output_signature: tuple[Mapping[str, object], ...]
    targets: tuple[Mapping[str, object], ...]
    roles: Mapping[str, str]
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise ValueError("TorchArtifactPlan source_kind is required")
        for name, values in (
            ("input_signature", self.input_signature),
            ("output_signature", self.output_signature),
            ("targets", self.targets),
        ):
            if any(not isinstance(value, Mapping) for value in values):
                raise ValueError(f"TorchArtifactPlan {name} entries must be mappings")
            normalized = tuple(dict(value) for value in values)
            object.__setattr__(self, name, normalized)
        target_names: list[str] = []
        for target in self.targets:
            target_name = target.get("name")
            if not isinstance(target_name, str) or not target_name:
                raise ValueError("TorchArtifactPlan targets require named targets")
            target_names.append(target_name)
        if len(set(target_names)) != len(target_names):
            raise ValueError("TorchArtifactPlan target names must be unique")
        for signature_name in ("input_signature", "output_signature"):
            for field_value in getattr(self, signature_name):
                if (
                    not isinstance(field_value.get("name"), str)
                    or not field_value["name"]
                    or not isinstance(field_value.get("dtype"), str)
                    or not field_value["dtype"]
                    or not isinstance(field_value.get("shape", ()), (list, tuple))
                ):
                    raise ValueError(f"TorchArtifactPlan {signature_name} is malformed")
        if not self.output_signature:
            raise ValueError("TorchArtifactPlan requires an output signature")
        if not isinstance(self.required, bool):
            raise ValueError("TorchArtifactPlan required must be boolean")
        roles = dict(self.roles)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(target, str)
            or not target
            for name, target in roles.items()
        ):
            raise ValueError("TorchArtifactPlan roles must map names to targets")
        if any(target not in target_names for target in roles.values()):
            raise ValueError("TorchArtifactPlan roles must reference declared targets")
        object.__setattr__(self, "roles", MappingProxyType(roles))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "input_signature": [dict(value) for value in self.input_signature],
            "output_signature": [dict(value) for value in self.output_signature],
            "targets": [dict(value) for value in self.targets],
            "roles": dict(self.roles),
            "required": self.required,
        }


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchBuildContext:
    """Model-construction context for a Core-managed Recipe."""

    runtime: TorchRuntimeContext
    stage: TorchStageContext


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchBatchContext:
    """Batch adaptation context with named input role information."""

    stage: TorchStageContext
    input_roles: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    label_name: str | None = None
    weight_name: str | None = None

    def __post_init__(self) -> None:
        roles = tuple(self.input_roles or self.stage.input_roles)
        if not roles or any(not isinstance(role, str) or not role for role in roles):
            raise ValueError("TorchBatchContext input_roles must be non-empty")
        if len(set(roles)) != len(roles):
            raise ValueError("TorchBatchContext input_roles must be unique")
        object.__setattr__(self, "input_roles", roles)


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchOptimizationPlan:
    """Optimizer and accumulation controls selected by one Recipe."""

    optimizer: object
    scheduler: object | None = None
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.gradient_accumulation_steps, int)
            or isinstance(self.gradient_accumulation_steps, bool)
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_gradient_norm is not None and (
            not isinstance(self.max_gradient_norm, (int, float))
            or isinstance(self.max_gradient_norm, bool)
            or not math.isfinite(float(self.max_gradient_norm))
            or float(self.max_gradient_norm) <= 0
        ):
            raise ValueError("max_gradient_norm must be finite and positive")


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchMetricPlan:
    """Metric names and reducer identities for one Torch Policy."""

    reducers: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized = dict(self.reducers)
        if any(not isinstance(k, str) or not k for k in normalized):
            raise ValueError("TorchMetricPlan reducer names must be non-empty")
        object.__setattr__(self, "reducers", MappingProxyType(normalized))


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class TorchArtifactContext:
    """Context used to construct a typed artifact plan."""

    stage: TorchStageContext
    checkpoint: TorchCheckpointRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, TorchStageContext):
            raise ValueError("TorchArtifactContext stage is required")


@PublicAPI(stability="alpha")
class TorchRecipe(ABC):
    """Core-owned PyTorch training hooks with no Runtime lifecycle control."""

    api_version = 1

    @abstractmethod
    def build_modules(self, context: TorchBuildContext) -> TorchModuleSet: ...

    @abstractmethod
    def adapt_batch(self, batch: object, context: TorchBatchContext) -> TorchBatch: ...

    @abstractmethod
    def training_step(
        self,
        modules: TorchModuleSet,
        batch: TorchBatch,
        context: TorchStepContext,
    ) -> TorchStepResult: ...

    @abstractmethod
    def validation_step(
        self,
        modules: TorchModuleSet,
        batch: TorchBatch,
        context: TorchStepContext,
    ) -> TorchStepResult: ...

    @abstractmethod
    def configure_optimizers(
        self,
        modules: TorchModuleSet,
        context: TorchBuildContext,
    ) -> TorchOptimizationPlan: ...

    @abstractmethod
    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan: ...

    @abstractmethod
    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan: ...


@PublicAPI(stability="alpha")
class RayTorchAdapter(ABC):
    """Framework-native Torch hooks driven by the Core Ray Train Runtime."""

    api_version = 1

    @abstractmethod
    def validate_environment(self, context: TorchRuntimeContext) -> None: ...

    @abstractmethod
    def bind_datasets(
        self,
        datasets: Mapping[str, object],
        context: TorchStageContext,
    ) -> Mapping[str, object]: ...

    @abstractmethod
    def worker_config(self, context: TorchStageContext) -> Mapping[str, object]: ...

    @abstractmethod
    def train_loop_per_worker(
        self,
        worker_config: Mapping[str, object],
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None: ...

    @abstractmethod
    def checkpoint_source(
        self,
        result: object,
        context: TorchCheckpointContext,
    ) -> object: ...

    @abstractmethod
    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan: ...

    @abstractmethod
    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan: ...

    @abstractmethod
    def open_export_source(
        self,
        checkpoint_ref: TorchCheckpointRef,
        artifact_context: TorchArtifactContext,
    ) -> Any: ...


__all__ = [
    "RayTorchAdapter",
    "TorchArtifactContext",
    "TorchArtifactPlan",
    "TorchBatch",
    "TorchBatchContext",
    "TorchBuildContext",
    "TorchCheckpointContext",
    "TorchMetricPlan",
    "TorchModuleSet",
    "TorchOptimizationPlan",
    "TorchRecipe",
    "TorchRuntimeContext",
    "TorchStageContext",
    "TorchStepContext",
    "TorchStepResult",
    "TorchWorkerCheckpointContext",
]
