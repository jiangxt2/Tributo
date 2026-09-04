"""Source Provider for final-stage Core Torch checkpoints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Generator

from pydantic import BaseModel, ConfigDict, Field

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    QualifiedReference,
    TorchCheckpointDescriptor,
    TorchCheckpointRef,
)
from tributo.algorithms.core.worker import _load_reference, _validate_module_digest
from tributo.algorithms.spi import (
    RayTorchAdapter,
    TorchArtifactContext,
    TorchArtifactPlan,
    TorchRecipe,
    TorchRuntimeContext,
)
from tributo.exporting.models import (
    CheckpointField,
    ExportCheckpointV1,
    ExportSource,
)
from tributo.training.checkpoint import checkpoint_directory
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class TorchSourceOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    implementation_ref: str = Field(min_length=1)
    implementation_code_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_id: str = Field(min_length=1)
    loop_owner: str = Field(default="core_recipe", pattern=r"^(core_recipe|adapter)$")
    algorithm_config: dict[str, Any] = Field(default_factory=dict)
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    output_config: dict[str, Any] = Field(default_factory=dict)
    policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_binding_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage_input_roles: tuple[str, ...] = ("train",)
    stage_index: int = Field(default=0, ge=0)


def _source_context_config(options: TorchSourceOptions) -> dict[str, Any]:
    """Keep Core Ray/output control fields out of implementation contexts."""
    return {
        str(key): value
        for key, value in options.algorithm_config.items()
        if str(key) not in {"ray", "output"}
    }


def _checkpoint_contract_from_artifact_plan(
    artifact_payload: Mapping[str, Any],
    descriptor: TorchCheckpointDescriptor,
    options: TorchSourceOptions,
    *,
    preprocessing: Mapping[str, Any] | None = None,
) -> ExportCheckpointV1:
    """Translate the typed Torch artifact declaration to Bundle metadata."""
    import torch

    try:
        input_schema = tuple(
            CheckpointField.model_validate(dict(field))
            for field in artifact_payload.get("input_signature", ())
        )
        output_schema = tuple(
            CheckpointField.model_validate(dict(field))
            for field in artifact_payload.get("output_signature", ())
        )
    except (TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "Torch artifact plan signature is malformed"
        ) from exc
    if not input_schema or not output_schema:
        raise AlgorithmConfigurationError(
            "Torch artifact plan requires input and output signatures"
        )
    targets = artifact_payload.get("targets", ())
    required_artifacts = tuple(
        str(target["name"])
        for target in targets
        if isinstance(target, Mapping) and isinstance(target.get("name"), str)
    )
    configured_task = options.algorithm_config.get("task_type")
    task_type = (
        configured_task
        if isinstance(configured_task, str) and configured_task
        else descriptor.identity.algorithm
    )
    return ExportCheckpointV1(
        trainer_type="ray_train_torch",
        architecture_id=descriptor.identity.implementation_id,
        input_schema=input_schema,
        output_schema=output_schema,
        preprocessing=dict(preprocessing or {}),
        task_type=task_type,
        framework="pytorch",
        framework_version=str(torch.__version__),
        checkpoint_format_version=1,
        required_artifacts=required_artifacts,
    )


@PublicAPI(stability="alpha")
class RayTorchSourceProvider:
    """Open an ExportSource while preserving the checkpoint lease lifetime."""

    api_version: ClassVar[int] = 1
    provider_id: ClassVar[str] = "ray-torch-v1"
    trainer_type: ClassVar[str] = "ray_train_torch"
    priority: ClassVar[int] = 100

    def open_source(self, result: Any, config: BaseModel | None = None) -> Any:
        options = TorchSourceOptions.model_validate(
            config.model_dump() if config is not None else {}
        )
        return self.open_export_source(result, options)

    def open_export_source(self, result: Any, config: TorchSourceOptions) -> Any:
        return _open_source(result, config)


@contextmanager
def _open_source(
    result: Any,
    options: TorchSourceOptions,
) -> Generator[ExportSource, None, None]:
    checkpoint = getattr(result, "checkpoint", result)
    if checkpoint is None:
        raise AlgorithmConfigurationError("Torch result has no final Checkpoint")
    with checkpoint_directory(checkpoint) as checkpoint_dir:
        descriptor = _read_descriptor(checkpoint_dir)
        reference = options.implementation_ref.partition(":")
        if not reference[1]:
            raise AlgorithmConfigurationError(
                "Torch implementation reference is invalid"
            )
        qualified = f"{reference[0]}:{reference[2]}"
        if (
            descriptor.adapter_identity is not None
            and descriptor.adapter_identity != options.implementation_id
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint implementation identity drifted"
            )
        if descriptor.identity.implementation_id != options.implementation_id:
            raise AlgorithmConfigurationError(
                "Torch checkpoint implementation identity drifted"
            )
        if (
            descriptor.identity.implementation_code_digest
            != options.implementation_code_digest
        ):
            raise AlgorithmConfigurationError(
                "Torch checkpoint implementation code digest drifted"
            )
        if options.policy_digest != descriptor.policy_digest:
            raise AlgorithmConfigurationError(
                "Torch export Policy digest does not match checkpoint"
            )
        if descriptor.identity.plan_digest != options.plan_digest:
            raise AlgorithmConfigurationError(
                "Torch export plan digest does not match checkpoint"
            )
        if descriptor.input_binding_digest != options.input_binding_digest:
            raise AlgorithmConfigurationError(
                "Torch export input binding digest does not match checkpoint"
            )
        implementation_ref = QualifiedReference.parse(qualified)
        _validate_module_digest(implementation_ref, options.implementation_code_digest)
        implementation = _load_reference(implementation_ref)
        if not isinstance(implementation, type) or not issubclass(
            implementation, (TorchRecipe, RayTorchAdapter)
        ):
            raise AlgorithmConfigurationError(
                "Torch export implementation is not a Recipe or Adapter"
            )
        checkpoint_ref = TorchCheckpointRef(
            checkpoint=checkpoint,
            descriptor_digest=descriptor.digest,
            source_stage_id=descriptor.identity.stage_id,
            descriptor=descriptor,
        )
        if options.loop_owner == "adapter":
            adapter = implementation()
            if not isinstance(adapter, RayTorchAdapter):
                raise AlgorithmConfigurationError(
                    "adapter export requires RayTorchAdapter"
                )
            runtime = TorchRuntimeContext(
                algorithm_config=_source_context_config(options),
                implementation_id=options.implementation_id,
                world_size=descriptor.world_size,
                policy_digest=descriptor.policy_digest,
                execution_plan_digest=descriptor.execution_plan_digest,
                run_identity=descriptor.identity,
                input_bindings=options.input_bindings,
                output_config=options.output_config,
                input_binding_digest=options.input_binding_digest,
                state_layout=descriptor.state_layout,
                adapter_identity=descriptor.adapter_identity,
                resume_supported=descriptor.resume_supported,
            )
            from tributo.algorithms.spi import TorchStageContext

            artifact_context = TorchArtifactContext(
                stage=TorchStageContext(
                    runtime=runtime,
                    stage_id=descriptor.identity.stage_id,
                    stage_index=options.stage_index,
                    is_final=True,
                    input_roles=options.stage_input_roles,
                ),
                checkpoint=checkpoint_ref,
            )
            artifact_plan = adapter.artifact_plan(artifact_context)
            if not isinstance(artifact_plan, TorchArtifactPlan):
                raise AlgorithmConfigurationError(
                    "RayTorchAdapter.artifact_plan must return TorchArtifactPlan"
                )
            source_context = adapter.open_export_source(
                checkpoint_ref, artifact_context
            )
            if not hasattr(source_context, "__enter__"):
                raise AlgorithmConfigurationError(
                    "Adapter open_export_source must return a context manager"
                )
            with source_context as source:
                if not isinstance(source, ExportSource):
                    raise AlgorithmConfigurationError(
                        "Adapter export source must be an ExportSource"
                    )
                artifact_payload = artifact_plan.to_dict()
                if source.source_kind != artifact_plan.source_kind:
                    raise AlgorithmConfigurationError(
                        "Adapter ExportSource source_kind does not match artifact plan"
                    )
                declared_payload = source.metadata.get("artifact_plan")
                if (
                    declared_payload is not None
                    and declared_payload != artifact_payload
                ):
                    raise AlgorithmConfigurationError(
                        "Adapter ExportSource artifact plan drifted"
                    )
                metadata = dict(source.metadata)
                metadata["artifact_plan"] = artifact_payload
                checkpoint_contract = _checkpoint_contract_from_artifact_plan(
                    artifact_payload,
                    descriptor,
                    options,
                    preprocessing=source.preprocessing_state,
                )
                yield source.model_copy(
                    update={
                        "metadata": metadata,
                        "checkpoint_contract": checkpoint_contract,
                    }
                )
            return
        recipe = implementation()
        if not isinstance(recipe, TorchRecipe):
            raise AlgorithmConfigurationError("core_recipe export requires TorchRecipe")
        model = _load_recipe_model(recipe, checkpoint_dir, descriptor, options)
        runtime = TorchRuntimeContext(
            algorithm_config=_source_context_config(options),
            implementation_id=options.implementation_id,
            world_size=descriptor.world_size,
            policy_digest=descriptor.policy_digest,
            execution_plan_digest=descriptor.execution_plan_digest,
            run_identity=descriptor.identity,
            input_bindings=options.input_bindings,
            output_config=options.output_config,
            input_binding_digest=options.input_binding_digest,
            state_layout=descriptor.state_layout,
            adapter_identity=descriptor.adapter_identity,
            resume_supported=descriptor.resume_supported,
        )
        from tributo.algorithms.spi import TorchStageContext

        artifact_context = TorchArtifactContext(
            stage=TorchStageContext(
                runtime=runtime,
                stage_id=descriptor.identity.stage_id,
                stage_index=options.stage_index,
                is_final=True,
                input_roles=options.stage_input_roles,
            ),
            checkpoint=checkpoint_ref,
        )
        artifact_plan = recipe.artifact_plan(artifact_context)
        if not isinstance(artifact_plan, TorchArtifactPlan):
            raise AlgorithmConfigurationError(
                "TorchRecipe.artifact_plan must return TorchArtifactPlan"
            )
        yield _export_source(
            model,
            checkpoint_dir,
            descriptor,
            artifact_plan,
            algorithm_config=options.algorithm_config,
        )


def _read_descriptor(root: Path) -> TorchCheckpointDescriptor:
    path = root / "torch_checkpoint_descriptor.json"
    if path.is_symlink() or not path.is_file():
        raise AlgorithmConfigurationError("Torch checkpoint descriptor is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptor = TorchCheckpointDescriptor.from_dict(payload)
        root_resolved = root.resolve()
        actual_files: dict[str, str] = {}
        for candidate in sorted(root.rglob("*")):
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(
                root_resolved
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint payload escapes its root"
                )
            if candidate.is_file() and candidate.name not in {
                "torch_checkpoint_descriptor.json",
                ".metadata.json",
            }:
                actual_files[candidate.relative_to(root).as_posix()] = _sha256_file(
                    candidate
                )
        for filename, expected_digest in descriptor.payload_files.items():
            artifact = root / filename
            if artifact.is_symlink() or not artifact.resolve().is_relative_to(
                root_resolved
            ):
                raise AlgorithmConfigurationError(
                    "Torch checkpoint payload escapes its root"
                )
            if not artifact.is_file() or _sha256_file(artifact) != expected_digest:
                raise AlgorithmConfigurationError(
                    "Torch checkpoint payload digest mismatch"
                )
        if actual_files != dict(descriptor.payload_files):
            raise AlgorithmConfigurationError(
                "Torch checkpoint payload files or digest mismatch"
            )
        return descriptor
    except (OSError, TypeError, ValueError) as exc:
        raise AlgorithmConfigurationError(
            "Torch checkpoint descriptor is malformed"
        ) from exc


def _load_recipe_model(
    recipe: TorchRecipe,
    root: Path,
    descriptor: TorchCheckpointDescriptor,
    options: TorchSourceOptions,
) -> object:
    import torch

    state_path = root / "model.pt"
    if state_path.is_symlink() or not state_path.is_file():
        raise AlgorithmConfigurationError("Torch export checkpoint is missing model.pt")
    runtime = TorchRuntimeContext(
        algorithm_config=_source_context_config(options),
        implementation_id=options.implementation_id,
        world_size=descriptor.world_size,
        policy_digest=descriptor.policy_digest,
        execution_plan_digest=descriptor.execution_plan_digest,
        run_identity=descriptor.identity,
        input_bindings=options.input_bindings,
        output_config=options.output_config,
        input_binding_digest=options.input_binding_digest,
        state_layout=descriptor.state_layout,
        adapter_identity=descriptor.adapter_identity,
        resume_supported=descriptor.resume_supported,
    )
    from tributo.algorithms.spi import (
        TorchBuildContext,
        TorchModuleSet,
        TorchStageContext,
    )

    modules = recipe.build_modules(
        TorchBuildContext(
            runtime=runtime,
            stage=TorchStageContext(
                runtime=runtime,
                stage_id=descriptor.identity.stage_id,
                stage_index=options.stage_index,
                is_final=True,
                input_roles=options.stage_input_roles,
            ),
        )
    )
    modules = (
        modules if isinstance(modules, TorchModuleSet) else TorchModuleSet(modules)
    )
    model = modules["model"]
    if not isinstance(model, torch.nn.Module):
        raise AlgorithmConfigurationError("TorchRecipe export model is not nn.Module")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise AlgorithmConfigurationError("Torch model checkpoint must be a state_dict")
    model.load_state_dict(state)
    return model


def _export_source(
    model: object,
    root: Path,
    descriptor: TorchCheckpointDescriptor,
    artifact_plan: object,
    *,
    algorithm_config: Mapping[str, Any],
) -> ExportSource:
    import torch

    metrics_path = root / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {}
    )
    artifact_payload = (
        artifact_plan.to_dict() if hasattr(artifact_plan, "to_dict") else artifact_plan
    )
    if not isinstance(artifact_payload, dict):
        raise AlgorithmConfigurationError("Torch artifact plan is malformed")
    input_signature = artifact_payload.get("input_signature", ())
    output_signature = artifact_payload.get("output_signature", ())
    if not isinstance(descriptor.identity.plan_digest, str):
        raise AlgorithmConfigurationError("Torch export checkpoint has no plan digest")
    checkpoint_contract = _checkpoint_contract_from_artifact_plan(
        artifact_payload,
        descriptor,
        TorchSourceOptions(
            implementation_ref="internal:torch",
            implementation_code_digest=descriptor.implementation_code_digest,
            implementation_id=descriptor.identity.implementation_id,
            policy_digest=descriptor.policy_digest,
            plan_digest=descriptor.identity.plan_digest,
            input_binding_digest=descriptor.input_binding_digest,
            algorithm_config=dict(algorithm_config),
        ),
    )
    sample_inputs: dict[str, object] = {}
    for field in input_signature:
        if not isinstance(field, dict):
            raise AlgorithmConfigurationError("Torch input signature is malformed")
        shape = tuple(
            1 if dim == "batch" else int(dim) for dim in field.get("shape", (1,))
        )
        dtype = getattr(torch, str(field.get("dtype", "float32")), None)
        if dtype is None:
            raise AlgorithmConfigurationError(
                "Torch artifact input dtype is unsupported"
            )
        field_name = str(field["name"])
        sample_inputs[field_name] = (
            torch.ones(shape, dtype=dtype)
            if (
                field_name == "input_ids"
                and dtype
                in {
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                }
            )
            else torch.zeros(shape, dtype=dtype)
        )
    return ExportSource(
        source_kind="torch_module",
        model_object=model,
        architecture_id=descriptor.identity.implementation_id,
        model_config_data=dict(algorithm_config.get("model", {}))
        if isinstance(algorithm_config.get("model", {}), Mapping)
        else {},
        feature_schema={
            "input_signature": input_signature,
            "output_signature": output_signature,
            "input_names": [
                field["name"]
                for field in input_signature
                if isinstance(field, Mapping) and isinstance(field.get("name"), str)
            ],
        },
        preprocessing_state={},
        sample_inputs=sample_inputs,
        checkpoint_contract=checkpoint_contract,
        metadata={
            "framework": "pytorch",
            "torch_runtime_api_version": 1,
            "artifact_plan": artifact_payload,
            "metrics": metrics,
        },
        source_fingerprint=_sha256_file(root / "model.pt"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["RayTorchSourceProvider", "TorchSourceOptions"]
