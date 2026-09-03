"""Core-owned Ray Train Torch runtime.

The module is intentionally the only Runtime implementation selected by
``DistributionStrategy.RAY_TRAIN_TORCH``.  It owns Trainer construction and
Stage ordering; algorithm Wheels only provide a ``TorchRecipe`` or
``RayTorchAdapter`` implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Generator, cast

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    ComponentStageEvidence,
    DistributionStrategy,
    QualifiedReference,
    ReplicatedTorchStateEvidence,
    ResultPolicy,
    TorchAccumulationWindow,
    TorchBackwardContext,
    TorchCheckpointDescriptor,
    TorchCheckpointLocator,
    TorchCheckpointProgress,
    TorchCheckpointRef,
    TorchCompositeGlobalState,
    TorchCompositeLossContribution,
    TorchExecutionEvidence,
    TorchGlobalLossContext,
    TorchGlobalLossReducer,
    TorchGlobalLossReduction,
    TorchLossContribution,
    TorchMetricContribution,
    TorchPreflightLease,
    TorchPreflightTokenData,
    TorchRankProgressStatistics,
    TorchRecoveryEnvelope,
    TorchRoleExecutionEvidence,
    TorchRuntimeExecutionEnvelope,
    TorchStageRunIdentity,
    TorchWorkerControlEnvelope,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    apply_torch_loss_backward,
    claim_torch_run_directory,
    describe_torch_checkpoint,
    invoke_torch_global_loss_reducer,
    report_torch_checkpoint,
    torch_run_config_name,
    validate_torch_retry_identity,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.core.worker import (
    _actual_environment_versions,
    _load_reference,
    _validate_module_digest,
)
from tributo.algorithms.spi import (
    PreparedInput,
    RayTorchAdapter,
    RuntimeExecutionEnvelope,
    TorchBatch,
    TorchBatchContext,
    TorchBuildContext,
    TorchCheckpointContext,
    TorchMetricPlan,
    TorchModuleSet,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchRuntimeContext,
    TorchStageContext,
    TorchStepContext,
    TorchStepResult,
    TorchWorkerCheckpointContext,
)
from tributo.integrations.algorithm_runtimes.portable_metrics import (
    portable_fit_only_metrics,
)
from tributo.util.annotations import DeveloperAPI

RAY_TRAIN_TORCH_RUNTIME_ID = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_TRAIN_TORCH
].runtime_id
_CORE_RECIPE_LOOP_REF = (
    "tributo.integrations.algorithm_runtimes.ray_train_torch:"
    "torch_recipe_train_loop_per_worker"
)
_CORE_ADAPTER_LOOP_REF = (
    "tributo.integrations.algorithm_runtimes.ray_train_torch:"
    "ray_torch_adapter_train_loop_per_worker"
)
logger = logging.getLogger(__name__)


def _load_torch_implementation(plan: Any) -> TorchRecipe | RayTorchAdapter:
    """Load a trusted recipe/adapter exactly once after preflight."""
    _validate_module_digest(
        plan.implementation.implementation_ref,
        plan.implementation.code_digest,
    )
    implementation = _load_reference(plan.implementation.implementation_ref)
    if not isinstance(implementation, type):
        raise AlgorithmConfigurationError(
            "Torch implementation reference must resolve to a class"
        )
    if not issubclass(implementation, (TorchRecipe, RayTorchAdapter)):
        raise AlgorithmConfigurationError(
            "Torch implementation must subclass TorchRecipe or RayTorchAdapter"
        )
    if getattr(implementation, "api_version", None) != 1:
        raise AlgorithmConfigurationError(
            "Torch implementation api_version must be exactly 1"
        )
    try:
        instance = implementation()
    except TypeError as exc:
        raise AlgorithmConfigurationError(
            "Torch implementation must have a no-argument constructor"
        ) from exc
    return cast(TorchRecipe | RayTorchAdapter, instance)


def _policy(plan: Any) -> Any:
    distribution = plan.distribution_spec
    if (
        distribution is None
        or distribution.strategy is not DistributionStrategy.RAY_TRAIN_TORCH
    ):
        raise AlgorithmConfigurationError(
            "Ray Train Torch runtime requires RAY_TRAIN_TORCH"
        )
    policy = distribution.policy
    if not hasattr(policy, "execution_plan"):
        raise AlgorithmConfigurationError("Torch runtime lost TorchPolicy")
    return policy


def _torch_algorithm_context_config(plan: Any) -> dict[str, object]:
    """Return only algorithm-owned config for Torch implementation contexts."""
    config = plan.algorithm_config
    if not isinstance(config, Mapping):
        raise AlgorithmConfigurationError("Torch algorithm config must be a mapping")
    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in {"ray", "output"}
    }


def _torch_input_bindings(plan: Any) -> dict[str, object]:
    """Expose credential-free InputBinding metadata to Torch implementations."""
    return {
        binding.name: binding.descriptor_payload()
        for binding in plan.input_bindings.bindings
    }


def _torch_output_config(plan: Any) -> dict[str, object]:
    """Expose output options separately from algorithm worker configuration."""
    config = plan.algorithm_config
    output = config.get("output", {}) if isinstance(config, Mapping) else {}
    if not isinstance(output, Mapping):
        raise AlgorithmConfigurationError("Torch output config must be a mapping")
    return {str(key): value for key, value in output.items()}


_ADAPTER_CONFIG_BLOCKED_KEYS = frozenset(
    {
        "ray",
        "output",
        "path",
        "uri",
        "locator",
        "storage_path",
        "resume",
        "resume_from",
        "checkpoint",
        "checkpoint_uri",
        "checkpoint_locator",
        "bundle_uri",
        "credential",
        "credentials",
        "secret",
        "secrets",
    }
)
_TORCH_INTERNAL_METRIC_NAMES = frozenset(
    {
        "torch_evidence",
        "checkpoint_descriptor",
        "reducer_id",
        "reducer_api_version",
        "reducer_schema_id",
        "reducer_code_digest",
        "reducer_branch",
        "reducer_evidence",
    }
)


def _validate_adapter_worker_config(
    value: object, *, path: str = "adapter_config"
) -> None:
    """Reject Core paths, recovery handles and credentials in Adapter config."""
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise AlgorithmConfigurationError(
                    "Adapter worker config keys must be strings"
                )
            key = raw_key.casefold()
            if key in _ADAPTER_CONFIG_BLOCKED_KEYS or key.endswith(
                ("_path", "_uri", "_locator")
            ):
                raise AlgorithmConfigurationError(
                    f"Adapter worker config contains a Core-owned path field: {path}.{raw_key}"
                )
            _validate_adapter_worker_config(nested, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_adapter_worker_config(nested, path=f"{path}[{index}]")


def _accumulate_metric_totals(
    target: dict[str, list[float]],
    contributions: Mapping[str, TorchMetricContribution],
    *,
    prefix: str = "",
) -> None:
    """Accumulate typed metric contributions without interpreting their meaning."""
    for name, contribution in contributions.items():
        if not isinstance(name, str) or not isinstance(
            contribution, TorchMetricContribution
        ):
            raise AlgorithmConfigurationError(
                "Torch reducer metric contribution is invalid"
            )
        key = f"{prefix}{name}"
        totals = target.setdefault(key, [0.0, 0.0])
        totals[0] += contribution.numerator
        totals[1] += contribution.normalizer


def _reduce_metric_totals(
    totals: Mapping[str, list[float]],
    reducers: Mapping[str, str],
    *,
    device: object,
    dist: Any,
    world_size: int,
) -> dict[str, float]:
    """Apply the reducer declared for each metric with aligned collectives."""
    import torch

    names = set(totals)
    if dist.is_available() and dist.is_initialized():
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, sorted(names))
        if any(value != sorted(names) for value in gathered):
            raise AlgorithmExecutionError("Torch metric keys differ across ranks")
    values: dict[str, float] = {}
    for name in sorted(names):
        numerator, normalizer = totals[name]
        reducer_value = reducers.get(name)
        if reducer_value is None and "_" in name:
            reducer_value = reducers.get(name.split("_", 1)[1])
        if reducer_value is None and name.endswith("_loss"):
            reducer_value = reducers.get("train_loss")
        if reducer_value is None:
            raise AlgorithmConfigurationError(
                f"Torch metric {name!r} has no declared reducer"
            )
        reducer = str(reducer_value)
        if reducer in {"sum_count", "weighted_mean"}:
            state = torch.tensor(
                [float(numerator), float(normalizer)],
                dtype=torch.float64,
                device=device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(state, op=dist.ReduceOp.SUM)
            if state[1].item() <= 0:
                raise AlgorithmExecutionError(
                    f"Torch metric {name!r} has a zero global normalizer"
                )
            values[name] = float(state[0].item() / state[1].item())
            continue
        if reducer not in {"min", "max"}:
            raise AlgorithmConfigurationError(
                f"unsupported Torch metric reducer {reducer!r}"
            )
        present = normalizer > 0
        sentinel = float("inf") if reducer == "min" else float("-inf")
        state = torch.tensor(
            float(numerator / normalizer) if present else sentinel,
            dtype=torch.float64,
            device=device,
        )
        present_state = torch.tensor(
            1 if present else 0, dtype=torch.int64, device=device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(present_state, op=dist.ReduceOp.MAX)
            dist.all_reduce(
                state,
                op=torch.distributed.ReduceOp.MIN
                if reducer == "min"
                else torch.distributed.ReduceOp.MAX,
            )
        if int(present_state.item()) == 0:
            raise AlgorithmExecutionError(
                f"Torch metric {name!r} has no global contribution"
            )
        values[name] = float(state.item())
    return values


def _reducer_metadata(policy: Any) -> dict[str, object]:
    """Load the qualified reducer metadata already attested by preflight."""
    if policy.global_loss_reducer_ref is None:
        return {}
    reference = QualifiedReference.parse(policy.global_loss_reducer_ref)
    _validate_module_digest(reference, policy.global_loss_reducer_code_digest)
    reducer = _load_reference(reference)
    if isinstance(reducer, type):
        reducer = reducer()
    values = {
        "reducer_id": getattr(reducer, "reducer_id", None),
        "reducer_api_version": getattr(reducer, "api_version", None),
        "reducer_schema_id": getattr(reducer, "component_schema_id", None),
        "reducer_code_digest": getattr(reducer, "code_digest", None),
    }
    if (
        not isinstance(values["reducer_id"], str)
        or values["reducer_api_version"] != policy.global_loss_reducer_api_version
        or values["reducer_schema_id"] != policy.composite_loss_schema_id
        or values["reducer_code_digest"] != policy.global_loss_reducer_code_digest
    ):
        raise AlgorithmConfigurationError("Torch global loss reducer metadata drifted")
    return values


def policy_result_policy(plan: Any) -> ResultPolicy:
    distribution = plan.distribution_spec
    if distribution is None:
        raise AlgorithmConfigurationError("Torch plan has no DistributionSpec")
    return cast(ResultPolicy, distribution.result_policy)


def _identity(
    plan: Any, run_id: str, invocation_id: str, stage_id: str
) -> TorchStageRunIdentity:
    code_digest = plan.implementation.code_digest
    if not isinstance(code_digest, str) or len(code_digest) != 64:
        raise AlgorithmConfigurationError(
            "Torch implementation code digest is required"
        )
    return TorchStageRunIdentity(
        run_id=run_id,
        invocation_id=invocation_id,
        stage_id=stage_id,
        torch_runtime_api_version=1,
        algorithm=plan.resolution.algorithm,
        implementation_id=plan.implementation.implementation_id,
        implementation_code_digest=code_digest,
        policy_digest=_policy(plan).digest,
        execution_plan_digest=_policy(plan).execution_plan.digest,
        plan_digest=plan.plan_id,
    )


def _stage_context(
    plan: Any,
    runtime_context: TorchRuntimeContext,
    stage: Any,
    index: int,
    *,
    predecessor: str | None = None,
    predecessor_descriptor: Mapping[str, Any] | None = None,
) -> TorchStageContext:
    return TorchStageContext(
        runtime=runtime_context,
        stage_id=stage.stage_id,
        stage_index=index,
        is_final=stage.stage_id == _policy(plan).execution_plan.final_stage_id,
        input_roles=tuple(stage.input_roles),
        predecessor_stage_id=predecessor,
        predecessor_checkpoint_descriptor=predecessor_descriptor,
        metric_mapping=dict(getattr(stage, "metric_mapping", {})),
        checkpoint_required=bool(getattr(stage, "checkpoint_required", True)),
        checkpoint_interval_windows=int(
            getattr(stage, "checkpoint_interval_windows", 1)
        ),
    )


def _control_for_stage(
    plan: Any,
    policy: Any,
    stage: Any,
    *,
    run_id: str,
    invocation_id: str,
    checkpoint: Mapping[str, Any] | None = None,
    purpose: str | None = None,
    source_stage_id: str | None = None,
    predecessor: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Create credential-free initial recovery control for a Stage."""
    if checkpoint is None:
        checkpoint = predecessor
        if checkpoint is not None and purpose is None:
            purpose = "stage_dependency"
        if checkpoint is not None and source_stage_id is None:
            source_stage_id = getattr(stage, "checkpoint_from_stage", None)
    if checkpoint is None:
        return None
    resume_uri = checkpoint.get("locator")
    descriptor_digest = checkpoint.get("descriptor_digest")
    if not isinstance(resume_uri, str) or not resume_uri:
        raise AlgorithmConfigurationError(
            "Torch recovery requires a credential-free locator"
        )
    if not isinstance(descriptor_digest, str):
        raise AlgorithmConfigurationError(
            "Torch recovery requires checkpoint_descriptor_digest"
        )
    locator = TorchCheckpointLocator(resume_uri, descriptor_digest)
    control = TorchWorkerControlEnvelope(
        schema_version=1,
        run_id=run_id,
        invocation_id=invocation_id,
        source_stage_id=source_stage_id,
        target_stage_id=stage.stage_id,
        purpose=purpose or "cross_run_initial_recovery",
        checkpoint_locator=locator,
        checkpoint_descriptor_digest=descriptor_digest,
        policy_digest=policy.digest,
        execution_plan_digest=policy.execution_plan.digest,
    )
    return control.to_dict()


def _describe_recovery_locator(
    locator: TorchCheckpointLocator,
    *,
    policy: Any,
    plan: Any,
    worker_count: int,
) -> TorchCheckpointDescriptor:
    """Open a recovery locator on the driver and validate its payload digest."""
    checkpoint = open_torch_checkpoint_locator(locator)
    try:
        descriptor = describe_torch_checkpoint(TorchCheckpointRef(checkpoint), object())
        _require_checkpoint_commit(checkpoint, descriptor)
    finally:
        closer = getattr(checkpoint, "close", None)
        if callable(closer):
            closer()
    if descriptor.digest != locator.descriptor_digest:
        raise AlgorithmExecutionError(
            "Torch recovery locator descriptor digest drifted"
        )
    if (
        descriptor.policy_digest != policy.digest
        or descriptor.execution_plan_digest != policy.execution_plan.digest
        or descriptor.world_size != worker_count
        or descriptor.implementation_code_digest != plan.implementation.code_digest
        or descriptor.identity.plan_digest != plan.plan_id
        or descriptor.input_binding_digest != _input_binding_digest(plan)
        or descriptor.state_layout != policy.state_layout
    ):
        raise AlgorithmExecutionError("Torch recovery descriptor identity mismatch")
    return descriptor


def _checkpoint_evidence_payload(checkpoint: object) -> dict[str, Any]:
    """Read the Core-owned, credential-free evidence sidecar when present."""
    opener = getattr(checkpoint, "as_directory", None)
    if not callable(opener):
        return {}
    try:
        with opener() as directory:
            path = Path(directory) / "torch_execution_evidence.json"
            if path.is_symlink() or not path.is_file():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise AlgorithmExecutionError(
            "Torch checkpoint execution evidence is malformed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AlgorithmExecutionError(
            "Torch checkpoint execution evidence is malformed"
        )
    return dict(payload)


def _require_checkpoint_commit(
    checkpoint: object, descriptor: TorchCheckpointDescriptor
) -> None:
    """Require the marker-last commit used by persistent Stage locators."""
    with _opened_checkpoint(checkpoint) as root:
        marker = root / "torch_stage_commit.json"
        if marker.is_symlink() or not marker.is_file():
            raise AlgorithmExecutionError(
                "Torch Stage locator references an uncommitted checkpoint"
            )
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "Torch Stage checkpoint commit marker is malformed"
            ) from exc
    if not isinstance(payload, Mapping) or (
        payload.get("identity") != descriptor.identity.to_dict()
        or payload.get("descriptor_digest") != descriptor.digest
    ):
        raise AlgorithmExecutionError(
            "Torch Stage checkpoint commit marker does not match descriptor"
        )


def _recovery_record_for_locator(
    locator: TorchCheckpointLocator,
    *,
    policy: Any,
    plan: Any,
    worker_count: int,
) -> dict[str, Any]:
    descriptor = _describe_recovery_locator(
        locator, policy=policy, plan=plan, worker_count=worker_count
    )
    checkpoint = open_torch_checkpoint_locator(locator)
    try:
        evidence = _checkpoint_evidence_payload(checkpoint)
    finally:
        closer = getattr(checkpoint, "close", None)
        if callable(closer):
            closer()
    return {
        "locator": locator.uri,
        "descriptor_digest": descriptor.digest,
        "descriptor": descriptor.to_dict(),
        "evidence": evidence,
    }


def _recovery_records(
    plan: Any,
    policy: Any,
    *,
    worker_count: int,
) -> tuple[tuple[str, ...], str | None, dict[str, dict[str, Any]]]:
    """Normalize the full Torch recovery envelope and legacy shorthand."""
    stages = tuple(policy.execution_plan.stages)
    stage_ids = tuple(stage.stage_id for stage in stages)
    raw_envelope = plan.runtime.torch_recovery
    if raw_envelope is not None:
        envelope = TorchRecoveryEnvelope.from_dict(raw_envelope)
        if not policy.resume_supported and (
            envelope.stage_checkpoints or envelope.active_checkpoint is not None
        ):
            raise AlgorithmConfigurationError(
                "Torch Policy does not support external recovery"
            )
        completed = tuple(envelope.completed_stage_ids)
        if any(stage_id not in stage_ids for stage_id in completed):
            raise AlgorithmExecutionError(
                "Torch recovery contains an unknown completed Stage"
            )
        if tuple(stage_ids[: len(completed)]) != completed:
            raise AlgorithmExecutionError(
                "Torch recovery completed stages must follow execution plan order"
            )
        if (
            envelope.active_stage_id is not None
            and envelope.active_stage_id not in stage_ids
        ):
            raise AlgorithmExecutionError("Torch recovery active Stage is unknown")
        if envelope.active_stage_id is not None:
            active_index = stage_ids.index(envelope.active_stage_id)
            if any(stage_id not in completed for stage_id in stage_ids[:active_index]):
                raise AlgorithmExecutionError(
                    "Torch recovery active Stage has an incomplete predecessor"
                )
            active_stage = stages[active_index]
            if any(dep not in completed for dep in active_stage.depends_on):
                raise AlgorithmExecutionError(
                    "Torch recovery active Stage dependencies are incomplete"
                )
        records: dict[str, dict[str, Any]] = {}
        for stage_id, locator in envelope.stage_checkpoints.items():
            record = _recovery_record_for_locator(
                locator, policy=policy, plan=plan, worker_count=worker_count
            )
            descriptor = TorchCheckpointDescriptor.from_dict(record["descriptor"])
            if descriptor.identity.stage_id != stage_id:
                raise AlgorithmExecutionError(
                    "Torch recovery checkpoint Stage mismatch"
                )
            records[stage_id] = record
        if (
            envelope.active_stage_id is not None
            and envelope.active_checkpoint is not None
        ):
            active_record = _recovery_record_for_locator(
                envelope.active_checkpoint,
                policy=policy,
                plan=plan,
                worker_count=worker_count,
            )
            active_descriptor = TorchCheckpointDescriptor.from_dict(
                active_record["descriptor"]
            )
            if active_descriptor.identity.stage_id != envelope.active_stage_id:
                raise AlgorithmExecutionError(
                    "Torch recovery active checkpoint Stage mismatch"
                )
            if not policy.resume_supported:
                raise AlgorithmConfigurationError(
                    "Torch Policy does not support cross-Run active recovery"
                )
            if not active_descriptor.resume_supported:
                raise AlgorithmExecutionError(
                    "Torch active recovery checkpoint is not externally recoverable"
                )
            records[envelope.active_stage_id] = active_record
        return completed, envelope.active_stage_id, records

    resume_uri = plan.runtime.resume_from
    ray_config = plan.algorithm_config.get("ray", {})
    resume_config = (
        ray_config.get("resume", {}) if isinstance(ray_config, Mapping) else {}
    )
    if resume_uri is None and isinstance(resume_config, Mapping):
        for key in ("uri", "checkpoint_uri"):
            if isinstance(resume_config.get(key), str):
                resume_uri = resume_config[key]
                break
    if resume_uri is None:
        return (), None, {}
    if not policy.resume_supported:
        raise AlgorithmConfigurationError(
            "Torch Policy does not support external recovery"
        )
    descriptor_digest = (
        resume_config.get("checkpoint_descriptor_digest")
        if isinstance(resume_config, Mapping)
        else None
    )
    if not isinstance(descriptor_digest, str):
        raise AlgorithmConfigurationError(
            "Torch resume shorthand requires ray.resume.checkpoint_descriptor_digest"
        )
    locator = TorchCheckpointLocator(resume_uri, descriptor_digest)
    record = _recovery_record_for_locator(
        locator, policy=policy, plan=plan, worker_count=worker_count
    )
    descriptor = TorchCheckpointDescriptor.from_dict(record["descriptor"])
    if descriptor.identity.stage_id not in stage_ids:
        raise AlgorithmExecutionError(
            "Torch resume checkpoint Stage is not in execution plan"
        )
    if not descriptor.resume_supported:
        raise AlgorithmExecutionError(
            "Torch resume checkpoint is not externally recoverable"
        )
    return (), descriptor.identity.stage_id, {descriptor.identity.stage_id: record}


def _payload_rows(value: object) -> int | None:
    count = getattr(value, "count", None)
    if callable(count):
        try:
            result = count()
            return int(result) if result is not None else None
        except Exception as exc:
            logger.debug("Torch Dataset row-count probe failed: %s", type(exc).__name__)
            return None
    return None


def _prepare_datasets(envelope: RuntimeExecutionEnvelope) -> PreparedInput:
    adapter = _load_reference(envelope.plan.runtime.worker_input_adapter_ref)
    if not callable(adapter):
        raise AlgorithmConfigurationError("Torch Worker input adapter is not callable")
    prepared = adapter(envelope.input_payloads[0])
    if not isinstance(prepared, PreparedInput) or not prepared.views:
        if isinstance(prepared, PreparedInput):
            prepared.close()
        raise AlgorithmConfigurationError(
            "Torch input adapter did not expose Ray datasets"
        )
    return prepared


def _resource_map(plan: Any) -> dict[str, float]:
    resources: dict[str, float] = {"CPU": plan.runtime.num_cpus}
    if plan.runtime.num_gpus:
        resources["GPU"] = plan.runtime.num_gpus
    if plan.runtime.memory_bytes is not None:
        resources["memory"] = plan.runtime.memory_bytes
    resources.update(plan.runtime.custom_resources)
    return resources


@contextmanager
def _opened_checkpoint(checkpoint: object) -> Generator[Path, None, None]:
    if isinstance(checkpoint, (str, Path)):
        root = Path(checkpoint)
        if not root.is_dir():
            raise AlgorithmExecutionError("Torch checkpoint directory is missing")
        yield root
        return
    opener = getattr(checkpoint, "as_directory", None)
    if not callable(opener):
        raise AlgorithmExecutionError("Torch checkpoint cannot be opened")
    with opener() as directory:
        yield Path(directory)


def _validate_stage_routes(
    policy: Any,
    stage: Any,
    datasets: Mapping[str, object],
    worker_count: int,
) -> dict[str, int]:
    """Validate role presence, exact coverage minimums and replication budgets."""
    routes = {route.role: route for route in policy.dataset_routing}
    rows: dict[str, int] = {}
    replicated_bytes = 0
    for role in stage.input_roles:
        route = routes.get(role)
        if route is None:
            raise AlgorithmConfigurationError(f"Torch Stage role {role!r} has no route")
        dataset = datasets.get(role)
        if dataset is None:
            if route.required:
                raise AlgorithmConfigurationError(
                    f"required Torch role {role!r} is absent"
                )
            continue
        count = _payload_rows(dataset)
        if count is None:
            raise AlgorithmConfigurationError(
                f"Torch role {role!r} row count is not verifiable"
            )
        rows[role] = count
        if count < route.min_total_rows_if_present:
            raise AlgorithmConfigurationError(f"Torch role {role!r} has too few rows")
        if (
            route.mode == "split_exact"
            and route.required
            and count < worker_count * route.min_rows_per_worker
        ):
            raise AlgorithmConfigurationError(
                f"Torch role {role!r} cannot cover every worker"
            )
        if route.mode == "replicate":
            if route.max_rows is None or count > route.max_rows:
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} exceeds max_rows"
                )
            limiter = getattr(dataset, "limit", None)
            if not callable(limiter):
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} cannot be bounded"
                )
            probe = limiter(route.max_rows + 1)
            probe_count = getattr(probe, "count", None)
            if not callable(probe_count):
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} bounded size is not verifiable"
                )
            observed_probe = int(probe_count())
            if observed_probe > route.max_rows:
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} exceeds max_rows"
                )
            if isinstance(datasets, dict):
                datasets[role] = limiter(route.max_rows)
            size_bytes = getattr(dataset, "size_bytes", None)
            if callable(size_bytes):
                size_bytes = size_bytes()
            if not isinstance(size_bytes, int) or size_bytes < 0:
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} size is not verifiable"
                )
            if (
                route.max_bytes_per_worker is None
                or size_bytes > route.max_bytes_per_worker
            ):
                raise AlgorithmConfigurationError(
                    f"Torch replicate role {role!r} exceeds max_bytes_per_worker"
                )
            replicated_bytes += size_bytes
    if (
        policy.max_replicated_bytes_per_worker is not None
        and replicated_bytes > policy.max_replicated_bytes_per_worker
    ):
        raise AlgorithmConfigurationError(
            "Torch replicate roles exceed aggregate byte budget"
        )
    return rows


def _worker_rows(payload: object, role: str) -> int:
    if hasattr(payload, "get") and callable(payload.get):
        payload = payload.get(role)
    value = getattr(payload, "value", payload)
    rows = _payload_rows(value)
    return rows if rows is not None else 0


def _worker_evidence(
    metrics: Mapping[str, Any],
    plan: Any,
    identity: TorchStageRunIdentity,
    stage: Any,
    expected_rows: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    raw_workers = metrics.get("execution_workers")
    if (
        not isinstance(raw_workers, (list, tuple))
        or len(raw_workers) != plan.runtime.worker_count
    ):
        raise AlgorithmExecutionError("Torch execution did not report every worker")
    workers = tuple(
        WorkerExecutionEvidence.from_dict(item)
        for item in _normalize_worker_evidence(raw_workers, plan)
    )
    role_evidence: list[TorchRoleExecutionEvidence] = []
    for role in stage.input_roles:
        observed = sum(item.input_rows.get(role, 0) for item in workers)
        role_evidence.append(
            TorchRoleExecutionEvidence(
                role=role,
                mode="split_exact",
                required=True,
                present=True,
                empty_rank_policy="reject",
                expected_rows=expected_rows.get(role),
                observed_rows=observed,
                rows_per_rank=tuple(item.input_rows.get(role, 0) for item in workers),
                binding_digest=_binding_digest_for_role(plan, role),
            )
        )
    global_digest = metrics.get("model_state_digest")
    if not isinstance(global_digest, str) or len(global_digest) != 64:
        raise AlgorithmExecutionError("Torch execution did not report a model digest")
    evidence = TorchExecutionEvidence(
        identity=identity,
        run_config_name=torch_run_config_name(identity),
        policy_digest=_policy(plan).digest,
        parallelism_id=_policy(plan).parallelism_id,
        state_layout=_policy(plan).state_layout,
        workers=workers,
        roles=tuple(role_evidence),
        replicated_state=(
            __import__(
                "tributo.algorithms.api.execution",
                fromlist=["ReplicatedTorchStateEvidence"],
            ).ReplicatedTorchStateEvidence(
                model_digests_by_rank={
                    item.rank: item.model_state_digest for item in workers
                },
                global_model_digest=global_digest,
            )
            if _policy(plan).state_layout == "replicated"
            else None
        ),
    )
    return (evidence.to_dict(),)


def _binding_digest_for_role(plan: Any, role: str) -> str:
    """Resolve a role digest, falling back to the primary binding for aliases."""
    if hasattr(plan.input_descriptors, "get"):
        try:
            descriptor = plan.input_descriptors.get(role)
        except AlgorithmConfigurationError:
            descriptor = None
        if descriptor is not None:
            return cast(str, descriptor.binding_digest)
    return cast(str, plan.primary_input_descriptor.binding_digest)


def _input_binding_digest(plan: Any) -> str:
    """Digest the complete role-keyed input descriptor set."""
    if not hasattr(plan, "input_descriptors"):
        bindings = cast(tuple[Any, ...], getattr(plan.input_bindings, "bindings", ()))
        if len(bindings) == 1:
            return hashlib.sha256(
                json.dumps(
                    bindings[0].descriptor_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        raise AlgorithmConfigurationError("Torch input binding descriptors are missing")
    descriptors = plan.input_descriptors.to_dict()
    if len(descriptors) == 1:
        return cast(str, plan.primary_input_descriptor.binding_digest)
    return hashlib.sha256(
        json.dumps(descriptors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _normalize_worker_evidence(
    raw_workers: list[object] | tuple[object, ...], plan: Any
) -> tuple[Mapping[str, object], ...]:
    """Add Core-owned declared resources when an Adapter omits that field."""
    resources: dict[str, object] = {
        "num_cpus": plan.runtime.num_cpus,
        "num_gpus": plan.runtime.num_gpus,
        "custom": dict(plan.runtime.custom_resources),
    }
    if plan.runtime.memory_bytes is not None:
        resources["memory_bytes"] = plan.runtime.memory_bytes
    normalized: list[Mapping[str, object]] = []
    for item in raw_workers:
        if not isinstance(item, Mapping):
            raise AlgorithmExecutionError("Torch worker evidence is malformed")
        payload = dict(item)
        payload.setdefault("resources", resources)
        normalized.append(payload)
    return tuple(normalized)


def _component_stage_evidence(
    *,
    plan: Any,
    policy: Any,
    stage: Any,
    identity: TorchStageRunIdentity,
    metrics: Mapping[str, Any],
    expected_rows: Mapping[str, int],
) -> ComponentStageEvidence:
    raw_workers = metrics.get("execution_workers")
    if not isinstance(raw_workers, (list, tuple)):
        raise AlgorithmExecutionError("Torch Stage did not report worker evidence")
    workers = tuple(
        WorkerExecutionEvidence.from_dict(item)
        for item in _normalize_worker_evidence(raw_workers, plan)
    )
    if len(workers) != plan.runtime.worker_count:
        raise AlgorithmExecutionError("Torch Stage worker evidence is incomplete")
    roles = _role_execution_evidence(
        plan=plan,
        policy=policy,
        stage=stage,
        workers=workers,
        expected_rows=expected_rows,
    )
    state_digest = metrics.get("model_state_digest")
    descriptor = metrics.get("checkpoint_descriptor")
    if not isinstance(state_digest, str) or len(state_digest) != 64:
        raise AlgorithmExecutionError("Torch Stage did not report a state digest")
    if getattr(stage, "checkpoint_required", True) and not isinstance(
        descriptor, Mapping
    ):
        raise AlgorithmExecutionError(
            "Torch Stage did not report a checkpoint descriptor"
        )
    checkpoint_descriptor_digest = (
        TorchCheckpointDescriptor.from_dict(descriptor).digest
        if isinstance(descriptor, Mapping)
        else None
    )
    return ComponentStageEvidence(
        stage_id=identity.stage_id,
        workers=workers,
        roles=roles,
        state_digest=state_digest,
        checkpoint_descriptor_digest=checkpoint_descriptor_digest,
    )


def _recovered_stage_evidence(
    *,
    plan: Any,
    policy: Any,
    stage: Any,
    descriptor: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> ComponentStageEvidence:
    """Rebuild component evidence persisted beside a completed Stage checkpoint."""
    workers_value = evidence.get("execution_workers")
    state_digest = evidence.get("model_state_digest")
    if not isinstance(workers_value, (list, tuple)) or not isinstance(
        state_digest, str
    ):
        raise AlgorithmExecutionError(
            f"Torch recovery checkpoint for Stage {stage.stage_id!r} has no evidence"
        )
    workers = tuple(
        WorkerExecutionEvidence.from_dict(item)
        for item in _normalize_worker_evidence(workers_value, plan)
    )
    if len(workers) != plan.runtime.worker_count:
        raise AlgorithmExecutionError(
            "Torch recovery Stage worker evidence is incomplete"
        )
    expected_rows: dict[str, int] = {}
    routes = {route.role: route for route in policy.dataset_routing}
    for role in stage.input_roles:
        route = routes[role]
        rows = tuple(worker.input_rows.get(role, 0) for worker in workers)
        if route.mode == "replicate":
            if rows and len(set(rows)) == 1:
                expected_rows[role] = rows[0]
        elif sum(rows) > 0:
            expected_rows[role] = sum(rows)
    metrics = dict(evidence)
    metrics["checkpoint_descriptor"] = dict(descriptor)
    identity = TorchStageRunIdentity.from_dict(descriptor["identity"])
    return _component_stage_evidence(
        plan=plan,
        policy=policy,
        stage=stage,
        identity=identity,
        metrics=metrics,
        expected_rows=expected_rows,
    )


def _component_state_details(
    stages: tuple[ComponentStageEvidence, ...],
) -> dict[str, str | int]:
    """Project component evidence into the scalar Core state receipt details."""
    if not stages:
        raise AlgorithmExecutionError("Torch component state requires Stage evidence")
    composition_digest = hashlib.sha256(
        json.dumps(
            [stage.to_dict() for stage in stages],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    details: dict[str, str | int] = {
        "framework": "torch_component",
        "component_stage_count": len(stages),
        "component_stages": ",".join(stage.stage_id for stage in stages),
        "anchor_stage": stages[-1].stage_id,
        "composition_digest": composition_digest,
    }
    for stage in stages:
        train_rows = sum(
            role.observed_rows
            for role in stage.roles
            if role.role == "train" and role.present
        )
        if train_rows == 0:
            train_rows = sum(worker.rows_processed or 0 for worker in stage.workers)
        details[f"stage.{stage.stage_id}.digest"] = stage.state_digest
        details[f"stage.{stage.stage_id}.rows"] = train_rows
    return details


def _role_execution_evidence(
    *,
    plan: Any,
    policy: Any,
    stage: Any,
    workers: tuple[WorkerExecutionEvidence, ...],
    expected_rows: Mapping[str, int],
) -> tuple[TorchRoleExecutionEvidence, ...]:
    """Build role evidence from the Policy route instead of a default split."""
    routes = {route.role: route for route in policy.dataset_routing}
    evidence: list[TorchRoleExecutionEvidence] = []
    for role in stage.input_roles:
        route = routes[role]
        present = role in expected_rows
        rows_per_rank = tuple(
            worker.input_rows.get(role, 0) if present else 0 for worker in workers
        )
        if route.mode == "replicate" and present:
            if len(set(rows_per_rank)) != 1:
                raise AlgorithmExecutionError(
                    f"Torch replicate role {role!r} is not identical across workers"
                )
            observed_rows = rows_per_rank[0]
            replicated_bytes = route.max_bytes_per_worker
        else:
            observed_rows = sum(rows_per_rank)
            replicated_bytes = None
        if (
            present
            and route.mode in {"split_exact", "replicate"}
            and expected_rows.get(role) != observed_rows
        ):
            raise AlgorithmExecutionError(
                f"Torch role {role!r} worker evidence does not prove exact coverage"
            )
        evidence.append(
            TorchRoleExecutionEvidence(
                role=role,
                mode=route.mode,
                required=route.required,
                present=present,
                empty_rank_policy=route.empty_rank_policy,
                expected_rows=expected_rows.get(role),
                observed_rows=observed_rows,
                rows_per_rank=rows_per_rank,
                replicated_bytes_per_worker=replicated_bytes,
                binding_digest=_binding_digest_for_role(plan, role),
            )
        )
    return tuple(evidence)


def _reduce_composite_loss(
    loss: TorchCompositeLossContribution,
    *,
    config: Mapping[str, Any],
    world_size: int,
    device: object,
    dist: Any,
    observation: dict[str, object] | None = None,
    expected_metrics: frozenset[str] = frozenset(),
) -> TorchGlobalLossReduction:
    """AllReduce generic component state, then invoke the Wheel-owned reducer."""
    import torch

    reducer_ref = config.get("_core_global_loss_reducer_ref")
    schema_id = config.get("_core_composite_loss_schema_id")
    if not isinstance(reducer_ref, str) or not isinstance(schema_id, str):
        raise AlgorithmExecutionError("Composite loss requires a qualified reducer")
    reducer_reference = QualifiedReference.parse(reducer_ref)
    expected_code_digest = config.get("_core_global_loss_reducer_code_digest")
    _validate_module_digest(reducer_reference, expected_code_digest)
    reducer = _load_reference(reducer_reference)
    if isinstance(reducer, type):
        reducer = reducer()
    if (
        getattr(reducer, "api_version", None)
        != config.get("_core_global_loss_reducer_api_version")
        or getattr(reducer, "component_schema_id", None) != schema_id
        or getattr(reducer, "code_digest", None) != expected_code_digest
    ):
        raise AlgorithmExecutionError("Torch global loss reducer identity drifted")
    # All ranks must agree on both component key sets before entering any
    # value collective.  Otherwise one rank can issue a different number of
    # all-reduces and deadlock the entire Stage.
    local_component_keys = sorted(str(name) for name in loss.differentiable_components)
    local_normalizer_keys = sorted(str(name) for name in loss.normalizer_components)
    if dist.is_available() and dist.is_initialized():
        gathered_keys: list[object] = [None] * world_size
        dist.all_gather_object(
            gathered_keys, (local_component_keys, local_normalizer_keys)
        )
        if any(
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or list(value[0]) != local_component_keys
            or list(value[1]) != local_normalizer_keys
            for value in gathered_keys
        ):
            raise AlgorithmExecutionError(
                "Composite loss component keys differ across ranks"
            )
    local_normalizers = dict(loss.normalizer_components)
    global_components: dict[str, float] = {}
    for name, value in loss.differentiable_components.items():
        tensor = cast(Any, value).detach().to(dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        global_components[name] = float(tensor.item())
    tensors: dict[str, torch.Tensor] = {}
    for name, value in local_normalizers.items():
        tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensors[name] = tensor
    global_state = TorchCompositeGlobalState(
        components=dict(global_components),
        normalizers={name: float(tensor.item()) for name, tensor in tensors.items()},
    )
    global_normalizer = sum(global_state.normalizers.values())
    if not math.isfinite(global_normalizer) or global_normalizer <= 0:
        raise AlgorithmExecutionError(
            "Composite loss global normalizer must be positive before backward"
        )
    algorithm_config = {
        str(key): value
        for key, value in config.items()
        if not str(key).startswith("_core_")
        and str(key) not in {"core_control", "ray", "output"}
    }
    context = TorchGlobalLossContext(
        world_size=world_size,
        policy_digest=str(config["_core_policy_digest"]),
        execution_plan_digest=str(config["_core_execution_plan_digest"]),
        config=algorithm_config,
    )
    reduction = invoke_torch_global_loss_reducer(
        loss,
        global_state,
        cast(TorchGlobalLossReducer, reducer),
        context,
    )
    if dist.is_available() and dist.is_initialized():
        payload = (
            reduction.to_dict()
            if hasattr(reduction, "to_dict")
            else {
                "status": reduction.status,
                "coefficients": dict(reduction.coefficients),
                "branch": reduction.branch,
                "failure_code": reduction.failure_code,
            }
        )
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, payload)
        if any(value != gathered[0] for value in gathered[1:]):
            raise AlgorithmExecutionError(
                "Composite reducer results differ across ranks"
            )
    if reduction.status == "rejected":
        raise AlgorithmExecutionError(
            f"Composite loss reducer rejected contribution: {reduction.failure_code}"
        )
    if not expected_metrics.issubset(set(reduction.metrics)):
        raise AlgorithmExecutionError(
            "Composite reducer did not return every declared metric"
        )
    if "train_loss" not in reduction.metrics:
        raise AlgorithmExecutionError(
            "Composite reducer must return the required train_loss metric"
        )
    if observation is not None:
        observation["branch"] = reduction.branch
        observation["evidence"] = dict(reduction.evidence)
    return reduction


def _composite_backward(
    loss: TorchCompositeLossContribution,
    *,
    config: Mapping[str, Any],
    world_size: int,
    device: object,
    dist: Any,
    observation: dict[str, object] | None = None,
    metric_totals: dict[str, list[float]] | None = None,
    expected_metrics: frozenset[str] = frozenset(),
) -> object:
    """Reduce a Composite loss and return its Core-scaled backward scalar."""
    reduction = _reduce_composite_loss(
        loss,
        config=config,
        world_size=world_size,
        device=device,
        dist=dist,
        observation=observation,
        expected_metrics=expected_metrics,
    )
    if metric_totals is not None:
        _accumulate_metric_totals(metric_totals, reduction.metrics)
    return (
        sum(
            reduction.coefficients[name]
            * cast(Any, loss.differentiable_components[name])
            for name in loss.differentiable_components
        )
        * world_size
    )


def _restore_torch_retry_checkpoint(
    checkpoint: object,
    *,
    stage_context: TorchStageContext,
    model: object,
    optimizer: object,
    scheduler: object | None,
    scaler: object,
    rank: int,
    strict_identity: bool = True,
    progress_sink: dict[str, object] | None = None,
    expected_accumulation: int | None = None,
) -> int:
    """Validate a Ray-injected retry checkpoint before loading any state."""
    if checkpoint is None:
        return 0
    identity = stage_context.runtime.run_identity
    if identity is None:
        raise AlgorithmExecutionError("Torch retry Stage context has no identity")
    ref = TorchCheckpointRef(checkpoint)
    descriptor = describe_torch_checkpoint(
        ref,
        object()
        if not strict_identity
        else TorchCheckpointContext(
            stage=stage_context,
            run_id=identity.run_id,
            invocation_id=identity.invocation_id,
            checkpoint_owner="core",
        ),
    )
    if strict_identity:
        validate_torch_retry_identity(
            descriptor,
            identity,
            world_size=stage_context.runtime.world_size,
        )
    elif descriptor.world_size != stage_context.runtime.world_size:
        raise AlgorithmExecutionError(
            "Torch source checkpoint world size does not match current Stage"
        )
    with _opened_checkpoint(checkpoint) as root:
        import torch

        model_path = root / "model.pt"
        optimizer_path = root / "optimizer.pt"
        scaler_path = root / "scaler.pt"
        rng_path = root / "rng_state.pt"
        if any(
            path.is_symlink() or not path.is_file()
            for path in (model_path, optimizer_path, scaler_path, rng_path)
        ):
            raise AlgorithmExecutionError(
                "Torch retry checkpoint is missing required model/optimizer/scaler/RNG state"
            )
        target_model = cast(Any, getattr(model, "module", model))
        target_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        cast(Any, optimizer).load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=True)
        )
        cast(Any, scaler).load_state_dict(
            torch.load(scaler_path, map_location="cpu", weights_only=True)
        )
        scheduler_path = root / "scheduler.pt"
        if scheduler is not None:
            if scheduler_path.is_symlink() or not scheduler_path.is_file():
                raise AlgorithmExecutionError(
                    "Torch retry checkpoint is missing configured scheduler state"
                )
            cast(Any, scheduler).load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=True)
            )
        payload = torch.load(rng_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("states"), list
        ):
            raise AlgorithmExecutionError("Torch retry RNG state is malformed")
        states = payload["states"]
        if (
            payload.get("world_size") != stage_context.runtime.world_size
            or len(states) != stage_context.runtime.world_size
            or rank >= len(states)
            or not isinstance(states[rank], bytes)
        ):
            raise AlgorithmExecutionError(
                "Torch retry checkpoint is missing the rank RNG state"
            )
        rng = torch.frombuffer(states[rank], dtype=torch.uint8).clone()
        torch.set_rng_state(rng)
        cuda_states = payload.get("cuda_states_by_rank")
        if torch.cuda.is_available():
            if (
                not isinstance(cuda_states, list)
                or len(cuda_states) != stage_context.runtime.world_size
                or rank >= len(cuda_states)
                or not isinstance(cuda_states[rank], list)
                or any(not isinstance(state, bytes) for state in cuda_states[rank])
            ):
                raise AlgorithmExecutionError("Torch retry CUDA RNG state is malformed")
            torch.cuda.set_rng_state_all(
                [
                    torch.frombuffer(state, dtype=torch.uint8).clone()
                    for state in cuda_states[rank]
                ]
            )
        progress_path = root / "torch_progress.json"
        if progress_path.is_symlink() or not progress_path.is_file():
            raise AlgorithmExecutionError(
                "Torch checkpoint is missing deterministic progress state"
            )
        try:
            progress = TorchCheckpointProgress.from_dict(
                json.loads(progress_path.read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "Torch checkpoint progress state is malformed"
            ) from exc
        if progress.optimizer_step != descriptor.completed_step:
            raise AlgorithmExecutionError(
                "Torch checkpoint progress and descriptor step differ"
            )
        if (
            expected_accumulation is not None
            and progress.accumulation_steps != expected_accumulation
        ):
            raise AlgorithmExecutionError(
                "Torch checkpoint accumulation configuration differs"
            )
        if expected_accumulation is not None and (
            len(progress.dataset_cursor_by_rank) != stage_context.runtime.world_size
            or str(rank) not in progress.dataset_cursor_by_rank
        ):
            raise AlgorithmExecutionError(
                "Torch checkpoint dataset cursor does not cover every rank"
            )
        if expected_accumulation is not None:
            raw_rank_statistics = progress.rank_statistics
            if (
                len(raw_rank_statistics) != stage_context.runtime.world_size
                or str(rank) not in raw_rank_statistics
            ):
                raise AlgorithmExecutionError(
                    "Torch checkpoint statistics do not cover every rank"
                )
        if progress_sink is not None:
            progress_sink.update(progress.to_dict())
            progress_sink["_typed_progress"] = progress
    return descriptor.completed_step


def _finalize_torch_window(
    *,
    scaler: Any,
    optimizer: Any,
    model: Any,
    max_gradient_norm: float | None,
    scale: float,
) -> None:
    """Apply the Core-owned unscale, normalize, clip, step, and reset order."""
    import torch

    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


def _should_apply_epoch_scheduler(
    *,
    restore_same_stage: bool,
    epoch: int,
    restored_epoch: int,
    restored_epoch_scheduler_applied: bool,
) -> bool:
    """Return whether the epoch boundary still owes one scheduler step."""
    return not (
        restore_same_stage
        and epoch == restored_epoch
        and restored_epoch_scheduler_applied
    )


def _select_worker_checkpoint(
    config: Mapping[str, Any],
    stage_context: TorchStageContext,
) -> TorchWorkerCheckpointContext:
    """Select retry first, then Core control, never the reverse."""
    import ray.train

    retry = ray.train.get_checkpoint()
    identity = stage_context.runtime.run_identity
    if identity is None:
        raise AlgorithmExecutionError("Torch Worker Stage context has no identity")
    if retry is not None:
        descriptor = describe_torch_checkpoint(
            TorchCheckpointRef(retry),
            TorchCheckpointContext(
                stage=stage_context,
                run_id=identity.run_id,
                invocation_id=identity.invocation_id,
                checkpoint_owner="core",
            ),
        )
        validate_torch_retry_identity(
            descriptor,
            identity,
            world_size=stage_context.runtime.world_size,
        )
        return TorchWorkerCheckpointContext(
            stage=stage_context,
            source="ray_failure_retry",
            checkpoint=TorchCheckpointRef(
                retry,
                descriptor_digest=descriptor.digest,
                source_stage_id=descriptor.identity.stage_id,
                descriptor=descriptor,
            ),
        )
    control_value = config.get("core_control")
    if control_value is None:
        return TorchWorkerCheckpointContext(stage=stage_context, source="none")
    if not isinstance(control_value, Mapping):
        raise AlgorithmExecutionError("Torch Core control envelope is malformed")
    control = TorchWorkerControlEnvelope.from_dict(control_value)
    if (
        control.target_stage_id != stage_context.stage_id
        or control.run_id != identity.run_id
        or control.invocation_id != identity.invocation_id
        or control.policy_digest != stage_context.runtime.policy_digest
        or control.execution_plan_digest != stage_context.runtime.execution_plan_digest
    ):
        raise AlgorithmExecutionError("Torch Core control envelope identity mismatch")
    opener = config.get("_core_checkpoint_opener")
    if opener is None:
        opener_ref = config.get("_core_checkpoint_opener_ref")
        if isinstance(opener_ref, str):
            opener = _load_reference(QualifiedReference.parse(opener_ref))
    if not callable(opener):
        raise AlgorithmExecutionError(
            "Torch Core control envelope has no verified checkpoint opener"
        )
    checkpoint = opener(control.checkpoint_locator)
    descriptor = describe_torch_checkpoint(
        TorchCheckpointRef(checkpoint),
        object(),
    )
    _require_checkpoint_commit(checkpoint, descriptor)
    if descriptor.digest != control.checkpoint_descriptor_digest:
        raise AlgorithmExecutionError("Torch Core control descriptor mismatch")
    expected_binding = stage_context.runtime.input_binding_digest
    if (
        descriptor.policy_digest != stage_context.runtime.policy_digest
        or descriptor.execution_plan_digest
        != stage_context.runtime.execution_plan_digest
        or (
            expected_binding is not None
            and descriptor.input_binding_digest != expected_binding
        )
        or descriptor.identity.implementation_id
        != stage_context.runtime.implementation_id
        or descriptor.implementation_code_digest != identity.implementation_code_digest
        or descriptor.world_size != stage_context.runtime.world_size
        or descriptor.state_layout != stage_context.runtime.state_layout
    ):
        raise AlgorithmExecutionError("Torch Core control descriptor identity mismatch")
    if control.purpose == "stage_dependency" and (
        descriptor.identity.stage_id != control.source_stage_id
        or control.source_stage_id != stage_context.predecessor_stage_id
        or descriptor.identity.run_id != identity.run_id
        or descriptor.identity.invocation_id != identity.invocation_id
    ):
        raise AlgorithmExecutionError("Torch Core control source Stage mismatch")
    if (
        control.purpose == "cross_run_initial_recovery"
        and not descriptor.resume_supported
    ):
        raise AlgorithmExecutionError("Torch checkpoint is not externally recoverable")
    return TorchWorkerCheckpointContext(
        stage=stage_context,
        source=(
            "stage_dependency"
            if control.purpose == "stage_dependency"
            else "cross_run_initial_recovery"
        ),
        checkpoint=TorchCheckpointRef(
            checkpoint,
            descriptor_digest=descriptor.digest,
            source_stage_id=descriptor.identity.stage_id,
            descriptor=descriptor,
        ),
    )


def open_torch_checkpoint_locator(locator: TorchCheckpointLocator) -> object:
    """Core-owned locator opener hook; credentials are resolved by the runtime."""
    if not isinstance(locator, TorchCheckpointLocator):
        raise AlgorithmExecutionError("Torch checkpoint locator is invalid")
    if locator.uri.startswith("ray://"):
        from ray.train import Checkpoint

        path = Path(locator.uri.removeprefix("ray://"))
        if path.is_dir():
            try:
                return Checkpoint.from_directory(str(path))
            except (OSError, ValueError, TypeError) as exc:
                raise AlgorithmExecutionError(
                    "Torch checkpoint locator could not be opened"
                ) from exc
    if locator.uri.startswith(("s3://", "gs://", "gcs://", "hdfs://")):
        from ray.train import Checkpoint

        try:
            return Checkpoint(locator.uri)
        except (OSError, ValueError, TypeError) as exc:
            raise AlgorithmExecutionError(
                "Torch checkpoint locator could not be opened"
            ) from exc
    raise AlgorithmExecutionError(
        "Torch checkpoint locator requires a configured Core storage opener"
    )


def _persist_stage_checkpoint(
    checkpoint: object,
    *,
    identity: TorchStageRunIdentity,
    storage_path: object,
    descriptor_digest: str,
) -> str | None:
    """Persist a same-invocation Stage checkpoint without replacing prior data."""
    if not isinstance(storage_path, (str, Path)):
        return None
    raw_storage = str(storage_path)
    if "://" in raw_storage and not raw_storage.startswith("file://"):
        from urllib.parse import urlsplit

        opener = getattr(checkpoint, "as_directory", None)
        if not callable(opener):
            raise AlgorithmExecutionError(
                "Torch remote Stage checkpoint cannot be opened for Core persistence"
            )
        try:
            import pyarrow.fs as pafs

            parsed = urlsplit(raw_storage)
            filesystem, prefix = pafs.FileSystem.from_uri(raw_storage)
            remote_staging = (
                f"{prefix.rstrip('/')}/{identity.run_config_name}/"
                f".stage_checkpoint.staging-{descriptor_digest}"
            )
            commit_path = f"{remote_staging}/torch_stage_commit.json"
            commit_info = filesystem.get_file_info(commit_path)
            commit_payload = {
                "schema_version": 1,
                "identity": identity.to_dict(),
                "descriptor_digest": descriptor_digest,
            }
            if commit_info.type is pafs.FileType.File:
                try:
                    existing_commit = json.loads(
                        filesystem.open_input_file(commit_path).read().decode("utf-8")
                    )
                except (OSError, TypeError, ValueError) as exc:
                    raise AlgorithmExecutionError(
                        "Torch remote Stage checkpoint commit marker is malformed"
                    ) from exc
                if existing_commit != commit_payload:
                    raise AlgorithmExecutionError(
                        "Torch remote Stage checkpoint identity collision"
                    )
                return f"{parsed.scheme}://{parsed.netloc}/{remote_staging.lstrip('/')}"
            filesystem.create_dir(remote_staging, recursive=True)
            with opener() as source:
                source_path = Path(source)
                for source_file in source_path.rglob("*"):
                    if source_file.is_symlink():
                        raise AlgorithmExecutionError(
                            "Torch Stage checkpoint payload contains a symlink"
                        )
                    if not source_file.is_file():
                        continue
                    relative = source_file.relative_to(source_path).as_posix()
                    destination = f"{remote_staging.rstrip('/')}/{relative}"
                    with filesystem.open_output_stream(destination) as stream:
                        stream.write(source_file.read_bytes())
            with filesystem.open_output_stream(commit_path) as stream:
                stream.write(
                    (json.dumps(commit_payload, sort_keys=True) + "\n").encode()
                )
            if not parsed.scheme or not parsed.netloc:
                raise AlgorithmExecutionError("Torch remote storage URI is invalid")
            return f"{parsed.scheme}://{parsed.netloc}/{remote_staging.lstrip('/')}"
        except AlgorithmExecutionError:
            raise
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "failed to persist Torch Stage checkpoint in Core storage"
            ) from exc
    root = Path(storage_path)
    if not root.is_absolute():
        return None
    target = root / identity.run_config_name / "stage_checkpoint"
    commit_path_local = target / "torch_stage_commit.json"
    commit_payload = {
        "schema_version": 1,
        "identity": identity.to_dict(),
        "descriptor_digest": descriptor_digest,
    }
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise AlgorithmExecutionError(
                "Torch Stage checkpoint destination is not a directory"
            )
        if not commit_path_local.is_file() or commit_path_local.is_symlink():
            raise AlgorithmExecutionError(
                "Torch Stage checkpoint destination is a partial or uncommitted snapshot"
            )
        try:
            if (
                json.loads(commit_path_local.read_text(encoding="utf-8"))
                != commit_payload
            ):
                raise AlgorithmExecutionError(
                    "Torch Stage checkpoint identity collision"
                )
        except (OSError, TypeError, ValueError) as exc:
            raise AlgorithmExecutionError(
                "Torch Stage checkpoint commit marker is malformed"
            ) from exc
        return f"ray://{target}"
    opener = getattr(checkpoint, "as_directory", None)
    if not callable(opener):
        return None
    temporary_target = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-{descriptor_digest}-",
            dir=target.parent,
        )
    )
    with opener() as source:
        source_path = Path(source)
        for source_file in source_path.rglob("*"):
            if source_file.is_symlink():
                raise AlgorithmExecutionError(
                    "Torch Stage checkpoint payload contains a symlink"
                )
            if not source_file.is_file():
                continue
            relative_local = source_file.relative_to(source_path)
            destination_local = temporary_target / relative_local
            destination_local.parent.mkdir(parents=True, exist_ok=True)
            destination_local.write_bytes(source_file.read_bytes())
        (temporary_target / "torch_stage_commit.json").write_text(
            json.dumps(commit_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    os.replace(temporary_target, target)
    return f"ray://{target}"


def _recipe_worker(config: Mapping[str, Any]) -> None:
    """Run a minimal Core-owned Recipe loop in a Ray Train worker."""
    import ray.train
    import torch
    from ray.train.torch import prepare_model

    recipe_ref = config.get("_core_implementation_ref")
    if not isinstance(recipe_ref, str):
        raise AlgorithmConfigurationError(
            "Core Worker implementation reference is missing"
        )
    recipe = _load_reference(QualifiedReference.parse(recipe_ref))
    if not isinstance(recipe, type) or not issubclass(recipe, TorchRecipe):
        raise AlgorithmConfigurationError("Core Worker did not receive a TorchRecipe")
    recipe_instance = recipe()
    stage_context_value = config.get("_core_stage_context")
    if not isinstance(stage_context_value, Mapping):
        raise AlgorithmConfigurationError("Core Worker stage context is missing")
    stage_context = TorchStageContext.from_dict(stage_context_value)
    runtime_context = stage_context.runtime
    modules = recipe_instance.build_modules(
        TorchBuildContext(runtime=runtime_context, stage=stage_context)
    )
    if isinstance(modules, TorchModuleSet):
        module_set = modules
    elif isinstance(modules, Mapping):
        module_set = TorchModuleSet(modules)
    else:
        raise AlgorithmConfigurationError(
            "TorchRecipe.build_modules must return TorchModuleSet"
        )
    model = module_set["model"]
    if not isinstance(model, torch.nn.Module):
        raise AlgorithmConfigurationError("TorchRecipe model must be torch.nn.Module")
    optimization = recipe_instance.configure_optimizers(
        module_set,
        TorchBuildContext(runtime=runtime_context, stage=stage_context),
    )
    if not isinstance(optimization, TorchOptimizationPlan):
        raise AlgorithmConfigurationError(
            "TorchRecipe.configure_optimizers returned invalid plan"
        )
    model = cast(Any, prepare_model(model))
    module_set = TorchModuleSet({**module_set.modules, "model": model})
    optimizer = cast(Any, optimization.optimizer)
    optimizer.zero_grad()
    stage_roles = tuple(stage_context.input_roles)
    if not stage_roles:
        raise AlgorithmConfigurationError("TorchRecipe Stage has no input roles")
    role_shards: dict[str, object] = {}
    for role in stage_roles:
        try:
            shard = ray.train.get_dataset_shard(role)
        except KeyError:
            shard = None
        if shard is not None:
            role_shards[role] = shard
    if stage_roles[0] not in role_shards:
        raise AlgorithmConfigurationError(
            f"TorchRecipe requires the {stage_roles[0]!r} dataset"
        )
    train = role_shards[stage_roles[0]]
    training_roles = tuple(role for role in stage_roles if role not in {"val", "test"})
    multi_role = len(training_roles) > 1
    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    training = config.get("training", {})
    if not isinstance(training, Mapping):
        raise AlgorithmConfigurationError("Torch training config must be a mapping")
    epochs = training.get("epochs", 1)
    shuffle = training.get("shuffle", False)
    if not isinstance(shuffle, bool):
        raise AlgorithmConfigurationError("Torch training shuffle must be boolean")
    if shuffle:
        raise AlgorithmConfigurationError(
            "Torch Runtime v1 requires an unshuffled Ray Dataset for exact recovery"
        )
    batch_size = training.get("batch_size", config.get("_core_batch_size", 32))
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise AlgorithmConfigurationError("Torch training epochs must be positive")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise AlgorithmConfigurationError("Torch training batch_size must be positive")
    amp = bool(training.get("amp", False))
    if amp and not torch.cuda.is_available():
        raise AlgorithmConfigurationError("Torch AMP requires CUDA")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    seed = training.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AlgorithmConfigurationError("Torch training seed must be an integer")
    torch.manual_seed(seed + rank)
    scheduler = optimization.scheduler
    accumulation = optimization.gradient_accumulation_steps
    checkpoint_context = _select_worker_checkpoint(config, stage_context)
    restored_progress: dict[str, object] = {}
    try:
        loaded_step = _restore_torch_retry_checkpoint(
            checkpoint_context.checkpoint.checkpoint
            if checkpoint_context.checkpoint is not None
            else None,
            stage_context=stage_context,
            model=model,
            optimizer=optimization.optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=rank,
            strict_identity=checkpoint_context.source == "ray_failure_retry",
            progress_sink=restored_progress,
            expected_accumulation=(
                accumulation
                if checkpoint_context.source
                in {"ray_failure_retry", "cross_run_initial_recovery"}
                else None
            ),
        )
        restored_step = (
            loaded_step
            if checkpoint_context.source
            in {"ray_failure_retry", "cross_run_initial_recovery"}
            else 0
        )
    finally:
        if checkpoint_context.checkpoint is not None:
            checkpoint_context.checkpoint.close()
    restore_same_stage = checkpoint_context.source in {
        "ray_failure_retry",
        "cross_run_initial_recovery",
    }
    typed_progress = restored_progress.get("_typed_progress")
    if restore_same_stage and not isinstance(typed_progress, TorchCheckpointProgress):
        raise AlgorithmExecutionError("Torch checkpoint progress is not typed")
    restored_checkpoint_progress = (
        cast(TorchCheckpointProgress, typed_progress) if restore_same_stage else None
    )
    rank_statistics = (
        restored_checkpoint_progress.rank_statistics.get(str(rank))
        if restored_checkpoint_progress is not None
        else None
    )
    if restore_same_stage and not isinstance(
        rank_statistics, TorchRankProgressStatistics
    ):
        raise AlgorithmExecutionError("Torch checkpoint rank statistics are missing")

    rows = rank_statistics.rows_processed if rank_statistics is not None else 0
    steps = 0
    coverage_totals = (
        dict(rank_statistics.coverage_totals) if rank_statistics is not None else {}
    )
    loss_numerator_total = (
        rank_statistics.loss_numerator_total if rank_statistics is not None else 0.0
    )
    loss_normalizer_total = (
        rank_statistics.loss_normalizer_total if rank_statistics is not None else 0.0
    )
    metric_totals = (
        {name: list(pair) for name, pair in rank_statistics.metric_totals.items()}
        if rank_statistics is not None
        else {}
    )
    evaluation_totals = (
        {name: list(pair) for name, pair in rank_statistics.evaluation_totals.items()}
        if rank_statistics is not None
        else {}
    )
    reducer_observation: dict[str, object] = (
        dict(rank_statistics.reducer_observation) if rank_statistics is not None else {}
    )
    composite_loss_seen = False
    batch_context = TorchBatchContext(
        stage=stage_context,
        input_roles=stage_roles,
        feature_names=tuple(
            name
            for name in config.get("_core_feature_names", ())
            if isinstance(name, str)
        ),
        label_name=(
            config.get("_core_label_name")
            if isinstance(config.get("_core_label_name"), str)
            else None
        ),
        weight_name=(
            config.get("_core_weight_name")
            if isinstance(config.get("_core_weight_name"), str)
            else None
        ),
    )
    checkpoint_interval = config.get("_core_checkpoint_interval_windows", 1)
    if (
        not isinstance(checkpoint_interval, int)
        or isinstance(checkpoint_interval, bool)
        or checkpoint_interval < 1
    ):
        raise AlgorithmConfigurationError(
            "Torch checkpoint interval must be a positive integer"
        )
    import torch.distributed as dist

    def emit_checkpoint(
        completed_step: int,
        *,
        epoch: int,
        micro_batch_cursor: int,
        scheduler_step: int,
        rows_processed: int,
        coverage_totals: Mapping[str, int],
        loss_numerator_total: float,
        loss_normalizer_total: float,
        metric_totals: Mapping[str, list[float]],
        evaluation_totals: Mapping[str, list[float]],
        reducer_observation: Mapping[str, object],
    ) -> None:
        """Report an optimizer-boundary checkpoint for Ray failure retry."""
        from ray.train import Checkpoint

        identity = runtime_context.run_identity
        if identity is None:
            raise AlgorithmExecutionError("Torch Worker stage context has no identity")
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="tributo_torch_checkpoint_"))
        try:
            target_model = getattr(model, "module", model)
            torch.save(target_model.state_dict(), checkpoint_dir / "model.pt")
            torch.save(
                cast(Any, optimizer).state_dict(), checkpoint_dir / "optimizer.pt"
            )
            torch.save(cast(Any, scaler).state_dict(), checkpoint_dir / "scaler.pt")
            if scheduler is not None:
                torch.save(
                    cast(Any, scheduler).state_dict(), checkpoint_dir / "scheduler.pt"
                )
            cursor_by_rank: list[object] = [None] * world_size
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(cursor_by_rank, micro_batch_cursor)
            else:
                cursor_by_rank = [micro_batch_cursor]
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in cursor_by_rank
            ):
                raise AlgorithmExecutionError(
                    "Torch dataset cursor collective is incomplete"
                )
            local_statistics = TorchRankProgressStatistics(
                rows_processed=rows_processed,
                coverage_totals=coverage_totals,
                loss_numerator_total=loss_numerator_total,
                loss_normalizer_total=loss_normalizer_total,
                metric_totals={
                    name: (values[0], values[1])
                    for name, values in metric_totals.items()
                },
                evaluation_totals={
                    name: (values[0], values[1])
                    for name, values in evaluation_totals.items()
                },
                reducer_observation=reducer_observation,
            )
            statistics_by_rank: list[object] = [None] * world_size
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(statistics_by_rank, local_statistics.to_dict())
            else:
                statistics_by_rank = [local_statistics.to_dict()]
            if any(not isinstance(value, Mapping) for value in statistics_by_rank):
                raise AlgorithmExecutionError(
                    "Torch checkpoint statistics collective is incomplete"
                )
            progress = TorchCheckpointProgress(
                epoch=epoch,
                micro_batch_cursor=micro_batch_cursor,
                optimizer_step=completed_step,
                scheduler_step=scheduler_step,
                accumulation_steps=accumulation,
                dataset_cursor_by_rank={
                    str(rank_id): cast(int, cursor)
                    for rank_id, cursor in enumerate(cursor_by_rank)
                },
                shuffle_seed=int(seed + epoch),
                rows_processed=rows_processed,
                coverage_totals=coverage_totals,
                loss_numerator_total=loss_numerator_total,
                loss_normalizer_total=loss_normalizer_total,
                metric_totals={
                    name: (values[0], values[1])
                    for name, values in metric_totals.items()
                },
                evaluation_totals={
                    name: (values[0], values[1])
                    for name, values in evaluation_totals.items()
                },
                rank_statistics={
                    str(rank_id): TorchRankProgressStatistics.from_dict(
                        cast(Mapping[str, Any], stats)
                    )
                    for rank_id, stats in enumerate(statistics_by_rank)
                },
                epoch_scheduler_applied=False,
            )
            (checkpoint_dir / "torch_progress.json").write_text(
                json.dumps(progress.to_dict(), sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            cpu_rng = torch.get_rng_state().cpu().numpy().tobytes()
            cpu_states: list[bytes | None] = [None] * world_size
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(cpu_states, cpu_rng)
            else:
                cpu_states = [cpu_rng]
            if any(not isinstance(value, bytes) for value in cpu_states):
                raise AlgorithmExecutionError(
                    "Torch RNG state collective is incomplete"
                )
            cuda_states_by_rank: list[list[bytes]] = []
            if torch.cuda.is_available():
                local_cuda = [
                    state.cpu().numpy().tobytes()
                    for state in torch.cuda.get_rng_state_all()
                ]
                gathered_cuda: list[object] = [None] * world_size
                if dist.is_available() and dist.is_initialized():
                    dist.all_gather_object(gathered_cuda, local_cuda)
                    if any(
                        not isinstance(value, list)
                        or any(not isinstance(state, bytes) for state in value)
                        for value in gathered_cuda
                    ):
                        raise AlgorithmExecutionError(
                            "Torch CUDA RNG state collective is incomplete"
                        )
                    cuda_states_by_rank = cast(list[list[bytes]], gathered_cuda)
                else:
                    cuda_states_by_rank = [local_cuda]
            torch.save(
                {
                    "world_size": world_size,
                    "states": cast(list[bytes], cpu_states),
                    "cuda_states_by_rank": cuda_states_by_rank,
                },
                checkpoint_dir / "rng_state.pt",
            )
            payload_names = [
                "model.pt",
                "optimizer.pt",
                "scaler.pt",
                "rng_state.pt",
                "torch_progress.json",
            ]
            if scheduler is not None:
                payload_names.append("scheduler.pt")
            descriptor = TorchCheckpointDescriptor(
                schema_version=1,
                identity=identity,
                run_config_name=identity.run_config_name,
                state_layout=str(config.get("_core_state_layout", "replicated")),
                world_size=world_size,
                completed_step=completed_step,
                policy_digest=runtime_context.policy_digest,
                execution_plan_digest=runtime_context.execution_plan_digest,
                input_binding_digest=str(config.get("_core_input_binding_digest", "")),
                implementation_code_digest=str(
                    config.get("_core_implementation_code_digest", "")
                ),
                payload_files={
                    name: hashlib.sha256(
                        (checkpoint_dir / name).read_bytes()
                    ).hexdigest()
                    for name in payload_names
                },
                adapter_identity=config.get("_core_adapter_identity"),
                resume_supported=runtime_context.resume_supported,
                same_world_size_resume=runtime_context.same_world_size_resume,
            )

            class _IntervalDraft:
                checkpoint_dir: str | os.PathLike[str]

                def __init__(self) -> None:
                    self.checkpoint_dir = str(checkpoint_dir)

                def report(
                    self,
                    *,
                    metrics: Mapping[str, object],
                    stage_context: object,
                    completed_step: int,
                ) -> None:
                    del stage_context, completed_step
                    checkpoint = (
                        Checkpoint.from_directory(str(checkpoint_dir))
                        if rank == int(config.get("_core_checkpoint_owner_rank", 0))
                        else None
                    )
                    ray.train.report(dict(metrics), checkpoint=checkpoint)

            report_torch_checkpoint(
                {
                    "train_loss": (
                        loss_numerator_total / loss_normalizer_total
                        if loss_normalizer_total > 0
                        else 0.0
                    ),
                    "model_state_digest": hashlib.sha256(
                        json.dumps(
                            {
                                name: str(
                                    cast(Any, value).detach().cpu().numpy().tobytes()
                                )
                                for name, value in target_model.state_dict().items()
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    "checkpoint_descriptor": descriptor.to_dict(),
                },
                _IntervalDraft(),
                stage_context,
                completed_step,
            )
        finally:
            import shutil

            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def reduce_window(value: float) -> float:
        tensor = torch.tensor(
            value, dtype=torch.float64, device=next(model.parameters()).device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def zero_batch(template: TorchBatch) -> TorchBatch:
        def zeros(value: object) -> object:
            if not isinstance(value, torch.Tensor):
                return value
            # Preserve the typed batch signature while making every tensor
            # genuinely zero-row.  This prevents empty ranks from producing
            # synthetic loss/metric contributions in Recipe or reducer code.
            if value.ndim > 0:
                return value[:0]
            return torch.zeros_like(value)

        return TorchBatch(
            positional=tuple(zeros(value) for value in template.positional),
            keyword={name: zeros(value) for name, value in template.keyword.items()},
            targets=zeros(template.targets) if template.targets is not None else None,
            weights=zeros(template.weights) if template.weights is not None else None,
            local_rows=0,
            coverage_counts=dict.fromkeys(template.coverage_counts, 0),
        )

    def aligned_metric_contributions(
        contributions: Mapping[str, TorchMetricContribution],
    ) -> dict[str, TorchMetricContribution]:
        """Make metric keys identical before any later rank-wise reduction."""
        for _name, contribution in contributions.items():
            if not isinstance(contribution, TorchMetricContribution):
                raise AlgorithmConfigurationError(
                    "TorchStepResult metrics must be TorchMetricContribution values"
                )
        names = set(contributions)
        if dist.is_available() and dist.is_initialized():
            gathered: list[object] = [None] * world_size
            dist.all_gather_object(gathered, sorted(names))
            for value in gathered:
                if not isinstance(value, list) or any(
                    not isinstance(name, str) for name in value
                ):
                    raise AlgorithmExecutionError(
                        "Torch metric key collective is incomplete"
                    )
                names.update(value)
        return {
            name: contributions.get(name, TorchMetricContribution(0.0, 0.0))
            for name in sorted(names)
        }

    raw_metric_mapping = config.get("_core_metric_mapping", {})
    metric_mapping = (
        {str(key): str(value) for key, value in raw_metric_mapping.items()}
        if isinstance(raw_metric_mapping, Mapping)
        else {}
    )
    if len(set(metric_mapping.values())) != len(metric_mapping):
        raise AlgorithmConfigurationError("Torch metric mapping targets must be unique")
    raw_metric_reducers = config.get("_core_metric_reducers", {})
    if not isinstance(raw_metric_reducers, Mapping):
        raise AlgorithmConfigurationError("Torch metric reducers must be a mapping")
    declared_metric_names = {str(name) for name in raw_metric_reducers} - {"train_loss"}
    expected_metric_names = frozenset(
        {
            source
            for source, target in metric_mapping.items()
            if target in declared_metric_names
        }
        | {
            name
            for name in declared_metric_names
            if name not in metric_mapping.values()
        }
    )
    restored_epoch = (
        restored_checkpoint_progress.epoch
        if restored_checkpoint_progress is not None
        else 0
    )
    remaining_skip_micro_batches = (
        restored_checkpoint_progress.dataset_cursor_by_rank[str(rank)]
        if restored_checkpoint_progress is not None
        else 0
    )
    scheduler_steps = (
        restored_checkpoint_progress.scheduler_step
        if restored_checkpoint_progress is not None
        else 0
    )
    restored_epoch_scheduler_applied = (
        restored_checkpoint_progress.epoch_scheduler_applied
        if restored_checkpoint_progress is not None
        else False
    )
    if restore_same_stage and restored_epoch >= epochs:
        raise AlgorithmExecutionError(
            "Torch checkpoint progress points past the configured epoch range"
        )
    if (
        checkpoint_context.source
        in {
            "ray_failure_retry",
            "cross_run_initial_recovery",
        }
        and restored_progress.get("shuffle_seed") != seed + restored_epoch
    ):
        raise AlgorithmExecutionError(
            "Torch checkpoint shuffle state does not match the current seed"
        )

    def map_metric_contributions(
        contributions: Mapping[str, TorchMetricContribution],
    ) -> dict[str, TorchMetricContribution]:
        return {
            metric_mapping.get(name, name): contribution
            for name, contribution in contributions.items()
        }

    epoch_micro_batch_cursor = 0
    for _epoch in range(epochs):
        if _epoch < restored_epoch:
            continue
        epoch_micro_batch_cursor = (
            remaining_skip_micro_batches if _epoch == restored_epoch else 0
        )
        next_payload: Any
        if multi_role:
            role_iterators = {
                role: iter(
                    cast(Any, shard).iter_torch_batches(
                        batch_size=batch_size, drop_last=False
                    )
                )
                for role, shard in role_shards.items()
                if role in training_roles
            }

            def _next_multi_payload(
                role_iterators: Mapping[str, Any] = role_iterators,
            ) -> object | None:
                payload = {
                    role: next(iterator, None)
                    for role, iterator in role_iterators.items()
                }
                return (
                    payload
                    if any(value is not None for value in payload.values())
                    else None
                )

            next_payload = _next_multi_payload

        else:
            iterator = iter(
                cast(Any, train).iter_torch_batches(
                    batch_size=batch_size, drop_last=False
                )
            )

            def _next_single_payload(iterator: Any = iterator) -> object | None:
                return next(iterator, None)

            next_payload = _next_single_payload

        raw = next_payload()
        first_active = torch.tensor(
            1 if raw is not None else 0,
            dtype=torch.int64,
            device=next(model.parameters()).device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(first_active, op=dist.ReduceOp.SUM)
        if int(first_active.item()) == 0:
            continue
        if remaining_skip_micro_batches == 0 and int(first_active.item()) != world_size:
            raise AlgorithmExecutionError(
                "TorchRecipe requires at least one batch on every rank"
            )
        template = recipe_instance.adapt_batch(raw, batch_context)
        if not isinstance(template, TorchBatch):
            raise AlgorithmConfigurationError(
                "TorchRecipe.adapt_batch must return TorchBatch"
            )
        window = TorchAccumulationWindow(
            index=restored_step + steps // accumulation,
            expected_micro_batches=accumulation,
        )
        while True:
            active = raw is not None
            active_tensor = torch.tensor(
                1 if active else 0,
                dtype=torch.int64,
                device=next(model.parameters()).device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(active_tensor, op=dist.ReduceOp.SUM)
            if int(active_tensor.item()) == 0:
                break
            next_raw = next_payload()
            if remaining_skip_micro_batches > 0:
                remaining_skip_micro_batches -= 1
                epoch_micro_batch_cursor += 1
                raw = next_raw
                continue
            next_active = torch.tensor(
                1 if next_raw is not None else 0,
                dtype=torch.int64,
                device=next(model.parameters()).device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(next_active, op=dist.ReduceOp.SUM)
            expected_micro_batches = window.expected_micro_batches
            if (
                int(next_active.item()) == 0
                and window.observed_micro_batches + 1 < expected_micro_batches
            ):
                expected_micro_batches = window.observed_micro_batches + 1
                window = TorchAccumulationWindow(
                    index=window.index,
                    expected_micro_batches=expected_micro_batches,
                    observed_micro_batches=window.observed_micro_batches,
                    normalizer_total=window.normalizer_total,
                )
            batch = (
                recipe_instance.adapt_batch(raw, batch_context)
                if active
                else zero_batch(template)
            )
            if not isinstance(batch, TorchBatch):
                raise AlgorithmConfigurationError(
                    "TorchRecipe.adapt_batch must return TorchBatch"
                )
            is_boundary = (
                window.observed_micro_batches + 1 == window.expected_micro_batches
            )
            sync_context = (
                model.no_sync()
                if hasattr(model, "no_sync") and not is_boundary
                else nullcontext()
            )
            with (
                sync_context,
                torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp),
            ):
                # Every rank invokes the typed step, including a synchronized
                # zero-contribution batch after its shard is exhausted.  This
                # keeps composite reducer collectives aligned across ranks.
                step = recipe_instance.training_step(
                    module_set,
                    batch,
                    TorchStepContext(
                        stage=stage_context,
                        window_index=restored_step + steps // accumulation,
                        micro_batch_index=steps % accumulation,
                    ),
                )
                if not isinstance(step, TorchStepResult):
                    raise AlgorithmConfigurationError(
                        "TorchRecipe.training_step returned invalid result"
                    )
                for name, count in step.coverage_counts.items():
                    if name in stage_roles:
                        raise AlgorithmConfigurationError(
                            "Torch coverage_counts cannot override Core role rows"
                        )
                    coverage_totals[name] = coverage_totals.get(name, 0) + count
                is_composite = isinstance(step.loss, TorchCompositeLossContribution)
                if is_composite and accumulation != 1:
                    raise AlgorithmConfigurationError(
                        "Core TorchRecipe composite loss requires accumulation_steps=1"
                    )
                if is_composite:
                    composite_loss_seen = True
                if isinstance(step.loss, TorchCompositeLossContribution):
                    normalizer = sum(
                        float(value)
                        for value in step.loss.normalizer_components.values()
                    )
                elif isinstance(step.loss, TorchLossContribution):
                    numerator = step.loss.numerator
                    normalizer = float(step.loss.normalizer)
                    loss_numerator_total += float(cast(Any, numerator).detach().item())
                    loss_normalizer_total += normalizer
                else:
                    raise AlgorithmConfigurationError(
                        "TorchRecipe loss contribution is invalid"
                    )
                for metric_name, contribution in aligned_metric_contributions(
                    map_metric_contributions(step.metrics)
                ).items():
                    totals = metric_totals.setdefault(metric_name, [0.0, 0.0])
                    totals[0] += contribution.numerator
                    totals[1] += contribution.normalizer
                mapped_step_metrics = map_metric_contributions(step.metrics)
                if (
                    not isinstance(step.loss, TorchCompositeLossContribution)
                    and set(mapped_step_metrics) != declared_metric_names
                ):
                    raise AlgorithmExecutionError(
                        "TorchStepResult.metrics do not match TorchMetricPlan"
                    )
                backward_context = TorchBackwardContext(
                    world_size=world_size,
                    backward=lambda value: scaler.scale(value).backward(),
                    reduce_normalizer=reduce_window,
                    reduce_window_normalizer=reduce_window,
                    compose_composite=(
                        lambda value: _composite_backward(
                            value,
                            config=config,
                            world_size=world_size,
                            device=next(model.parameters()).device,
                            dist=dist,
                            observation=reducer_observation,
                            metric_totals=metric_totals,
                            expected_metrics=expected_metric_names,
                        )
                    ),
                    finalize_window=lambda scale: _finalize_torch_window(
                        scaler=scaler,
                        optimizer=optimizer,
                        model=model,
                        max_gradient_norm=optimization.max_gradient_norm,
                        scale=scale,
                    ),
                )
                # The public helper owns normalizer accumulation and the
                # optimizer-window boundary; the callback owns only Torch's
                # unscale/clip/step sequence.
                result = apply_torch_loss_backward(
                    step.loss,
                    window,
                    backward_context,
                )
            if result.window_complete:
                window = TorchAccumulationWindow(
                    index=window.index + 1,
                    expected_micro_batches=accumulation,
                )
            else:
                window = window.add(result.local_normalizer)
            rows += batch.local_rows
            steps += 1
            epoch_micro_batch_cursor += 1
            if (
                result.window_complete
                and (steps // accumulation) % checkpoint_interval == 0
            ):
                emit_checkpoint(
                    restored_step + steps // accumulation,
                    epoch=_epoch,
                    micro_batch_cursor=epoch_micro_batch_cursor,
                    scheduler_step=scheduler_steps,
                    rows_processed=rows,
                    coverage_totals=coverage_totals,
                    loss_numerator_total=loss_numerator_total,
                    loss_normalizer_total=loss_normalizer_total,
                    metric_totals=metric_totals,
                    evaluation_totals=evaluation_totals,
                    reducer_observation=reducer_observation,
                )
            raw = next_raw
        if optimization.scheduler is not None and _should_apply_epoch_scheduler(
            restore_same_stage=restore_same_stage,
            epoch=_epoch,
            restored_epoch=restored_epoch,
            restored_epoch_scheduler_applied=restored_epoch_scheduler_applied,
        ):
            scheduler_step = getattr(optimization.scheduler, "step", None)
            if not callable(scheduler_step):
                raise AlgorithmConfigurationError(
                    "Torch scheduler must implement step()"
                )
            scheduler_step()
            scheduler_steps += 1
    for split in ("val", "test"):
        try:
            evaluation_data = ray.train.get_dataset_shard(split)
        except KeyError:
            evaluation_data = None
        if evaluation_data is None:
            continue
        iterator = iter(
            evaluation_data.iter_torch_batches(batch_size=batch_size, drop_last=False)
        )
        raw = next(iterator, None)
        active = torch.tensor(
            1 if raw is not None else 0,
            dtype=torch.int64,
            device=next(model.parameters()).device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(active, op=dist.ReduceOp.SUM)
        if int(active.item()) == 0:
            continue
        template_raw: object | None = raw
        if dist.is_available() and dist.is_initialized():
            templates: list[object | None] = [None] * world_size
            dist.all_gather_object(templates, raw)
            template_raw = next(
                (candidate for candidate in templates if candidate is not None),
                None,
            )
        if template_raw is None:
            raise AlgorithmExecutionError(
                f"Torch {split} evaluation has no typed batch template"
            )
        template_batch = recipe_instance.adapt_batch(template_raw, batch_context)
        if not isinstance(template_batch, TorchBatch):
            raise AlgorithmConfigurationError(
                "TorchRecipe evaluation batch template is invalid"
            )
        while True:
            local_active = raw is not None
            active = torch.tensor(
                1 if local_active else 0,
                dtype=torch.int64,
                device=next(model.parameters()).device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(active, op=dist.ReduceOp.SUM)
            if int(active.item()) == 0:
                break
            next_raw = next(iterator, None) if local_active else None
            batch = (
                recipe_instance.adapt_batch(raw, batch_context)
                if local_active
                else zero_batch(template_batch)
            )
            if not isinstance(batch, TorchBatch):
                raise AlgorithmConfigurationError(
                    "TorchRecipe evaluation batch is invalid"
                )
            with torch.no_grad():
                validation = recipe_instance.validation_step(
                    module_set,
                    batch,
                    TorchStepContext(
                        stage=stage_context, window_index=0, micro_batch_index=0
                    ),
                )
            if not isinstance(validation, TorchStepResult):
                raise AlgorithmConfigurationError(
                    "TorchRecipe.validation_step returned invalid result"
                )
            mapped_validation_metrics = map_metric_contributions(validation.metrics)
            if (
                not isinstance(validation.loss, TorchCompositeLossContribution)
                and set(mapped_validation_metrics) != declared_metric_names
            ):
                raise AlgorithmExecutionError(
                    "TorchRecipe.validation_step.metrics do not match TorchMetricPlan"
                )
            for metric_name, contribution in aligned_metric_contributions(
                mapped_validation_metrics
            ).items():
                _accumulate_metric_totals(
                    evaluation_totals,
                    {metric_name: contribution},
                    prefix=f"{split}_",
                )
            if isinstance(validation.loss, TorchLossContribution):
                totals = evaluation_totals.setdefault(f"{split}_loss", [0.0, 0.0])
                totals[0] += float(cast(Any, validation.loss.numerator).detach().item())
                totals[1] += float(validation.loss.normalizer)
            elif isinstance(validation.loss, TorchCompositeLossContribution):
                reduction = _reduce_composite_loss(
                    validation.loss,
                    config=config,
                    world_size=world_size,
                    device=next(model.parameters()).device,
                    dist=dist,
                    observation=reducer_observation,
                    expected_metrics=expected_metric_names,
                )
                evaluation_metrics = {
                    name: contribution
                    for name, contribution in reduction.metrics.items()
                    if name != "train_loss"
                }
                _accumulate_metric_totals(
                    evaluation_totals,
                    evaluation_metrics,
                    prefix=f"{split}_",
                )
                # The generic reducer's objective metric is reported under
                # the established ``val_loss``/``test_loss`` name.
                objective = reduction.metrics.get("train_loss")
                if objective is not None:
                    _accumulate_metric_totals(
                        evaluation_totals,
                        {"loss": objective},
                        prefix=f"{split}_",
                    )
            else:
                raise AlgorithmExecutionError(
                    "TorchRecipe evaluation loss contribution is invalid"
                )
            raw = next_raw
    state_digest = hashlib.sha256(
        json.dumps(
            {
                name: str(cast(Any, value).detach().cpu().numpy().tobytes())
                for name, value in model.state_dict().items()
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assigned = ray.get_runtime_context().get_assigned_resources()
    resources = {
        "num_cpus": float(assigned.get("CPU", assigned.get("cpu", 1.0))),
        "num_gpus": float(assigned.get("GPU", assigned.get("gpu", 0.0))),
        "custom": {
            str(name): float(value)
            for name, value in assigned.items()
            if str(name).upper() not in {"CPU", "GPU"}
        },
    }
    node_id = ray.get_runtime_context().get_node_id()
    input_rows_evidence = dict(coverage_totals)
    input_rows_evidence[stage_roles[0]] = rows
    execution_worker = {
        "worker_id": f"torch-{rank}",
        "node_id": str(node_id),
        "rank": rank,
        "world_size": world_size,
        "shard_id": f"train-{rank}",
        "resources": resources,
        "model_state_digest": state_digest,
        "rows_processed": rows,
        "input_rows": input_rows_evidence,
        "batch_count": steps,
        "collective_steps": steps,
    }
    worker_records_raw: list[dict[str, object] | None] = [None] * world_size
    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(worker_records_raw, execution_worker)
    else:
        worker_records_raw = [execution_worker]
    if any(not isinstance(item, Mapping) for item in worker_records_raw):
        raise AlgorithmExecutionError("Torch worker evidence collective is incomplete")
    worker_records = [cast(dict[str, object], item) for item in worker_records_raw]
    from ray.train import Checkpoint

    checkpoint_dir = Path(tempfile.mkdtemp(prefix="tributo_torch_checkpoint_"))
    try:
        torch.save(
            getattr(model, "module", model).state_dict(), checkpoint_dir / "model.pt"
        )
        torch.save(cast(Any, optimizer).state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(cast(Any, scaler).state_dict(), checkpoint_dir / "scaler.pt")
        if optimization.scheduler is not None:
            torch.save(
                cast(Any, optimization.scheduler).state_dict(),
                checkpoint_dir / "scheduler.pt",
            )
        final_cursor_values: list[object] = [None] * world_size
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(final_cursor_values, epoch_micro_batch_cursor)
        else:
            final_cursor_values = [epoch_micro_batch_cursor]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in final_cursor_values
        ):
            raise AlgorithmExecutionError("Torch final cursor collective is incomplete")
        final_local_statistics = TorchRankProgressStatistics(
            rows_processed=rows,
            coverage_totals=coverage_totals,
            loss_numerator_total=loss_numerator_total,
            loss_normalizer_total=loss_normalizer_total,
            metric_totals={
                name: (values[0], values[1]) for name, values in metric_totals.items()
            },
            evaluation_totals={
                name: (values[0], values[1])
                for name, values in evaluation_totals.items()
            },
            reducer_observation=reducer_observation,
        )
        final_statistics_by_rank: list[object] = [None] * world_size
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(
                final_statistics_by_rank, final_local_statistics.to_dict()
            )
        else:
            final_statistics_by_rank = [final_local_statistics.to_dict()]
        if any(not isinstance(value, Mapping) for value in final_statistics_by_rank):
            raise AlgorithmExecutionError(
                "Torch final statistics collective is incomplete"
            )
        final_progress = TorchCheckpointProgress(
            epoch=max(epochs - 1, 0),
            micro_batch_cursor=epoch_micro_batch_cursor,
            optimizer_step=restored_step + steps // accumulation,
            scheduler_step=scheduler_steps,
            accumulation_steps=accumulation,
            dataset_cursor_by_rank={
                str(rank_id): cast(int, cursor)
                for rank_id, cursor in enumerate(final_cursor_values)
            },
            shuffle_seed=int(seed + max(epochs - 1, 0)),
            rows_processed=rows,
            coverage_totals=coverage_totals,
            loss_numerator_total=loss_numerator_total,
            loss_normalizer_total=loss_normalizer_total,
            metric_totals={
                name: (values[0], values[1]) for name, values in metric_totals.items()
            },
            evaluation_totals={
                name: (values[0], values[1])
                for name, values in evaluation_totals.items()
            },
            rank_statistics={
                str(rank_id): TorchRankProgressStatistics.from_dict(
                    cast(Mapping[str, Any], stats)
                )
                for rank_id, stats in enumerate(final_statistics_by_rank)
            },
            epoch_scheduler_applied=True,
        )
        (checkpoint_dir / "torch_progress.json").write_text(
            json.dumps(final_progress.to_dict(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        rng_payload = torch.get_rng_state().cpu().numpy().tobytes()
        rng_states: list[bytes | None] = [None] * world_size
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(rng_states, rng_payload)
        else:
            rng_states = [rng_payload]
        if any(not isinstance(value, bytes) for value in rng_states):
            raise AlgorithmExecutionError("Torch RNG state collective is incomplete")
        cuda_rng_payload: list[list[bytes]] = []
        if torch.cuda.is_available():
            local_cuda = [
                state.cpu().numpy().tobytes()
                for state in torch.cuda.get_rng_state_all()
            ]
            if dist.is_available() and dist.is_initialized():
                all_cuda: list[object] = [None] * world_size
                dist.all_gather_object(all_cuda, local_cuda)
                if any(
                    not isinstance(value, list)
                    or any(not isinstance(state, bytes) for state in value)
                    for value in all_cuda
                ):
                    raise AlgorithmExecutionError(
                        "Torch CUDA RNG state collective is incomplete"
                    )
                cuda_rng_payload = cast(list[list[bytes]], all_cuda)
            else:
                cuda_rng_payload = [local_cuda]
        torch.save(
            {
                "world_size": world_size,
                "states": rng_states,
                "cuda_states_by_rank": cuda_rng_payload,
            },
            checkpoint_dir / "rng_state.pt",
        )
        identity = stage_context.runtime.run_identity
        if identity is None:
            raise AlgorithmExecutionError(
                "Torch Worker stage context has no run identity"
            )
        payload_names = [
            "model.pt",
            "optimizer.pt",
            "scaler.pt",
            "rng_state.pt",
            "torch_progress.json",
        ]
        if optimization.scheduler is not None:
            payload_names.append("scheduler.pt")
        payload_files = {
            name: hashlib.sha256((checkpoint_dir / name).read_bytes()).hexdigest()
            for name in payload_names
        }
        descriptor = TorchCheckpointDescriptor(
            schema_version=1,
            identity=identity,
            run_config_name=torch_run_config_name(identity),
            state_layout=str(config.get("_core_state_layout", "replicated")),
            world_size=world_size,
            completed_step=restored_step + steps // accumulation,
            policy_digest=stage_context.runtime.policy_digest,
            execution_plan_digest=stage_context.runtime.execution_plan_digest,
            input_binding_digest=str(config.get("_core_input_binding_digest", "")),
            implementation_code_digest=str(
                config.get("_core_implementation_code_digest", "")
            ),
            payload_files=payload_files,
            adapter_identity=config.get("_core_adapter_identity"),
            resume_supported=stage_context.runtime.resume_supported,
            same_world_size_resume=stage_context.runtime.same_world_size_resume,
        )

        class _Draft:
            checkpoint_dir: str | os.PathLike[str]

            def __init__(
                self, checkpoint_dir: Path, checkpoint_owner_rank: int
            ) -> None:
                self.checkpoint_dir = str(checkpoint_dir)
                self._checkpoint_owner_rank = checkpoint_owner_rank

            def report(
                self,
                *,
                metrics: Mapping[str, object],
                stage_context: object,
                completed_step: int,
            ) -> None:
                del stage_context, completed_step
                checkpoint = (
                    Checkpoint.from_directory(str(checkpoint_dir))
                    if rank == self._checkpoint_owner_rank
                    else None
                )
                ray.train.report(dict(metrics), checkpoint=checkpoint)

        loss_state = torch.tensor(
            [loss_numerator_total, loss_normalizer_total],
            dtype=torch.float64,
            device=next(model.parameters()).device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_state, op=dist.ReduceOp.SUM)
        train_loss = (
            float(loss_state[0].item() / loss_state[1].item())
            if loss_state[1].item() > 0
            else 0.0
        )
        metric_values: dict[str, float] = {}
        metric_reducers = config.get(
            "_core_metric_reducers", {"train_loss": "sum_count"}
        )
        if not isinstance(metric_reducers, Mapping):
            raise AlgorithmConfigurationError("Torch metric reducers must be a mapping")
        metric_values.update(
            _reduce_metric_totals(
                metric_totals,
                metric_reducers,
                device=next(model.parameters()).device,
                dist=dist,
                world_size=world_size,
            )
        )
        if composite_loss_seen and "train_loss" not in metric_values:
            raise AlgorithmExecutionError(
                "Composite Torch training produced no train_loss metric"
            )
        metric_values.update(
            _reduce_metric_totals(
                evaluation_totals,
                metric_reducers,
                device=next(model.parameters()).device,
                dist=dist,
                world_size=world_size,
            )
        )
        metric_values.setdefault("train_loss", train_loss)
        for source_name, target_name in metric_mapping.items():
            if source_name in metric_values:
                metric_values[target_name] = metric_values.pop(source_name)
        reducer_report = {
            "reducer_id": config.get("_core_global_loss_reducer_id"),
            "reducer_api_version": config.get("_core_global_loss_reducer_api_version"),
            "reducer_schema_id": config.get("_core_global_loss_reducer_schema_id"),
            "reducer_code_digest": config.get("_core_global_loss_reducer_code_digest"),
            "reducer_branch": reducer_observation.get("branch"),
            "reducer_evidence": reducer_observation.get("evidence"),
        }
        reducer_report = {
            key: value for key, value in reducer_report.items() if value is not None
        }
        report_torch_checkpoint(
            {
                "train_loss": train_loss,
                **metric_values,
                "execution_workers": worker_records,
                "model_state_digest": state_digest,
                "checkpoint_descriptor": descriptor.to_dict(),
                **reducer_report,
            },
            _Draft(
                checkpoint_dir,
                int(config.get("_core_checkpoint_owner_rank", 0)),
            ),
            stage_context,
            restored_step + steps // accumulation,
        )
    finally:
        import shutil

        shutil.rmtree(checkpoint_dir, ignore_errors=True)


@DeveloperAPI
def torch_recipe_train_loop_per_worker(config: Mapping[str, Any]) -> None:
    """Public Core worker entrypoint referenced by ``TorchStageSpec``."""
    _recipe_worker(config)


@DeveloperAPI
def ray_torch_adapter_train_loop_per_worker(config: Mapping[str, Any]) -> None:
    """Public Core wrapper entrypoint for Adapter-owned Stage loops."""
    reference = config.get("_core_implementation_ref")
    if not isinstance(reference, str):
        raise AlgorithmConfigurationError(
            "Adapter Worker implementation reference is missing"
        )
    implementation = _load_reference(QualifiedReference.parse(reference))
    if not isinstance(implementation, type):
        raise AlgorithmConfigurationError("Adapter Worker reference is invalid")
    adapter = implementation()
    if not isinstance(adapter, RayTorchAdapter):
        raise AlgorithmConfigurationError("Adapter Worker reference is invalid")
    stage_context_value = config.get("_core_stage_context")
    if not isinstance(stage_context_value, Mapping):
        raise AlgorithmConfigurationError("Adapter Worker stage context is missing")
    stage_context = TorchStageContext.from_dict(stage_context_value)
    checkpoint_context = _select_worker_checkpoint(config, stage_context)
    try:
        adapter.train_loop_per_worker(
            config.get("adapter_config", {}),
            checkpoint_context,
        )
    finally:
        if checkpoint_context.checkpoint is not None:
            checkpoint_context.checkpoint.close()


@DeveloperAPI
class RayTrainTorchRuntime:
    """Unified Core-owned Runtime for Recipe and Adapter implementations."""

    @property
    def runtime_id(self) -> str:
        return RAY_TRAIN_TORCH_RUNTIME_ID

    def preflight(
        self,
        plan: Any,
        run_id: str,
        invocation_id: str,
    ) -> TorchPreflightLease:
        policy = _policy(plan)
        if policy.state_layout == "sharded":
            raise AlgorithmConfigurationError(
                "Torch sharded state is reserved and not supported by Runtime v1"
            )
        if policy.evidence_adapter_ref is not None:
            raise AlgorithmConfigurationError(
                "Torch evidence_adapter_ref is reserved until a Core evidence adapter protocol is gated"
            )
        ray_config = plan.algorithm_config.get("ray", {})
        resume_config = (
            ray_config.get("resume", {}) if isinstance(ray_config, Mapping) else {}
        )
        has_external_recovery = plan.runtime.resume_from is not None or (
            isinstance(resume_config, Mapping)
            and any(
                resume_config.get(name) is not None
                for name in ("uri", "checkpoint_uri", "checkpoint_descriptor_digest")
            )
        )
        if plan.runtime.torch_recovery is not None:
            recovery = TorchRecoveryEnvelope.from_dict(plan.runtime.torch_recovery)
            has_external_recovery = has_external_recovery or bool(
                recovery.stage_checkpoints or recovery.active_checkpoint is not None
            )
        if has_external_recovery and not policy.resume_supported:
            raise AlgorithmConfigurationError(
                "Torch Policy does not support external recovery"
            )
        implementation = _load_torch_implementation(plan)
        for stage in policy.execution_plan.stages:
            expected_loop_ref = (
                _CORE_ADAPTER_LOOP_REF
                if policy.loop_owner == "adapter"
                else _CORE_RECIPE_LOOP_REF
            )
            if stage.worker_loop_ref != expected_loop_ref:
                raise AlgorithmConfigurationError(
                    "Torch Stage worker_loop_ref must use the Core-owned loop wrapper"
                )
            stage_worker = _load_reference(
                QualifiedReference.parse(stage.worker_loop_ref)
            )
            if not callable(stage_worker):
                raise AlgorithmConfigurationError(
                    f"Torch Stage {stage.stage_id!r} worker_loop_ref is not callable"
                )
        identity = _identity(
            plan, run_id, invocation_id, policy.execution_plan.final_stage_id
        )
        context = TorchRuntimeContext(
            algorithm_config=_torch_algorithm_context_config(plan),
            implementation_id=plan.implementation.implementation_id,
            world_size=plan.runtime.worker_count,
            policy_digest=policy.digest,
            execution_plan_digest=policy.execution_plan.digest,
            run_identity=identity,
            input_bindings=_torch_input_bindings(plan),
            output_config=_torch_output_config(plan),
            input_binding_digest=_input_binding_digest(plan),
            state_layout=policy.state_layout,
            adapter_identity=(
                plan.implementation.implementation_id
                if isinstance(implementation, RayTorchAdapter)
                else None
            ),
            resume_supported=policy.resume_supported,
            same_world_size_resume=policy.same_world_size_resume,
        )
        if isinstance(implementation, RayTorchAdapter):
            implementation.validate_environment(context)
            metric_plan = implementation.metric_plan(context)
        else:
            self._validate_recipe_environment(context)
            metric_plan = implementation.metric_plan(context)
        if not isinstance(metric_plan, TorchMetricPlan):
            raise AlgorithmConfigurationError(
                "Torch metric_plan must return TorchMetricPlan"
            )
        declared_reducers = {
            name: reduction.value for name, reduction in policy.metric_reducers.items()
        }
        if dict(metric_plan.reducers) != declared_reducers:
            raise AlgorithmConfigurationError(
                "Torch metric_plan reducers do not match TorchPolicy"
            )
        if policy.global_loss_reducer_ref:
            reducer_reference = QualifiedReference.parse(policy.global_loss_reducer_ref)
            _validate_module_digest(
                reducer_reference,
                policy.global_loss_reducer_code_digest,
            )
            reducer = _load_reference(reducer_reference)
            if isinstance(reducer, type):
                reducer = reducer()
            if not callable(getattr(reducer, "reduce", None)):
                raise AlgorithmConfigurationError(
                    "Torch global loss reducer is not callable"
                )
            if (
                getattr(reducer, "api_version", None)
                != policy.global_loss_reducer_api_version
            ):
                raise AlgorithmConfigurationError(
                    "Torch global reducer API version mismatch"
                )
            if (
                getattr(reducer, "component_schema_id", None)
                != policy.composite_loss_schema_id
            ):
                raise AlgorithmConfigurationError(
                    "Torch global reducer schema mismatch"
                )
            if (
                getattr(reducer, "code_digest", None)
                != policy.global_loss_reducer_code_digest
            ):
                raise AlgorithmConfigurationError(
                    "Torch global reducer code digest mismatch"
                )
        token = TorchPreflightTokenData(
            run_id=run_id,
            invocation_id=invocation_id,
            algorithm=plan.resolution.algorithm,
            implementation_ref=str(plan.implementation.implementation_ref),
            implementation_code_digest=cast(str, plan.implementation.code_digest),
            policy_digest=policy.digest,
            execution_plan_digest=policy.execution_plan.digest,
            runtime_id=self.runtime_id,
            reducer_identity=policy.global_loss_reducer_ref,
            plan_digest=plan.plan_id,
        )
        return TorchPreflightLease(token)

    @staticmethod
    def _validate_recipe_environment(context: TorchRuntimeContext) -> None:
        if context.world_size < 1:
            raise AlgorithmConfigurationError("Torch world size must be positive")
        try:
            import ray
            import ray.train
            import torch
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "TorchRecipe requires Ray and PyTorch"
            ) from exc
        if not hasattr(ray, "train") or not hasattr(torch, "nn"):
            raise AlgorithmConfigurationError("TorchRecipe environment is incomplete")

    def execute(self, envelope: TorchRuntimeExecutionEnvelope) -> WorkerExecutionResult:
        if not isinstance(envelope, TorchRuntimeExecutionEnvelope):
            raise AlgorithmConfigurationError(
                "Ray Train Torch requires TorchRuntimeExecutionEnvelope"
            )
        base = envelope.base
        if base.cancelled:
            raise AlgorithmExecutionError("Torch execution was cancelled")
        invocation_id = envelope.preflight_lease.data.invocation_id
        token = envelope.preflight_lease.consume(
            run_id=base.run_id,
            invocation_id=invocation_id,
            plan_digest=base.plan.plan_id,
            runtime_id=self.runtime_id,
        )
        implementation = _load_torch_implementation(base.plan)
        policy = _policy(base.plan)
        if policy.state_layout == "sharded":
            raise AlgorithmConfigurationError(
                "Torch sharded state is reserved and not supported by Runtime v1"
            )
        if policy.evidence_adapter_ref is not None:
            raise AlgorithmConfigurationError(
                "Torch evidence_adapter_ref is reserved until a Core evidence adapter protocol is gated"
            )
        reducer_metadata = _reducer_metadata(policy)
        completed_stage_ids, active_stage_id, recovery_records = _recovery_records(
            base.plan,
            policy,
            worker_count=base.plan.runtime.worker_count,
        )
        prepared = _prepare_datasets(base)
        try:
            stages = policy.execution_plan.stages
            stage_records: dict[str, dict[str, Any]] = dict(recovery_records)
            last_result: Any = None
            final_result: Any = None
            stage_evidence: list[ComponentStageEvidence] = []
            final_expected_rows: Mapping[str, int] = {}
            for index, stage in enumerate(stages):
                if stage.stage_id in completed_stage_ids:
                    recovered_record = stage_records.get(stage.stage_id)
                    if recovered_record is None:
                        raise AlgorithmExecutionError(
                            f"Torch recovery is missing Stage {stage.stage_id!r}"
                        )
                    recovered_evidence = _recovered_stage_evidence(
                        plan=base.plan,
                        policy=policy,
                        stage=stage,
                        descriptor=recovered_record["descriptor"],
                        evidence=recovered_record.get("evidence", {}),
                    )
                    if policy.state_layout == "component":
                        stage_evidence.append(recovered_evidence)
                    if stage.stage_id == policy.execution_plan.final_stage_id:
                        final_expected_rows = {
                            role.role: int(role.expected_rows or 0)
                            for role in recovered_evidence.roles
                            if role.present and role.expected_rows is not None
                        }
                    continue
                identity = _identity(
                    base.plan, token.run_id, token.invocation_id, stage.stage_id
                )
                runtime_context = TorchRuntimeContext(
                    algorithm_config=_torch_algorithm_context_config(base.plan),
                    implementation_id=base.plan.implementation.implementation_id,
                    world_size=base.plan.runtime.worker_count,
                    policy_digest=policy.digest,
                    execution_plan_digest=policy.execution_plan.digest,
                    run_identity=identity,
                    input_bindings=_torch_input_bindings(base.plan),
                    output_config=_torch_output_config(base.plan),
                    input_binding_digest=_input_binding_digest(base.plan),
                    state_layout=policy.state_layout,
                    adapter_identity=(
                        base.plan.implementation.implementation_id
                        if isinstance(implementation, RayTorchAdapter)
                        else None
                    ),
                    resume_supported=policy.resume_supported,
                    same_world_size_resume=policy.same_world_size_resume,
                )
                predecessor_id = stage.checkpoint_from_stage
                predecessor_record = (
                    stage_records.get(predecessor_id)
                    if predecessor_id is not None
                    else None
                )
                context_descriptor = (
                    {
                        key: value
                        for key, value in predecessor_record["descriptor"].items()
                        if key not in {"locator", "checkpoint_locator"}
                    }
                    if predecessor_record is not None
                    else None
                )
                context = _stage_context(
                    base.plan,
                    runtime_context,
                    stage,
                    index,
                    predecessor=predecessor_id,
                    predecessor_descriptor=context_descriptor,
                )
                stage_datasets: Mapping[str, object] = prepared.views
                if isinstance(implementation, RayTorchAdapter):
                    stage_datasets = implementation.bind_datasets(
                        prepared.views,
                        context,
                    )
                    if not isinstance(stage_datasets, Mapping) or not stage_datasets:
                        raise AlgorithmConfigurationError(
                            "RayTorchAdapter.bind_datasets must return named datasets"
                        )
                expected_rows = _validate_stage_routes(
                    policy,
                    stage,
                    stage_datasets,
                    base.plan.runtime.worker_count,
                )
                if stage.stage_id == policy.execution_plan.final_stage_id:
                    final_expected_rows = dict(expected_rows)
                if (
                    stage.checkpoint_from_stage is not None
                    and predecessor_record is None
                ):
                    raise AlgorithmExecutionError(
                        f"Torch Stage {stage.stage_id!r} is missing checkpoint from "
                        f"{stage.checkpoint_from_stage!r}"
                    )
                core_control = _control_for_stage(
                    base.plan,
                    policy,
                    stage,
                    run_id=token.run_id,
                    invocation_id=token.invocation_id,
                    checkpoint=(
                        stage_records.get(stage.stage_id)
                        if active_stage_id == stage.stage_id
                        else stage_records.get(stage.checkpoint_from_stage)
                        if stage.checkpoint_from_stage is not None
                        else None
                    ),
                    purpose=(
                        "cross_run_initial_recovery"
                        if active_stage_id == stage.stage_id
                        else "stage_dependency"
                        if stage.checkpoint_from_stage is not None
                        else None
                    ),
                    source_stage_id=(
                        None
                        if active_stage_id == stage.stage_id
                        else stage.checkpoint_from_stage
                    ),
                )
                train_config = _torch_algorithm_context_config(base.plan)
                stage_binding = base.plan.input_bindings.get(stage.input_roles[0])
                if stage_binding is None:
                    raise AlgorithmConfigurationError(
                        f"Torch Stage role {stage.input_roles[0]!r} has no input binding"
                    )
                train_config.update(
                    {
                        "_core_implementation_ref": str(
                            base.plan.implementation.implementation_ref
                        ),
                        "_core_implementation_code_digest": base.plan.implementation.code_digest,
                        "_core_state_layout": policy.state_layout,
                        "_core_checkpoint_owner_rank": policy.checkpoint_owner_rank,
                        "_core_input_binding_digest": _input_binding_digest(base.plan),
                        "_core_feature_names": list(stage_binding.feature_names),
                        "_core_label_name": stage_binding.label_name,
                        "_core_weight_name": stage_binding.sample_weight_name,
                        "_core_stage_input_roles": list(stage.input_roles),
                        "_core_input_role_bindings": _torch_input_bindings(base.plan),
                        "_core_checkpoint_interval_windows": int(
                            getattr(stage, "checkpoint_interval_windows", 1)
                        ),
                        "_core_policy_digest": policy.digest,
                        "_core_execution_plan_digest": policy.execution_plan.digest,
                        "_core_global_loss_reducer_ref": policy.global_loss_reducer_ref,
                        "_core_global_loss_reducer_api_version": policy.global_loss_reducer_api_version,
                        "_core_global_loss_reducer_code_digest": policy.global_loss_reducer_code_digest,
                        "_core_composite_loss_schema_id": policy.composite_loss_schema_id,
                        "_core_global_loss_reducer_id": reducer_metadata.get(
                            "reducer_id"
                        ),
                        "_core_global_loss_reducer_schema_id": reducer_metadata.get(
                            "reducer_schema_id"
                        ),
                        "_core_metric_reducers": {
                            name: reduction.value
                            for name, reduction in policy.metric_reducers.items()
                        },
                        "_core_metric_mapping": dict(
                            getattr(stage, "metric_mapping", {})
                        ),
                        "_core_adapter_identity": (
                            base.plan.implementation.implementation_id
                            if isinstance(implementation, RayTorchAdapter)
                            else None
                        ),
                        "_core_stage_context": context.to_dict(),
                        "_core_batch_size": int(
                            base.plan.algorithm_config.get("training", {}).get(
                                "batch_size", 32
                            )
                            if isinstance(
                                base.plan.algorithm_config.get("training", {}), Mapping
                            )
                            else 32
                        ),
                        "_core_torch_evidence": {},
                        "core_control": core_control,
                        "_core_checkpoint_opener_ref": (
                            "tributo.integrations.algorithm_runtimes.ray_train_torch:"
                            "open_torch_checkpoint_locator"
                            if core_control is not None
                            else None
                        ),
                    }
                )
                loop: Any
                if isinstance(implementation, TorchRecipe):
                    loop_ref = _load_reference(
                        QualifiedReference.parse(stage.worker_loop_ref)
                    )
                    if not callable(loop_ref):
                        raise AlgorithmConfigurationError(
                            "Recipe Stage worker_loop_ref is not callable"
                        )
                    loop = loop_ref
                else:
                    adapter = cast(RayTorchAdapter, implementation)
                    adapter_config = adapter.worker_config(context)
                    if not isinstance(adapter_config, Mapping):
                        raise AlgorithmConfigurationError(
                            "RayTorchAdapter.worker_config must return a mapping"
                        )
                    if any(not isinstance(key, str) for key in adapter_config):
                        raise AlgorithmConfigurationError(
                            "Adapter worker config keys must be strings"
                        )
                    _validate_adapter_worker_config(adapter_config)
                    try:
                        json.dumps(adapter_config, allow_nan=False)
                    except (TypeError, ValueError) as exc:
                        raise AlgorithmConfigurationError(
                            "Adapter worker config must be JSON-compatible"
                        ) from exc
                    train_config["adapter_config"] = dict(adapter_config)
                    loop_ref = _load_reference(
                        QualifiedReference.parse(stage.worker_loop_ref)
                    )
                    if not callable(loop_ref):
                        raise AlgorithmConfigurationError(
                            "Adapter Stage worker_loop_ref is not callable"
                        )
                    loop = loop_ref
                from ray.train import FailureConfig, RunConfig, ScalingConfig
                from ray.train.torch import TorchConfig, TorchTrainer

                from tributo.integrations.algorithm_runtimes.ray_data_config import (
                    TorchRoleDataConfig,
                )

                storage_path = None
                ray_config = base.plan.algorithm_config.get("ray", {})
                max_failures = 0
                if isinstance(ray_config, Mapping):
                    storage_path = ray_config.get("storage_path")
                    configured_failures = ray_config.get("max_failures", 0)
                    if (
                        not isinstance(configured_failures, int)
                        or isinstance(configured_failures, bool)
                        or configured_failures < -1
                    ):
                        raise AlgorithmConfigurationError(
                            "ray.max_failures must be -1 or a non-negative integer"
                        )
                    max_failures = configured_failures
                if (
                    storage_path is None
                    and cast(Any, base.plan.runtime.execution_profile).value == "local"
                ):
                    storage_path = tempfile.mkdtemp(prefix="tributo_torch_runs_")
                if storage_path is not None:
                    claim_torch_run_directory(storage_path, identity)
                trainer = TorchTrainer(
                    train_loop_per_worker=loop,
                    train_loop_config=train_config,
                    scaling_config=ScalingConfig(
                        num_workers=base.plan.runtime.worker_count,
                        use_gpu=base.plan.runtime.num_gpus > 0,
                        resources_per_worker=_resource_map(base.plan),
                        placement_strategy="SPREAD",
                    ),
                    datasets=cast(Any, dict(stage_datasets)),
                    run_config=RunConfig(
                        name=torch_run_config_name(identity),
                        storage_path=str(storage_path)
                        if storage_path is not None
                        else None,
                        failure_config=FailureConfig(max_failures=max_failures),
                    ),
                    torch_config=TorchConfig(
                        backend=None if policy.backend == "auto" else policy.backend
                    ),
                    dataset_config=TorchRoleDataConfig(
                        {route.role: route for route in policy.dataset_routing}
                    ),
                )
                last_result = trainer.fit()
                if stage.stage_id == policy.execution_plan.final_stage_id:
                    final_result = last_result
                metrics = last_result.metrics or {}
                stage_descriptor_payload = metrics.get("checkpoint_descriptor")
                checkpoint = getattr(last_result, "checkpoint", None)
                if checkpoint is not None and isinstance(
                    stage_descriptor_payload, Mapping
                ):
                    validated_descriptor = describe_torch_checkpoint(
                        TorchCheckpointRef(checkpoint),
                        TorchCheckpointContext(
                            stage=context,
                            run_id=identity.run_id,
                            invocation_id=identity.invocation_id,
                            checkpoint_owner="core",
                        ),
                    )
                    if validated_descriptor.to_dict() != dict(stage_descriptor_payload):
                        raise AlgorithmExecutionError(
                            "Torch Stage checkpoint descriptor differs from embedded payload"
                        )
                    persisted_locator = _persist_stage_checkpoint(
                        checkpoint,
                        identity=identity,
                        storage_path=storage_path,
                        descriptor_digest=validated_descriptor.digest,
                    )
                    if persisted_locator is not None:
                        stage_descriptor_payload = dict(stage_descriptor_payload)
                        stage_descriptor_payload["locator"] = persisted_locator
                        stage_descriptor_payload["descriptor_digest"] = (
                            validated_descriptor.digest
                        )
                    stage_records[stage.stage_id] = {
                        "locator": (
                            stage_descriptor_payload.get("locator")
                            if isinstance(stage_descriptor_payload, Mapping)
                            else None
                        ),
                        "descriptor_digest": validated_descriptor.digest,
                        "descriptor": validated_descriptor.to_dict(),
                        "evidence": {
                            key: metrics[key]
                            for key in (
                                "execution_workers",
                                "model_state_digest",
                                "reducer_id",
                                "reducer_api_version",
                                "reducer_schema_id",
                                "reducer_code_digest",
                                "reducer_branch",
                                "reducer_evidence",
                            )
                            if key in metrics
                        },
                    }
                elif stage.checkpoint_required:
                    raise AlgorithmExecutionError(
                        f"Torch Stage {stage.stage_id!r} requires a Core checkpoint"
                    )
                stage_evidence.append(
                    _component_stage_evidence(
                        plan=base.plan,
                        policy=policy,
                        stage=stage,
                        identity=identity,
                        metrics=metrics,
                        expected_rows=expected_rows,
                    )
                )
            final_stage = next(
                stage
                for stage in stages
                if stage.stage_id == policy.execution_plan.final_stage_id
            )
            if final_result is None and final_stage.stage_id in stage_records:
                recovered = stage_records[final_stage.stage_id]
                locator = TorchCheckpointLocator(
                    cast(str, recovered["locator"]),
                    cast(str, recovered["descriptor_digest"]),
                )
                recovered_checkpoint = open_torch_checkpoint_locator(locator)
                final_result = _CheckpointResultProxy(
                    recovered_checkpoint,
                    recovered_checkpoint,
                    metrics={
                        **dict(recovered.get("evidence", {})),
                        "checkpoint_descriptor": recovered["descriptor"],
                    },
                )
            if final_result is None:
                raise AlgorithmExecutionError(
                    "Torch execution plan has no Stage result"
                )
            last_result = final_result
            metrics = dict(final_result.metrics or {})
            checkpoint = getattr(final_result, "checkpoint", None)
            if checkpoint is not None:
                raw_descriptor = metrics.get("checkpoint_descriptor")
                if not isinstance(raw_descriptor, Mapping):
                    raise AlgorithmExecutionError(
                        "Torch final Checkpoint is missing its Core descriptor"
                    )
                descriptor = TorchCheckpointDescriptor.from_dict(raw_descriptor)
                checkpoint_ref = TorchCheckpointRef(
                    checkpoint=checkpoint,
                    descriptor_digest=descriptor.digest,
                    source_stage_id=descriptor.identity.stage_id,
                    descriptor=descriptor,
                )
                describe_torch_checkpoint(
                    checkpoint_ref,
                    TorchCheckpointContext(
                        stage=_stage_context(
                            base.plan,
                            TorchRuntimeContext(
                                algorithm_config=_torch_algorithm_context_config(
                                    base.plan
                                ),
                                implementation_id=base.plan.implementation.implementation_id,
                                world_size=base.plan.runtime.worker_count,
                                policy_digest=policy.digest,
                                execution_plan_digest=policy.execution_plan.digest,
                                run_identity=descriptor.identity,
                                input_bindings=_torch_input_bindings(base.plan),
                                output_config=_torch_output_config(base.plan),
                                input_binding_digest=_input_binding_digest(base.plan),
                                state_layout=policy.state_layout,
                                adapter_identity=(
                                    base.plan.implementation.implementation_id
                                    if isinstance(implementation, RayTorchAdapter)
                                    else None
                                ),
                                resume_supported=policy.resume_supported,
                                same_world_size_resume=policy.same_world_size_resume,
                            ),
                            final_stage,
                            stages.index(final_stage),
                        ),
                        run_id=descriptor.identity.run_id,
                        invocation_id=descriptor.identity.invocation_id,
                        checkpoint_owner="core",
                    ),
                )
            final_identity = _identity(
                base.plan,
                token.run_id,
                token.invocation_id,
                policy.execution_plan.final_stage_id,
            )
            supplied_evidence = metrics.get("torch_evidence")
            if supplied_evidence is not None:
                raise AlgorithmExecutionError(
                    "Torch execution evidence is Core-owned and must not be supplied by an algorithm"
                )
            raw_workers = metrics.get("execution_workers")
            if (
                not isinstance(raw_workers, (list, tuple))
                or len(raw_workers) != base.plan.runtime.worker_count
            ):
                raise AlgorithmExecutionError(
                    "Torch execution did not report every worker"
                )
            workers = tuple(
                WorkerExecutionEvidence.from_dict(item)
                for item in _normalize_worker_evidence(raw_workers, base.plan)
            )
            role_evidence = _role_execution_evidence(
                plan=base.plan,
                policy=policy,
                stage=final_stage,
                workers=workers,
                expected_rows=final_expected_rows,
            )
            global_digest = metrics.get("model_state_digest")
            if not isinstance(global_digest, str) or len(global_digest) != 64:
                raise AlgorithmExecutionError(
                    "Torch execution did not report a model digest"
                )
            composition_digest = None
            if policy.state_layout == "component":
                composition_digest = hashlib.sha256(
                    json.dumps(
                        [item.to_dict() for item in stage_evidence],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            metrics["torch_evidence"] = TorchExecutionEvidence(
                identity=final_identity,
                run_config_name=torch_run_config_name(final_identity),
                policy_digest=policy.digest,
                parallelism_id=policy.parallelism_id,
                state_layout=policy.state_layout,
                workers=workers,
                roles=role_evidence,
                replicated_state=(
                    ReplicatedTorchStateEvidence(
                        model_digests_by_rank={
                            worker.rank: cast(str, worker.model_state_digest)
                            for worker in workers
                        },
                        global_model_digest=global_digest,
                    )
                    if policy.state_layout == "replicated"
                    else None
                ),
                stages=tuple(stage_evidence)
                if policy.state_layout == "component"
                else (),
                composition_digest=composition_digest,
                final_stage_id=(
                    policy.execution_plan.final_stage_id
                    if policy.state_layout == "component"
                    else None
                ),
                reducer_id=cast(str | None, reducer_metadata.get("reducer_id")),
                reducer_api_version=cast(
                    int | None, reducer_metadata.get("reducer_api_version")
                ),
                reducer_schema_id=cast(
                    str | None, reducer_metadata.get("reducer_schema_id")
                ),
                reducer_code_digest=cast(
                    str | None, reducer_metadata.get("reducer_code_digest")
                ),
                reducer_branch=(
                    metrics.get("reducer_branch")
                    if isinstance(metrics.get("reducer_branch"), str)
                    else None
                ),
                reducer_evidence=(
                    dict(metrics["reducer_evidence"])
                    if isinstance(metrics.get("reducer_evidence"), Mapping)
                    else {}
                ),
            ).to_dict()
            execution = AlgorithmExecutionResult(
                status="succeeded",
                metrics=portable_fit_only_metrics(
                    metrics, extra_evidence_names=_TORCH_INTERNAL_METRIC_NAMES
                ),
            )
            export_result = _CheckpointResultProxy(
                last_result,
                checkpoint,
                metrics=metrics,
                core_evidence_attested=True,
            )
            export_state_details: dict[str, object] = {}
            if policy_result_policy(base.plan) is ResultPolicy.BUNDLE_REQUIRED:
                if isinstance(implementation, RayTorchAdapter):
                    final_context = TorchCheckpointContext(
                        stage=_stage_context(
                            base.plan,
                            TorchRuntimeContext(
                                algorithm_config=_torch_algorithm_context_config(
                                    base.plan
                                ),
                                implementation_id=base.plan.implementation.implementation_id,
                                world_size=base.plan.runtime.worker_count,
                                policy_digest=policy.digest,
                                execution_plan_digest=policy.execution_plan.digest,
                                run_identity=final_identity,
                                input_bindings=_torch_input_bindings(base.plan),
                                output_config=_torch_output_config(base.plan),
                                input_binding_digest=_input_binding_digest(base.plan),
                                state_layout=policy.state_layout,
                                adapter_identity=(
                                    base.plan.implementation.implementation_id
                                    if isinstance(implementation, RayTorchAdapter)
                                    else None
                                ),
                                resume_supported=policy.resume_supported,
                                same_world_size_resume=policy.same_world_size_resume,
                            ),
                            final_stage,
                            stages.index(final_stage),
                        ),
                        run_id=token.run_id,
                        invocation_id=token.invocation_id,
                        checkpoint_owner="core",
                    )
                    checkpoint = implementation.checkpoint_source(
                        last_result, final_context
                    )
                    if checkpoint is None:
                        raise AlgorithmExecutionError(
                            "RayTorchAdapter checkpoint_source returned no checkpoint"
                        )
                    checkpoint_ref = (
                        checkpoint
                        if isinstance(checkpoint, TorchCheckpointRef)
                        else TorchCheckpointRef(checkpoint)
                    )
                    describe_torch_checkpoint(checkpoint_ref, final_context)
                    export_result = _CheckpointResultProxy(
                        last_result,
                        checkpoint_ref.checkpoint,
                        metrics=metrics,
                        core_evidence_attested=True,
                    )
                execution = export_ray_train_torch_result(
                    result=export_result,
                    plan=base.plan,
                    run_id=token.run_id,
                    state_details_sink=export_state_details,
                )
            raw_worker_metadata = metrics.get("execution_workers", [])
            normalized_worker_metadata = (
                [
                    dict(item)
                    for item in _normalize_worker_evidence(
                        raw_worker_metadata, base.plan
                    )
                ]
                if isinstance(raw_worker_metadata, (list, tuple))
                else []
            )
            state_details = (
                _component_state_details(tuple(stage_evidence))
                if policy.state_layout == "component"
                else export_state_details
            )
            return WorkerExecutionResult(
                execution=execution,
                actual_versions=_actual_environment_versions(
                    base.plan.environment.python,
                    base.plan.environment.dependencies,
                ),
                worker_metadata={
                    "topology": "ray_train_torch",
                    "workers": normalized_worker_metadata,
                    "state": {
                        "coordination": "torch_managed",
                        "synchronized": True,
                        "bounded": True,
                        "global_model_digest": metrics.get("model_state_digest"),
                        "details": state_details,
                    },
                    "torch_evidence": metrics.get("torch_evidence", {}),
                    "input_complete": True,
                    "driver_materialized_training_rows": 0,
                },
            )
        finally:
            prepared.close()


@DeveloperAPI
def create_torch_algorithm(
    *, plan: Any, implementation: object, artifacts: tuple[object, ...]
) -> object:
    """Factory validation hook retained for the unified Builder."""
    del artifacts
    expected = _load_torch_implementation(plan)
    if implementation is not _load_reference(plan.implementation.implementation_ref):
        raise AlgorithmConfigurationError(
            "Torch implementation drifted after descriptor resolution"
        )
    return expected


class _CheckpointResultProxy:
    """Preserve Ray Result metrics while replacing the exported Checkpoint."""

    def __init__(
        self,
        result: object,
        checkpoint: object,
        *,
        metrics: Mapping[str, object] | None = None,
        core_evidence_attested: bool = False,
    ) -> None:
        self.metrics = dict(metrics or getattr(result, "metrics", {}) or {})
        self.checkpoint = checkpoint
        self.core_evidence_attested = core_evidence_attested


def _component_composition_digest(result: object) -> str | None:
    """Read a validated component composition digest from a Torch result."""
    if getattr(result, "core_evidence_attested", False) is not True:
        return None
    metrics = getattr(result, "metrics", {}) or {}
    evidence = metrics.get("torch_evidence") if isinstance(metrics, Mapping) else None
    digest = (
        evidence.get("composition_digest") if isinstance(evidence, Mapping) else None
    )
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        return None
    return digest


def _source_state_details(metadata: Mapping[str, object]) -> dict[str, object]:
    """Project Adapter-declared scalar export evidence into receipt state details."""
    details: dict[str, object] = {}
    for source_name, detail_name in (
        ("sampling", "sampling"),
        ("topology_kind", "topology_kind"),
        ("sparse_routing", "routing"),
    ):
        value = metadata.get(source_name)
        if isinstance(value, (str, int, float, bool)):
            details[detail_name] = value
    if "routing" in details:
        details["jagged"] = True
    return details


@DeveloperAPI
def export_ray_train_torch_result(
    *,
    result: object,
    plan: Any,
    run_id: str,
    state_details_sink: dict[str, object] | None = None,
) -> AlgorithmExecutionResult:
    """Export the final Stage through the Core BundleExportService."""
    if (
        plan.distribution_spec is None
        or plan.distribution_spec.result_policy is ResultPolicy.FIT_ONLY
    ):
        return AlgorithmExecutionResult(
            status="succeeded",
            metrics=portable_fit_only_metrics(
                getattr(result, "metrics", {}) or {},
                extra_evidence_names=_TORCH_INTERNAL_METRIC_NAMES,
            ),
        )
    import importlib.metadata

    from tributo.exporting.models import BundleOutputConfig, ExportTarget
    from tributo.exporting.service import BundleExportService
    from tributo.integrations.sources.ray_torch import (
        RayTorchSourceProvider,
        TorchSourceOptions,
    )

    output = plan.algorithm_config.get("output", {})
    if not isinstance(output, Mapping) or not isinstance(output.get("bundle_uri"), str):
        raise AlgorithmConfigurationError(
            "Torch Bundle export requires output.bundle_uri"
        )
    composition_digest = None
    if _policy(plan).state_layout == "component":
        composition_digest = _component_composition_digest(result)
        if composition_digest is None:
            raise AlgorithmExecutionError(
                "Torch component Bundle is missing composition_digest"
            )
    final_stage_id = getattr(_policy(plan).execution_plan, "final_stage_id", None)
    final_stage = next(
        (
            stage
            for stage in _policy(plan).execution_plan.stages
            if getattr(stage, "stage_id", final_stage_id) == final_stage_id
        ),
        _policy(plan).execution_plan.stages[-1],
    )
    options = TorchSourceOptions(
        implementation_ref=str(plan.implementation.implementation_ref),
        implementation_code_digest=plan.implementation.code_digest,
        implementation_id=plan.implementation.implementation_id,
        loop_owner=_policy(plan).loop_owner,
        algorithm_config=_torch_algorithm_context_config(plan),
        input_bindings=_torch_input_bindings(plan),
        output_config=_torch_output_config(plan),
        policy_digest=_policy(plan).digest,
        plan_digest=plan.plan_id,
        input_binding_digest=_input_binding_digest(plan),
        stage_input_roles=tuple(final_stage.input_roles),
        stage_index=_policy(plan).execution_plan.stages.index(final_stage),
    )
    provider = RayTorchSourceProvider()
    with provider.open_source(result, options) as source:
        if state_details_sink is not None:
            state_details_sink.update(_source_state_details(source.metadata))
        artifact_payload = source.metadata.get("artifact_plan", {})
        if not isinstance(artifact_payload, Mapping):
            raise AlgorithmExecutionError("Torch artifact plan is missing")
        raw_targets = artifact_payload.get("targets")
        raw_roles = artifact_payload.get("roles")
        if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
            raise AlgorithmExecutionError("Torch artifact plan declares no targets")
        if not isinstance(raw_roles, Mapping) or not raw_roles:
            raise AlgorithmExecutionError(
                "Torch artifact plan declares no Bundle roles"
            )
        targets = tuple(
            ExportTarget(**dict(target))
            for target in raw_targets
            if isinstance(target, Mapping)
        )
        if len(targets) != len(raw_targets):
            raise AlgorithmExecutionError("Torch artifact plan target is malformed")
        published = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=output["bundle_uri"],
                request_id=run_id,
                run_id=run_id,
                targets=list(targets),
                roles=dict(raw_roles),
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    outputs: dict[str, object] = {
        "bundle_id": published.bundle_id,
        "bundle_uri": published.canonical_uri,
        "execution_id": published.execution_id,
        "manifest_sha256": published.manifest_sha256,
    }
    if composition_digest is not None:
        outputs["composition_digest"] = composition_digest
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=portable_fit_only_metrics(
            getattr(result, "metrics", {}) or {},
            extra_evidence_names=_TORCH_INTERNAL_METRIC_NAMES,
        ),
        outputs=outputs,
    )


__all__ = [
    "RAY_TRAIN_TORCH_RUNTIME_ID",
    "RayTrainTorchRuntime",
    "ray_torch_adapter_train_loop_per_worker",
    "torch_recipe_train_loop_per_worker",
    "create_torch_algorithm",
    "export_ray_train_torch_result",
]
