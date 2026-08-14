"""Shared Ray Train/PyTorch collective primitives for DNN and PU."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from tributo.exceptions import JobConfigurationError


def collective_available() -> bool:
    """Return whether the current worker belongs to an initialized process group."""
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized())


def _collective_device() -> Any:
    import torch
    import torch.distributed as dist

    if collective_available() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce_values(values: Iterable[float]) -> tuple[float, ...]:
    """Sum finite scalar values over the current Ray Train worker group."""
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(
        tuple(values), dtype=torch.float64, device=_collective_device()
    )
    if collective_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = tuple(float(value) for value in tensor.tolist())
    if any(not math.isfinite(value) for value in result):
        raise JobConfigurationError(
            "collective metric reduction produced non-finite data"
        )
    return result


def all_reduce_max(value: int) -> int:
    """Return the maximum integer value over the current worker group."""
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(value, dtype=torch.int64, device=_collective_device())
    if collective_available():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def broadcast_bool(value: bool, *, source_rank: int = 0) -> bool:
    """Broadcast one rank-owned control decision to the worker group."""
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(
        1 if value else 0,
        dtype=torch.int64,
        device=_collective_device(),
    )
    if collective_available():
        dist.broadcast(tensor, src=source_rank)
    return bool(tensor.item())


def all_gather_objects(value: Any) -> tuple[Any, ...]:
    """Gather one bounded checkpoint/control object from every rank."""
    import torch.distributed as dist

    if not collective_available():
        return (value,)
    gathered: list[Any | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered):
        raise JobConfigurationError(
            "distributed object gather did not return every worker value"
        )
    return tuple(gathered)


def fit_global_feature_transformer(features: list[Any], data: dict[str, Any]) -> Any:
    """Fit one deterministic preprocessor from mergeable worker-local statistics."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.distributed as dist

    from tributo.training.features.column_types import DenseFeat, NormMethod, SparseFeat
    from tributo.training.features.transformer import FeatureTransformer

    transformer = FeatureTransformer(features)
    if not collective_available():
        return transformer.fit(data)

    for feature in features:
        values = np.asarray(data[feature.name])
        if isinstance(feature, SparseFeat):
            if feature.use_hash:
                continue
            mask = pd.isna(values)
            local_values = tuple(np.unique(values[~mask]).tolist())
            if len(local_values) >= feature.vocab_size:
                raise JobConfigurationError(
                    f"Sparse feature {feature.name!r} has {len(local_values)} "
                    f"worker-local categories but vocab_size={feature.vocab_size} "
                    "must reserve one unknown-category index"
                )
            gathered: list[tuple[object, ...] | None] = [
                None for _ in range(dist.get_world_size())
            ]
            dist.all_gather_object(gathered, local_values)
            unique: dict[tuple[str, str], object] = {}
            for worker_values in gathered:
                for value in worker_values or ():
                    unique[(type(value).__qualname__, repr(value))] = value
            ordered = [unique[key] for key in sorted(unique)]
            if len(ordered) >= feature.vocab_size:
                raise JobConfigurationError(
                    f"Sparse feature {feature.name!r} has {len(ordered)} global "
                    f"categories but vocab_size={feature.vocab_size} must reserve "
                    "one unknown-category index"
                )
            transformer.label_encoders[feature.name] = {
                value: index for index, value in enumerate(ordered)
            }
            continue
        if not isinstance(feature, DenseFeat) or feature.norm is NormMethod.NONE:
            continue
        numeric = values.astype(np.float64)
        valid = numeric[np.isfinite(numeric)]
        if feature.norm in {NormMethod.MINMAX, NormMethod.LOG}:
            local_min = float(np.min(valid)) if valid.size else float("inf")
            minimum = torch.tensor(
                local_min, dtype=torch.float64, device=_collective_device()
            )
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            if not math.isfinite(float(minimum.item())):
                raise JobConfigurationError(
                    f"Dense feature {feature.name!r} has no finite values"
                )
            params = {"min": float(minimum.item())}
            if feature.norm is NormMethod.MINMAX:
                local_max = float(np.max(valid)) if valid.size else float("-inf")
                maximum = torch.tensor(
                    local_max, dtype=torch.float64, device=_collective_device()
                )
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                params["max"] = float(maximum.item())
            transformer.norm_params[feature.name] = params
            continue
        summary = torch.tensor(
            [
                float(valid.size),
                float(valid.sum(dtype=np.float64)),
                float(np.square(valid).sum(dtype=np.float64)),
            ],
            dtype=torch.float64,
            device=_collective_device(),
        )
        dist.all_reduce(summary, op=dist.ReduceOp.SUM)
        count, total, square_total = (float(value) for value in summary.tolist())
        if count <= 0:
            raise JobConfigurationError(
                f"Dense feature {feature.name!r} has no finite values"
            )
        mean = total / count
        variance = max(0.0, square_total / count - mean * mean)
        transformer.norm_params[feature.name] = {
            "mean": mean,
            "std": math.sqrt(variance),
        }
    transformer.fitted = True
    return transformer


def prepare_model(model: Any) -> tuple[Any, Any]:
    """Move and wrap one model with Ray Train DDP when world size exceeds one."""
    if not collective_available():
        import torch

        # Formal Ray Train always initializes a process group, including for one
        # worker. Keep this defensive path on CPU so an undeclared GPU can never
        # be consumed outside Ray's assigned-resource contract.
        device = torch.device("cpu")
        return model.to(device), device
    from ray.train.torch import get_device
    from ray.train.torch import prepare_model as ray_prepare_model

    return ray_prepare_model(model), get_device()


def unwrapped_model(model: Any) -> Any:
    """Return the underlying module used for consolidated checkpoints."""
    return getattr(model, "module", model)


def collective_execution_evidence(
    model: Any,
    *,
    shard_rows: int,
    input_binding_digest: str | None = None,
    input_rows: Mapping[str, int] | None = None,
    batch_count: int,
    collective_steps: int,
) -> tuple[tuple[dict[str, object], ...], str]:
    """Collect real worker/node/shard facts and verify one synchronized model."""
    import ray
    import torch.distributed as dist

    module = unwrapped_model(model)
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    model_digest = digest.hexdigest()
    runtime = ray.get_runtime_context()
    rank = dist.get_rank() if collective_available() else 0
    world_size = dist.get_world_size() if collective_available() else 1
    assigned = runtime.get_assigned_resources()
    custom_resources = {
        str(name): float(value)
        for name, value in assigned.items()
        if name not in {"CPU", "GPU", "memory", "object_store_memory"}
    }
    shard_id = hashlib.sha256(
        f"{input_binding_digest or 'ray-train'}:{rank}/{world_size}".encode("ascii")
    ).hexdigest()
    local: dict[str, object] = {
        "worker_id": str(runtime.get_worker_id()),
        "node_id": str(runtime.get_node_id()),
        "rank": rank,
        "world_size": world_size,
        "shard_id": shard_id,
        "rows_processed": shard_rows,
        "input_rows": dict(input_rows or {}),
        "batch_count": batch_count,
        "collective_steps": collective_steps,
        "model_state_digest": model_digest,
        "resources": {
            "num_cpus": float(assigned.get("CPU", 0.0)),
            "num_gpus": float(assigned.get("GPU", 0.0)),
            "custom": custom_resources,
        },
    }
    workers: list[dict[str, object] | None] = [None for _ in range(world_size)]
    if collective_available():
        dist.all_gather_object(workers, local)
    else:
        workers[0] = local
    normalized = tuple(worker for worker in workers if worker is not None)
    if (
        len(normalized) != world_size
        or len({str(worker["model_state_digest"]) for worker in normalized}) != 1
    ):
        raise JobConfigurationError(
            "DDP workers did not converge to one synchronized model state"
        )
    return normalized, model_digest


def equalized_batches(
    loader: Any,
    *,
    collective_steps: int | None = None,
) -> Iterator[tuple[Any, bool]]:
    """Yield every local batch once, then empty batches for collective alignment.

    Exhausted ranks still perform a zero-contribution DDP forward/backward and
    every custom all-reduce. No observed row is replayed merely to equalize
    uneven shard lengths.
    """
    local_steps = len(loader)
    global_steps = (
        all_reduce_max(local_steps) if collective_steps is None else collective_steps
    )
    if global_steps < local_steps:
        raise JobConfigurationError(
            "collective_steps cannot be smaller than the local DataLoader"
        )
    if local_steps < 1 or global_steps < 1:
        raise JobConfigurationError("distributed training received an empty shard")
    iterator = iter(loader)
    template: Any | None = None
    for step in range(global_steps):
        if step < local_steps:
            batch = next(iterator)
            template = batch
            yield batch, True
            continue
        if not isinstance(template, Mapping):
            raise JobConfigurationError(
                "distributed DataLoader batches must be mappings"
            )
        empty: dict[str, Any] = {}
        for name, value in template.items():
            try:
                empty[name] = value[:0]
            except (IndexError, TypeError) as exc:
                raise JobConfigurationError(
                    f"distributed batch value {name!r} cannot form an empty shard"
                ) from exc
        yield empty, False


def distributed_pu_loss(criterion: Any, logits: Any, labels: Any) -> Any:
    """Compute one global nnPU/uPU risk with autograd-aware sum collectives."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    positive = labels == 1
    unlabeled = labels == 0
    if not bool(torch.all(positive | unlabeled)):
        raise JobConfigurationError(
            "PU labels must contain only 1 (positive) or 0 (unlabeled)"
        )
    positive_count = positive.sum().to(dtype=logits.dtype)
    unlabeled_count = unlabeled.sum().to(dtype=logits.dtype)
    positive_loss_sum = functional.softplus(-logits[positive]).sum()
    positive_as_negative_sum = functional.softplus(logits[positive]).sum()
    unlabeled_negative_sum = functional.softplus(logits[unlabeled]).sum()
    global_sums = torch.stack(
        (
            positive_loss_sum.detach(),
            positive_as_negative_sum.detach(),
            unlabeled_negative_sum.detach(),
        )
    )
    if collective_available():
        positive_count = positive_count.clone()
        unlabeled_count = unlabeled_count.clone()
        dist.all_reduce(positive_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(unlabeled_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_sums, op=dist.ReduceOp.SUM)
    if positive_count.item() <= 0 or unlabeled_count.item() <= 0:
        raise JobConfigurationError(
            "every global PU optimization batch requires positive and unlabeled rows"
        )
    global_negative_risk = global_sums[2] / unlabeled_count - (
        criterion.class_prior * global_sums[1] / positive_count
    )
    # DDP averages parameter gradients.  Scale each rank-local contribution by
    # world size so the average equals the gradient of the global empirical risk.
    world_size = dist.get_world_size() if collective_available() else 1
    positive_contribution = (
        world_size * criterion.class_prior * positive_loss_sum / positive_count
    )
    negative_contribution = world_size * (
        unlabeled_negative_sum / unlabeled_count
        - criterion.class_prior * positive_as_negative_sum / positive_count
    )
    global_positive_risk = criterion.class_prior * global_sums[0] / positive_count
    if (
        criterion.loss_type == "nnpu"
        and float(global_negative_risk.item()) < -criterion.beta
    ):
        local_gradient_value = -criterion.gamma * negative_contribution
        global_value = -criterion.gamma * global_negative_risk
    else:
        local_gradient_value = positive_contribution + negative_contribution
        global_value = global_positive_risk + global_negative_risk
    # Keep the globally meaningful scalar value while differentiating through
    # only this rank's contribution. DDP then averages the world-size-scaled
    # local gradients into the gradient of the global empirical risk.
    return local_gradient_value + global_value.detach() - local_gradient_value.detach()
