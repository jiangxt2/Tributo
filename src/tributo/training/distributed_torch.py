"""Algorithm-neutral Ray Train and PyTorch collective primitives."""

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


def prepare_model(model: Any) -> tuple[Any, Any]:
    """Move and wrap one model with Ray Train DDP when world size exceeds one."""
    if not collective_available():
        import torch

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
    """Collect real worker/node/shard facts and verify synchronized state."""
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
    """Yield local batches once, then empty batches for collective alignment."""
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


__all__: list[str] = []
