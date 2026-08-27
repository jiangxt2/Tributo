"""Run algorithm-neutral third-party conformance Wheels on Ray."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import ray


class _BorrowedDockerRayRuntimeManager:
    """Expose the IT-owned cluster without taking lifecycle ownership."""

    def open(self, profile: object, **kwargs: object) -> Any:
        from tributo.algorithms.api import ExecutionProfile, WorkerResources
        from tributo.algorithms.core import RayRuntimeManager, RayRuntimeSession

        if profile is not ExecutionProfile.CLUSTER:
            raise AssertionError("borrowed runtime requires cluster profile")
        resources = kwargs.get("resources_per_worker")
        worker_count = kwargs.get("worker_count")
        if not isinstance(resources, WorkerResources) or not isinstance(
            worker_count, int
        ):
            raise AssertionError("borrowed runtime requires resources")
        RayRuntimeManager.validate_resources(
            resources,
            worker_count,
            cluster_resources=ray.cluster_resources(),
            nodes=ray.nodes(),
        )
        return RayRuntimeSession(
            cast(RayRuntimeManager, self),
            ExecutionProfile.CLUSTER,
            owned=False,
            cluster_resources=ray.cluster_resources(),
            resource_preflight="validated",
        )

    def _release(self) -> None:
        return None


def _stage_parquet(path: str, records: dict[str, list[object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(target, index=False)


def _execute(
    algorithm: str,
    data_path: str,
    bundle_uri: str,
    *,
    profile: str,
    worker_count: int,
    runtime_manager: object | None = None,
    local_num_cpus: int | None = None,
    expected_rows: int = 64,
) -> dict[str, object]:
    from tributo.algorithms import build_algorithm_dispatcher
    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
    )
    from tributo.algorithms.core import LocalRuntimeOptions, RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    if algorithm not in {
        "third_party_mean_regressor",
        "third_party_binary_linear",
    }:
        raise ValueError("conformance job accepts only third-party fixture algorithms")
    key = f"conformance-{algorithm}"
    values = {
        key: IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=data_path),
                engine="ray",
            )
        )
    }
    config: dict[str, object] = {}
    if algorithm == "third_party_binary_linear":
        root = str(Path(bundle_uri).parent / f"ray-{worker_count}")
        config = {
            "model": {"input_features": 2},
            "loss": {},
            "optimizer": {"learning_rate": 0.1},
            "metrics": {},
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "prefetch_batches": 0,
                "seed": 7,
            },
            "ray": {
                "max_failures": 0,
                "storage_path": root,
                "resume": {"checkpoint_interval": 1},
            },
            "output": {"bundle_uri": bundle_uri},
        }
    execution_profile = ExecutionProfile(profile)
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=algorithm,
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference=key,
                feature_names=("f0", "f1"),
                label_name="label",
            ),
            algorithm_config=config,
        ),
        profile=execution_profile,
        worker_count=worker_count,
    )
    manager = runtime_manager
    if manager is None:
        manager = RayRuntimeManager(
            default_local_options=LocalRuntimeOptions(
                num_cpus=local_num_cpus,
                num_gpus=0,
            )
        )
    result = build_algorithm_dispatcher(
        runtime_manager=cast(RayRuntimeManager, manager)
    ).execute(
        request,
        InputExecutionContext(values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None:
        raise AssertionError("conformance execution did not return a receipt")
    if (
        sum(worker.input_rows.get("train", 0) for worker in receipt.workers)
        != expected_rows
    ):
        raise AssertionError("conformance execution did not cover every input row")
    if algorithm == "third_party_binary_linear" and not Path(bundle_uri).is_dir():
        raise AssertionError("third-party recipe did not publish its Bundle")
    return {
        "algorithm": algorithm,
        "worker_count": worker_count,
        "status": result.execution.status,
        "receipt": receipt.to_dict(),
    }


def main() -> int:
    mode = os.environ.get("TRIBUTO_DISTRIBUTED_GATE_PROFILE", "local")
    if mode not in {"local", "docker-distributed"}:
        raise ValueError(f"unsupported conformance profile: {mode}")
    if mode == "docker-distributed":
        ray.init()
    elif ray.is_initialized():
        raise AssertionError("owned local Gate must start without Ray")
    root = Path(os.environ["TRIBUTO_DISTRIBUTED_GATE_ROOT"])
    rng = random.Random(42)
    records: dict[str, list[object]] = {
        "f0": [rng.gauss(0.0, 1.0) for _ in range(64)],
        "f1": [rng.gauss(0.0, 1.0) for _ in range(64)],
        "label": [float(index % 3 == 0) for index in range(64)],
    }
    data_path = str(root / "data.parquet")
    _stage_parquet(data_path, records)
    if mode == "local":
        algorithms = (
            "third_party_mean_regressor",
            "third_party_binary_linear",
        )
        results = [
            _execute(
                algorithm,
                data_path,
                str(root / f"bundle-{algorithm}-{worker_count}"),
                profile="local",
                worker_count=worker_count,
                local_num_cpus=4,
            )
            for algorithm in algorithms
            for worker_count in (1, 2)
        ]
    else:
        manager = _BorrowedDockerRayRuntimeManager()
        results = [
            _execute(
                "third_party_mean_regressor",
                data_path,
                str(root / f"unused-{worker_count}"),
                profile="cluster",
                worker_count=worker_count,
                runtime_manager=manager,
            )
            for worker_count in (1, 2)
        ]
    print(f"RESULT: {json.dumps(results, sort_keys=True)}")
    if mode == "docker-distributed":
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
