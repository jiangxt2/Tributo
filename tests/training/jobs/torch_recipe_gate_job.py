"""Run an isolated out-of-tree Torch recipe on an existing Ray cluster."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import ray

from tests.training.jobs.distributed_algorithm_gate_job import (
    _BorrowedDockerRayRuntimeManager,
    _execute,
    _stage_parquet,
)


def main() -> int:
    if os.environ.get("TRIBUTO_DISTRIBUTED_GATE_PROFILE") != "torch-recipe":
        raise AssertionError("Torch recipe Gate profile is missing")
    ray.init()
    from tributo.plugin import discover_algorithm_descriptors

    matches = [
        descriptor
        for descriptor in discover_algorithm_descriptors()
        if descriptor.name == "third_party_binary_linear"
    ]
    if len(matches) != 1:
        raise AssertionError("Torch recipe descriptor was not isolated and discovered")
    root = Path(os.environ["TRIBUTO_DISTRIBUTED_GATE_ROOT"])
    rng = random.Random(42)
    records = {
        "f0": [rng.gauss(0.0, 1.0) for _ in range(65)],
        "f1": [rng.gauss(0.0, 1.0) for _ in range(65)],
        "label": [float(index % 3 == 0) for index in range(65)],
    }
    data_path = str(root / "data.parquet")
    _stage_parquet(data_path, records)
    result = _execute(
        "third_party_binary_linear",
        data_path,
        str(root / "bundle"),
        profile="cluster",
        worker_count=2,
        runtime_manager=_BorrowedDockerRayRuntimeManager(),
        expected_rows=65,
    )
    print(f"RESULT: {json.dumps([result], sort_keys=True)}")
    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
