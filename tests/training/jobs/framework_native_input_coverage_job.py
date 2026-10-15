"""Exercise role counts and row identities using a live Ray Data cluster."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import ray
import ray.data

from tributo.algorithms.api import (
    FrameworkNativeInputRoute,
    FrameworkNativePolicy,
    WorkerResources,
)
from tributo.algorithms.spi import PreparedInput
from tributo.integrations.algorithm_runtimes.framework_native import (
    _framework_input_coverage_snapshot,
    _validated_framework_evidence,
)


def main() -> None:
    ray.init(address="auto")
    run_id = os.environ["TRIBUTO_RUN_ID"]
    train = ray.data.range(5, override_num_blocks=5)
    validation = ray.data.range(5, override_num_blocks=5)
    train_shards = train.split(2, equal=False)
    validation_rows = [
        [int(row["id"]) for row in validation.iter_rows()] for _ in range(2)
    ]
    train_rows = [
        [int(row["id"]) for row in shard.iter_rows()] for shard in train_shards
    ]
    prepared = PreparedInput({"train": train, "validation": validation})
    plan = SimpleNamespace(
        input_bindings=SimpleNamespace(
            bindings=(
                SimpleNamespace(name="train"),
                SimpleNamespace(name="validation"),
            )
        )
    )
    policy = FrameworkNativePolicy(
        framework="ray-data-integration",
        evidence_collector_ref="tests.example:collect",
        input_routes=(
            FrameworkNativeInputRoute("train", "partitioned"),
            FrameworkNativeInputRoute("validation", "replicated"),
        ),
    )
    expected_training_rows, role_modes, expected_role_rows = (
        _framework_input_coverage_snapshot(
            plan=plan,
            prepared_views=prepared.views,
            datasets={"train": train, "validation": validation},
            policy=policy,
        )
    )
    workers = [
        {
            "worker_id": f"{run_id}-worker-{rank}",
            "node_id": f"ray-node-{rank}",
            "rank": rank,
            "world_size": 2,
            "shard_id": f"{run_id}-shard-{rank}",
            "rows_processed": len(train_rows[rank]),
            "input_rows": {
                "train": len(train_rows[rank]),
                "validation": len(validation_rows[rank]),
            },
            "model_state_digest": "a" * 64,
            "resources": {"num_cpus": 1.0, "num_gpus": 0.0, "custom": {}},
            "train_row_ids": train_rows[rank],
            "validation_row_ids": validation_rows[rank],
        }
        for rank in range(2)
    ]
    evidence = {
        "workers": workers,
        "state": {
            "coordination": "framework_native",
            "synchronized": True,
            "bounded": True,
            "global_model_digest": "a" * 64,
            "details": {"integration": "ray_data"},
        },
        "input_complete": True,
    }
    validated_workers, _ = _validated_framework_evidence(
        evidence,
        worker_count=2,
        resources_per_worker=WorkerResources(),
        expected_training_rows=expected_training_rows,
        input_role_modes=role_modes,
        expected_input_role_rows=expected_role_rows,
    )
    result_workers = []
    for worker, validated in zip(workers, validated_workers, strict=True):
        result_workers.append(
            {
                **validated,
                "train_row_ids": worker["train_row_ids"],
                "validation_row_ids": worker["validation_row_ids"],
            }
        )
    print(
        "FRAMEWORK_NATIVE_COVERAGE_RESULT: "
        + json.dumps(
            {
                "cluster_nodes": len(
                    [node for node in ray.nodes() if node.get("Alive")]
                ),
                "canonical_rows": expected_role_rows,
                "workers": result_workers,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    prepared.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
