"""Evidence tests that prevent task parallelism from masquerading as training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
    ComponentStageEvidence,
    DistributionStrategy,
    ExecutionProfile,
    ExecutionReceipt,
    ExecutionRequest,
    ResultPolicy,
    StateCoordination,
    StateCoordinationEvidence,
    WorkerExecutionEvidence,
    WorkerExecutionResult,
    WorkerResources,
)
from tributo.algorithms.core import AlgorithmRunCoordinator

from .conftest import request_for


def _worker(rank: int, node_id: str = "node-a") -> WorkerExecutionEvidence:
    return WorkerExecutionEvidence(
        worker_id=f"worker-{rank}",
        node_id=node_id,
        rank=rank,
        world_size=2,
        shard_id=f"train-shard-{rank}",
        resources=WorkerResources(),
        model_state_digest="a" * 64,
        rows_processed=4,
        input_rows={"train": 4},
        batch_count=1,
        collective_steps=1,
    )


def _receipt(profile: ExecutionProfile) -> ExecutionReceipt:
    return ExecutionReceipt(
        run_id="run-1",
        plan_id="b" * 64,
        requested_algorithm="dnn",
        canonical_algorithm="dnn",
        profile=profile,
        strategy=DistributionStrategy.RAY_TRAIN_COLLECTIVE,
        requested_worker_count=2,
        distributed_min_workers=2,
        requested_resources_per_worker=WorkerResources(),
        workers=(_worker(0), _worker(1)),
        input_complete=True,
        state=StateCoordinationEvidence(
            coordination=StateCoordination.ALL_REDUCE,
            synchronized=True,
            bounded=True,
            global_model_digest="a" * 64,
        ),
        artifact_ids=("bundle-1",),
        resource_preflight=(
            "validated" if profile is ExecutionProfile.LOCAL else "deferred_to_ray"
        ),
    )


def test_execution_request_wraps_algorithm_request_without_duplicate_identity() -> None:
    algorithm_request = request_for("example", AlgorithmOperation.FIT)
    request = ExecutionRequest(
        algorithm_request=algorithm_request,
        profile=ExecutionProfile.LOCAL,
        worker_count=2,
    )

    assert request.algorithm_request is algorithm_request
    assert not hasattr(request, "algorithm")
    assert not hasattr(request, "algorithm_config")


def test_execution_request_rejects_standalone() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="local.*cluster"):
        ExecutionRequest(
            algorithm_request=request_for("example", AlgorithmOperation.FIT),
            profile=cast(ExecutionProfile, "standalone"),
            worker_count=1,
        )


def test_worker_and_state_evidence_reject_truthy_string_coercion() -> None:
    worker = _worker(0).to_dict()
    worker["rank"] = "0"
    with pytest.raises(AlgorithmConfigurationError, match="rank"):
        WorkerExecutionEvidence.from_dict(worker)

    state = {
        "coordination": "all_reduce",
        "synchronized": "false",
        "bounded": True,
        "global_model_digest": "a" * 64,
    }
    with pytest.raises(AlgorithmConfigurationError, match="boolean"):
        StateCoordinationEvidence.from_dict(state)


@pytest.mark.parametrize("worker_digest", [None, "c" * 64])
def test_component_stage_rejects_unbound_worker_state_digest(
    worker_digest: str | None,
) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="model digests"):
        ComponentStageEvidence(
            stage_id="teacher",
            workers=(
                _worker(0),
                replace(_worker(1), model_state_digest=worker_digest),
            ),
            roles=(),
            state_digest="a" * 64,
        )


def test_local_multi_worker_proves_model_distribution_not_cross_node() -> None:
    receipt = _receipt(ExecutionProfile.LOCAL)

    assert receipt.distributed is True
    assert receipt.cross_node is False
    assert receipt.cluster_distributed is False
    assert receipt.runtime_owned is False
    assert receipt.resource_preflight == "validated"
    assert receipt.to_dict()["distributed"] is True
    assert receipt.to_dict()["runtime_owned"] is False
    assert receipt.to_dict()["resource_preflight"] == "validated"
    assert "torch_evidence" not in receipt.to_dict()


def test_coordinator_receipt_preserves_requested_alias_and_canonical_algorithm() -> (
    None
):
    plan = SimpleNamespace(
        plan_id="b" * 64,
        resolution=SimpleNamespace(
            requested_algorithm="dnn",
            algorithm="pu",
        ),
        distribution_spec=SimpleNamespace(
            strategy=DistributionStrategy.RAY_TRAIN_COLLECTIVE,
            distributed_min_workers=2,
            result_policy=ResultPolicy.BUNDLE_REQUIRED,
        ),
        runtime=SimpleNamespace(
            execution_profile=ExecutionProfile.LOCAL,
            worker_count=2,
            num_cpus=1.0,
            num_gpus=0.0,
            custom_resources={},
        ),
    )
    result = WorkerExecutionResult(
        execution=AlgorithmExecutionResult(
            status="succeeded",
            outputs={"bundle_id": "bundle-1"},
        ),
        actual_versions={},
        worker_metadata={
            "workers": [_worker(0).to_dict(), _worker(1).to_dict()],
            "state": {
                "coordination": "all_reduce",
                "synchronized": True,
                "bounded": True,
                "global_model_digest": "a" * 64,
            },
            "input_complete": True,
            "driver_materialized_training_rows": 0,
        },
    )

    receipt = AlgorithmRunCoordinator._execution_receipt("run-1", plan, result)

    assert receipt is not None
    assert receipt.requested_algorithm == "dnn"
    assert receipt.canonical_algorithm == "pu"
    assert receipt.to_dict()["requested_algorithm"] == "dnn"
    assert receipt.to_dict()["canonical_algorithm"] == "pu"


def test_runtime_ownership_evidence_requires_a_boolean() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="runtime_owned"):
        replace(_receipt(ExecutionProfile.LOCAL), runtime_owned=cast(bool, "true"))


@pytest.mark.parametrize("api_version", (True, "1"))
def test_execution_receipt_requires_strict_integer_api_version(
    api_version: object,
) -> None:
    with pytest.raises(AlgorithmConfigurationError, match="api_version"):
        replace(
            _receipt(ExecutionProfile.LOCAL),
            api_version=cast(int, api_version),
        )


def test_cluster_distribution_requires_two_actual_nodes() -> None:
    same_node = _receipt(ExecutionProfile.CLUSTER)
    cross_node = replace(
        same_node,
        workers=(_worker(0, "node-a"), _worker(1, "node-b")),
    )

    assert same_node.cluster_distributed is False
    assert cross_node.cluster_distributed is True
    assert cross_node.to_dict()["cluster_distributed"] is True


def test_staged_framework_composite_digest_proves_distribution() -> None:
    details = {
        "framework": "staged_composite",
        "component_stage_count": 2,
        "component_stages": "mu0,propensity",
        "anchor_stage": "propensity",
        "composition_digest": "d" * 64,
        "stage.mu0.digest": "c" * 64,
        "stage.mu0.rows": 4,
        "stage.propensity.digest": "a" * 64,
        "stage.propensity.rows": 8,
    }
    composite_digest = hashlib.sha256(
        json.dumps(
            {
                "composition_digest": "d" * 64,
                "stages": {
                    "mu0": {"digest": "c" * 64, "rows": 4},
                    "propensity": {"digest": "a" * 64, "rows": 8},
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt = replace(
        _receipt(ExecutionProfile.LOCAL),
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        state=StateCoordinationEvidence(
            coordination=StateCoordination.FRAMEWORK_NATIVE,
            synchronized=True,
            bounded=True,
            global_model_digest=composite_digest,
            details=details,
        ),
    )

    assert {worker.model_state_digest for worker in receipt.workers} == {"a" * 64}
    assert receipt.distributed is True


def test_staged_framework_receipt_rejects_unbound_composite_digest() -> None:
    receipt = replace(
        _receipt(ExecutionProfile.LOCAL),
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        state=StateCoordinationEvidence(
            coordination=StateCoordination.FRAMEWORK_NATIVE,
            synchronized=True,
            bounded=True,
            global_model_digest="c" * 64,
            details={
                "framework": "staged_composite",
                "component_stage_count": 2,
                "component_stages": "mu0,propensity",
                "anchor_stage": "propensity",
                "composition_digest": "d" * 64,
                "stage.mu0.digest": "c" * 64,
                "stage.mu0.rows": 4,
                "stage.propensity.digest": "b" * 64,
                "stage.propensity.rows": 8,
            },
        ),
    )

    assert receipt.distributed is False


def test_local_receipt_cannot_defer_resource_preflight() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="local.*preflight"):
        replace(
            _receipt(ExecutionProfile.LOCAL),
            resource_preflight="deferred_to_ray",
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_complete": False}, ""),
        (
            {
                "state": StateCoordinationEvidence(
                    coordination=StateCoordination.ALL_REDUCE,
                    synchronized=False,
                    bounded=True,
                )
            },
            "",
        ),
        ({"driver_materialized_training_rows": 8}, ""),
    ],
)
def test_incomplete_evidence_never_claims_distributed(
    changes: dict[str, object],
    message: str,
) -> None:
    del message
    assert replace(_receipt(ExecutionProfile.LOCAL), **changes).distributed is False


def test_duplicate_shards_are_rejected_instead_of_counted_as_distribution() -> None:
    duplicate = replace(_worker(1), shard_id="train-shard-0")

    with pytest.raises(AlgorithmConfigurationError, match="shard IDs"):
        replace(
            _receipt(ExecutionProfile.LOCAL),
            workers=(_worker(0), duplicate),
        )


def test_collective_worker_model_digest_mismatch_cannot_claim_distributed() -> None:
    divergent = replace(_worker(1), model_state_digest="c" * 64)
    receipt = replace(
        _receipt(ExecutionProfile.LOCAL),
        workers=(_worker(0), divergent),
    )

    assert receipt.distributed is False


def test_distribution_threshold_resources_and_bundle_are_evidence_not_claims() -> None:
    receipt = _receipt(ExecutionProfile.LOCAL)

    assert replace(receipt, distributed_min_workers=3).distributed is False
    assert replace(receipt, artifact_ids=()).distributed is False
    fit_only = replace(
        receipt,
        result_policy=ResultPolicy.FIT_ONLY,
        artifact_ids=(),
    )
    assert fit_only.distributed is True
    assert fit_only.to_dict()["result_policy"] == "fit_only"
    assert (
        replace(
            receipt,
            workers=(replace(_worker(0), rows_processed=0), _worker(1)),
        ).distributed
        is False
    )
    assert (
        replace(
            receipt,
            state=replace(receipt.state, global_model_digest=None),
        ).distributed
        is False
    )
    with pytest.raises(AlgorithmConfigurationError, match="requested_resources"):
        replace(
            receipt,
            requested_resources_per_worker=WorkerResources(num_cpus=2),
        )
