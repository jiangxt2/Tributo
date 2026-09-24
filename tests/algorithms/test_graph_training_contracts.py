"""Contracts for Core-owned partitioned node-level graph training."""

from __future__ import annotations

import pytest

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmInputError,
    GraphBatch,
    GraphInputSpec,
    GraphPartitionEvidence,
    GraphSamplingEvidence,
    GraphSamplingSpec,
    GraphWorkerEvidence,
    InputBinding,
    InputBindingSet,
    MetricReduction,
    ReplicatedTorchStateEvidence,
    SingleStageTorchPlan,
    TorchDatasetRoute,
    TorchExecutionEvidence,
    TorchPolicy,
    TorchRoleExecutionEvidence,
    TorchStageRunIdentity,
    TorchStageSpec,
    WorkerExecutionEvidence,
    WorkerResources,
)


def test_graph_contracts_round_trip_with_seed_rows_first() -> None:
    batch = GraphBatch(
        node_ids=(10, 12, 11),
        edge_index=((2, 0), (1, 0)),
        node_features=((1.0, 0.0), (0.5, 0.5), (-1.0, 2.0)),
        seed_count=1,
        seed_labels=(1,),
    )
    assert batch.seed_node_ids == (10,)
    assert batch.node_ids[batch.edge_index[0][0]] == 11
    assert GraphSamplingSpec((5, 3), seed_batch_size=4, random_seed=7).fanouts == (
        5,
        3,
    )


def test_graph_contracts_reject_invalid_sampling_and_local_edges() -> None:
    with pytest.raises(AlgorithmConfigurationError, match="fanouts"):
        GraphSamplingSpec((0, 2), seed_batch_size=4)
    with pytest.raises(AlgorithmConfigurationError, match="direction"):
        GraphSamplingSpec((2,), seed_batch_size=4, direction="both")
    with pytest.raises(AlgorithmInputError, match="local ids"):
        GraphBatch(
            node_ids=(1,),
            edge_index=((0, 1),),
            node_features=((1.0,),),
            seed_count=1,
            seed_labels=(0,),
        )


def test_graph_input_policy_is_versioned_and_keeps_graph_roles_core_owned() -> None:
    policy = TorchPolicy(
        torch_runtime_api_version=1,
        loop_owner="adapter",
        parallelism_id="torch.ddp.replicated",
        dataset_routing=(TorchDatasetRoute("train", "split_exact"),),
        execution_plan=SingleStageTorchPlan(stage=TorchStageSpec("train", ("train",))),
        state_layout="replicated",
        metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
        graph_input=GraphInputSpec(
            node_role="nodes",
            edge_role="edges",
            seed_role="train",
            partition_count=2,
        ),
    )
    restored = TorchPolicy.from_dict(policy.to_dict())
    assert restored.digest == policy.digest
    assert restored.graph_input == policy.graph_input
    assert (
        "graph_input"
        not in TorchPolicy(
            torch_runtime_api_version=1,
            loop_owner="adapter",
            parallelism_id="torch.ddp.replicated",
            dataset_routing=(TorchDatasetRoute("train", "split_exact"),),
            execution_plan=SingleStageTorchPlan(
                stage=TorchStageSpec("train", ("train",))
            ),
            state_layout="replicated",
            metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
        ).to_dict()
    )


def test_graph_evidence_rejects_single_owner_holding_complete_graph() -> None:
    with pytest.raises(AlgorithmInputError, match="complete node or edge table"):
        GraphPartitionEvidence(
            graph_version="a" * 64,
            seed_role="train",
            partition_count=2,
            total_nodes=8,
            total_edges=8,
            owner_node_rows=(8, 0),
            owner_edge_rows=(4, 4),
        )
    evidence = GraphWorkerEvidence(
        graph_version="a" * 64,
        partition_count=2,
        seed_rows=4,
        sampled_node_rows=7,
        sampled_edge_rows=6,
        batch_count=2,
        touched_partitions=(0, 1),
        sampling_profiles=(
            GraphSamplingEvidence(
                fanouts=(5, 3),
                seed_batch_size=4,
                direction="incoming",
                request_count=2,
                random_seed_min=7,
                random_seed_max=8,
                random_seed_digest="c" * 64,
            ),
        ),
    )
    assert GraphWorkerEvidence.from_dict(evidence.to_dict()) == evidence


def test_torch_graph_evidence_rejects_worker_graph_version_drift() -> None:
    """The Torch receipt must bind each worker's graph evidence to Core's graph."""
    digest = "a" * 64
    model_digest = "b" * 64
    identity = TorchStageRunIdentity(
        run_id="1" * 32,
        invocation_id="2" * 32,
        stage_id="train",
        torch_runtime_api_version=1,
        algorithm="graph_test",
        implementation_id="tributo.test.graph",
        implementation_code_digest="c" * 64,
        policy_digest="d" * 64,
        execution_plan_digest="e" * 64,
        plan_digest="f" * 64,
    )
    partition = GraphPartitionEvidence(
        graph_version=digest,
        seed_role="train",
        partition_count=2,
        total_nodes=2,
        total_edges=0,
        owner_node_rows=(1, 1),
        owner_edge_rows=(0, 0),
    )
    profiles = (
        GraphSamplingEvidence(
            fanouts=(2,),
            seed_batch_size=1,
            direction="incoming",
            request_count=1,
            random_seed_min=7,
            random_seed_max=7,
            random_seed_digest="9" * 64,
        ),
    )
    workers = tuple(
        WorkerExecutionEvidence(
            worker_id=f"worker-{rank}",
            node_id=f"node-{rank}",
            rank=rank,
            world_size=2,
            shard_id=f"train-{rank}",
            resources=WorkerResources(num_cpus=1),
            model_state_digest=model_digest,
            input_rows={"train": 1},
            graph=GraphWorkerEvidence(
                graph_version=digest if rank == 0 else "8" * 64,
                partition_count=2,
                seed_rows=1,
                sampled_node_rows=1,
                sampled_edge_rows=0,
                batch_count=1,
                touched_partitions=(0, 1),
                sampling_profiles=profiles,
            ),
        )
        for rank in range(2)
    )
    roles = (
        TorchRoleExecutionEvidence(
            role="train",
            mode="split_exact",
            required=True,
            present=True,
            empty_rank_policy="reject",
            expected_rows=2,
            observed_rows=2,
            rows_per_rank=(1, 1),
            binding_digest="7" * 64,
        ),
    )

    with pytest.raises(AlgorithmConfigurationError, match="graph worker evidence"):
        TorchExecutionEvidence(
            identity=identity,
            run_config_name=identity.run_config_name,
            policy_digest=identity.policy_digest,
            parallelism_id="torch.ddp.replicated",
            state_layout="replicated",
            workers=workers,
            roles=roles,
            replicated_state=ReplicatedTorchStateEvidence(
                model_digests_by_rank={0: model_digest, 1: model_digest},
                global_model_digest=model_digest,
            ),
            graph_partition=partition,
        )


def test_planner_accepts_core_graph_roles_without_torch_dataset_routes() -> None:
    from tributo.algorithms.core.planner import AlgorithmPlanner

    policy = TorchPolicy(
        torch_runtime_api_version=1,
        loop_owner="adapter",
        parallelism_id="torch.ddp.replicated",
        dataset_routing=(TorchDatasetRoute("train", "split_exact"),),
        execution_plan=SingleStageTorchPlan(stage=TorchStageSpec("train", ("train",))),
        state_layout="replicated",
        metric_reducers={"train_loss": MetricReduction.SUM_COUNT},
        graph_input=GraphInputSpec("nodes", "edges", "train", partition_count=2),
    )
    bindings = InputBindingSet(
        bindings=(
            InputBinding(
                "nodes",
                "tributo.test.ingestion",
                "nodes",
                ("node_id", "f0"),
            ),
            InputBinding(
                "edges",
                "tributo.test.ingestion",
                "edges",
                ("source", "destination"),
            ),
            InputBinding(
                "train",
                "tributo.test.ingestion",
                "train",
                ("node_id",),
                label_name="label",
            ),
        ),
        primary_role="train",
    )
    AlgorithmPlanner._validate_torch_binding_roles(bindings, policy)

    with pytest.raises(AlgorithmConfigurationError, match="graph input binding"):
        AlgorithmPlanner._validate_torch_binding_roles(
            InputBindingSet(
                bindings=(bindings.get("nodes"), bindings.get("train")),
                primary_role="train",
            ),
            policy,
        )

    with pytest.raises(AlgorithmConfigurationError, match="not declared"):
        AlgorithmPlanner._validate_torch_binding_roles(
            InputBindingSet(
                bindings=(
                    *bindings.bindings,
                    InputBinding(
                        "extra",
                        "tributo.test.ingestion",
                        "extra",
                        ("value",),
                    ),
                ),
                primary_role="train",
            ),
            policy,
        )
