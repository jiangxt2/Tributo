"""Two-worker Ray Jobs Gate for Core-owned partitioned graph training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    DistributedAlgorithmDescriptor,
    GraphReadHandle,
    GraphSamplingSpec,
    TorchCheckpointPayloadDraft,
    TorchCheckpointRef,
    report_torch_graph_checkpoint,
)
from tributo.algorithms.spi import (
    RayTorchAdapter,
    TorchArtifactContext,
    TorchArtifactPlan,
    TorchCheckpointContext,
    TorchMetricPlan,
    TorchRuntimeContext,
    TorchStageContext,
    TorchWorkerCheckpointContext,
)


class _BorrowedClusterRuntimeManager:
    """Use the Ray Jobs cluster without taking ownership of its lifecycle."""

    def open(self, profile: object, **kwargs: object) -> Any:
        from tributo.algorithms.api import ExecutionProfile, WorkerResources
        from tributo.algorithms.core import RayRuntimeManager, RayRuntimeSession

        if profile is not ExecutionProfile.CLUSTER:
            raise AssertionError("graph core gate requires the cluster profile")
        resources = kwargs.get("resources_per_worker")
        worker_count = kwargs.get("worker_count")
        if not isinstance(resources, WorkerResources) or not isinstance(
            worker_count, int
        ):
            raise AssertionError("graph core gate requires worker resources")
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


class GraphCoreAdapter(RayTorchAdapter):
    """Test-only adapter proving the public partitioned graph worker contract."""

    api_version = 1

    def validate_environment(self, context: TorchRuntimeContext) -> None:
        del context
        import torch

        if not hasattr(torch, "nn"):
            raise RuntimeError("PyTorch is unavailable")

    def bind_datasets(
        self,
        datasets: Mapping[str, object],
        context: TorchStageContext,
    ) -> Mapping[str, object]:
        del context
        if set(datasets) != {"train"}:
            raise ValueError("Core graph adapter should receive seed rows only")
        return dict(datasets)

    def worker_config(self, context: TorchStageContext) -> Mapping[str, object]:
        del context
        return {"batch_size": 2, "fanouts": [2], "random_seed": 17}

    def train_loop_per_worker(
        self,
        worker_config: Mapping[str, object],
        checkpoint_context: TorchWorkerCheckpointContext,
    ) -> None:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as functional
        from ray import train
        from ray.train.torch import get_device, prepare_model

        graph_reader = getattr(checkpoint_context, "graph_reader", None)
        if not isinstance(graph_reader, GraphReadHandle):
            raise AssertionError("Core did not attach the graph reader handle")
        stage = checkpoint_context.stage
        training_shard = train.get_dataset_shard("train")
        raw_batch_size = worker_config.get("batch_size")
        raw_fanouts = worker_config.get("fanouts")
        raw_random_seed = worker_config.get("random_seed")
        if (
            not isinstance(raw_batch_size, int)
            or isinstance(raw_batch_size, bool)
            or not isinstance(raw_fanouts, (list, tuple))
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_fanouts
            )
            or not isinstance(raw_random_seed, int)
            or isinstance(raw_random_seed, bool)
        ):
            raise AssertionError("Core graph adapter worker config is invalid")
        batch_size = raw_batch_size
        fanouts = tuple(raw_fanouts)
        random_seed = raw_random_seed
        device = get_device()
        torch.manual_seed(11)
        model = prepare_model(torch.nn.Linear(2, 2).to(device))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_sum = 0.0
        correct = 0
        seed_rows = 0
        batch_count = 0
        for raw_batch in training_shard.iter_torch_batches(
            batch_size=batch_size,
            dtypes={"node_id": torch.int64, "label": torch.int64},
        ):
            batch = cast(Mapping[str, torch.Tensor], raw_batch)
            seed_ids = tuple(int(value) for value in batch["node_id"].tolist())
            labels = tuple(int(value) for value in batch["label"].tolist())
            graph_batch = graph_reader.sample(
                seed_ids,
                labels,
                GraphSamplingSpec(
                    fanouts=fanouts,
                    seed_batch_size=batch_size,
                    random_seed=random_seed + batch_count,
                ),
            )
            features = torch.tensor(
                graph_batch.node_features,
                dtype=torch.float32,
                device=device,
            )
            aggregated = features.clone()
            degrees = torch.ones((len(graph_batch.node_ids), 1), device=device)
            if graph_batch.edge_index:
                source = torch.tensor(
                    [edge[0] for edge in graph_batch.edge_index],
                    dtype=torch.long,
                    device=device,
                )
                destination = torch.tensor(
                    [edge[1] for edge in graph_batch.edge_index],
                    dtype=torch.long,
                    device=device,
                )
                aggregated.index_add_(0, destination, features[source])
                degrees.index_add_(
                    0,
                    destination,
                    torch.ones((len(destination), 1), device=device),
                )
            aggregated = aggregated / degrees
            logits = model(aggregated[: graph_batch.seed_count])
            target = torch.tensor(labels, dtype=torch.long, device=device)
            loss = functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(labels)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            seed_rows += len(labels)
            batch_count += 1

        metrics_tensor = torch.tensor(
            [loss_sum, float(correct), float(seed_rows)],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        world_loss, world_correct, world_seeds = (
            float(value) for value in metrics_tensor.tolist()
        )
        runtime_context = ray.get_runtime_context()
        rank_context = train.get_context()
        worker = {
            "worker_id": str(runtime_context.get_worker_id()),
            "node_id": str(runtime_context.get_node_id()),
            "rank": rank_context.get_world_rank(),
            "world_size": rank_context.get_world_size(),
            "shard_id": f"train-{rank_context.get_world_rank()}",
            "rows_processed": seed_rows,
            "input_rows": {"train": seed_rows},
            "batch_count": batch_count,
            "collective_steps": batch_count,
            "resources": {
                "num_cpus": float(
                    runtime_context.get_assigned_resources().get("CPU", 0)
                ),
                "num_gpus": float(
                    runtime_context.get_assigned_resources().get("GPU", 0)
                ),
            },
            "model_state_digest": _model_digest(model),
        }
        workers: list[object] = [worker]
        if dist.is_available() and dist.is_initialized():
            workers = [None] * rank_context.get_world_size()
            dist.all_gather_object(workers, worker)
        with tempfile.TemporaryDirectory(
            prefix="tributo-graph-core-checkpoint-"
        ) as directory:
            root = Path(directory)
            torch.save(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                root / "model.pt",
            )
            report_torch_graph_checkpoint(
                {
                    "train_loss": world_loss / max(world_seeds, 1.0),
                    "accuracy": world_correct / max(world_seeds, 1.0),
                    "execution_workers": workers,
                    "model_state_digest": _model_digest(model),
                },
                TorchCheckpointPayloadDraft(root),
                stage,
                batch_count,
                graph_reader=graph_reader,
            )

    def checkpoint_source(
        self,
        result: object,
        context: TorchCheckpointContext,
    ) -> object:
        del context
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise RuntimeError("graph test adapter did not report a checkpoint")
        return checkpoint

    def metric_plan(self, context: TorchRuntimeContext) -> TorchMetricPlan:
        del context
        return TorchMetricPlan({"train_loss": "sum_count", "accuracy": "sum_count"})

    def artifact_plan(self, context: TorchArtifactContext) -> TorchArtifactPlan:
        del context
        return TorchArtifactPlan(
            source_kind="torch_module",
            input_signature=(
                {"name": "node_id", "dtype": "int64", "shape": ("batch",)},
            ),
            output_signature=(
                {"name": "logits", "dtype": "float32", "shape": ("batch", 2)},
            ),
            targets=(),
            roles={},
        )

    def open_export_source(
        self,
        checkpoint_ref: TorchCheckpointRef,
        artifact_context: TorchArtifactContext,
    ) -> Any:
        del checkpoint_ref, artifact_context
        raise RuntimeError("fit-only graph fixture does not export a Bundle")


def _model_digest(model: object) -> str:
    module = cast(Any, model)
    state = (
        module.module.state_dict() if hasattr(module, "module") else module.state_dict()
    )
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _write_graph_sources(root: Path) -> tuple[Path, Path, Path]:
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    nodes = root / "nodes.parquet"
    edges = root / "edges.parquet"
    seeds = root / "seeds.parquet"
    pd.DataFrame(
        {
            "node_id": list(range(8)),
            "f0": [float(index % 2) for index in range(8)],
            "f1": [float((index + 1) % 2) for index in range(8)],
        }
    ).to_parquet(nodes, index=False)
    pd.DataFrame(
        {
            "source": list(range(8)),
            "destination": [(index + 1) % 8 for index in range(8)],
            "relation": [index % 2 for index in range(8)],
        }
    ).to_parquet(edges, index=False)
    pd.DataFrame(
        {"node_id": list(range(8)), "label": [index % 2 for index in range(8)]}
    ).to_parquet(seeds, index=False)
    return nodes, edges, seeds


def _register_test_algorithm() -> DistributedAlgorithmDescriptor:
    from tributo.algorithms import AlgorithmBuilder
    from tributo.algorithms.api import (
        EnvironmentSpec,
        ExecutionProfile,
        GraphInputSpec,
        MetricReduction,
        ResultPolicy,
        SingleStageTorchPlan,
        TorchDatasetRoute,
        TorchPolicy,
        TorchStageSpec,
        WorkerRange,
        WorkerResources,
    )
    from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType
    from tributo.training.registry import get_execution_registry

    input_spec = GraphInputSpec("nodes", "edges", "train", partition_count=2)
    policy = TorchPolicy(
        torch_runtime_api_version=1,
        loop_owner="adapter",
        parallelism_id="torch.ddp.replicated",
        dataset_routing=(TorchDatasetRoute("train", "split_exact"),),
        execution_plan=SingleStageTorchPlan(stage=TorchStageSpec("train", ("train",))),
        state_layout="replicated",
        metric_reducers={
            "train_loss": MetricReduction.SUM_COUNT,
            "accuracy": MetricReduction.SUM_COUNT,
        },
        graph_input=input_spec,
    )
    spec = AlgorithmSpec(
        name="graph_core_partition_fixture",
        trainer_cls=None,
        version="1.0.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.NODE_CLASSIFICATION,),
        capabilities=(Capability.DISTRIBUTED,),
        learning_paradigm="supervised",
        model_family="test_graph_model",
        data_modalities=("graph",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=("ray_train_torch",),
        config_contract_ref="tributo.test.graph-core.config.v1",
        input_contract_ref="tributo.test.graph-core.input.v1",
        output_contract_ref="tributo.test.graph-core.output.v1",
    )
    descriptor = AlgorithmBuilder.from_torch_adapter(
        spec=spec,
        implementation_id="tributo.test.graph_core_partition",
        implementation_version="1.0.0",
        adapter="tests.training.jobs.graph_core_gate_job:GraphCoreAdapter",
        environment=EnvironmentSpec(
            environment_id="tributo.test.graph-core.v1",
            dependencies=("tributo>=1,<2", "torch>=2.5"),
        ),
        metric_reducers={
            "train_loss": MetricReduction.SUM_COUNT,
            "accuracy": MetricReduction.SUM_COUNT,
        },
        supported_worker_range=WorkerRange(2, 2),
        supported_execution_profiles=(ExecutionProfile.CLUSTER,),
        resources_per_worker=WorkerResources(num_cpus=1),
        package_name="tributo",
        package_version="1.0.0",
        tributo_version_spec=">=1,<2",
        policy=policy,
        code_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        descriptor_api_version=1,
        is_default=True,
    )
    if descriptor.registration.distribution_spec is None:
        raise AssertionError("Torch test descriptor has no distribution specification")
    registration = replace(
        descriptor.registration,
        distribution_spec=replace(
            descriptor.registration.distribution_spec,
            result_policy=ResultPolicy.FIT_ONLY,
        ),
    )
    get_execution_registry().register(registration)
    return descriptor


def main() -> None:
    from tributo.algorithms import build_algorithm_dispatcher
    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
        InputBindingSet,
        WorkerResources,
    )
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    root_value = os.environ.get("TRIBUTO_OFFICIAL_ALGORITHM_GATE_ROOT")
    if not root_value:
        raise RuntimeError("Core graph Gate root is missing")
    root = Path(root_value)
    ray.init(address="auto")
    nodes, edges, seeds = _write_graph_sources(root)
    paths = {"nodes": nodes, "edges": edges, "train": seeds}
    values = {
        f"graph-core-{role}": IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=str(path)),
                engine="ray",
            )
        )
        for role, path in paths.items()
    }
    bindings = InputBindingSet(
        bindings=(
            InputBinding(
                name="nodes",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="graph-core-nodes",
                feature_names=("node_id", "f0", "f1"),
            ),
            InputBinding(
                name="edges",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="graph-core-edges",
                feature_names=("source", "destination", "relation"),
            ),
            InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="graph-core-train",
                feature_names=("node_id",),
                label_name="label",
            ),
        ),
        primary_role="train",
    )
    descriptor = _register_test_algorithm()
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=descriptor.registration.spec.name,
            operation=AlgorithmOperation.FIT,
            implementation_id=descriptor.registration.implementation.implementation_id,
            input_binding=bindings,
            algorithm_config={
                "training": {"batch_size": 2, "random_seed": 17},
                "ray": {"storage_path": str(root / "ray-results")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        resources_per_worker=WorkerResources(num_cpus=1),
    )
    result = build_algorithm_dispatcher(
        runtime_manager=cast(Any, _BorrowedClusterRuntimeManager())
    ).execute(
        request,
        InputExecutionContext(values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None:
        raise AssertionError("graph core gate produced no execution receipt")
    portable = receipt.to_dict()
    if not receipt.distributed or not portable["cross_node"]:
        raise AssertionError(f"graph core training did not cross nodes: {portable}")
    if portable["driver_materialized_training_rows"] != 0:
        raise AssertionError("graph training materialized seed rows on the driver")
    evidence = portable["torch_evidence"]
    graph = evidence.get("graph_partition")
    if not isinstance(graph, Mapping):
        raise AssertionError("graph partition evidence is missing")
    if (
        graph["partition_count"] != 2
        or graph["total_nodes"] != 8
        or graph["total_edges"] != 8
    ):
        raise AssertionError(f"graph partition totals are incorrect: {graph}")
    if any(
        nodes_rows >= graph["total_nodes"] for nodes_rows in graph["owner_node_rows"]
    ):
        raise AssertionError(
            f"one graph owner retained the complete node table: {graph}"
        )
    if any(
        edges_rows >= graph["total_edges"] for edges_rows in graph["owner_edge_rows"]
    ):
        raise AssertionError(
            f"one graph owner retained the complete edge table: {graph}"
        )
    workers = evidence["workers"]
    if len(workers) != 2 or len({worker["node_id"] for worker in workers}) != 2:
        raise AssertionError("graph training workers did not run on separate Ray nodes")
    if any(set(worker["input_rows"]) != {"train"} for worker in workers):
        raise AssertionError("a graph role was copied into a training worker")
    if sum(worker["graph"]["seed_rows"] for worker in workers) != 8:
        raise AssertionError("graph worker seed coverage is incomplete")
    if any(
        worker["graph"]["seed_rows"] != worker["input_rows"]["train"]
        for worker in workers
    ):
        raise AssertionError("graph worker sample rows differ from its seed shard")
    if any(
        worker["graph"]["sampled_edge_rows"] < 1
        or worker["graph"]["touched_partitions"] != [0, 1]
        for worker in workers
    ):
        raise AssertionError("graph workers did not read across graph partitions")
    print("GRAPH_CORE_RESULT: " + json.dumps(portable, sort_keys=True))


if __name__ == "__main__":
    main()
