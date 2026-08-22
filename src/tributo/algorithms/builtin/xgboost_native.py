"""First-party framework-native XGBoost distributed algorithm adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import ray

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    FrameworkNativePolicy,
    ImplementationDescriptor,
    InputDistribution,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    StateCoordination,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import FrameworkNativeAlgorithm
from tributo.integrations.algorithm_runtimes.legacy_descriptors import (
    XGBOOST_DESCRIPTOR as LEGACY_XGBOOST_DESCRIPTOR,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

_FRAMEWORK_NATIVE_CONTRACT = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.FRAMEWORK_NATIVE
]
_RAY_TRAIN_INPUT = QualifiedReference.parse(
    _FRAMEWORK_NATIVE_CONTRACT.worker_input_adapter_ref
)


class _EvidenceCollectorState:
    """Collect one bounded framework evidence record per worker rank."""

    def __init__(self) -> None:
        self._records: dict[int, dict[str, object]] = {}

    def record(self, value: Mapping[str, object]) -> None:
        rank = cast(int, value["rank"])
        if rank in self._records:
            raise ValueError(f"duplicate XGBoost worker evidence rank: {rank}")
        self._records[rank] = dict(value)

    def snapshot(self) -> list[dict[str, object]]:
        return [self._records[rank] for rank in sorted(self._records)]


@PublicAPI(stability="alpha")
class DistributedXGBoost(FrameworkNativeAlgorithm):
    """Bind Tributo's established Ray Train XGBoost implementation to the SPI."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self._collector: Any | None = None
        ray_config = plan.algorithm_config.get("ray", {})
        if not isinstance(ray_config, Mapping):
            raise AlgorithmConfigurationError("ray config must be a mapping")
        if (
            "num_workers" in ray_config
            and int(ray_config["num_workers"]) != plan.runtime.worker_count
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.num_workers must match "
                "ExecutionRequest.worker_count"
            )
        if "use_gpu" in ray_config and bool(ray_config["use_gpu"]) != (
            plan.runtime.num_gpus > 0
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.use_gpu must match the formal per-worker "
                "GPU request"
            )
        if int(ray_config.get("max_failures", 0)) != 0:
            raise AlgorithmConfigurationError(
                "formal framework-native XGBoost requires ray.max_failures=0 "
                "until failure-injection evidence is gated"
            )
        resume = ray_config.get("resume", {})
        if not isinstance(resume, Mapping):
            raise AlgorithmConfigurationError("ray.resume config must be a mapping")
        configured_checkpoint = resume.get("checkpoint_path")
        if configured_checkpoint is not None and (
            configured_checkpoint != plan.runtime.resume_from
        ):
            raise AlgorithmConfigurationError(
                "algorithm_config.ray.resume.checkpoint_path must match "
                "ExecutionRequest.resume_from"
            )
        if plan.runtime.worker_count > 1 and (
            plan.runtime.resume_from is not None or configured_checkpoint is not None
        ):
            raise AlgorithmConfigurationError(
                "multi-worker XGBoost resume is not supported by the formal "
                "framework-native adapter"
            )

    def validate_environment(self) -> None:
        """Verify XGBoost's Ray Train integration before opening training data."""
        try:
            import xgboost
            from ray.train.xgboost import XGBoostTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "framework-native XGBoost requires xgboost and ray.train.xgboost"
            ) from exc
        if not xgboost.__version__ or XGBoostTrainer is None:
            raise AlgorithmConfigurationError("XGBoost environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        """Filter, project, and split one canonical Ray Dataset on Driver plans."""
        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "XGBoost requires one canonical Ray Dataset input"
            )
        dataset: Any = next(iter(datasets.values()))
        from tributo.training.xgboost_evaluator import (
            filter_invalid_labels,
            split_dataset,
        )
        from tributo.training.xgboost_trainer import XGBoostTrainingConfig

        cfg = XGBoostTrainingConfig.model_validate(self.plan.algorithm_config)
        dataset = filter_invalid_labels(dataset, label_col=cfg.data.label_col)
        if cfg.data.feature_columns:
            dataset = dataset.select_columns(
                [*cfg.data.feature_columns, cfg.data.label_col]
            )
        train, validation, test = split_dataset(
            dataset,
            val_size=cfg.training.val_size,
            test_size=cfg.training.test_size,
            seed=cfg.training.seed,
        )
        resolved: dict[str, object] = {"train": train}
        if validation is not None:
            resolved["val"] = validation
        if test is not None:
            resolved["test"] = test
        return resolved

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        """Build the established XGBoostTrainer with formal resource choices."""
        from tributo.training.checkpoint import load_initial_checkpoint
        from tributo.training.xgboost_trainer import (
            XGBoostTrainingConfig,
        )

        cfg = XGBoostTrainingConfig.model_validate(config)
        collector_actor: Any = ray.remote(_EvidenceCollectorState).options(num_cpus=0)
        self._collector = collector_actor.remote()
        typed_datasets = cast(dict[str, Any], dict(datasets))
        label_col = cfg.data.label_col
        xgb_params = {
            key: value
            for key, value in cfg.model.model_dump(exclude={"objective"}).items()
            if not key.startswith("_")
        }
        xgb_params["objective"] = cfg.model.objective
        train_config: dict[str, Any] = {
            "label_col": label_col,
            "xgb_params": xgb_params,
            "num_rounds": cfg.training.num_rounds,
            "max_rows_per_worker": cfg.training.max_rows_per_worker,
            "resource": cfg.resource.model_dump(),
            "resume": {
                **cfg.ray.resume.model_dump(),
                **({"enabled": True} if self.plan.runtime.resume_from else {}),
            },
            "_tributo_evidence_actor": self._collector,
            "_tributo_input_binding_digest": (
                self.plan.input_descriptor.binding_digest
            ),
        }
        if "val" in datasets and cfg.training.early_stopping_rounds:
            train_config["early_stopping_rounds"] = cfg.training.early_stopping_rounds
        from tributo.training.xgboost_trainer import _build_trainer

        return _build_trainer(
            ray_dataset=typed_datasets["train"],
            train_config=train_config,
            val_dataset=typed_datasets.get("val"),
            test_dataset=typed_datasets.get("test"),
            num_workers=self.plan.runtime.worker_count,
            use_gpu=self.plan.runtime.num_gpus > 0,
            resources_per_worker={
                "CPU": self.plan.runtime.num_cpus,
                "GPU": self.plan.runtime.num_gpus,
                **dict(self.plan.runtime.custom_resources),
            },
            storage_path=cfg.ray.storage_path,
            max_failures=cfg.ray.max_failures,
            resume_from_checkpoint=load_initial_checkpoint(
                self.plan.runtime.resume_from or cfg.ray.resume.checkpoint_path
            ),
            run_name="tributo-xgboost",
        )

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        """Validate every framework worker reported one synchronized Booster."""
        del result
        if self._collector is None:
            raise AlgorithmExecutionError("XGBoost evidence collector was not created")
        workers = ray.get(self._collector.snapshot.remote())
        expected = self.plan.runtime.worker_count
        if len(workers) != expected:
            raise AlgorithmExecutionError(
                "XGBoost did not report evidence for every requested worker"
            )
        ranks = {int(item["rank"]) for item in workers}
        shards = {str(item["shard_id"]) for item in workers}
        digests = {str(item["model_state_digest"]) for item in workers}
        if ranks != set(range(expected)) or len(shards) != expected:
            raise AlgorithmExecutionError(
                "XGBoost did not prove unique framework-owned input shards"
            )
        if len(digests) != 1:
            raise AlgorithmExecutionError(
                "XGBoost workers did not produce one synchronized Booster"
            )
        return {
            "workers": workers,
            "state": {
                "coordination": "framework_native",
                "synchronized": True,
                "bounded": True,
                "global_model_digest": next(iter(digests)),
                "details": {"framework": "xgboost", "collective": "rabit"},
            },
            "input_complete": True,
        }

    def checkpoint_source(self, result: object) -> object:
        """Return the consolidated XGBoost checkpoint owned by Ray Train."""
        checkpoint = getattr(result, "checkpoint", None)
        if checkpoint is None:
            raise AlgorithmExecutionError("XGBoost result has no checkpoint")
        return checkpoint


@DeveloperAPI
def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> FrameworkNativeAlgorithm:
    """Instantiate only the declared first-party XGBoost SPI implementation."""
    del artifacts
    if implementation is not DistributedXGBoost:
        raise AlgorithmConfigurationError(
            "XGBoost implementation reference does not match the SPI class"
        )
    return DistributedXGBoost(plan)


@DeveloperAPI
def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    """Publish the framework checkpoint through the existing Bundle service."""
    del checkpoint
    output = plan.algorithm_config.get("output", {})
    if not isinstance(output, Mapping):
        raise AlgorithmConfigurationError("output config must be a mapping")
    bundle_uri = output.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not bundle_uri:
        raise AlgorithmConfigurationError(
            "formal XGBoost training requires output.bundle_uri"
        )
    from tributo.training.lifecycle import publish_existing_training_result
    from tributo.training.xgboost_trainer import (
        XGBoostTrainerImpl,
        XGBoostTrainingConfig,
    )

    cfg = XGBoostTrainingConfig.model_validate(plan.algorithm_config)
    trainer = XGBoostTrainerImpl(
        datasets={},
        config=dict(plan.algorithm_config),
        _validated_config=cfg,
    )
    published = publish_existing_training_result(
        trainer,
        result,
        bundle_uri=bundle_uri,
        run_id=run_id,
    )
    metrics = getattr(result, "metrics", None) or {}
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            key: value
            for key, value in metrics.items()
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        },
        outputs={
            "bundle_id": published["bundle_id"],
            "bundle_uri": published["canonical_uri"],
            "execution_id": published["execution_id"],
            "manifest_sha256": published["manifest_sha256"],
        },
    )


XGBOOST_REGISTRATION = AlgorithmRegistration(
    spec=LEGACY_XGBOOST_DESCRIPTOR.registration.spec,
    implementation=ImplementationDescriptor(
        implementation_id="tributo.xgboost.framework_native",
        version="1.0.0",
        execution_mode=ExecutionMode.FRAMEWORK_NATIVE,
        implementation_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.xgboost_native:DistributedXGBoost"
        ),
        executable_factory_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.xgboost_native:create_algorithm"
        ),
        operations=(AlgorithmOperation.FIT,),
        input_compatibility=BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=("shardable",),
            supported_explicit_adapters=(_RAY_TRAIN_INPUT,),
            distribution_policy=(RuntimeTopology.FRAMEWORK_NATIVE,),
        ),
        distribution="xgboost",
        framework="xgboost",
        allowed_config_keys=("data", "model", "training", "ray", "resource", "output"),
        runtime_id=_FRAMEWORK_NATIVE_CONTRACT.runtime_id,
        worker_input_adapter_ref=_RAY_TRAIN_INPUT,
        exporter_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.xgboost_native:export_result"
        ),
        flavor_id="onnx-runtime-v1",
    ),
    environment=EnvironmentSpec(
        environment_id="tributo.xgboost.framework_native.v1",
        dependencies=(
            "onnx>=1.16.0",
            "onnxruntime>=1.20.0",
            "onnxmltools>=1.13.0",
            "ray==2.55.1",
            "skl2onnx>=1.17.0",
            "xgboost>=2.1.0",
        ),
    ),
    distribution_spec=DistributionSpec(
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        supported_worker_range=WorkerRange(1, 1024),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.CLUSTER,
        ),
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
        input_distribution=InputDistribution.FRAMEWORK_OWNED,
        state_coordination=StateCoordination.FRAMEWORK_NATIVE,
        policy=FrameworkNativePolicy(
            framework="xgboost-rabit",
            evidence_collector_ref=(
                "tributo.algorithms.builtin.xgboost_native:"
                "DistributedXGBoost.collect_evidence"
            ),
        ),
    ),
    is_default=True,
)

XGBOOST_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=XGBOOST_REGISTRATION,
    package_name="tributo",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(
        ExecutionProfile.LOCAL,
        ExecutionProfile.CLUSTER,
    ),
    limitations=(
        "CPU distributed training is supported; GPU requires a separate gate.",
        "Multi-worker checkpoint resume is not supported.",
        "Automatic worker retries are rejected until failure injection is gated.",
    ),
)


__all__ = [
    "DistributedXGBoost",
    "create_algorithm",
    "export_result",
]
