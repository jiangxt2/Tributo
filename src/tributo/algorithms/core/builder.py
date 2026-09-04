"""High-level builders that lower user integrations to one registration model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    CollectivePolicy,
    ContractBindingSet,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    FrameworkNativePolicy,
    ImplementationDescriptor,
    InputDistribution,
    IterativeOptimizationPolicy,
    JoblibEstimatorPolicy,
    MapReducePolicy,
    MetricReduction,
    ParallelEnsemblePolicy,
    QualifiedReference,
    ResultPolicy,
    RuntimeBinding,
    RuntimeTopology,
    SingleStageTorchPlan,
    TorchDatasetRoute,
    TorchPolicy,
    TorchStageSpec,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.training.algorithm_spec import AlgorithmSpec
from tributo.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class AlgorithmBuilder:
    """Construct provisional registrations without exposing provider internals."""

    _PRODUCTION_INPUT_ADAPTER = QualifiedReference.parse(
        "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input"
    )
    _PRODUCTION_INPUT_ADAPTERS = tuple(
        QualifiedReference.parse(reference)
        for reference in (
            "tributo.integrations.algorithm_inputs.ingestion:prepare_daft_input",
            "tributo.integrations.algorithm_inputs.ingestion:prepare_ingestion_input",
            "tributo.integrations.algorithm_inputs.ingestion:prepare_ray_data_input",
        )
    )

    @staticmethod
    def _default_input_compatibility(
        *topologies: RuntimeTopology,
    ) -> BackendInputCompatibility:
        return BackendInputCompatibility(
            accepted_input_views=(
                "daft_dataframe",
                "ray_data",
            ),
            accepted_ingestion_engines=(
                "tributo.daft",
                "tributo.ray_data",
            ),
            required_input_capabilities=("materializable",),
            supported_explicit_adapters=AlgorithmBuilder._PRODUCTION_INPUT_ADAPTERS,
            distribution_policy=topologies,
        )

    @staticmethod
    def _formal_input_compatibility(
        topology: RuntimeTopology,
        adapter: QualifiedReference,
        *,
        sharded: bool,
    ) -> BackendInputCompatibility:
        return BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=(
                ("shardable",) if sharded else ("materializable",)
            ),
            supported_explicit_adapters=(adapter,),
            distribution_policy=(topology,),
        )

    @staticmethod
    def from_distributed_algorithm(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        implementation: str,
        executable_factory: str,
        distribution: str,
        framework: str | None,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...],
        strategy: DistributionStrategy,
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        policy: (
            CollectivePolicy
            | MapReducePolicy
            | FrameworkNativePolicy
            | JoblibEstimatorPolicy
            | ParallelEnsemblePolicy
            | IterativeOptimizationPolicy
            | TorchPolicy
        ),
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED,
        input_compatibility: BackendInputCompatibility | None = None,
        exporter: str | None = None,
        flavor_id: str | None = None,
        distributed_min_workers: int = 2,
        stability: Literal["alpha", "beta", "stable"] = "alpha",
        tested: bool = False,
        supported: bool = False,
        validated_execution_profiles: tuple[ExecutionProfile, ...] = (),
        limitations: tuple[str, ...] = (),
        is_default: bool = False,
        code_digest: str | None = None,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 1,
    ) -> DistributedAlgorithmDescriptor:
        """Lower one formal distributed implementation to its public descriptor.

        The current formal distributed Builder supports only the ``fit``
        operation. Implementations that also expose ``evaluate`` or
        ``predict`` must construct an ``AlgorithmRegistration`` directly.
        """
        try:
            resolved_strategy = DistributionStrategy(strategy)
            operations = tuple(
                AlgorithmOperation(operation) for operation in spec.operations
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"formal distributed algorithm has an invalid enum value: {exc}"
            ) from exc
        if operations != (AlgorithmOperation.FIT,):
            raise AlgorithmConfigurationError(
                "formal distributed algorithm Builder supports only operation='fit'"
            )

        contract = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[resolved_strategy]
        execution_mode = contract.execution_mode
        runtime_id = contract.runtime_id
        topology = contract.topology
        input_distribution = contract.input_distribution
        coordination = contract.state_coordination
        adapter = QualifiedReference.parse(contract.worker_input_adapter_ref)
        compatibility = input_compatibility or (
            AlgorithmBuilder._formal_input_compatibility(
                topology,
                adapter,
                sharded=input_distribution is not InputDistribution.FULL_DATASET,
            )
        )
        if compatibility.distribution_policy != (topology,):
            raise AlgorithmConfigurationError(
                "input compatibility conflicts with the distributed strategy topology"
            )
        if adapter not in compatibility.supported_explicit_adapters:
            raise AlgorithmConfigurationError(
                "input compatibility must include the strategy's standard Ray Data "
                "Worker adapter"
            )
        required_capability = (
            "shardable"
            if input_distribution is not InputDistribution.FULL_DATASET
            else "materializable"
        )
        if (
            "ray_data" not in compatibility.accepted_input_views
            or "tributo.ray_data" not in compatibility.accepted_ingestion_engines
            or required_capability not in compatibility.required_input_capabilities
        ):
            raise AlgorithmConfigurationError(
                "formal distributed input compatibility must retain the standard "
                f"{required_capability} Ray Data contract"
            )

        registration = AlgorithmRegistration(
            spec=spec,
            implementation=ImplementationDescriptor(
                implementation_id=implementation_id,
                version=implementation_version,
                execution_mode=execution_mode,
                implementation_ref=QualifiedReference.parse(implementation),
                executable_factory_ref=QualifiedReference.parse(executable_factory),
                operations=operations,
                input_compatibility=compatibility,
                distribution=distribution,
                framework=framework,
                code_digest=code_digest,
                artifact_format="none",
                allowed_config_keys=allowed_config_keys,
                runtime_id=runtime_id,
                worker_input_adapter_ref=adapter,
                exporter_ref=(
                    QualifiedReference.parse(exporter) if exporter is not None else None
                ),
                flavor_id=flavor_id,
            ),
            environment=environment,
            distribution_spec=DistributionSpec(
                strategy=resolved_strategy,
                supported_worker_range=supported_worker_range,
                supported_execution_profiles=supported_execution_profiles,
                resources_per_worker=resources_per_worker,
                input_distribution=input_distribution,
                state_coordination=coordination,
                policy=policy,
                distributed_min_workers=distributed_min_workers,
                result_policy=result_policy,
            ),
            contract_bindings=contract_bindings,
            is_default=is_default,
        )
        return DistributedAlgorithmDescriptor(
            registration=registration,
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            stability=stability,
            tested=tested,
            supported=supported,
            validated_execution_profiles=validated_execution_profiles,
            limitations=limitations,
            api_version=descriptor_api_version,
        )

    @staticmethod
    def from_torch(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        recipe: str,
        environment: EnvironmentSpec,
        metric_reducers: Mapping[str, MetricReduction],
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        policy: TorchPolicy | None = None,
        backend: Literal["auto", "gloo", "nccl"] = "auto",
        distributed_min_workers: int = 2,
        stability: Literal["alpha", "beta", "stable"] = "alpha",
        tested: bool = False,
        supported: bool = False,
        validated_execution_profiles: tuple[ExecutionProfile, ...] = (),
        limitations: tuple[str, ...] = (),
        is_default: bool = False,
        code_digest: str | None = None,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 2,
    ) -> DistributedAlgorithmDescriptor:
        """Build a registration for the unified Core-owned Torch Runtime."""
        if code_digest is None:
            raise AlgorithmConfigurationError(
                "Torch implementations require an explicit code_digest"
            )
        normalized_reducers = {
            name: MetricReduction(reduction)
            for name, reduction in metric_reducers.items()
        }
        if "train_loss" in normalized_reducers and (
            normalized_reducers["train_loss"] is not MetricReduction.SUM_COUNT
        ):
            raise AlgorithmConfigurationError(
                "Torch train_loss uses the fixed sum_count reducer"
            )
        normalized_reducers["train_loss"] = MetricReduction.SUM_COUNT
        resolved_policy = policy or TorchPolicy(
            torch_runtime_api_version=1,
            loop_owner="core_recipe",
            parallelism_id="torch.ddp.replicated",
            dataset_routing=(
                TorchDatasetRoute(
                    role="train",
                    mode="split_exact",
                    required=True,
                    min_total_rows_if_present=1,
                    min_rows_per_worker=1,
                    empty_rank_policy="reject",
                ),
            ),
            execution_plan=SingleStageTorchPlan(
                stage=TorchStageSpec(
                    stage_id="train",
                    input_roles=("train",),
                )
            ),
            state_layout="replicated",
            metric_reducers=normalized_reducers,
            backend=backend,
            resume_supported=False,
        )
        if resolved_policy.backend != backend and policy is not None:
            raise AlgorithmConfigurationError(
                "Torch Policy backend conflicts with the requested backend"
            )
        if dict(resolved_policy.metric_reducers) != normalized_reducers:
            raise AlgorithmConfigurationError(
                "Torch Policy metric reducers conflict with the Builder declaration"
            )
        return AlgorithmBuilder.from_distributed_algorithm(
            spec=spec,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            implementation=recipe,
            executable_factory=(
                "tributo.integrations.algorithm_runtimes.ray_train_torch:"
                "create_torch_algorithm"
            ),
            distribution=package_name,
            framework="pytorch",
            environment=environment,
            allowed_config_keys=(
                "data",
                "loss",
                "metrics",
                "model",
                "optimizer",
                "output",
                "ray",
                "training",
            ),
            strategy=DistributionStrategy.RAY_TRAIN_TORCH,
            supported_worker_range=supported_worker_range,
            supported_execution_profiles=supported_execution_profiles,
            resources_per_worker=resources_per_worker,
            policy=resolved_policy,
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            result_policy=ResultPolicy.BUNDLE_REQUIRED,
            exporter=(
                "tributo.integrations.algorithm_runtimes.ray_train_torch:"
                "export_ray_train_torch_result"
            ),
            flavor_id="onnx-runtime-v1",
            distributed_min_workers=distributed_min_workers,
            stability=stability,
            tested=tested,
            supported=supported,
            validated_execution_profiles=validated_execution_profiles,
            limitations=limitations,
            is_default=is_default,
            code_digest=code_digest,
            contract_bindings=contract_bindings,
            descriptor_api_version=descriptor_api_version,
        )

    @staticmethod
    def from_torch_adapter(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        adapter: str,
        environment: EnvironmentSpec,
        metric_reducers: Mapping[str, MetricReduction],
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        policy: TorchPolicy,
        distributed_min_workers: int = 2,
        stability: Literal["alpha", "beta", "stable"] = "alpha",
        tested: bool = False,
        supported: bool = False,
        validated_execution_profiles: tuple[ExecutionProfile, ...] = (),
        limitations: tuple[str, ...] = (),
        is_default: bool = False,
        code_digest: str | None = None,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 2,
    ) -> DistributedAlgorithmDescriptor:
        """Build a registration for a framework-owned ``RayTorchAdapter``."""
        if policy.loop_owner != "adapter":
            raise AlgorithmConfigurationError(
                "Torch adapter registrations require policy.loop_owner='adapter'"
            )
        return AlgorithmBuilder.from_torch(
            spec=spec,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            recipe=adapter,
            environment=environment,
            metric_reducers=metric_reducers,
            supported_worker_range=supported_worker_range,
            supported_execution_profiles=supported_execution_profiles,
            resources_per_worker=resources_per_worker,
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            policy=policy,
            distributed_min_workers=distributed_min_workers,
            stability=stability,
            tested=tested,
            supported=supported,
            validated_execution_profiles=validated_execution_profiles,
            limitations=limitations,
            is_default=is_default,
            code_digest=code_digest,
            contract_bindings=contract_bindings,
            descriptor_api_version=descriptor_api_version,
        )

    @staticmethod
    def from_joblib_estimator_recipe(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        recipe: str,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...],
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        policy: JoblibEstimatorPolicy | None = None,
        input_compatibility: BackendInputCompatibility | None = None,
        result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED,
        exporter: str | None = None,
        flavor_id: str | None = None,
        distributed_min_workers: int = 2,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 2,
        is_default: bool = False,
    ) -> DistributedAlgorithmDescriptor:
        """Lower estimator mathematics to the Core Ray Joblib Runtime."""
        return AlgorithmBuilder.from_distributed_algorithm(
            spec=spec,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            implementation=recipe,
            executable_factory=(
                "tributo.integrations.algorithm_runtimes.decomposition:create_algorithm"
            ),
            distribution=package_name,
            framework="sklearn",
            environment=environment,
            allowed_config_keys=allowed_config_keys,
            strategy=DistributionStrategy.RAY_JOBLIB_ESTIMATOR,
            supported_worker_range=supported_worker_range,
            supported_execution_profiles=supported_execution_profiles,
            resources_per_worker=resources_per_worker,
            policy=policy or JoblibEstimatorPolicy(),
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            result_policy=result_policy,
            input_compatibility=input_compatibility,
            exporter=exporter,
            flavor_id=flavor_id,
            distributed_min_workers=distributed_min_workers,
            contract_bindings=contract_bindings,
            descriptor_api_version=descriptor_api_version,
            is_default=is_default,
        )

    @staticmethod
    def from_parallel_ensemble(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        algorithm: str,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...],
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        policy: ParallelEnsemblePolicy,
        input_compatibility: BackendInputCompatibility | None = None,
        result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED,
        exporter: str | None = None,
        flavor_id: str | None = None,
        distributed_min_workers: int = 2,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 2,
        is_default: bool = False,
    ) -> DistributedAlgorithmDescriptor:
        """Lower independent unit Hook methods to the Core Ensemble Runtime."""
        return AlgorithmBuilder.from_distributed_algorithm(
            spec=spec,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            implementation=algorithm,
            executable_factory=(
                "tributo.integrations.algorithm_runtimes.decomposition:create_algorithm"
            ),
            distribution=package_name,
            framework=None,
            environment=environment,
            allowed_config_keys=allowed_config_keys,
            strategy=DistributionStrategy.RAY_PARALLEL_ENSEMBLE,
            supported_worker_range=supported_worker_range,
            supported_execution_profiles=supported_execution_profiles,
            resources_per_worker=resources_per_worker,
            policy=policy,
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            result_policy=result_policy,
            input_compatibility=input_compatibility,
            exporter=exporter,
            flavor_id=flavor_id,
            distributed_min_workers=distributed_min_workers,
            contract_bindings=contract_bindings,
            descriptor_api_version=descriptor_api_version,
            is_default=is_default,
        )

    @staticmethod
    def from_iterative_optimization(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        algorithm: str,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...],
        supported_worker_range: WorkerRange,
        supported_execution_profiles: tuple[ExecutionProfile, ...],
        resources_per_worker: WorkerResources,
        package_name: str,
        package_version: str,
        tributo_version_spec: str,
        policy: IterativeOptimizationPolicy,
        input_compatibility: BackendInputCompatibility | None = None,
        result_policy: ResultPolicy = ResultPolicy.BUNDLE_REQUIRED,
        exporter: str | None = None,
        flavor_id: str | None = None,
        distributed_min_workers: int = 2,
        contract_bindings: ContractBindingSet | None = None,
        descriptor_api_version: int = 2,
        is_default: bool = False,
    ) -> DistributedAlgorithmDescriptor:
        """Lower shard-update Hook methods to the Core Iterative Runtime."""
        return AlgorithmBuilder.from_distributed_algorithm(
            spec=spec,
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            implementation=algorithm,
            executable_factory=(
                "tributo.integrations.algorithm_runtimes.decomposition:create_algorithm"
            ),
            distribution=package_name,
            framework=None,
            environment=environment,
            allowed_config_keys=allowed_config_keys,
            strategy=DistributionStrategy.RAY_ITERATIVE_OPTIMIZATION,
            supported_worker_range=supported_worker_range,
            supported_execution_profiles=supported_execution_profiles,
            resources_per_worker=resources_per_worker,
            policy=policy,
            package_name=package_name,
            package_version=package_version,
            tributo_version_spec=tributo_version_spec,
            result_policy=result_policy,
            input_compatibility=input_compatibility,
            exporter=exporter,
            flavor_id=flavor_id,
            distributed_min_workers=distributed_min_workers,
            contract_bindings=contract_bindings,
            descriptor_api_version=descriptor_api_version,
            is_default=is_default,
        )

    @staticmethod
    def from_sklearn(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        estimator_factory: str,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...],
        input_compatibility: BackendInputCompatibility | None = None,
        runtime: RuntimeBinding | None = None,
        framework_parallelism: int = 1,
        num_cpus: float | None = None,
        num_gpus: float | None = None,
        trusted_pickle: bool = False,
        is_default: bool = False,
        code_digest: str | None = None,
    ) -> AlgorithmRegistration:
        """Lower a module-qualified sklearn factory to AlgorithmRegistration."""
        try:
            operations = tuple(
                AlgorithmOperation(operation) for operation in spec.operations
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"managed sklearn has an invalid operation: {exc}"
            ) from exc
        supported = {
            AlgorithmOperation.FIT,
            AlgorithmOperation.EVALUATE,
            AlgorithmOperation.PREDICT,
        }
        unsupported = sorted(
            operation.value for operation in operations if operation not in supported
        )
        if unsupported:
            raise AlgorithmConfigurationError(
                f"managed sklearn does not support operation(s): {unsupported}"
            )
        if not trusted_pickle and any(
            operation in {AlgorithmOperation.EVALUATE, AlgorithmOperation.PREDICT}
            for operation in operations
        ):
            raise AlgorithmConfigurationError(
                "managed sklearn evaluate/predict requires an explicitly trusted "
                "model persistence mode"
            )
        if runtime is not None and (
            framework_parallelism != 1 or num_cpus is not None or num_gpus is not None
        ):
            raise AlgorithmConfigurationError(
                "pass either runtime or Builder topology/resources, not both"
            )
        resolved_runtime = runtime or RuntimeBinding(
            runtime_id="tributo.ray_task",
            worker_input_adapter_ref=AlgorithmBuilder._PRODUCTION_INPUT_ADAPTER,
            topology=(
                RuntimeTopology.FRAMEWORK_MANAGED
                if framework_parallelism > 1
                else RuntimeTopology.SINGLE_WORKER
            ),
            framework_parallelism=framework_parallelism,
            num_cpus=(
                num_cpus
                if num_cpus is not None
                else (0 if framework_parallelism > 1 else 1)
            ),
            num_gpus=num_gpus if num_gpus is not None else 0,
        )
        return AlgorithmRegistration(
            spec=spec,
            implementation=ImplementationDescriptor(
                implementation_id=implementation_id,
                version=implementation_version,
                execution_mode=ExecutionMode.MANAGED_ESTIMATOR,
                implementation_ref=QualifiedReference.parse(estimator_factory),
                executable_factory_ref=QualifiedReference.parse(
                    "tributo.integrations.algorithm_runtimes.sklearn:create_executable"
                ),
                operations=operations,
                input_compatibility=(
                    input_compatibility
                    or AlgorithmBuilder._default_input_compatibility(
                        RuntimeTopology.SINGLE_WORKER,
                        RuntimeTopology.FRAMEWORK_MANAGED,
                    )
                ),
                distribution="scikit-learn",
                framework="sklearn",
                code_digest=code_digest,
                artifact_format="trusted_pickle" if trusted_pickle else "none",
                allowed_config_keys=allowed_config_keys,
            ),
            environment=environment,
            runtime=resolved_runtime,
            is_default=is_default,
        )

    @staticmethod
    def from_ray_function(
        *,
        spec: AlgorithmSpec,
        implementation_id: str,
        implementation_version: str,
        function: str,
        environment: EnvironmentSpec,
        allowed_config_keys: tuple[str, ...] = (),
        input_compatibility: BackendInputCompatibility | None = None,
        runtime: RuntimeBinding | None = None,
        worker_count: int = 1,
        result_reducer: str | None = None,
        num_cpus: float | None = None,
        num_gpus: float | None = None,
        distribution: str | None = None,
        is_default: bool = False,
        code_digest: str | None = None,
    ) -> AlgorithmRegistration:
        """Lower a trusted module-qualified user function to a registration."""
        try:
            operations = tuple(
                AlgorithmOperation(operation) for operation in spec.operations
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmConfigurationError(
                f"custom Ray function has an invalid operation: {exc}"
            ) from exc
        if operations != (AlgorithmOperation.FIT,):
            raise AlgorithmConfigurationError(
                "the first Custom Ray Function slice supports only operation='fit'"
            )
        if runtime is not None and (
            worker_count != 1
            or result_reducer is not None
            or num_cpus is not None
            or num_gpus is not None
        ):
            raise AlgorithmConfigurationError(
                "pass either runtime or Builder topology/resources, not both"
            )
        if runtime is None:
            topology = (
                RuntimeTopology.DATA_PARALLEL
                if worker_count > 1
                else RuntimeTopology.SINGLE_WORKER
            )
            reducer_ref = (
                QualifiedReference.parse(result_reducer)
                if result_reducer is not None
                else None
            )
            runtime = RuntimeBinding(
                runtime_id="tributo.ray_task",
                worker_input_adapter_ref=AlgorithmBuilder._PRODUCTION_INPUT_ADAPTER,
                topology=topology,
                worker_count=worker_count,
                result_reducer_ref=reducer_ref,
                num_cpus=num_cpus if num_cpus is not None else 1,
                num_gpus=num_gpus if num_gpus is not None else 0,
            )
        return AlgorithmRegistration(
            spec=spec,
            implementation=ImplementationDescriptor(
                implementation_id=implementation_id,
                version=implementation_version,
                execution_mode=ExecutionMode.CUSTOM_RAY_FUNCTION,
                implementation_ref=QualifiedReference.parse(function),
                executable_factory_ref=QualifiedReference.parse(
                    "tributo.integrations.algorithm_runtimes.ray_function:create_executable"
                ),
                operations=operations,
                input_compatibility=(
                    input_compatibility
                    or AlgorithmBuilder._default_input_compatibility(
                        RuntimeTopology.SINGLE_WORKER,
                        RuntimeTopology.DATA_PARALLEL,
                    )
                ),
                distribution=distribution,
                framework=None,
                code_digest=code_digest,
                artifact_format="none",
                allowed_config_keys=allowed_config_keys,
            ),
            environment=environment,
            runtime=runtime,
            is_default=is_default,
        )


__all__ = ["AlgorithmBuilder"]
