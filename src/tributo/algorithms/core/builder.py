"""High-level builders that lower user integrations to one registration model."""

from __future__ import annotations

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmOperation,
    AlgorithmRegistration,
    BackendInputCompatibility,
    EnvironmentSpec,
    ExecutionMode,
    ImplementationDescriptor,
    QualifiedReference,
    RuntimeBinding,
    RuntimeTopology,
)
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
