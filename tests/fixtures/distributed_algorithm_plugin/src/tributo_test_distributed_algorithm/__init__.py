"""Independent package fixture implementing Tributo's public MapReduce SPI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmInputError,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    MapReducePolicy,
    ResolvedAlgorithmPlan,
    ResultPolicy,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.spi import AlgorithmExecutionContext, MapReduceAlgorithm
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType

_STATE_SCHEMA = (
    StateField("feature_sum", "float64", (None,)),
    StateField("target_sum", "float64", ()),
    StateField("row_count", "int64", ()),
)


@dataclass(frozen=True)
class MeanRegressionModel:
    """Finalized in-memory result used to prove fit semantics completed."""

    feature_means: tuple[float, ...]
    target_mean: float
    row_count: int


class ThirdPartyMeanRegressor(
    MapReduceAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        MeanRegressionModel,
    ]
):
    """Compute bounded global statistics without depending on Tributo builtins."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self._feature_names = plan.input_binding.feature_names
        label_name = plan.input_binding.label_name
        if label_name is None:
            raise AlgorithmConfigurationError(
                "ThirdPartyMeanRegressor requires InputBinding.label_name"
            )
        self._label_name = label_name

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        """Reduce one exclusive input shard to finite numeric sums."""
        del context
        import numpy as np

        state = self.empty_partition()
        for batch in batches:
            missing = [
                name
                for name in (*self._feature_names, self._label_name)
                if name not in batch
            ]
            if missing:
                raise AlgorithmInputError(
                    f"mean-regression batch is missing column(s): {missing}"
                )
            try:
                features = np.column_stack(
                    [
                        np.asarray(batch[name], dtype=np.float64)
                        for name in self._feature_names
                    ]
                )
                targets = np.asarray(batch[self._label_name], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise AlgorithmInputError(
                    f"mean-regression batch is not numeric: {exc}"
                ) from exc
            if features.ndim != 2 or targets.ndim != 1:
                raise AlgorithmInputError(
                    "mean-regression features and target must be columnar"
                )
            if features.shape[0] != targets.shape[0]:
                raise AlgorithmInputError(
                    "mean-regression feature and target row counts disagree"
                )
            if not np.isfinite(features).all() or not np.isfinite(targets).all():
                raise AlgorithmInputError(
                    "mean-regression input must contain only finite numbers"
                )
            state = self.merge_states(
                state,
                {
                    "feature_sum": features.sum(axis=0, dtype=np.float64),
                    "target_sum": np.asarray(
                        targets.sum(dtype=np.float64), dtype=np.float64
                    ),
                    "row_count": np.asarray(targets.shape[0], dtype=np.int64),
                },
            )
        return state

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Associatively add bounded partial states."""
        import numpy as np

        left_features = np.asarray(left["feature_sum"], dtype=np.float64)
        right_features = np.asarray(right["feature_sum"], dtype=np.float64)
        if left_features.shape != right_features.shape:
            raise AlgorithmExecutionError(
                "mean-regression feature state dimensions disagree"
            )
        return {
            "feature_sum": left_features + right_features,
            "target_sum": np.asarray(
                float(np.asarray(left["target_sum"], dtype=np.float64))
                + float(np.asarray(right["target_sum"], dtype=np.float64)),
                dtype=np.float64,
            ),
            "row_count": np.asarray(
                int(np.asarray(left["row_count"], dtype=np.int64))
                + int(np.asarray(right["row_count"], dtype=np.int64)),
                dtype=np.int64,
            ),
        }

    def finalize_model(self, state: Mapping[str, object]) -> MeanRegressionModel:
        """Build one global in-memory model even when publication is disabled."""
        import numpy as np

        row_count = int(np.asarray(state["row_count"], dtype=np.int64))
        if row_count < 1:
            raise AlgorithmInputError("mean regression requires at least one row")
        feature_sum = np.asarray(state["feature_sum"], dtype=np.float64)
        target_sum = float(np.asarray(state["target_sum"], dtype=np.float64))
        return MeanRegressionModel(
            feature_means=tuple(float(value) / row_count for value in feature_sum),
            target_mean=target_sum / row_count,
            row_count=row_count,
        )

    def state_schema(self) -> tuple[StateField, ...]:
        """Return the descriptor-owned bounded state schema."""
        return _STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        """Return the identity element for associative reduction."""
        import numpy as np

        return {
            "feature_sum": np.zeros(len(self._feature_names), dtype=np.float64),
            "target_sum": np.asarray(0.0, dtype=np.float64),
            "row_count": np.asarray(0, dtype=np.int64),
        }

    @property
    def retry_safe(self) -> bool:
        """The map and merge stages are deterministic and side-effect free."""
        return True


def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> ThirdPartyMeanRegressor:
    """Construct only the implementation named by this package descriptor."""
    del artifacts
    if implementation is not ThirdPartyMeanRegressor:
        raise AlgorithmConfigurationError(
            "third-party implementation does not match its descriptor"
        )
    return ThirdPartyMeanRegressor(plan)


DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=AlgorithmSpec(
        name="third_party_mean_regressor",
        trainer_cls=None,
        version="0.1.0",
        default_config={},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(ProblemType.REGRESSION,),
        capabilities=(Capability.DISTRIBUTED,),
        extras_group="training",
        learning_paradigm="supervised",
        model_family="mean_regression",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
        config_contract_ref="example.mean_regression.config.v1",
        input_contract_ref="tributo.tabular.numeric.v1",
        output_contract_ref="example.mean_regression.fit_only.v1",
    ),
    implementation_id="example.mean_regression.map_reduce",
    implementation_version="0.1.0",
    implementation="tributo_test_distributed_algorithm:ThirdPartyMeanRegressor",
    executable_factory="tributo_test_distributed_algorithm:create_algorithm",
    distribution="tributo-test-distributed-algorithm",
    framework=None,
    environment=EnvironmentSpec(
        environment_id="example.mean_regression.v1",
        dependencies=("tributo-test-distributed-algorithm==0.1.0",),
    ),
    allowed_config_keys=(),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(1, 32),
    supported_execution_profiles=(
        ExecutionProfile.LOCAL,
        ExecutionProfile.KUBERNETES,
    ),
    resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
    policy=MapReducePolicy(
        state_schema=_STATE_SCHEMA,
        max_partial_state_bytes=64 * 1024,
        reducer_ref=(
            "tributo_test_distributed_algorithm:ThirdPartyMeanRegressor.merge_states"
        ),
        finalizer_ref=(
            "tributo_test_distributed_algorithm:ThirdPartyMeanRegressor.finalize_model"
        ),
        commutative=True,
        max_retries=0,
    ),
    package_name="tributo-test-distributed-algorithm",
    package_version="0.1.0",
    tributo_version_spec=">=1,<2",
    result_policy=ResultPolicy.FIT_ONLY,
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(ExecutionProfile.LOCAL,),
    limitations=("CPU-only fit-only conformance fixture.",),
    is_default=True,
)
REGISTRATION = DESCRIPTOR.registration

__all__ = [
    "DESCRIPTOR",
    "MeanRegressionModel",
    "REGISTRATION",
    "ThirdPartyMeanRegressor",
]
