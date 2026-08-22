"""First-party distributed X-Learner over native Ray Train XGBoost stages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
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
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType
from tributo.training.x_learner import (
    X_LEARNER_FORMULA,
    X_LEARNER_STAGES,
    XLearnerConfig,
    XLearnerFitDriver,
    XLearnerTrainingResult,
    split_x_learner_dataset,
    validate_x_learner_dataset,
)
from tributo.util.annotations import DeveloperAPI, PublicAPI

_CONTRACT = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[DistributionStrategy.FRAMEWORK_NATIVE]
_INPUT_ADAPTER = QualifiedReference.parse(_CONTRACT.worker_input_adapter_ref)


def _validate_local_control_plane_headroom(
    *,
    profile: ExecutionProfile | None,
    worker_count: int,
    cluster_resources: Mapping[str, float],
) -> None:
    """Fail before data execution when owned local Ray lacks control headroom."""
    if profile is None:
        raise AlgorithmConfigurationError("X-Learner execution profile is unresolved")
    if profile is not ExecutionProfile.LOCAL:
        return
    required_cpus = worker_count + 1
    available_cpus = float(cluster_resources.get("CPU", 0.0))
    if available_cpus < required_cpus:
        raise AlgorithmConfigurationError(
            "X-Learner local execution requires one Ray control-plane CPU beyond "
            f"its workers: required={required_cpus:g} available={available_cpus:g}"
        )


@PublicAPI(stability="alpha")
class DistributedXLearner(FrameworkNativeAlgorithm):
    """Bind one fixed X-Learner stage plan to Ray's XGBoost trainers."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self.plan = plan
        self.config = XLearnerConfig.model_validate(plan.algorithm_config)
        if self.config.ray.num_workers != plan.runtime.worker_count:
            raise AlgorithmConfigurationError(
                "X-Learner ray.num_workers must match ExecutionRequest.worker_count"
            )
        if plan.runtime.num_gpus != 0:
            raise AlgorithmConfigurationError(
                "X-Learner is CPU-only until a dedicated GPU gate passes"
            )
        if plan.input_binding.sample_weight_name is not None:
            raise AlgorithmConfigurationError(
                "X-Learner does not support sample weights"
            )
        data = self.config.data
        projected = set(plan.input_binding.feature_names)
        expected = {*data.feature_columns, data.treatment_col, data.identity_col}
        if projected != expected or plan.input_binding.label_name != data.outcome_col:
            raise AlgorithmConfigurationError(
                "X-Learner InputBinding must project model features, treatment, "
                "identity, and the outcome label"
            )

    def validate_environment(self) -> None:
        try:
            import xgboost
            from ray.train.xgboost import XGBoostTrainer
        except ImportError as exc:
            raise AlgorithmConfigurationError(
                "distributed X-Learner requires xgboost and ray.train.xgboost"
            ) from exc
        if not xgboost.__version__ or XGBoostTrainer is None:
            raise AlgorithmConfigurationError("X-Learner environment is invalid")

    def bind_datasets(self, datasets: Mapping[str, object]) -> Mapping[str, object]:
        import ray

        _validate_local_control_plane_headroom(
            profile=self.plan.runtime.execution_profile,
            worker_count=self.plan.runtime.worker_count,
            cluster_resources=ray.cluster_resources(),
        )
        if len(datasets) != 1:
            raise AlgorithmConfigurationError(
                "X-Learner requires one canonical Ray Dataset"
            )
        dataset: Any = next(iter(datasets.values()))
        data = self.config.data
        required = [
            *data.feature_columns,
            data.treatment_col,
            data.outcome_col,
            data.identity_col,
        ]
        names = set(dataset.schema().names)
        missing = sorted(set(required) - names)
        if missing:
            raise AlgorithmConfigurationError(
                f"X-Learner input is missing columns: {missing}"
            )
        selected = dataset.select_columns(required)
        validate_x_learner_dataset(selected, data)
        train, validation, test = split_x_learner_dataset(
            selected,
            identity_name=data.identity_col,
            val_size=self.config.training.val_size,
            test_size=self.config.training.test_size,
            seed=self.config.training.seed,
        )
        result: dict[str, object] = {"train": train, "test": test}
        if validation is not None:
            result["val"] = validation
        return result

    def build_trainer(
        self,
        config: Mapping[str, Any],
        datasets: Mapping[str, object],
    ) -> object:
        del config
        return XLearnerFitDriver(
            datasets=datasets,
            config=self.config,
            worker_count=self.plan.runtime.worker_count,
            resources_per_worker=WorkerResources(
                num_cpus=self.plan.runtime.num_cpus,
                num_gpus=self.plan.runtime.num_gpus,
                custom=self.plan.runtime.custom_resources,
            ),
            run_identity=self.plan.plan_id[:16],
            input_binding_digest=self.plan.input_descriptor.binding_digest,
        )

    def collect_evidence(self, result: object) -> Mapping[str, Any]:
        if not isinstance(result, XLearnerTrainingResult):
            raise AlgorithmConfigurationError(
                "X-Learner fit returned an unexpected result"
            )
        composition = {
            "feature_names": list(result.feature_names),
            "formula": X_LEARNER_FORMULA,
            "input_binding_digest": self.plan.input_descriptor.binding_digest,
            "propensity_clip": list(result.propensity_clip),
            "response_threshold": result.response_threshold,
        }
        composition_digest = hashlib.sha256(
            json.dumps(
                composition,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "stages": dict(result.stage_evidence),
            "composition_digest": composition_digest,
        }

    def checkpoint_source(self, result: object) -> object:
        if not isinstance(result, XLearnerTrainingResult):
            raise AlgorithmConfigurationError(
                "X-Learner fit returned an unexpected result"
            )
        return result


@DeveloperAPI
def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> FrameworkNativeAlgorithm:
    del artifacts
    if implementation is not DistributedXLearner:
        raise AlgorithmConfigurationError("X-Learner implementation identity drift")
    return DistributedXLearner(plan)


@DeveloperAPI
def export_result(
    *,
    result: object,
    checkpoint: object,
    plan: ResolvedAlgorithmPlan,
    run_id: str,
) -> AlgorithmExecutionResult:
    del checkpoint
    if not isinstance(result, XLearnerTrainingResult):
        raise AlgorithmConfigurationError("X-Learner export requires training result")
    import importlib.metadata

    from tributo.exporting.models import BundleOutputConfig, ExportTarget
    from tributo.exporting.service import BundleExportService
    from tributo.integrations.sources.ray_x_learner import (
        RayXLearnerSourceProvider,
    )

    config = XLearnerConfig.model_validate(plan.algorithm_config)
    with RayXLearnerSourceProvider().open_source(result) as source:
        published = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=config.output.bundle_uri,
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="x-learner-model",
                        format="x-learner",
                        exporter_id="x-learner-v1",
                    ),
                    ExportTarget(
                        name="causal-report",
                        format="json",
                        exporter_id="causal-report-v1",
                    ),
                ],
                roles={
                    "inference": "x-learner-model",
                    "causal_report": "causal-report",
                },
            ),
            tributo_version=importlib.metadata.version("tributo"),
        )
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics=dict(result.metrics),
        outputs={
            "bundle_id": published.bundle_id,
            "bundle_uri": published.canonical_uri,
            "execution_id": published.execution_id,
            "manifest_sha256": published.manifest_sha256,
        },
    )


X_LEARNER_SPEC = AlgorithmSpec(
    name="x_learner",
    trainer_cls=None,
    version="1.0.0",
    default_config={},
    supported_tasks=("fit",),
    operations=("fit",),
    problem_types=(ProblemType.CAUSAL_EFFECT_ESTIMATION,),
    capabilities=(Capability.DISTRIBUTED, Capability.EXPORTABLE),
    extras_group="training",
    learning_paradigm="causal_meta_learner",
    model_family="x_learner",
    data_modalities=("tabular",),
    lifecycle_kind="causal_estimate",
    allowed_execution_modes=(ExecutionMode.FRAMEWORK_NATIVE.value,),
    config_contract_ref="tributo.x_learner.config.v1",
    input_contract_ref="tributo.causal.binary_tabular.v1",
    output_contract_ref="tributo.causal.x_learner_bundle.v1",
)

X_LEARNER_REGISTRATION = AlgorithmRegistration(
    spec=X_LEARNER_SPEC,
    implementation=ImplementationDescriptor(
        implementation_id="tributo.x_learner.xgboost",
        version="1.0.0",
        execution_mode=ExecutionMode.FRAMEWORK_NATIVE,
        implementation_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.x_learner:DistributedXLearner"
        ),
        executable_factory_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.x_learner:create_algorithm"
        ),
        operations=(AlgorithmOperation.FIT,),
        input_compatibility=BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=("shardable",),
            supported_explicit_adapters=(_INPUT_ADAPTER,),
            distribution_policy=(RuntimeTopology.FRAMEWORK_NATIVE,),
        ),
        distribution="xgboost",
        framework="xgboost",
        allowed_config_keys=("data", "model", "training", "ray", "output"),
        runtime_id=_CONTRACT.runtime_id,
        worker_input_adapter_ref=_INPUT_ADAPTER,
        exporter_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.x_learner:export_result"
        ),
        flavor_id="x-learner-v1",
    ),
    environment=EnvironmentSpec(
        environment_id="tributo.x_learner.xgboost.v1",
        dependencies=("ray==2.55.1", "xgboost>=2.1.0"),
    ),
    distribution_spec=DistributionSpec(
        strategy=DistributionStrategy.FRAMEWORK_NATIVE,
        supported_worker_range=WorkerRange(1, 1024),
        supported_execution_profiles=(ExecutionProfile.LOCAL, ExecutionProfile.CLUSTER),
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
        input_distribution=InputDistribution.FRAMEWORK_OWNED,
        state_coordination=StateCoordination.FRAMEWORK_NATIVE,
        policy=FrameworkNativePolicy(
            framework="xgboost-x-learner",
            evidence_collector_ref=(
                "tributo.algorithms.builtin.x_learner:"
                "DistributedXLearner.collect_evidence"
            ),
            component_stages=X_LEARNER_STAGES,
        ),
    ),
    is_default=True,
)

X_LEARNER_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=X_LEARNER_REGISTRATION,
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
        "Binary treatment and outcome with numeric tabular features only.",
        "Owned-local Ray requires one control-plane CPU beyond the worker CPUs.",
        "No cross-fitting, sample weights, automatic retries, or multi-worker resume.",
    ),
)


__all__ = ["DistributedXLearner"]
