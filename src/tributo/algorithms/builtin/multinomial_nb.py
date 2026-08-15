"""True distributed MultinomialNB using bounded sufficient statistics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tributo.algorithms.api import (
    AlgorithmConfigurationError,
    AlgorithmExecutionError,
    AlgorithmExecutionResult,
    AlgorithmInputError,
    AlgorithmOperation,
    AlgorithmRegistration,
    ArtifactDraft,
    BackendInputCompatibility,
    DistributedAlgorithmDescriptor,
    DistributionSpec,
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionMode,
    ExecutionProfile,
    ImplementationDescriptor,
    InputDistribution,
    MapReducePolicy,
    QualifiedReference,
    ResolvedAlgorithmPlan,
    RuntimeTopology,
    StateCoordination,
    StateField,
    WorkerRange,
    WorkerResources,
)
from tributo.algorithms.api.models import FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS
from tributo.algorithms.spi import AlgorithmExecutionContext, MapReduceAlgorithm
from tributo.training.algorithm_spec import AlgorithmSpec, Capability, ProblemType
from tributo.util.annotations import DeveloperAPI, PublicAPI

_STATE_SCHEMA = (
    StateField("classes", "int64", (None,)),
    StateField("class_count", "float64", (None,)),
    StateField("feature_count", "float64", (None, None)),
    StateField("row_count", "int64", ()),
)


@dataclass(frozen=True)
class MultinomialNBModel:
    """Bounded finalizer output passed to the declared exporter."""

    estimator: Any
    feature_names: tuple[str, ...]
    row_count: int
    sample_weight_sum: float


@PublicAPI(stability="alpha")
class DistributedMultinomialNB(
    MapReduceAlgorithm[
        Mapping[str, object],
        Mapping[str, object],
        MultinomialNBModel,
    ]
):
    """Map and merge exact class/feature counts before public-API finalization."""

    def __init__(self, plan: ResolvedAlgorithmPlan) -> None:
        self._plan = plan
        self._feature_names = plan.input_binding.feature_names
        label_name = plan.input_binding.label_name
        self._label_name: str = label_name or ""
        self._weight_name = plan.input_binding.sample_weight_name
        if label_name is None:
            raise AlgorithmConfigurationError(
                "MultinomialNB fit requires InputBinding.label_name"
            )

    def map_partition(
        self,
        batches: Iterable[Mapping[str, object]],
        context: AlgorithmExecutionContext,
    ) -> Mapping[str, object]:
        """Aggregate one shard into class and feature sufficient statistics."""
        del context
        import numpy as np

        state = self.empty_partition()
        for batch in batches:
            features, labels, weights = self._batch_arrays(batch)
            if labels.size == 0:
                continue
            classes = np.unique(labels)
            class_count = np.zeros(classes.shape[0], dtype=np.float64)
            feature_count = np.zeros(
                (classes.shape[0], len(self._feature_names)),
                dtype=np.float64,
            )
            for index, label in enumerate(classes):
                mask = labels == label
                selected_weights = weights[mask]
                class_count[index] = selected_weights.sum(dtype=np.float64)
                feature_count[index] = self._weighted_feature_sum(
                    features[mask],
                    selected_weights,
                    np,
                )
            state = self.merge_states(
                state,
                {
                    "classes": classes.astype(np.int64, copy=False),
                    "class_count": class_count,
                    "feature_count": feature_count,
                    "row_count": np.asarray(labels.shape[0], dtype=np.int64),
                },
            )
        return state

    def _batch_arrays(self, batch: Mapping[str, object]) -> tuple[Any, Any, Any]:
        import numpy as np
        from scipy import sparse

        missing = [
            name
            for name in (*self._feature_names, self._label_name)
            if name not in batch
        ]
        if self._weight_name is not None and self._weight_name not in batch:
            missing.append(self._weight_name)
        if missing:
            raise AlgorithmInputError(
                f"MultinomialNB batch is missing required column(s): {missing}"
            )
        try:
            feature_columns = [
                self._feature_column(batch[name], np, sparse)
                for name in self._feature_names
            ]
            if any(sparse.issparse(column) for column in feature_columns):
                sparse_columns = [
                    column
                    if sparse.issparse(column)
                    else sparse.csr_matrix(column[:, None])
                    for column in feature_columns
                ]
                features = sparse.hstack(sparse_columns, format="csr")
            else:
                features = np.column_stack(feature_columns)
            raw_labels = np.asarray(batch[self._label_name])
            labels = raw_labels.astype(np.int64)
            if raw_labels.ndim != 1 or not np.array_equal(raw_labels, labels):
                raise ValueError("labels must contain exact integer values")
            weights = (
                np.asarray(batch[self._weight_name], dtype=np.float64)
                if self._weight_name is not None
                else np.ones(labels.shape[0], dtype=np.float64)
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmInputError(
                f"MultinomialNB could not convert a tabular batch: {exc}"
            ) from exc
        if features.shape[0] != labels.shape[0] or weights.shape != labels.shape:
            raise AlgorithmInputError(
                "MultinomialNB feature, label, and sample-weight rows disagree"
            )
        feature_values = features.data if sparse.issparse(features) else features
        if not np.isfinite(feature_values).all() or (feature_values < 0).any():
            raise AlgorithmInputError(
                "MultinomialNB features must be finite and non-negative"
            )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise AlgorithmInputError(
                "MultinomialNB sample weights must be finite and non-negative"
            )
        return features, labels, weights

    @staticmethod
    def _feature_column(value: object, np: Any, sparse: Any) -> Any:
        """Normalize a dense or sparse batch column without batch densification."""
        if sparse.issparse(value):
            column = sparse.csr_matrix(value, dtype=np.float64)
            if column.ndim != 2 or column.shape[1] != 1:
                raise ValueError("sparse feature columns must have shape (rows, 1)")
            return column
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError("feature columns must be one-dimensional")
        return array

    @staticmethod
    def _weighted_feature_sum(features: Any, weights: Any, np: Any) -> Any:
        """Return one bounded dense statistic while preserving sparse batches."""
        from scipy import sparse

        if sparse.issparse(features):
            return np.asarray(
                features.multiply(weights[:, None]).sum(axis=0),
                dtype=np.float64,
            ).reshape(-1)
        return (features * weights[:, None]).sum(axis=0, dtype=np.float64)

    def merge_states(
        self,
        left: Mapping[str, object],
        right: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Associatively align class identities and sum exact counts."""
        import numpy as np

        left_classes = np.asarray(left["classes"], dtype=np.int64)
        right_classes = np.asarray(right["classes"], dtype=np.int64)
        classes = np.union1d(left_classes, right_classes).astype(np.int64)
        class_count = np.zeros(classes.shape[0], dtype=np.float64)
        feature_count = np.zeros(
            (classes.shape[0], len(self._feature_names)),
            dtype=np.float64,
        )
        row_count = 0
        for state, state_classes in (
            (left, left_classes),
            (right, right_classes),
        ):
            counts = np.asarray(state["class_count"], dtype=np.float64)
            features = np.asarray(state["feature_count"], dtype=np.float64)
            if counts.shape != (state_classes.shape[0],) or features.shape != (
                state_classes.shape[0],
                len(self._feature_names),
            ):
                raise AlgorithmExecutionError(
                    "MultinomialNB state dimensions are internally inconsistent"
                )
            if state_classes.size:
                indices = np.searchsorted(classes, state_classes)
                class_count[indices] += counts
                feature_count[indices] += features
            raw_row_count = np.asarray(state["row_count"], dtype=np.int64)
            if raw_row_count.shape != ():
                raise AlgorithmExecutionError(
                    "MultinomialNB row_count state must be scalar"
                )
            row_count += int(raw_row_count)
        return {
            "classes": classes,
            "class_count": class_count,
            "feature_count": feature_count,
            "row_count": np.asarray(row_count, dtype=np.int64),
        }

    def finalize_model(self, state: Mapping[str, object]) -> MultinomialNBModel:
        """Reconstruct sklearn state through public ``partial_fit`` only."""
        import numpy as np
        from sklearn.naive_bayes import MultinomialNB

        classes = np.asarray(state["classes"], dtype=np.int64)
        class_count = np.asarray(state["class_count"], dtype=np.float64)
        feature_count = np.asarray(state["feature_count"], dtype=np.float64)
        row_count = int(np.asarray(state["row_count"], dtype=np.int64))
        if classes.size == 0 or class_count.sum() <= 0:
            raise AlgorithmInputError(
                "MultinomialNB requires at least one positive-weight training row"
            )
        config = dict(self._plan.algorithm_config)
        class_prior_value = config.get("class_prior")
        class_prior = (
            list(class_prior_value)
            if isinstance(class_prior_value, (list, tuple))
            else class_prior_value
        )
        estimator = MultinomialNB(
            alpha=float(config.get("alpha", 1.0)),
            force_alpha=bool(config.get("force_alpha", True)),
            fit_prior=bool(config.get("fit_prior", True)),
            class_prior=class_prior,
        )
        synthetic = np.divide(
            feature_count,
            class_count[:, None],
            out=np.zeros_like(feature_count),
            where=class_count[:, None] != 0,
        )
        estimator.partial_fit(
            synthetic,
            classes,
            classes=classes,
            sample_weight=class_count,
        )
        return MultinomialNBModel(
            estimator=estimator,
            feature_names=self._feature_names,
            row_count=row_count,
            sample_weight_sum=float(class_count.sum()),
        )

    def state_schema(self) -> tuple[StateField, ...]:
        """Return the exact descriptor-owned state schema."""
        return _STATE_SCHEMA

    def empty_partition(self) -> Mapping[str, object]:
        """Return the identity state for an empty shard."""
        import numpy as np

        return {
            "classes": np.empty((0,), dtype=np.int64),
            "class_count": np.empty((0,), dtype=np.float64),
            "feature_count": np.empty((0, len(self._feature_names)), dtype=np.float64),
            "row_count": np.asarray(0, dtype=np.int64),
        }

    @property
    def retry_safe(self) -> bool:
        """Map and merge stages are deterministic and side-effect free."""
        return True


@DeveloperAPI
def create_algorithm(
    *,
    plan: ResolvedAlgorithmPlan,
    implementation: object,
    artifacts: tuple[object, ...],
) -> DistributedMultinomialNB:
    """Construct the descriptor-selected MapReduce implementation."""
    del artifacts
    if not isinstance(implementation, type) or not issubclass(
        implementation, DistributedMultinomialNB
    ):
        raise AlgorithmConfigurationError(
            "MultinomialNB implementation reference must resolve to its SPI class"
        )
    return implementation(plan)


@DeveloperAPI
def export_model(
    *,
    model: MultinomialNBModel,
    plan: ResolvedAlgorithmPlan,
    run_id: str | None = None,
) -> AlgorithmExecutionResult:
    """Convert the finalized estimator to a bounded ONNX artifact."""
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn import __version__ as sklearn_version

    if not isinstance(model, MultinomialNBModel):
        raise AlgorithmExecutionError(
            "MultinomialNB exporter received an incompatible model"
        )
    try:
        onnx_model = convert_sklearn(
            model.estimator,
            initial_types=[
                (
                    "float_input",
                    FloatTensorType([None, len(model.feature_names)]),
                )
            ],
            options={id(model.estimator): {"zipmap": False}},
            target_opset=18,
        )
        onnx_model.graph.name = "tributo-multinomial-nb"
        onnx_model.doc_string = ""
        for node in onnx_model.graph.node:
            node.doc_string = ""
        metadata = {
            "feature_names": json.dumps(model.feature_names),
            "distribution_spec_digest": plan.runtime.distribution_digest or "",
        }
        for name, value in sorted(metadata.items()):
            item = onnx_model.metadata_props.add()
            item.key = name
            item.value = value
        payload = onnx_model.SerializeToString()
    except Exception as exc:
        raise AlgorithmExecutionError(
            f"MultinomialNB ONNX export failed: {type(exc).__name__}"
        ) from exc
    estimator = model.estimator
    artifact = ArtifactDraft.from_payload(
        name="model",
        kind="model",
        format="application/onnx",
        payload=payload,
    )
    output_config = plan.algorithm_config.get("output", {})
    bundle_outputs: dict[str, object] = {}
    if isinstance(output_config, Mapping) and output_config.get("bundle_uri"):
        from tributo.exporting.models import (
            BundleOutputConfig,
            ExportSource,
            ExportTarget,
        )
        from tributo.exporting.service import BundleExportService

        bundle_uri = output_config["bundle_uri"]
        if not isinstance(bundle_uri, str) or not bundle_uri:
            raise AlgorithmConfigurationError(
                "output.bundle_uri must be a non-empty string"
            )
        source = ExportSource(
            source_kind="prebuilt_onnx",
            model_object=payload,
            feature_schema={"feature_names": list(model.feature_names)},
            metadata={
                "framework": "sklearn",
                "framework_versions": {"scikit-learn": sklearn_version},
                "task_type": "classification",
            },
            source_fingerprint=artifact.sha256,
        )
        bundle = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=bundle_uri,
                request_id=run_id,
                run_id=run_id,
                targets=[
                    ExportTarget(
                        name="onnx-model",
                        format="onnx",
                        exporter_id="prebuilt-onnx-v1",
                    )
                ],
                roles={"inference": "onnx-model"},
            ),
        )
        bundle_outputs = {
            "bundle_id": bundle.bundle_id,
            "bundle_uri": bundle.canonical_uri,
            "execution_id": bundle.execution_id,
            "manifest_sha256": bundle.manifest_sha256,
        }
    return AlgorithmExecutionResult(
        status="succeeded",
        metrics={
            "row_count": model.row_count,
            "sample_weight_sum": model.sample_weight_sum,
        },
        outputs={
            "classes": np.asarray(estimator.classes_).tolist(),
            "class_count": np.asarray(estimator.class_count_).tolist(),
            "feature_count": np.asarray(estimator.feature_count_).tolist(),
            **bundle_outputs,
        },
        artifacts=(artifact,),
    )


_MAP_REDUCE_CONTRACT = FORMAL_DISTRIBUTED_STRATEGY_CONTRACTS[
    DistributionStrategy.RAY_MAP_REDUCE
]
_INPUT_ADAPTER = QualifiedReference.parse(_MAP_REDUCE_CONTRACT.worker_input_adapter_ref)

MULTINOMIAL_NB_REGISTRATION = AlgorithmRegistration(
    spec=AlgorithmSpec(
        name="multinomial_nb",
        trainer_cls=None,
        version="1.0.0",
        default_config={"alpha": 1.0, "force_alpha": True, "fit_prior": True},
        supported_tasks=("fit",),
        operations=("fit",),
        problem_types=(
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTI_CLASS_CLASSIFICATION,
        ),
        capabilities=(Capability.EXPORTABLE, Capability.DISTRIBUTED),
        extras_group="training",
        learning_paradigm="supervised",
        model_family="naive_bayes",
        data_modalities=("tabular",),
        lifecycle_kind="batch_fit",
        allowed_execution_modes=(ExecutionMode.MAP_REDUCE.value,),
        config_contract_ref="tributo.multinomial_nb.config.v1",
        input_contract_ref="tributo.tabular.nonnegative.v1",
        output_contract_ref="tributo.classification.onnx.v1",
    ),
    implementation=ImplementationDescriptor(
        implementation_id="tributo.multinomial_nb.map_reduce",
        version="1.0.0",
        execution_mode=ExecutionMode.MAP_REDUCE,
        implementation_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.multinomial_nb:DistributedMultinomialNB"
        ),
        executable_factory_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.multinomial_nb:create_algorithm"
        ),
        operations=(AlgorithmOperation.FIT,),
        input_compatibility=BackendInputCompatibility(
            accepted_input_views=("ray_data",),
            accepted_ingestion_engines=("tributo.ray_data",),
            required_input_capabilities=("shardable",),
            supported_explicit_adapters=(_INPUT_ADAPTER,),
            distribution_policy=(RuntimeTopology.RAY_MAP_REDUCE,),
        ),
        distribution="scikit-learn",
        framework="sklearn",
        artifact_format="none",
        allowed_config_keys=(
            "alpha",
            "class_prior",
            "fit_prior",
            "force_alpha",
            "output",
        ),
        runtime_id=_MAP_REDUCE_CONTRACT.runtime_id,
        worker_input_adapter_ref=_INPUT_ADAPTER,
        exporter_ref=QualifiedReference.parse(
            "tributo.algorithms.builtin.multinomial_nb:export_model"
        ),
        flavor_id="onnx-runtime-v1",
    ),
    environment=EnvironmentSpec(
        environment_id="tributo.multinomial_nb.v1",
        dependencies=(
            "onnx>=1.16.0",
            "onnxruntime>=1.20.0",
            "scikit-learn>=1.4,<2",
            "skl2onnx>=1.17",
        ),
    ),
    distribution_spec=DistributionSpec(
        strategy=DistributionStrategy.RAY_MAP_REDUCE,
        supported_worker_range=WorkerRange(1, 1024),
        supported_execution_profiles=(
            ExecutionProfile.LOCAL,
            ExecutionProfile.KUBERNETES,
        ),
        resources_per_worker=WorkerResources(num_cpus=1, num_gpus=0),
        input_distribution=InputDistribution.SHARDED,
        state_coordination=StateCoordination.ASSOCIATIVE_REDUCE,
        policy=MapReducePolicy(
            state_schema=_STATE_SCHEMA,
            max_partial_state_bytes=64 * 1024 * 1024,
            reducer_ref=(
                "tributo.algorithms.builtin.multinomial_nb:"
                "DistributedMultinomialNB.merge_states"
            ),
            finalizer_ref=(
                "tributo.algorithms.builtin.multinomial_nb:"
                "DistributedMultinomialNB.finalize_model"
            ),
            commutative=True,
            max_retries=0,
        ),
    ),
    is_default=True,
)

MULTINOMIAL_NB_DESCRIPTOR = DistributedAlgorithmDescriptor(
    registration=MULTINOMIAL_NB_REGISTRATION,
    package_name="tributo",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    stability="alpha",
    tested=True,
    supported=True,
    validated_execution_profiles=(ExecutionProfile.LOCAL,),
)


__all__ = [
    "DistributedMultinomialNB",
    "create_algorithm",
    "export_model",
]
