"""Run official Wheel algorithms through the attached multi-node Ray cluster."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import ray

from tributo.inference.batch_predictor import XGBoostONNXPredictor


class _NodeProofONNXPredictor(XGBoostONNXPredictor):
    """Wrap the stable ONNX predictor and expose actual actor node identity."""

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = super().__call__(batch)
        row_count = len(next(iter(result.values())))
        result["__tributo_node_id"] = np.full(
            row_count,
            str(ray.get_runtime_context().get_node_id()),
        )
        return result


class _BorrowedClusterRuntimeManager:
    """Expose the IT-owned cluster without starting or stopping Ray."""

    def open(self, profile: object, **kwargs: object) -> Any:
        from tributo.algorithms.api import ExecutionProfile, WorkerResources
        from tributo.algorithms.core import RayRuntimeManager, RayRuntimeSession

        if profile is not ExecutionProfile.CLUSTER:
            raise AssertionError("official algorithm Gate requires cluster profile")
        resources = kwargs.get("resources_per_worker")
        worker_count = kwargs.get("worker_count")
        if not isinstance(resources, WorkerResources) or not isinstance(
            worker_count, int
        ):
            raise AssertionError("official algorithm Gate requires resources")
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


def _stage_data(root: Path) -> Path:
    path = root / "train.parquet"
    discovery_x0 = [float(index) for index in range(32)]
    discovery_x1 = [
        1.5 * value + ((index % 3) - 1) * 0.1
        for index, value in enumerate(discovery_x0)
    ]
    discovery_x2 = [
        -0.8 * value + ((index % 5) - 2) * 0.05
        for index, value in enumerate(discovery_x1)
    ]
    pd.DataFrame(
        {
            "x0": [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0] * 4,
            "x1": [-1.0, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1.0] * 4,
            "count0": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0] * 4,
            "count1": [0.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 2.0] * 4,
            "lag_3": [-2.0, -1.5, -1.0, -0.5, 0.1, 0.5, 1.0, 1.5] * 4,
            "lag_2": [-1.8, -1.3, -0.8, -0.3, 0.2, 0.7, 1.2, 1.7] * 4,
            "lag_1": [-1.6, -1.1, -0.6, -0.1, 0.3, 0.9, 1.4, 1.9] * 4,
            "lag_0": [-1.4, -0.9, -0.4, 0.1, 0.4, 1.1, 1.6, 2.1] * 4,
            "label": [0, 0, 0, 0, 1, 1, 1, 1] * 4,
            "user_id": [0, 0, 1, 1, 2, 2, 3, 3] * 4,
            "item_id": [0, 1, 1, 2, 2, 3, 3, 0] * 4,
            "item_history": [
                [0],
                [1, 2],
                [2, 3, 4],
                [3, 4],
                [4, 5, 6, 7],
                [5],
                [6, 7],
                [7, 0, 1],
            ]
            * 4,
            "token_0": [1, 4, 2, 5, 3, 6, 7, 8] * 4,
            "token_1": [2, 5, 3, 6, 4, 7, 8, 9] * 4,
            "token_2": [3, 0, 4, 0, 5, 0, 9, 0] * 4,
            "token_3": [0, 0, 0, 0, 0, 0, 0, 0] * 4,
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1] * 4,
            "instrument": [0, 1, 0, 1, 0, 1, 0, 1] * 4,
            "outcome": [1.0, 4.0, 3.0, 6.0, 1.0, 4.0, 3.0, 6.0] * 4,
            "discover_x0": discovery_x0,
            "discover_x1": discovery_x1,
            "discover_x2": discovery_x2,
            "identity": list(range(32)),
        }
    ).to_parquet(path, index=False)
    return path


def _stage_graph_data(root: Path) -> tuple[Path, Path, Path]:
    nodes = root / "graph-nodes.parquet"
    edges = root / "graph-edges.parquet"
    seeds = root / "graph-train.parquet"
    pd.DataFrame(
        {
            "node_id": list(range(8)),
            "f0": [1.0, 0.0, 1.0, 0.5, -1.0, 0.0, -1.0, -0.5],
            "f1": [0.0, 1.0, 1.0, 0.5, 0.0, -1.0, -1.0, -0.5],
        }
    ).to_parquet(nodes, index=False)
    pd.DataFrame(
        {
            "source": [0, 1, 2, 3, 4, 5, 6, 7],
            "destination": [1, 2, 3, 0, 5, 6, 7, 4],
            "relation": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_parquet(edges, index=False)
    pd.DataFrame(
        {
            "node_id": list(range(8)),
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    ).to_parquet(seeds, index=False)
    return nodes, edges, seeds


def _stage_gcm_anomalies(root: Path) -> Path:
    path = root / "gcm-anomalies.parquet"
    pd.DataFrame(
        {
            "x0": [4.0, -4.0, 3.5, -3.5],
            "x1": [2.0, -2.0, 1.5, -1.5],
            "outcome": [13.0, -13.0, 11.0, -11.0],
        }
    ).to_parquet(path, index=False)
    return path


def _stage_inference_data(root: Path) -> Path:
    directory = root / "inference-input"
    directory.mkdir()
    for part in range(4):
        pd.DataFrame(
            {
                "x0": [float(part), float(part) + 0.5] * 2,
                "x1": [float(part) * 0.25, float(part) * 0.5] * 2,
            }
        ).to_parquet(directory / f"part-{part}.parquet", index=False)
    return directory


def _execute(
    *,
    algorithm: str,
    implementation_id: str | None,
    feature_names: tuple[str, ...],
    data_path: Path,
    root: Path,
    config: dict[str, object],
    label_name: str | None = "label",
    resume_from: str | None = None,
    require_onnx: bool = False,
) -> dict[str, object]:
    from tributo.algorithms import build_algorithm_dispatcher
    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
        WorkerResources,
    )
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    key = f"official-{algorithm}-{implementation_id or 'default'}"
    invocation = IngestionInputInvocation(
        request=IngestionRequest(
            source=ParquetSourceConfig(path=str(data_path)),
            engine="ray",
        )
    )
    values = {key: invocation}
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=algorithm,
            operation=AlgorithmOperation.FIT,
            implementation_id=implementation_id,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference=key,
                feature_names=feature_names,
                label_name=label_name,
            ),
            algorithm_config=config,
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        resources_per_worker=WorkerResources(num_cpus=1),
        resume_from=resume_from,
    )
    from tributo.algorithms.core import RayRuntimeManager

    result = build_algorithm_dispatcher(
        runtime_manager=cast(RayRuntimeManager, _BorrowedClusterRuntimeManager())
    ).execute(
        request,
        InputExecutionContext(values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None or not receipt.distributed:
        raise AssertionError(
            f"{algorithm}/{implementation_id} did not prove distributed training: "
            f"{receipt.to_dict() if receipt else None}"
        )
    bundle_uri = result.execution.outputs.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not Path(bundle_uri).is_dir():
        raise AssertionError("official algorithm did not publish a readable Bundle")
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.exporting.runtime import BundleModelLoader

    manifest = BundleReader().read_manifest(bundle_uri)
    onnx_exported = any(artifact.format == "onnx" for artifact in manifest.artifacts)
    if require_onnx and not onnx_exported:
        raise AssertionError(
            "official algorithm Bundle did not contain an ONNX artifact"
        )
    inference_roundtrip = False
    if (
        "inference" in manifest.roles
        and manifest.input_signature.input_fields
        and manifest.output_signature.output_fields
    ):
        import numpy as np

        runtime = BundleModelLoader().open(
            bundle_uri,
            role="inference",
            use_case="batch",
        )
        try:
            inputs = {
                name: np.zeros(
                    tuple(1 if size is None else size for size in shape),
                    dtype=np.dtype(dtype),
                )
                for name, dtype, shape in zip(
                    runtime.model.input_names,
                    runtime.model.input_dtypes,
                    runtime.model.input_shapes,
                    strict=True,
                )
            }
            prediction = runtime.predict(inputs)
            if not prediction:
                raise AssertionError("Bundle inference returned no outputs")
            inference_roundtrip = True
        finally:
            runtime.close()
    return {
        "algorithm": algorithm,
        "implementation_id": implementation_id,
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "onnx_exported": onnx_exported,
        "inference_roundtrip": inference_roundtrip,
        "receipt": receipt.to_dict(),
    }


def _execute_portable_tune(
    *,
    data_path: Path,
    root: Path,
) -> dict[str, object]:
    from tributo_algorithms_classical import LINEAR_REGRESSION_DESCRIPTOR

    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
    )
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )
    from tributo.training.portable_tune import PortableTuneRunner
    from tributo.training.tune_config import TuneSearchConfig
    from tributo.training.tune_space import SearchParamSpec, SearchSpaceSpec

    invocation = IngestionInputInvocation(
        request=IngestionRequest(
            source=ParquetSourceConfig(path=str(data_path)),
            engine="ray",
        )
    )
    values = {"official-tune-train": invocation}
    forbidden_bundle = root / "tune-must-not-publish"
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="linear_regression",
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="official-tune-train",
                feature_names=("x0", "x1"),
                label_name="label",
            ),
            algorithm_config={
                "feature_count": 2,
                "learning_rate": 0.1,
                "tolerance": 0.2,
                "runtime": {"checkpoint_dir": str(root / "unused-base-checkpoint")},
                "output": {"bundle_uri": str(forbidden_bundle)},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    grid = PortableTuneRunner(
        LINEAR_REGRESSION_DESCRIPTOR,
        request,
        TuneSearchConfig(
            metric="loss",
            mode="min",
            num_samples=2,
            max_concurrent_trials=1,
        ),
        SearchSpaceSpec(
            parameters=(
                SearchParamSpec(
                    path="learning_rate",
                    kind="choice",
                    values=(0.05, 0.1),
                ),
            )
        ),
        InputExecutionContext(values),
        InputResolutionContext(values=values),
    ).run(
        output_path=str(root / "tune-output"),
        experiment_name="official-linear-regression-tune",
    )
    grid_results = list(cast(Any, grid))
    if len(grid_results) != 2 or any(item.checkpoint is None for item in grid_results):
        raise AssertionError("portable Tune trials did not publish checkpoints")
    if forbidden_bundle.exists():
        raise AssertionError("portable Tune trial published a formal Bundle")
    best = grid.get_best_result(metric="loss", mode="min")
    best_metrics = cast(Mapping[str, object], best.metrics)
    return {
        "trial_count": len(grid_results),
        "checkpoint_count": sum(item.checkpoint is not None for item in grid_results),
        "best_loss": float(cast(int | float, best_metrics["loss"])),
        "formal_bundle_published": False,
    }


def _execute_required_bundle_failure(
    *,
    data_path: Path,
    root: Path,
) -> dict[str, object]:
    blocked = root / "required-bundle-blocked"
    blocked.write_text("not-a-directory", encoding="utf-8")
    try:
        _execute(
            algorithm="multinomial_nb",
            implementation_id="tributo.official.multinomial_nb.map_reduce",
            feature_names=("count0", "count1"),
            data_path=data_path,
            root=root,
            config={
                "alpha": 1.0,
                "fit_prior": True,
                "force_alpha": True,
                "output": {"bundle_uri": str(blocked / "bundle")},
            },
        )
    except Exception as exc:
        if tuple(root.rglob("manifest.json")):
            manifests = tuple(
                path
                for path in root.rglob("manifest.json")
                if "required-bundle-blocked" in str(path)
            )
            if manifests:
                raise AssertionError(
                    "required artifact failure published a manifest"
                ) from exc
        return {
            "failed_closed": True,
            "error_type": type(exc).__name__,
            "manifest_published": False,
        }
    raise AssertionError("required Bundle publication failure unexpectedly succeeded")


def _receipt_details(record: Mapping[str, object]) -> Mapping[str, object]:
    receipt = cast(Mapping[str, object], record["receipt"])
    state = cast(Mapping[str, object], receipt["state"])
    return cast(Mapping[str, object], state["details"])


def _execute_checkpoint_recovery(
    *,
    data_path: Path,
    root: Path,
) -> dict[str, object]:
    ensemble_checkpoint = root / "rf-recovery-checkpoint"
    ensemble_config: dict[str, object] = {
        "task": "classification",
        "unit_count": 8,
        "seed": 17,
        "runtime": {"checkpoint_dir": str(ensemble_checkpoint)},
        "output": {"bundle_uri": str(root / "rf-recovery-first-bundle")},
    }
    _execute(
        algorithm="random_forest",
        implementation_id="tributo.official.random_forest.native_ensemble",
        feature_names=("x0", "x1"),
        data_path=data_path,
        root=root,
        config=ensemble_config,
    )
    ensemble_resumed = _execute(
        algorithm="random_forest",
        implementation_id="tributo.official.random_forest.native_ensemble",
        feature_names=("x0", "x1"),
        data_path=data_path,
        root=root,
        config={
            **ensemble_config,
            "output": {"bundle_uri": str(root / "rf-recovery-resumed-bundle")},
        },
        resume_from=str(ensemble_checkpoint),
    )
    (ensemble_checkpoint / "workers" / "rank-0.bin").write_bytes(b"corrupted")
    ensemble_corruption_rejected = False
    try:
        _execute(
            algorithm="random_forest",
            implementation_id="tributo.official.random_forest.native_ensemble",
            feature_names=("x0", "x1"),
            data_path=data_path,
            root=root,
            config={
                **ensemble_config,
                "output": {"bundle_uri": str(root / "rf-corrupt-must-not-publish")},
            },
            resume_from=str(ensemble_checkpoint),
        )
    except Exception:
        ensemble_corruption_rejected = True
    if (root / "rf-corrupt-must-not-publish").exists():
        raise AssertionError("corrupted Ensemble checkpoint published a Bundle")

    iterative_checkpoint = root / "lr-recovery-checkpoint"
    iterative_config: dict[str, object] = {
        "feature_count": 2,
        "learning_rate": 0.2,
        "tolerance": 1_000_000.0,
        "runtime": {"checkpoint_dir": str(iterative_checkpoint)},
        "output": {"bundle_uri": str(root / "lr-recovery-first-bundle")},
    }
    first_iterative = _execute(
        algorithm="logistic_regression",
        implementation_id=None,
        feature_names=("x0", "x1"),
        data_path=data_path,
        root=root,
        config=iterative_config,
    )
    iterative_resumed = _execute(
        algorithm="logistic_regression",
        implementation_id=None,
        feature_names=("x0", "x1"),
        data_path=data_path,
        root=root,
        config={
            **iterative_config,
            "output": {"bundle_uri": str(root / "lr-recovery-resumed-bundle")},
        },
        resume_from=str(iterative_checkpoint),
    )
    (iterative_checkpoint / "state.bin").write_bytes(b"corrupted")
    iterative_corruption_rejected = False
    try:
        _execute(
            algorithm="logistic_regression",
            implementation_id=None,
            feature_names=("x0", "x1"),
            data_path=data_path,
            root=root,
            config={
                **iterative_config,
                "output": {"bundle_uri": str(root / "lr-corrupt-must-not-publish")},
            },
            resume_from=str(iterative_checkpoint),
        )
    except Exception:
        iterative_corruption_rejected = True
    if (root / "lr-corrupt-must-not-publish").exists():
        raise AssertionError("corrupted Iterative checkpoint published a Bundle")

    ensemble_details = _receipt_details(ensemble_resumed)
    first_iterative_details = _receipt_details(first_iterative)
    iterative_details = _receipt_details(iterative_resumed)
    return {
        "ensemble_resumed": ensemble_details["resumed"],
        "ensemble_restored_unit_count": ensemble_details["restored_unit_count"],
        "ensemble_corruption_rejected": ensemble_corruption_rejected,
        "iterative_first_rounds": first_iterative_details["rounds_completed"],
        "iterative_resumed": iterative_details["resumed"],
        "iterative_resumed_rounds": iterative_details["rounds_completed"],
        "iterative_corruption_rejected": iterative_corruption_rejected,
    }


def _classification_predictions(
    bundle_uri: str,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from tributo.exporting.runtime import BundleModelLoader

    runtime = BundleModelLoader().open(bundle_uri, role="inference", use_case="batch")
    try:
        outputs = runtime.predict(
            {runtime.model.input_names[0]: np.asarray(features, dtype=np.float32)}
        )
    finally:
        runtime.close()
    arrays = [np.asarray(value) for value in outputs.values()]
    probabilities = next(
        (value for value in arrays if value.ndim == 2 and value.shape[1] == 2),
        None,
    )
    labels = next(
        (value.reshape(-1) for value in arrays if value.ndim == 1),
        None,
    )
    if probabilities is None or labels is None:
        raise AssertionError("classification Bundle output is incomplete")
    return labels, probabilities


def _execute_baseline_equivalence(
    *,
    data_path: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    frame = pd.read_parquet(data_path)
    features = frame[["x0", "x1"]].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    joblib_record = next(
        record
        for record in records
        if record["implementation_id"] == "tributo.official.random_forest.joblib"
    )
    logistic_record = next(
        record for record in records if record["algorithm"] == "logistic_regression"
    )
    rf_labels, rf_probabilities = _classification_predictions(
        cast(str, joblib_record["bundle_uri"]),
        features,
    )
    rf_baseline = RandomForestClassifier(
        n_estimators=8,
        max_features="sqrt",
        random_state=7,
    ).fit(features, labels)
    rf_probability_delta = float(
        np.max(np.abs(rf_probabilities - rf_baseline.predict_proba(features)))
    )
    if not np.array_equal(rf_labels.astype(np.int64), rf_baseline.predict(features)):
        raise AssertionError("Ray Joblib Random Forest changed sklearn predictions")
    if rf_probability_delta > 1e-6:
        raise AssertionError("Ray Joblib Random Forest changed sklearn probabilities")

    lr_labels, lr_probabilities = _classification_predictions(
        cast(str, logistic_record["bundle_uri"]),
        features,
    )
    lr_baseline = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        tol=1e-8,
    ).fit(features, labels)
    lr_probability_delta = float(
        np.max(np.abs(lr_probabilities - lr_baseline.predict_proba(features)))
    )
    if not np.array_equal(lr_labels.astype(np.int64), lr_baseline.predict(features)):
        raise AssertionError("distributed Logistic Regression changed class decisions")
    if lr_probability_delta > 0.15:
        raise AssertionError(
            "distributed Logistic Regression exceeded declared sklearn tolerance"
        )
    return {
        "random_forest_exact": True,
        "random_forest_max_probability_delta": rf_probability_delta,
        "logistic_prediction_equivalent": True,
        "logistic_max_probability_delta": lr_probability_delta,
        "logistic_probability_tolerance": 0.15,
    }


def _execute_distributed_inference(
    *,
    bundle_uri: str,
    input_path: Path,
    root: Path,
) -> dict[str, object]:
    from tributo.data import ParquetSourceConfig
    from tributo.inference.pipeline import InferenceConfig, run_batch_inference

    output = root / "inference-output"
    result = run_batch_inference(
        InferenceConfig(
            source=ParquetSourceConfig(
                path=str(input_path),
                columns=["x0", "x1"],
            ),
            output_uri=str(output),
            bundle_uri=bundle_uri,
            predictor_config={
                "return_probs": True,
                "prediction_column": "prediction",
            },
            batch_size=4,
            concurrency=2,
            num_cpus_per_actor=2,
            output_format="parquet",
            output_mode="overwrite",
        ),
        predictor_cls=_NodeProofONNXPredictor,
    )
    frame = pd.read_parquet(output)
    node_ids = tuple(sorted(set(frame["__tributo_node_id"].astype(str))))
    if len(node_ids) != 2 or len(frame) != 16:
        raise AssertionError(
            "batch inference did not prove complete cross-node Actor execution"
        )
    return {
        "status": result["status"],
        "row_count": len(frame),
        "node_count": len(node_ids),
        "node_ids": node_ids,
    }


def _execute_graph(
    *,
    node_path: Path,
    edge_path: Path,
    seed_path: Path,
    root: Path,
    algorithm: str = "graphsage_node_classifier",
    relational: bool = False,
) -> dict[str, object]:
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
    from tributo.algorithms.core import RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    paths = {"nodes": node_path, "edges": edge_path, "train": seed_path}
    values = {
        f"official-graph-{role}": IngestionInputInvocation(
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
                reference="official-graph-nodes",
                feature_names=("node_id", "f0", "f1"),
            ),
            InputBinding(
                name="edges",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="official-graph-edges",
                feature_names=(
                    ("source", "destination", "relation")
                    if relational
                    else ("source", "destination")
                ),
            ),
            InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference="official-graph-train",
                feature_names=("node_id",),
                label_name="label",
            ),
        ),
        primary_role="train",
    )
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=algorithm,
            operation=AlgorithmOperation.FIT,
            input_binding=bindings,
            algorithm_config={
                "model": {
                    "hidden_features": 8,
                    "num_classes": 2,
                    **({"num_relations": 2} if relational else {}),
                },
                "training": {"epochs": 1, "learning_rate": 0.01},
                "ray": {
                    "storage_path": str(root / f"{algorithm}-ray-results"),
                },
                "output": {"bundle_uri": str(root / f"{algorithm}-bundle")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
        resources_per_worker=WorkerResources(num_cpus=1),
    )
    result = build_algorithm_dispatcher(
        runtime_manager=cast(RayRuntimeManager, _BorrowedClusterRuntimeManager())
    ).execute(
        request,
        InputExecutionContext(values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None or not receipt.distributed:
        raise AssertionError(
            f"{algorithm} did not prove distributed training: "
            f"{receipt.to_dict() if receipt else None}"
        )
    bundle_uri = result.execution.outputs.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not Path(bundle_uri).is_dir():
        raise AssertionError(f"{algorithm} did not publish a readable Bundle")
    return {
        "algorithm": algorithm,
        "implementation_id": receipt.to_dict()["strategy"],
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "receipt": receipt.to_dict(),
    }


def _execute_gcm(
    *,
    train_path: Path,
    anomaly_path: Path,
    root: Path,
) -> dict[str, object]:
    from tributo.algorithms import build_algorithm_dispatcher
    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
        InputBindingSet,
    )
    from tributo.algorithms.core import RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    paths = {"train": train_path, "anomaly": anomaly_path}
    values = {
        f"official-gcm-{role}": IngestionInputInvocation(
            request=IngestionRequest(
                source=ParquetSourceConfig(path=str(path)),
                engine="ray",
            )
        )
        for role, path in paths.items()
    }
    variables = ("x0", "x1", "outcome")
    bindings = InputBindingSet(
        bindings=tuple(
            InputBinding(
                name=role,
                resolver_id=INGESTION_RESOLVER_ID,
                reference=f"official-gcm-{role}",
                feature_names=variables,
            )
            for role in ("train", "anomaly")
        ),
        primary_role="train",
    )
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm="gcm_root_cause",
            operation=AlgorithmOperation.FIT,
            input_binding=bindings,
            algorithm_config={
                "data": {
                    "nodes": list(variables),
                    "edges": [["x0", "outcome"], ["x1", "outcome"]],
                    "target_node": "outcome",
                    "interventions": {"x0": 0.0},
                },
                "gcm": {
                    "quality": "good",
                    "distribution_samples": 50,
                    "shapley_permutations": 3,
                },
                "runtime": {},
                "output": {"bundle_uri": str(root / "causal-gcm-bundle")},
            },
        ),
        profile=ExecutionProfile.CLUSTER,
        worker_count=2,
    )
    result = build_algorithm_dispatcher(
        runtime_manager=cast(RayRuntimeManager, _BorrowedClusterRuntimeManager())
    ).execute(
        request,
        InputExecutionContext(values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None or not receipt.cluster_distributed:
        raise AssertionError(
            "gcm_root_cause did not prove cross-node training: "
            f"{receipt.to_dict() if receipt else None}"
        )
    bundle_uri = result.execution.outputs.get("bundle_uri")
    if not isinstance(bundle_uri, str) or not Path(bundle_uri).is_dir():
        raise AssertionError("gcm_root_cause did not publish a readable Bundle")
    if float(result.execution.metrics["counterfactual_target_absolute_delta"]) <= 0:
        raise AssertionError("gcm_root_cause did not produce counterfactual evidence")
    return {
        "algorithm": "gcm_root_cause",
        "implementation_id": receipt.to_dict()["strategy"],
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "receipt": receipt.to_dict(),
    }


def main() -> None:
    root = Path(os.environ["TRIBUTO_OFFICIAL_ALGORITHM_GATE_ROOT"])
    if root.parent != Path("/workspace/tributo-work") or not root.name.startswith(
        "tributo-official-algorithm-gate-"
    ):
        raise ValueError("official algorithm Gate root is outside the owned workspace")
    if root.exists():
        raise FileExistsError(f"refusing to reuse Gate root: {root}")
    root.mkdir(parents=True)
    ray.init(address="auto")
    try:
        data_path = _stage_data(root)
        graph_nodes, graph_edges, graph_seeds = _stage_graph_data(root)
        gcm_anomalies = _stage_gcm_anomalies(root)
        inference_data = _stage_inference_data(root)
        results = [
            _execute(
                algorithm="random_forest",
                implementation_id="tributo.official.random_forest.joblib",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 8,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "rf-joblib-bundle")},
                },
            ),
            _execute(
                algorithm="extra_trees",
                implementation_id="tributo.official.extra_trees.joblib",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 8,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "extra-joblib-bundle")},
                },
            ),
            _execute(
                algorithm="extra_trees",
                implementation_id=("tributo.official.extra_trees.native_ensemble"),
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 8,
                    "unit_count": 8,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "extra-native-bundle")},
                },
            ),
            _execute(
                algorithm="random_forest",
                implementation_id="tributo.official.random_forest.native_ensemble",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 8,
                    "unit_count": 8,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "rf-native-bundle")},
                },
            ),
            _execute(
                algorithm="logistic_regression",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "feature_count": 2,
                    "learning_rate": 0.4,
                    "tolerance": 0.2,
                    "runtime": {"checkpoint_dir": str(root / "lr-checkpoint")},
                    "output": {"bundle_uri": str(root / "lr-bundle")},
                },
            ),
            _execute(
                algorithm="linear_regression",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "feature_count": 2,
                    "learning_rate": 0.1,
                    "tolerance": 1_000_000.0,
                    "runtime": {"checkpoint_dir": str(root / "linear-checkpoint")},
                    "output": {"bundle_uri": str(root / "linear-bundle")},
                },
            ),
            _execute(
                algorithm="multinomial_nb",
                implementation_id="tributo.official.multinomial_nb.map_reduce",
                feature_names=("count0", "count1"),
                data_path=data_path,
                root=root,
                config={
                    "alpha": 1.0,
                    "fit_prior": True,
                    "force_alpha": True,
                    "output": {"bundle_uri": str(root / "multinomial-nb-bundle")},
                },
            ),
            _execute(
                algorithm="tabular_autoencoder",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "model": {"input_features": 2, "latent_features": 1},
                    "optimizer": {"learning_rate": 0.01},
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 4,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "autoencoder-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "autoencoder-bundle")},
                },
            ),
            _execute(
                algorithm="temporal_conv_classifier",
                implementation_id=None,
                feature_names=("lag_3", "lag_2", "lag_1", "lag_0"),
                data_path=data_path,
                root=root,
                config={
                    "model": {"input_features": 4, "channels": 4},
                    "optimizer": {
                        "learning_rate": 0.05,
                        "accumulation_steps": 2,
                        "max_gradient_norm": 1.0,
                    },
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 4,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "timeseries-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "timeseries-bundle")},
                },
            ),
            _execute(
                algorithm="dnn",
                implementation_id="tributo.official.tabular_torch.dnn",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "model": {"input_features": 2, "hidden_units": [8, 4]},
                    "optimizer": {"learning_rate": 0.01},
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "dnn-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "dnn-v2-bundle")},
                },
            ),
            _execute(
                algorithm="pu",
                implementation_id="tributo.official.tabular_torch.pu",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "model": {"input_features": 2, "hidden_units": [8, 4]},
                    "loss": {"type": "nnpu", "class_prior": 0.4},
                    "optimizer": {"learning_rate": 0.01},
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "pu-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "pu-v2-bundle")},
                },
            ),
            _execute(
                algorithm="two_tower_recommender",
                implementation_id=None,
                feature_names=("user_id", "item_id"),
                data_path=data_path,
                root=root,
                config={
                    "model": {
                        "user_count": 4,
                        "item_count": 4,
                        "embedding_dim": 4,
                    },
                    "optimizer": {"learning_rate": 0.01},
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "two-tower-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "two-tower-bundle")},
                },
            ),
            _execute(
                algorithm="jagged_embedding_recommender",
                implementation_id=None,
                feature_names=("user_id", "item_history", "item_id"),
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "user_col": "user_id",
                        "history_col": "item_history",
                        "candidate_col": "item_id",
                        "label_col": "label",
                    },
                    "model": {
                        "user_count": 4,
                        "item_count": 8,
                        "embedding_dim": 4,
                    },
                    "training": {"learning_rate": 0.01},
                    "ray": {
                        "storage_path": str(root / "jagged-ray-results"),
                    },
                    "output": {"bundle_uri": str(root / "jagged-bundle")},
                },
            ),
            _execute(
                algorithm="token_transformer_classifier",
                implementation_id=None,
                feature_names=("token_0", "token_1", "token_2", "token_3"),
                data_path=data_path,
                root=root,
                config={
                    "model": {
                        "vocab_size": 32,
                        "sequence_length": 4,
                        "hidden_size": 8,
                        "heads": 2,
                    },
                    "optimizer": {"learning_rate": 0.001},
                    "metrics": {},
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "prefetch_batches": 0,
                        "seed": 7,
                    },
                    "ray": {
                        "max_failures": 0,
                        "storage_path": str(root / "transformer-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "transformer-bundle")},
                },
            ),
            _execute(
                algorithm="difference_in_means_ate",
                implementation_id=None,
                feature_names=("treatment", "x0"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "treatment_col": "treatment",
                    "policy_cost": 0.5,
                    "output": {"bundle_uri": str(root / "causal-ate-bundle")},
                },
            ),
            _execute(
                algorithm="linear_dml_ate",
                implementation_id=None,
                feature_names=("treatment", "x0"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "treatment_col": "treatment",
                    "policy_cost": 0.5,
                    "output": {"bundle_uri": str(root / "causal-dml-bundle")},
                },
            ),
            _execute(
                algorithm="linear_iv_ate",
                implementation_id=None,
                feature_names=("treatment", "instrument", "x0"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "treatment_col": "treatment",
                    "instrument_col": "instrument",
                    "policy_cost": 0.5,
                    "output": {"bundle_uri": str(root / "causal-iv-bundle")},
                },
            ),
            _execute(
                algorithm="pc_stability_discovery",
                implementation_id=None,
                feature_names=("discover_x0", "discover_x1", "discover_x2"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "alpha": 0.05,
                    "vote_threshold": 0.5,
                    "output": {"bundle_uri": str(root / "causal-pc-bundle")},
                },
            ),
            _execute(
                algorithm="teacher_student_distillation",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "model": {
                        "input_features": 2,
                        "teacher_hidden": 8,
                        "student_hidden": 3,
                    },
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 0.01,
                        "supervised_weight": 0.5,
                    },
                    "ray": {"storage_path": str(root / "distillation-ray-results")},
                    "output": {"bundle_uri": str(root / "distillation-bundle")},
                },
            ),
            _execute(
                algorithm="pretrain_finetune_classifier",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "model": {"input_features": 2, "hidden_features": 4},
                    "training": {
                        "pretrain_epochs": 1,
                        "finetune_epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 0.01,
                    },
                    "ray": {
                        "storage_path": str(root / "pretrain-finetune-ray-results")
                    },
                    "output": {"bundle_uri": str(root / "pretrain-finetune-bundle")},
                },
            ),
            _execute(
                algorithm="xgboost",
                implementation_id="tributo.official.boosting.xgboost",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "label_col": "label",
                        "feature_columns": ["x0", "x1"],
                    },
                    "model": {
                        "objective": "binary:logistic",
                        "tree_method": "hist",
                        "max_depth": 2,
                        "eta": 0.3,
                        "eval_metric": "logloss",
                    },
                    "training": {"num_rounds": 3},
                    "ray": {"storage_path": str(root / "xgboost-ray-results")},
                    "output": {"bundle_uri": str(root / "xgboost-v2-bundle")},
                },
            ),
            _execute(
                algorithm="x_learner",
                implementation_id="tributo.official.causal_xlearner.xgboost",
                feature_names=("x0", "x1", "treatment", "identity"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "feature_columns": ["x0", "x1"],
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                        "identity_col": "identity",
                    },
                    "model": {},
                    "training": {"num_rounds": 1},
                    "ray": {"storage_path": str(root / "xlearner-ray-results")},
                    "output": {"bundle_uri": str(root / "xlearner-v2-bundle")},
                },
            ),
            _execute(
                algorithm="doubly_robust_ate",
                implementation_id=None,
                feature_names=("x0", "x1", "treatment"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "feature_columns": ["x0", "x1"],
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                    },
                    "model": {},
                    "training": {"num_rounds": 1},
                    "ray": {"storage_path": str(root / "causal-dr-ray-results")},
                    "output": {"bundle_uri": str(root / "causal-dr-bundle")},
                },
            ),
            _execute(
                algorithm="dowhy_linear_refutation",
                implementation_id=None,
                feature_names=("x0", "treatment"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "common_causes": ["x0"],
                        "treatment_col": "treatment",
                        "outcome_col": "outcome",
                    },
                    "refutation": {"seed": 7, "policy_cost": 0.5},
                    "runtime": {},
                    "output": {"bundle_uri": str(root / "causal-dowhy-bundle")},
                },
            ),
            _execute_gcm(
                train_path=data_path,
                anomaly_path=gcm_anomalies,
                root=root,
            ),
            _execute_graph(
                node_path=graph_nodes,
                edge_path=graph_edges,
                seed_path=graph_seeds,
                root=root,
            ),
            _execute_graph(
                node_path=graph_nodes,
                edge_path=graph_edges,
                seed_path=graph_seeds,
                root=root,
                algorithm="rgcn_node_classifier",
                relational=True,
            ),
        ]
        baseline_result = _execute_baseline_equivalence(
            data_path=data_path,
            records=results,
        )
        tune_result = _execute_portable_tune(data_path=data_path, root=root)
        recovery_result = _execute_checkpoint_recovery(
            data_path=data_path,
            root=root,
        )
        xgboost_bundle = next(
            str(record["bundle_uri"])
            for record in results
            if record["algorithm"] == "xgboost"
        )
        inference_result = _execute_distributed_inference(
            bundle_uri=xgboost_bundle,
            input_path=inference_data,
            root=root,
        )
        failure_result = _execute_required_bundle_failure(
            data_path=data_path,
            root=root,
        )
        print("TUNE_RESULT: " + json.dumps(tune_result, sort_keys=True), flush=True)
        print(
            "BASELINE_RESULT: " + json.dumps(baseline_result, sort_keys=True),
            flush=True,
        )
        print(
            "RECOVERY_RESULT: " + json.dumps(recovery_result, sort_keys=True),
            flush=True,
        )
        print(
            "INFERENCE_RESULT: " + json.dumps(inference_result, sort_keys=True),
            flush=True,
        )
        print(
            "FAILURE_RESULT: " + json.dumps(failure_result, sort_keys=True),
            flush=True,
        )
        print("RESULT: " + json.dumps(results, sort_keys=True), flush=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
