"""Run official Wheel algorithms through the attached multi-node Ray cluster."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import ray
from official_algorithm_matrix import (
    ALL_ENTRY_POINTS,
    CATEGORY_ENTRY_POINTS,
    ENTRY_POINT_DISTRIBUTIONS,
    OFFICIAL_ALGORITHM_IDENTITIES,
    category_for_entry_point,
    entry_point_for,
    entry_points_for_gate,
)
from official_algorithm_output_contract import (
    OutputExpectation,
    build_output_expectation,
    validate_output_value,
)
from packaging.utils import canonicalize_name

from tributo.algorithms.conformance import validate_installed_algorithm_package
from tributo.inference.batch_predictor import XGBoostONNXPredictor
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    TensorInputBinding,
    TensorOutputBinding,
)


def _gate_category() -> str:
    category = os.environ.get("TRIBUTO_OFFICIAL_ALGORITHM_CATEGORY", "all")
    if category != "all" and category not in CATEGORY_ENTRY_POINTS:
        raise ValueError(f"unknown official algorithm Gate category: {category!r}")
    return category


def _entry_point_enabled(entry_point: str) -> bool:
    category = _gate_category()
    expected_entry_points = entry_points_for_gate(
        category,
        os.environ.get("TRIBUTO_OFFICIAL_ALGORITHM_ENTRY_POINTS", ""),
    )
    return entry_point in expected_entry_points


def _tensor_columns(
    *,
    name: str,
    shape: tuple[int | str, ...],
) -> tuple[str, ...]:
    if not shape or shape[0] != "batch":
        raise AssertionError(f"model input {name!r} must declare dynamic batch first")
    trailing = shape[1:]
    if any(not isinstance(value, int) or value < 1 for value in trailing):
        raise AssertionError(
            f"model input {name!r} has unsupported dynamic trailing shape {shape}"
        )
    width = math.prod(cast(tuple[int, ...], trailing)) if trailing else 1
    return tuple(f"{name}__{index}" for index in range(width))


def _inference_value(
    dtype: str,
    *,
    row: int,
    column: int,
    entry_point: str,
) -> object:
    normalized = dtype.lower()
    if entry_point == "token_transformer_classifier":
        token = (row + column) % 8 + 1
        if normalized.startswith("float"):
            return float(token)
        if normalized.startswith("int") or normalized.startswith("uint"):
            return token
        raise AssertionError(
            f"token Transformer probe requires a numeric dtype, got {dtype!r}"
        )
    if normalized.startswith("float") or normalized in {"double", "half"}:
        return float(((row + column) % 7) - 3) / 4.0
    if normalized.startswith("int") or normalized.startswith("uint"):
        return int((row + column) % 2)
    if normalized in {"bool", "boolean"}:
        return bool((row + column) % 2)
    if normalized in {"str", "string"}:
        return str((row + column) % 2)
    raise AssertionError(f"unsupported inference probe dtype: {dtype!r}")


def _stage_bundle_inference_input(
    *,
    bundle_uri: str,
    root: Path,
    entry_point: str,
) -> tuple[
    Path,
    InputBindingSpec,
    OutputBindingSpec,
    tuple[OutputExpectation, ...],
]:
    from tributo.exporting.bundle_reader import BundleReader

    manifest = BundleReader().read_manifest(bundle_uri)
    if "inference" not in manifest.roles:
        raise AssertionError(f"{entry_point} Bundle omitted the inference role")
    if not manifest.input_signature.input_fields:
        raise AssertionError(f"{entry_point} Bundle omitted its input signature")
    if not manifest.output_signature.output_fields:
        raise AssertionError(f"{entry_point} Bundle omitted its output signature")

    bindings = []
    values: dict[str, list[object]] = {}
    projected: list[str] = []
    for field_index, field in enumerate(manifest.input_signature.input_fields):
        columns = _tensor_columns(name=field.name, shape=field.shape)
        projected.extend(columns)
        for column_index, column in enumerate(columns):
            values[column] = [
                _inference_value(
                    field.dtype,
                    row=row,
                    column=field_index + column_index,
                    entry_point=entry_point,
                )
                for row in range(16)
            ]
        bindings.append(
            TensorInputBinding(
                tensor_name=field.name,
                columns=columns,
                dtype=field.dtype,
                single_column_mode=(
                    "scalar" if field.shape == ("batch",) else "vector"
                ),
            )
        )

    output_expectations = tuple(
        build_output_expectation(
            tensor_name=field.name,
            column=f"result__{field.name}",
            dtype=field.dtype,
            shape=field.shape,
        )
        for field in manifest.output_signature.output_fields
    )
    output_bindings = tuple(
        TensorOutputBinding(
            tensor_name=expectation.tensor_name,
            column=expectation.column,
            semantic="tensor",
            dtype=expectation.dtype,
            squeeze_singleton=False,
        )
        for expectation in output_expectations
    )
    input_root = root / "inference" / entry_point.replace(".", "-") / "input"
    input_root.mkdir(parents=True)
    frame = pd.DataFrame(values)
    for part in range(4):
        frame.iloc[part * 4 : (part + 1) * 4].to_parquet(
            input_root / f"part-{part}.parquet",
            index=False,
        )
    return (
        input_root,
        InputBindingSpec(tensors=tuple(bindings)),
        OutputBindingSpec(tensors=output_bindings),
        output_expectations,
    )


def _actor_snapshot() -> dict[str, Any]:
    from ray.util.state import list_actors

    job_id = str(ray.get_runtime_context().get_job_id())
    return {
        str(actor.actor_id): actor
        for actor in list_actors(
            filters=[("job_id", "=", job_id)],
            limit=10_000,
            detail=True,
        )
    }


def _execute_bundle_inference(
    *,
    bundle_uri: str,
    root: Path,
    entry_point: str,
) -> dict[str, object]:
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.inference.api import run_inference
    from tributo.inference.contracts import (
        BundleModelReference,
        InferenceRequest,
        ParquetResultSinkRequest,
        RayExecutionPolicy,
    )

    input_root, input_binding, output_binding, output_expectations = (
        _stage_bundle_inference_input(
            bundle_uri=bundle_uri,
            root=root,
            entry_point=entry_point,
        )
    )
    sink_root = root / "inference" / entry_point.replace(".", "-") / "sink"
    before = _actor_snapshot()
    result = run_inference(
        InferenceRequest(
            model=BundleModelReference(uri=bundle_uri, role="inference"),
            input=IngestionRequest(
                source=ParquetSourceConfig(
                    path=str(input_root),
                    columns=list(input_binding.projected_columns()),
                ),
                engine="ray",
            ),
            input_binding=input_binding,
            output_binding=output_binding,
            result_sink=ParquetResultSinkRequest(uri=str(sink_root)),
            execution=RayExecutionPolicy(
                batch_size=2,
                concurrency=2,
                num_cpus_per_actor=2,
            ),
            run_id=f"inference-{entry_point.replace('.', '-')}",
        )
    )
    if result.status != "succeeded" or result.sink_receipt is None:
        raise AssertionError(
            f"{entry_point} formal distributed inference failed: {result}"
        )
    receipt = result.sink_receipt
    if (
        receipt.sink_id != "parquet-v1"
        or receipt.uri != str(sink_root)
        or receipt.metadata.get("format") != "parquet"
    ):
        raise AssertionError(f"{entry_point} returned an invalid Parquet receipt")
    materialized = ray.data.read_parquet(str(sink_root)).materialize()
    rows = cast(list[dict[str, object]], materialized.take_all())
    if len(rows) != 16:
        raise AssertionError(f"{entry_point} inference returned {len(rows)} rows")
    for row in rows:
        for expectation in output_expectations:
            if expectation.column not in row:
                raise AssertionError(
                    f"{entry_point} inference omitted output column "
                    f"{expectation.column!r}"
                )
            validate_output_value(expectation, row[expectation.column])
    after = _actor_snapshot()
    actors = [
        actor
        for actor_id, actor in after.items()
        if actor_id not in before
        and "KernelBatchPredictor" in str(actor.class_name)
        and actor.node_id is not None
    ]
    node_ids = sorted({str(actor.node_id) for actor in actors})
    if len(node_ids) != 2:
        raise AssertionError(
            f"{entry_point} inference did not create actors on two nodes: {node_ids}"
        )
    return {
        "status": result.status,
        "row_count": len(rows),
        "node_count": len(node_ids),
        "node_ids": node_ids,
        "output_columns": [expectation.column for expectation in output_expectations],
        "flavor_id": result.flavor_id,
        "manifest_sha256": result.manifest_sha256,
        "result_id": receipt.result_id,
    }


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
) -> dict[str, object] | None:
    entry_point = entry_point_for(algorithm, implementation_id)
    if not _entry_point_enabled(entry_point):
        return None
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
    distributed_inference = None
    if os.environ.get("TRIBUTO_OFFICIAL_DISTRIBUTED_INFERENCE") == "1":
        distributed_inference = _execute_bundle_inference(
            bundle_uri=bundle_uri,
            root=root,
            entry_point=entry_point,
        )
    return {
        "entry_point": entry_point,
        "category": category_for_entry_point(entry_point),
        "algorithm": algorithm,
        "implementation_id": implementation_id,
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "onnx_exported": onnx_exported,
        "inference_roundtrip": inference_roundtrip,
        "distributed_inference": distributed_inference,
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

    if ensemble_resumed is None or first_iterative is None or iterative_resumed is None:
        raise AssertionError("classical recovery Gate unexpectedly skipped a record")
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
) -> dict[str, object] | None:
    entry_point = entry_point_for(algorithm, None)
    if not _entry_point_enabled(entry_point):
        return None
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
    distributed_inference = None
    if os.environ.get("TRIBUTO_OFFICIAL_DISTRIBUTED_INFERENCE") == "1":
        distributed_inference = _execute_bundle_inference(
            bundle_uri=bundle_uri,
            root=root,
            entry_point=entry_point,
        )
    return {
        "entry_point": entry_point,
        "category": category_for_entry_point(entry_point),
        "algorithm": algorithm,
        "implementation_id": receipt.to_dict()["strategy"],
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "receipt": receipt.to_dict(),
        "distributed_inference": distributed_inference,
    }


def _execute_gcm(
    *,
    train_path: Path,
    anomaly_path: Path,
    root: Path,
) -> dict[str, object] | None:
    entry_point = entry_point_for("gcm_root_cause", None)
    if not _entry_point_enabled(entry_point):
        return None
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
    distributed_inference = None
    if os.environ.get("TRIBUTO_OFFICIAL_DISTRIBUTED_INFERENCE") == "1":
        distributed_inference = _execute_bundle_inference(
            bundle_uri=bundle_uri,
            root=root,
            entry_point=entry_point,
        )
    return {
        "entry_point": entry_point,
        "category": category_for_entry_point(entry_point),
        "algorithm": "gcm_root_cause",
        "implementation_id": receipt.to_dict()["strategy"],
        "status": result.execution.status,
        "bundle_uri": bundle_uri,
        "receipt": receipt.to_dict(),
        "distributed_inference": distributed_inference,
    }


def _validate_installed_official_entry_points() -> tuple[dict[str, object], ...]:
    entry_points = tuple(
        entry_point
        for entry_point in importlib.metadata.entry_points(group="tributo.algorithms")
        if (distribution := getattr(entry_point, "dist", None)) is not None
        and canonicalize_name(str(distribution.metadata["Name"])).startswith(
            "tributo-algorithms-"
        )
    )
    by_name: dict[str, list[object]] = {}
    installed_pairs: set[tuple[str, str]] = set()
    for entry_point in entry_points:
        distribution = entry_point.dist
        if distribution is None:
            raise AssertionError(
                f"official Entry Point {entry_point.name!r} has no distribution owner"
            )
        distribution_name = canonicalize_name(str(distribution.metadata["Name"]))
        by_name.setdefault(str(entry_point.name), []).append(entry_point)
        installed_pairs.add((str(entry_point.name), distribution_name))

    duplicates = sorted(name for name, values in by_name.items() if len(values) != 1)
    expected_pairs = {
        (entry_point, canonicalize_name(distribution))
        for entry_point, distribution in ENTRY_POINT_DISTRIBUTIONS.items()
    }
    if duplicates or installed_pairs != expected_pairs:
        raise AssertionError(
            "installed official algorithm Entry Point ownership drifted: "
            f"duplicates={duplicates} "
            f"missing={sorted(expected_pairs - installed_pairs)} "
            f"unexpected={sorted(installed_pairs - expected_pairs)}"
        )
    identities: list[dict[str, object]] = []
    for entry_point_name in sorted(ALL_ENTRY_POINTS):
        entry_point = cast(Any, by_name[entry_point_name][0])
        expected_identity = OFFICIAL_ALGORITHM_IDENTITIES[entry_point_name]
        expected_distribution = canonicalize_name(expected_identity.distribution)
        report = validate_installed_algorithm_package(
            entry_point.load(),
            entry_point_name=entry_point_name,
        )
        if (
            canonicalize_name(report.distribution) != expected_distribution
            or report.algorithm_id != expected_identity.algorithm_id
            or report.implementation_id != expected_identity.implementation_id
            or report.implementation_loaded
        ):
            raise AssertionError(
                f"installed official descriptor identity drifted for {entry_point_name!r}"
            )
        identities.append(
            {
                "entry_point": entry_point_name,
                "distribution": expected_distribution,
                "algorithm_id": report.algorithm_id,
                "implementation_id": report.implementation_id,
                "package_version": report.package_version,
            }
        )
    return tuple(identities)


def main() -> None:
    root = Path(os.environ["TRIBUTO_OFFICIAL_ALGORITHM_GATE_ROOT"])
    if root.parent != Path("/workspace/tributo-work") or not root.name.startswith(
        "tributo-official-algorithm-gate-"
    ):
        raise ValueError("official algorithm Gate root is outside the owned workspace")
    if root.exists():
        raise FileExistsError(f"refusing to reuse Gate root: {root}")
    root.mkdir(parents=True)
    installed_identities = _validate_installed_official_entry_points()
    print(
        "INSTALLATION_RESULT: "
        + json.dumps(
            {
                "record_count": len(installed_identities),
                "records": installed_identities,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    ray.init(address="auto")
    try:
        data_path = _stage_data(root)
        graph_nodes, graph_edges, graph_seeds = _stage_graph_data(root)
        gcm_anomalies = _stage_gcm_anomalies(root)
        records = [
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
                algorithm="pca",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "feature_count": 2,
                    "n_components": 2,
                    "output": {"bundle_uri": str(root / "pca-bundle")},
                },
            ),
            _execute(
                algorithm="kmeans",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "feature_count": 2,
                    "n_clusters": 2,
                    "max_iter": 3,
                    "seed": 7,
                    "runtime": {"checkpoint_dir": str(root / "kmeans-checkpoint")},
                    "output": {"bundle_uri": str(root / "kmeans-bundle")},
                },
            ),
            _execute(
                algorithm="kmeans_minibatch",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "batch_size": 8,
                    "feature_count": 2,
                    "learning_rate": 0.5,
                    "n_clusters": 2,
                    "max_iter": 3,
                    "seed": 7,
                    "runtime": {
                        "checkpoint_dir": str(root / "minibatch-kmeans-checkpoint")
                    },
                    "output": {"bundle_uri": str(root / "minibatch-kmeans-bundle")},
                },
            ),
            _execute(
                algorithm="sgd_classifier",
                implementation_id=None,
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "learning_rate": 0.2,
                    "max_iter": 3,
                    "seed": 7,
                    "runtime": {
                        "checkpoint_dir": str(root / "sgd-classifier-checkpoint")
                    },
                    "output": {"bundle_uri": str(root / "sgd-classifier-bundle")},
                },
            ),
            _execute(
                algorithm="sgd_regressor",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name="outcome",
                data_path=data_path,
                root=root,
                config={
                    "learning_rate": 0.1,
                    "max_iter": 3,
                    "seed": 7,
                    "runtime": {
                        "checkpoint_dir": str(root / "sgd-regressor-checkpoint")
                    },
                    "output": {"bundle_uri": str(root / "sgd-regressor-bundle")},
                },
            ),
            _execute(
                algorithm="isolation_forest",
                implementation_id=None,
                feature_names=("x0", "x1"),
                label_name=None,
                data_path=data_path,
                root=root,
                config={
                    "contamination": "auto",
                    "max_samples": 16,
                    "n_estimators": 4,
                    "unit_count": 4,
                    "seed": 7,
                    "output": {"bundle_uri": str(root / "isolation-forest-bundle")},
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
                algorithm="lstm_classifier",
                implementation_id=None,
                feature_names=("lag_3", "lag_2", "lag_1", "lag_0"),
                data_path=data_path,
                root=root,
                config={
                    "model": {
                        "input_features": 4,
                        "hidden_size": 4,
                        "num_layers": 1,
                    },
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
                        "storage_path": str(root / "lstm-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "lstm-bundle")},
                },
            ),
            _execute(
                algorithm="gru_classifier",
                implementation_id=None,
                feature_names=("lag_3", "lag_2", "lag_1", "lag_0"),
                data_path=data_path,
                root=root,
                config={
                    "model": {
                        "input_features": 4,
                        "hidden_size": 4,
                        "num_layers": 1,
                    },
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
                        "storage_path": str(root / "gru-ray-results"),
                        "resume": {"checkpoint_interval": 1},
                    },
                    "output": {"bundle_uri": str(root / "gru-bundle")},
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
                        "inference_history_width": 8,
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
                algorithm="lightgbm",
                implementation_id="tributo.official.boosting.lightgbm",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "data": {
                        "label_col": "label",
                        "feature_columns": ["x0", "x1"],
                    },
                    "model": {
                        "task": "classification",
                        "objective": "binary",
                        "num_leaves": 3,
                        "min_data_in_leaf": 1,
                        "num_threads": 1,
                        "verbosity": -1,
                    },
                    "training": {"num_rounds": 3},
                    "ray": {"storage_path": str(root / "lightgbm-ray-results")},
                    "output": {"bundle_uri": str(root / "lightgbm-bundle")},
                },
            ),
            _execute(
                algorithm="catboost",
                implementation_id="tributo.official.catboost.parallel_ensemble",
                feature_names=("x0", "x1"),
                data_path=data_path,
                root=root,
                config={
                    "task": "classification",
                    "n_estimators": 2,
                    "unit_count": 2,
                    "seed": 7,
                    "model": {
                        "iterations": 5,
                        "depth": 2,
                        "learning_rate": 0.2,
                    },
                    "output": {"bundle_uri": str(root / "catboost-bundle")},
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
        results = [record for record in records if record is not None]
        category = _gate_category()
        expected_entry_points = entry_points_for_gate(
            category,
            os.environ.get("TRIBUTO_OFFICIAL_ALGORITHM_ENTRY_POINTS", ""),
        )
        actual_entry_points = {str(record["entry_point"]) for record in results}
        if actual_entry_points != expected_entry_points:
            raise AssertionError(
                f"{category} Gate record drift: "
                f"missing={sorted(expected_entry_points - actual_entry_points)} "
                f"unexpected={sorted(actual_entry_points - expected_entry_points)}"
            )
        run_classical_controls = os.environ.get(
            "TRIBUTO_OFFICIAL_EXTENDED_CONTROLS"
        ) == "1" and category in {"all", "classical"}
        baseline_result = (
            _execute_baseline_equivalence(data_path=data_path, records=results)
            if run_classical_controls
            else {"skipped": True}
        )
        tune_result = (
            _execute_portable_tune(data_path=data_path, root=root)
            if run_classical_controls
            else {"skipped": True}
        )
        recovery_result = (
            _execute_checkpoint_recovery(data_path=data_path, root=root)
            if run_classical_controls
            else {"skipped": True}
        )
        inference_result = {
            "record_count": len(results),
            "all_distributed": all(
                isinstance(record.get("distributed_inference"), Mapping)
                and cast(Mapping[str, object], record["distributed_inference"]).get(
                    "node_count"
                )
                == 2
                for record in results
            ),
        }
        failure_result = (
            _execute_required_bundle_failure(data_path=data_path, root=root)
            if run_classical_controls
            else {"skipped": True}
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
