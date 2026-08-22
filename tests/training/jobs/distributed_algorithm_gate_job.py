"""Run every first-party formal distributed strategy on a real Ray cluster."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import ray


class _BorrowedDockerRayRuntimeManager:
    """Expose the IT-owned Docker cluster as borrowed cluster evidence."""

    def open(self, _profile: object, **_kwargs: object) -> Any:
        from tributo.algorithms.api import ExecutionProfile, WorkerResources
        from tributo.algorithms.core import RayRuntimeManager, RayRuntimeSession

        if _profile is not ExecutionProfile.CLUSTER:
            raise AssertionError("Docker gate requires the cluster profile")
        resources = _kwargs.get("resources_per_worker")
        worker_count = _kwargs.get("worker_count")
        if not isinstance(resources, WorkerResources) or not isinstance(
            worker_count, int
        ):
            raise AssertionError("Docker gate requires explicit worker resources")
        RayRuntimeManager.validate_resources(
            resources,
            worker_count,
            cluster_resources=ray.cluster_resources(),
            nodes=ray.nodes(),
        )

        return RayRuntimeSession(
            self,
            ExecutionProfile.CLUSTER,
            owned=False,
            cluster_resources=ray.cluster_resources(),
            resource_preflight="validated",
        )

    def _release(self) -> None:
        return None


def _stage_parquet(path: str, records: dict[str, list[Any]]) -> None:
    """Create the shared test fixture once, outside the Tributo execution path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(destination, index=False)


def _remove_fixture(root: str) -> None:
    path = Path(root)
    if (
        path.parent == Path("/workspace/tributo-work")
        and path.name.startswith("tributo-distributed-gate-")
        and path.exists()
    ):
        shutil.rmtree(path, ignore_errors=True)


def _algorithm_config(
    algorithm: str,
    bundle_uri: str,
    *,
    worker_count: int,
) -> dict[str, Any]:
    if algorithm == "third_party_mean_regressor":
        return {}
    output = {"bundle_uri": bundle_uri}
    storage_path = str(Path(bundle_uri).parent / f"ray-results-{algorithm}")
    if algorithm == "multinomial_nb":
        return {"alpha": 1.0, "output": output}
    if algorithm == "third_party_binary_linear":
        return {
            "model": {"input_features": 2},
            "optimizer": {"learning_rate": 0.1},
            "training": {
                "epochs": 2,
                "batch_size": 8,
                "prefetch_batches": 1,
                "seed": 42,
            },
            "ray": {
                "max_failures": 0,
                "storage_path": storage_path,
                "resume": {"checkpoint_interval": 1},
            },
            "output": output,
        }
    if algorithm == "xgboost":
        return {
            "data": {"label_col": "label", "feature_columns": ["f0", "f1"]},
            "model": {"objective": "binary:logistic", "max_depth": 2},
            "training": {
                "num_rounds": 2,
                "val_size": 0.25,
                "test_size": 0.0,
                "seed": 42,
            },
            "ray": {
                "num_workers": worker_count,
                "use_gpu": False,
                "max_failures": 0,
                "storage_path": storage_path,
            },
            "output": output,
        }
    if algorithm == "x_learner":
        return {
            "data": {
                "feature_columns": ["f0", "f1"],
                "treatment_col": "treatment",
                "outcome_col": "outcome",
                "identity_col": "identity",
            },
            "model": {
                "outcome": {"objective": "binary:logistic", "max_depth": 2},
                "effect": {"objective": "reg:squarederror", "max_depth": 2},
                "propensity": {"objective": "binary:logistic", "max_depth": 2},
            },
            "training": {
                "num_rounds": 2,
                "val_size": 0.125,
                "test_size": 0.25,
                "seed": 42,
                "curve_points": 16,
            },
            "ray": {
                "num_workers": worker_count,
                "max_failures": 0,
                "storage_path": storage_path,
            },
            "output": output,
        }
    config: dict[str, Any] = {
        "features": [
            {"name": "f0", "type": "dense", "norm": "standard"},
            {"name": "f1", "type": "dense", "norm": "minmax"},
        ],
        "label_col": "label",
        "model": {"dnn_hidden_units": [8], "dnn_dropout": 0.0},
        "training": {
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.01,
            "val_size": 0.25,
            "seed": 42,
        },
        "ray": {
            "num_workers": worker_count,
            "use_gpu": False,
            "max_failures": 0,
            "storage_path": storage_path,
        },
        "output": output,
    }
    if algorithm == "dnn":
        config["features"].append(
            {
                "name": "segment",
                "type": "sparse",
                "vocab_size": 5,
                "embedding_dim": 2,
            }
        )
        config["features"].append(
            {
                "name": "numeric_code",
                "type": "sparse",
                "vocab_size": 5,
                "embedding_dim": 2,
            }
        )
        config["loss"] = {"type": "bce"}
    elif algorithm == "pu":
        config["pu"] = {
            "loss_type": "nnpu",
            "class_prior_method": "label_frequency",
        }
    else:
        raise ValueError(f"unsupported gate algorithm: {algorithm}")
    return config


def _execution_request(
    algorithm: str,
    data_path: str,
    bundle_uri: str,
    *,
    profile: str,
    worker_count: int,
) -> tuple[object, dict[str, object]]:
    from tributo.algorithms.api import (
        AlgorithmOperation,
        AlgorithmRequest,
        ExecutionProfile,
        ExecutionRequest,
        InputBinding,
    )
    from tributo.data import IngestionRequest, ParquetSourceConfig
    from tributo.integrations.algorithm_inputs import (
        INGESTION_RESOLVER_ID,
        IngestionInputInvocation,
    )

    request_key = f"tributo.gate-{algorithm.replace('_', '-')}"
    invocation = IngestionInputInvocation(
        request=IngestionRequest(
            source=ParquetSourceConfig(path=data_path),
            engine="ray",
        )
    )
    values = {request_key: invocation}
    request = ExecutionRequest(
        algorithm_request=AlgorithmRequest(
            algorithm=algorithm,
            operation=AlgorithmOperation.FIT,
            input_binding=InputBinding(
                name="train",
                resolver_id=INGESTION_RESOLVER_ID,
                reference=request_key,
                feature_names=(
                    ("f0", "f1", "segment", "numeric_code")
                    if algorithm == "dnn"
                    else (
                        ("f0", "f1", "treatment", "identity")
                        if algorithm == "x_learner"
                        else ("f0", "f1")
                    )
                ),
                label_name=("outcome" if algorithm == "x_learner" else "label"),
            ),
            algorithm_config=_algorithm_config(
                algorithm,
                bundle_uri,
                worker_count=worker_count,
            ),
        ),
        profile=ExecutionProfile(profile),
        worker_count=worker_count,
    )
    return request, values


def _execute(
    algorithm: str,
    data_path: str,
    bundle_uri: str,
    *,
    profile: str,
    worker_count: int,
    runtime_manager: object | None = None,
    local_num_cpus: int | None = None,
    expected_rows: int = 64,
) -> dict[str, Any]:
    from tributo.algorithms.api import ResultPolicy
    from tributo.algorithms.composition import build_algorithm_dispatcher
    from tributo.algorithms.core import LocalRuntimeOptions, RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext
    from tributo.exporting.bundle_reader import BundleReader

    request, values = _execution_request(
        algorithm,
        data_path,
        bundle_uri,
        profile=profile,
        worker_count=worker_count,
    )
    resolved_runtime_manager = runtime_manager
    if resolved_runtime_manager is None and local_num_cpus is not None:
        resolved_runtime_manager = RayRuntimeManager(
            default_local_options=LocalRuntimeOptions(
                num_cpus=local_num_cpus,
                num_gpus=0,
            )
        )
    dispatcher = build_algorithm_dispatcher(runtime_manager=resolved_runtime_manager)
    result = dispatcher.execute(
        request,
        InputExecutionContext(values=values),
        resolution_context=InputResolutionContext(values=values),
    )
    receipt = result.execution_receipt
    if receipt is None:
        raise AssertionError(f"{algorithm} returned no ExecutionReceipt")
    preprocessor_state: dict[str, object] | None = None
    if receipt.result_policy is ResultPolicy.BUNDLE_REQUIRED:
        from tributo.exporting.service import bundle_id_for_request

        if result.execution.outputs.get("bundle_id") != bundle_id_for_request(
            receipt.run_id
        ):
            raise AssertionError(
                f"{algorithm} Bundle identity does not match its execution receipt"
            )
        with BundleReader().open_artifact(
            str(result.execution.outputs["bundle_uri"]), role="inference"
        ) as artifact:
            model_name = "x_learner.json" if algorithm == "x_learner" else "model.onnx"
            model = artifact.path_for(model_name)
            if not model.is_file() or model.stat().st_size <= 0:
                raise AssertionError(f"{algorithm} Bundle has no readable model")
            if algorithm == "dnn":
                raw_preprocessor = json.loads(
                    artifact.path_for("preprocessor.json").read_text(encoding="utf-8")
                )
                if not isinstance(raw_preprocessor, dict):
                    raise AssertionError("DNN preprocessor is not a JSON object")
                preprocessor_state = raw_preprocessor
    elif result.execution.artifacts or result.execution.outputs or receipt.artifact_ids:
        raise AssertionError(f"{algorithm} FIT_ONLY execution published artifacts")
    evidence = receipt.to_dict()
    if profile == "local" and (runtime_manager is None or local_num_cpus is not None):
        if not receipt.runtime_owned:
            raise AssertionError("local[*] receipt did not prove runtime ownership")
        if receipt.cross_node:
            raise AssertionError("local[*] execution incorrectly claimed cross-node")
        if worker_count >= 2 and not receipt.distributed:
            raise AssertionError(f"{algorithm} did not prove local distribution")
        if worker_count == 1 and receipt.distributed:
            raise AssertionError(f"{algorithm} single worker claimed distribution")
    elif profile == "cluster":
        if receipt.runtime_owned:
            raise AssertionError("borrowed Ray cluster connection was marked owned")
        if worker_count >= 2:
            if not receipt.distributed or not receipt.cross_node:
                raise AssertionError(
                    f"{algorithm} did not prove cross-node distribution"
                )
            if not receipt.cluster_distributed:
                raise AssertionError(f"{algorithm} did not prove cluster distribution")
        elif receipt.distributed or receipt.cross_node or receipt.cluster_distributed:
            raise AssertionError(
                f"{algorithm} single worker claimed cluster distribution"
            )
    elif worker_count >= 2 and (not receipt.distributed or not receipt.cross_node):
        raise AssertionError(f"{algorithm} did not prove cross-node distribution")
    elif worker_count == 1 and receipt.distributed:
        raise AssertionError(f"{algorithm} single worker claimed distribution")
    if profile == "local" and receipt.cluster_distributed:
        raise AssertionError("Docker local-profile evidence masqueraded as cluster")
    if receipt.driver_materialized_training_rows != 0:
        raise AssertionError("Driver materialized training rows")
    if len({worker.shard_id for worker in receipt.workers}) != worker_count:
        raise AssertionError(f"{algorithm} did not report unique shards")
    if sum(worker.rows_processed or 0 for worker in receipt.workers) <= 0:
        raise AssertionError(f"{algorithm} did not report processed rows")
    if algorithm in {"dnn", "pu", "third_party_binary_linear"}:
        if (
            sum(sum(worker.input_rows.values()) for worker in receipt.workers)
            != expected_rows
        ):
            raise AssertionError(f"{algorithm} did not prove complete input coverage")
        if any(
            worker.batch_count is None
            or worker.collective_steps is None
            or worker.batch_count < 1
            or worker.batch_count > worker.collective_steps
            for worker in receipt.workers
        ):
            raise AssertionError(
                f"{algorithm} did not prove aligned no-replay batch execution"
            )
    if algorithm == "pu":
        class_prior = result.execution.metrics.get("class_prior")
        if not isinstance(class_prior, (int, float)) or abs(class_prior - 0.25) > 1e-12:
            raise AssertionError(
                "PU label_frequency did not all-reduce the exact global class prior"
            )
        observed_splits = {
            name: sum(worker.input_rows.get(name, 0) for worker in receipt.workers)
            for name in ("positive", "unlabeled", "positive_val", "unlabeled_val")
        }
        if observed_splits != {
            "positive": 12,
            "unlabeled": 36,
            "positive_val": 4,
            "unlabeled_val": 12,
        }:
            raise AssertionError(
                "PU global stratified split did not preserve the expected class "
                f"counts: {observed_splits}"
            )
    if algorithm in {"multinomial_nb", "third_party_mean_regressor"}:
        if sum(worker.rows_processed or 0 for worker in receipt.workers) != 64:
            raise AssertionError(f"{algorithm} did not consume every input row")
        if worker_count >= 2 and receipt.state.details.get("tree_depth") != 1:
            raise AssertionError(f"{algorithm} did not execute a reduction tree")
    if (
        algorithm == "xgboost"
        and sum(worker.rows_processed or 0 for worker in receipt.workers) != 48
    ):
        raise AssertionError("XGBoost train shards did not cover the global split")
    if algorithm == "x_learner":
        stage_names = ("mu0", "mu1", "tau0", "tau1", "propensity")
        details = receipt.state.details
        for stage in stage_names:
            if details.get(f"stage.{stage}.workers") != worker_count:
                raise AssertionError(
                    f"X-Learner stage {stage} did not report every worker"
                )
            node_count = details.get(f"stage.{stage}.nodes")
            if profile == "cluster" and worker_count >= 2:
                if not isinstance(node_count, int) or node_count < 2:
                    raise AssertionError(
                        f"X-Learner stage {stage} did not prove cross-node execution"
                    )
            elif node_count != 1:
                raise AssertionError(
                    f"X-Learner stage {stage} reported unexpected node count"
                )
            stage_rows = details.get(f"stage.{stage}.rows")
            stage_digest = details.get(f"stage.{stage}.digest")
            if not isinstance(stage_rows, int) or stage_rows < 1:
                raise AssertionError(f"X-Learner stage {stage} has no row evidence")
            if not isinstance(stage_digest, str) or len(stage_digest) != 64:
                raise AssertionError(f"X-Learner stage {stage} has no digest evidence")
        if details["stage.mu0.rows"] != details["stage.tau0.rows"]:
            raise AssertionError("X-Learner control stage row coverage drifted")
        if details["stage.mu1.rows"] != details["stage.tau1.rows"]:
            raise AssertionError("X-Learner treated stage row coverage drifted")
        if details["stage.propensity.rows"] != (
            details["stage.mu0.rows"] + details["stage.mu1.rows"]
        ):
            raise AssertionError("X-Learner full-input stage row coverage drifted")
        if result.execution.metrics.get("ate_definition") != "model_mean_cate":
            raise AssertionError("X-Learner did not report its ATE definition")
        if not isinstance(result.execution.metrics.get("qini"), (int, float)):
            raise AssertionError("X-Learner did not report Qini")
    execution_summary = {
        "algorithm": algorithm,
        "worker_count": worker_count,
        "status": result.execution.status,
        "receipt": evidence,
    }
    if algorithm == "dnn":
        from tributo.serving.identity_predictor import IdentityPredictor

        predictor = IdentityPredictor(
            bundle_uri=str(result.execution.outputs["bundle_uri"])
        )
        try:
            predictions = predictor.predict_batch(
                [
                    {
                        "f0": 0.75,
                        "f1": 0.60,
                        "segment": 0,
                        "numeric_code": "001",
                    },
                    {
                        "f0": 1.10,
                        "f1": 0.85,
                        "segment": 1,
                        "numeric_code": "1",
                    },
                    {
                        "f0": 1.40,
                        "f1": 1.00,
                        "segment": 2,
                        "numeric_code": "10",
                    },
                ]
            )
        finally:
            predictor.close()
        execution_summary["probe_probabilities"] = [
            prediction["probability"] for prediction in predictions
        ]
        execution_summary["preprocessor_state"] = preprocessor_state
    if algorithm == "x_learner":
        import numpy as np

        from tributo.exporting.models import BundleRef
        from tributo.inference.bundle_predictor import BundleBatchPredictor
        from tributo.inference.contracts import (
            InputBindingSpec,
            OutputBindingSpec,
            ResolvedModelSelection,
            TensorInputBinding,
            TensorOutputBinding,
        )

        predictor = BundleBatchPredictor(
            ResolvedModelSelection(
                bundle_ref=BundleRef(
                    canonical_uri=str(result.execution.outputs["bundle_uri"]),
                    bundle_id=str(result.execution.outputs["bundle_id"]),
                    manifest_sha256=str(result.execution.outputs["manifest_sha256"]),
                ),
                role="inference",
                flavor_id="x-learner-v1",
                source_provenance="tributo-bundle",
            ),
            InputBindingSpec(
                tensors=(
                    TensorInputBinding(
                        tensor_name="float_input",
                        columns=("f0", "f1"),
                        dtype="float32",
                    ),
                )
            ),
            OutputBindingSpec(
                tensors=(
                    TensorOutputBinding(
                        tensor_name="cate",
                        column="cate",
                        semantic="score",
                    ),
                    TensorOutputBinding(
                        tensor_name="quadrant",
                        column="quadrant",
                        semantic="label",
                    ),
                )
            ),
        )
        try:
            prediction = predictor(
                {
                    "f0": np.asarray([0.75, 1.40], dtype=np.float32),
                    "f1": np.asarray([0.60, 1.00], dtype=np.float32),
                }
            )
        finally:
            predictor.close()
        if prediction["cate"].shape != (2,) or prediction["quadrant"].shape != (2,):
            raise AssertionError("X-Learner Bundle prediction shape is invalid")
    return execution_summary


def _assert_dnn_single_multi_numerical_consistency(
    single: dict[str, Any],
    multi: dict[str, Any],
) -> None:
    """Compare one controlled full-batch update across one and two workers."""
    single_probabilities = single.get("probe_probabilities")
    multi_probabilities = multi.get("probe_probabilities")
    if not isinstance(single_probabilities, list) or not isinstance(
        multi_probabilities, list
    ):
        raise AssertionError("DNN numerical Gate did not return probe predictions")
    if len(single_probabilities) != len(multi_probabilities) or any(
        abs(float(left) - float(right)) > 1e-5
        for left, right in zip(single_probabilities, multi_probabilities, strict=True)
    ):
        raise AssertionError(
            "DNN two-worker full-batch update diverged from the single-worker "
            f"reference: single={single_probabilities}, multi={multi_probabilities}"
        )
    single_preprocessor = single.get("preprocessor_state")
    multi_preprocessor = multi.get("preprocessor_state")
    if not isinstance(single_preprocessor, dict) or not isinstance(
        multi_preprocessor, dict
    ):
        raise AssertionError("DNN preprocessing Gate returned invalid state")
    if single_preprocessor.get("label_encoders") != multi_preprocessor.get(
        "label_encoders"
    ):
        raise AssertionError("DNN global categorical preprocessing depends on sharding")
    expected_encoder = {
        "segment": {str(index): index for index in range(4)},
        "numeric_code": {"001": 0, "1": 1, "10": 2, "2": 3},
    }
    if single_preprocessor.get("label_encoders") != expected_encoder:
        raise AssertionError(
            "DNN categorical preprocessing differs from the deterministic reference: "
            f"{single_preprocessor.get('label_encoders')}"
        )
    if single_preprocessor.get("label_encoder_key_types") != {
        "segment": "int",
        "numeric_code": "str",
    }:
        raise AssertionError("DNN categorical key type metadata is missing")
    single_norm = single_preprocessor.get("norm_params")
    multi_norm = multi_preprocessor.get("norm_params")
    if not isinstance(single_norm, dict) or not isinstance(multi_norm, dict):
        raise AssertionError("DNN preprocessing Gate returned invalid norm params")
    if set(single_norm) != set(multi_norm):
        raise AssertionError("DNN global normalization fields depend on sharding")
    for feature_name in single_norm:
        single_values = single_norm[feature_name]
        multi_values = multi_norm[feature_name]
        if not isinstance(single_values, dict) or not isinstance(multi_values, dict):
            raise AssertionError("DNN normalization parameters are malformed")
        if set(single_values) != set(multi_values) or any(
            abs(float(single_values[name]) - float(multi_values[name])) > 1e-12
            for name in single_values
        ):
            raise AssertionError(
                "DNN global preprocessing changed with shard arrangement: "
                f"single={single_norm}, multi={multi_norm}"
            )


def _assert_local_failure_releases_runtime(
    data_path: str,
    bundle_uri: str,
) -> None:
    from tributo.algorithms.api import AlgorithmExecutionError
    from tributo.algorithms.composition import build_algorithm_dispatcher
    from tributo.algorithms.core import LocalRuntimeOptions, RayRuntimeManager
    from tributo.algorithms.spi import InputExecutionContext, InputResolutionContext

    request, values = _execution_request(
        "multinomial_nb",
        data_path,
        bundle_uri,
        profile="local",
        worker_count=1,
    )
    try:
        build_algorithm_dispatcher(
            runtime_manager=RayRuntimeManager(
                default_local_options=LocalRuntimeOptions(num_cpus=4, num_gpus=0)
            )
        ).execute(
            request,
            InputExecutionContext(values=values),
            resolution_context=InputResolutionContext(values=values),
            cancelled=True,
        )
    except AlgorithmExecutionError:
        pass
    else:
        raise AssertionError("cancelled local execution unexpectedly succeeded")
    if ray.is_initialized():
        raise AssertionError("failed local execution leaked its owned Ray runtime")


def _assert_third_party_descriptor_discovered() -> None:
    from tributo.plugin import discover_algorithm_descriptors

    matches = [
        descriptor
        for descriptor in discover_algorithm_descriptors()
        if descriptor.name == "third_party_mean_regressor"
    ]
    if len(matches) != 1:
        raise AssertionError(
            "Ray Job runtime did not discover exactly one third-party descriptor"
        )


def _assert_algorithm_distribution_environment() -> None:
    """Prove the Ray Job received the validated code-only Wheel contract."""
    import json

    expected_mode = os.environ.get("TRIBUTO_EXPECTED_ALGORITHM_MODE")
    if expected_mode is None:
        return
    if os.environ.get("TRIBUTO_ALGORITHM_DISTRIBUTION_MODE") != expected_mode:
        raise AssertionError("algorithm distribution mode was not propagated")
    if os.environ.get("TRIBUTO_PLUGINS") != "third_party_mean_regressor":
        raise AssertionError("algorithm entry-point filter was not propagated")
    receipt = json.loads(os.environ["TRIBUTO_ALGORITHM_PREFLIGHT_RECEIPT"])
    if receipt["mode"] != expected_mode:
        raise AssertionError("algorithm preflight receipt mode disagrees")
    if receipt["package_name"] != "tributo-test-distributed-algorithm":
        raise AssertionError("algorithm preflight receipt package disagrees")


def main() -> int:
    mode = os.environ.get("TRIBUTO_DISTRIBUTED_GATE_PROFILE", "docker-distributed")
    if mode not in {
        "docker-distributed",
        "docker-required-artifact-failure",
        "local",
    }:
        raise ValueError(f"unsupported distributed Gate profile: {mode}")
    if mode.startswith("docker-"):
        ray.init()
    elif ray.is_initialized():
        raise AssertionError(
            f"{mode} Gate must begin without an initialized Ray driver"
        )
    rng = random.Random(42)
    _assert_algorithm_distribution_environment()
    f0 = [abs(rng.gauss(1.0, 0.3)) for _ in range(64)]
    f1 = [abs(rng.gauss(0.8, 0.2)) for _ in range(64)]
    segment = [index % 4 for index in range(64)]
    numeric_code = [("001", "1", "2", "10")[index % 4] for index in range(64)]
    labels = [float(index % 4 == 0) for index in range(64)]
    treatment = [index % 2 for index in range(64)]
    outcome = [
        float(
            (f0[index] + (0.45 if treatment[index] and f1[index] > 0.75 else 0.0))
            > 1.05
        )
        for index in range(64)
    ]
    records = {
        "f0": f0,
        "f1": f1,
        "segment": segment,
        "numeric_code": numeric_code,
        "label": labels,
        "treatment": treatment,
        "outcome": outcome,
        "identity": [f"user-{index:03d}" for index in range(64)],
    }
    configured_root = os.environ.get("TRIBUTO_DISTRIBUTED_GATE_ROOT")
    root = configured_root or (
        f"/workspace/tributo-work/tributo-distributed-gate-{uuid.uuid4().hex}"
    )
    data_path = f"{root}/data.parquet"
    try:
        _stage_parquet(data_path, records)
        if mode == "local":
            _assert_third_party_descriptor_discovered()
            only = os.environ.get("TRIBUTO_ALGORITHM_LOCAL_ONLY")
            if only:
                if only != "x_learner":
                    raise ValueError("TRIBUTO_ALGORITHM_LOCAL_ONLY must be x_learner")
                results = [
                    _execute(
                        "x_learner",
                        data_path,
                        f"{root}/bundle-x-learner-{worker_count}",
                        profile="local",
                        worker_count=worker_count,
                        local_num_cpus=worker_count + 1,
                    )
                    for worker_count in (1, 2)
                ]
                print(f"RESULT: {json.dumps(results, sort_keys=True)}")
                return 0
            results = [
                _execute(
                    algorithm,
                    data_path,
                    f"{root}/bundle-{algorithm}-single",
                    profile="local",
                    worker_count=1,
                    local_num_cpus=4,
                )
                for algorithm in ("dnn", "pu", "xgboost", "multinomial_nb")
            ]
            results.append(
                _execute(
                    "x_learner",
                    data_path,
                    f"{root}/bundle-x-learner-single",
                    profile="local",
                    worker_count=1,
                    local_num_cpus=2,
                )
            )
            results.extend(
                _execute(
                    algorithm,
                    data_path,
                    f"{root}/bundle-{algorithm}-multi",
                    profile="local",
                    worker_count=2,
                    local_num_cpus=4,
                )
                for algorithm in ("dnn", "multinomial_nb")
            )
            results.append(
                _execute(
                    "x_learner",
                    data_path,
                    f"{root}/bundle-x-learner-multi",
                    profile="local",
                    worker_count=2,
                    local_num_cpus=3,
                )
            )
            results.extend(
                _execute(
                    "third_party_mean_regressor",
                    data_path,
                    f"{root}/unused-fit-only-output-{worker_count}",
                    profile="local",
                    worker_count=worker_count,
                    local_num_cpus=4,
                )
                for worker_count in (1, 2)
            )
            results.extend(
                _execute(
                    "third_party_binary_linear",
                    data_path,
                    f"{root}/bundle-torch-recipe-{worker_count}",
                    profile="local",
                    worker_count=worker_count,
                    local_num_cpus=4,
                )
                for worker_count in (1, 2)
            )
            dnn_results = [result for result in results if result["algorithm"] == "dnn"]
            _assert_dnn_single_multi_numerical_consistency(
                next(result for result in dnn_results if result["worker_count"] == 1),
                next(result for result in dnn_results if result["worker_count"] == 2),
            )
            _assert_local_failure_releases_runtime(
                data_path,
                f"{root}/bundle-cancelled",
            )
        elif mode == "docker-required-artifact-failure":
            blocked = Path(root) / "blocked-output"
            blocked.write_text("not a directory", encoding="utf-8")
            _execute(
                "multinomial_nb",
                data_path,
                str(blocked / "bundle"),
                profile="local",
                worker_count=2,
                runtime_manager=_BorrowedDockerRayRuntimeManager(),
            )
            raise AssertionError("required artifact failure unexpectedly succeeded")
        else:
            profile = "cluster"
            runtime_manager = _BorrowedDockerRayRuntimeManager()
            _assert_third_party_descriptor_discovered()
            algorithms = [
                "dnn",
                "pu",
                "xgboost",
                "multinomial_nb",
                "x_learner",
            ]
            results = [
                _execute(
                    algorithm,
                    data_path,
                    f"{root}/bundle-{algorithm}",
                    profile=profile,
                    worker_count=2,
                    runtime_manager=runtime_manager,
                )
                for algorithm in algorithms
            ]
            results.extend(
                _execute(
                    "third_party_mean_regressor",
                    data_path,
                    f"{root}/unused-fit-only-output-{worker_count}",
                    profile=profile,
                    worker_count=worker_count,
                    runtime_manager=runtime_manager,
                )
                for worker_count in (1, 2)
            )
        print(f"RESULT: {json.dumps(results, sort_keys=True)}")
    finally:
        if not configured_root:
            _remove_fixture(root)
        if mode.startswith("docker-"):
            ray.shutdown()
        elif mode == "local" and ray.is_initialized():
            raise AssertionError("local[*] Gate leaked an owned Ray runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
