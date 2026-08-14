"""Numerical and contract tests for distributed MultinomialNB."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from tributo.algorithms.api import (
    AlgorithmExecutionError,
    AlgorithmInputError,
    ResolvedAlgorithmPlan,
    WorkerExecutionEvidence,
    WorkerResources,
)
from tributo.algorithms.builtin.multinomial_nb import (
    DistributedMultinomialNB,
    export_model,
)
from tributo.algorithms.spi import AlgorithmExecutionContext
from tributo.integrations.algorithm_runtimes.map_reduce import (
    _MapReduceStageResult,
    _validate_input_coverage,
    _validate_partition_row_count,
    _validate_state,
)


def _plan(
    *,
    config: dict[str, object] | None = None,
    sample_weight_name: str | None = None,
) -> ResolvedAlgorithmPlan:
    value = SimpleNamespace(
        input_binding=SimpleNamespace(
            feature_names=("f0", "f1", "f2"),
            label_name="label",
            sample_weight_name=sample_weight_name,
        ),
        algorithm_config=config or {},
        runtime=SimpleNamespace(distribution_digest="0" * 64),
    )
    return cast(ResolvedAlgorithmPlan, value)


def _context() -> AlgorithmExecutionContext:
    return AlgorithmExecutionContext(inputs={})


def _batch(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    sparse_columns: bool = False,
) -> dict[str, object]:
    columns: dict[str, object] = {
        f"f{index}": (
            sparse.csr_matrix(features[:, index : index + 1])
            if sparse_columns
            else features[:, index]
        )
        for index in range(features.shape[1])
    }
    columns["label"] = labels
    if weights is not None:
        columns["weight"] = weights
    return columns


def _distributed_fit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict[str, object] | None = None,
    weights: np.ndarray | None = None,
    sparse_columns: bool = False,
) -> tuple[DistributedMultinomialNB, Any, dict[str, object]]:
    algorithm = DistributedMultinomialNB(
        _plan(
            config=config, sample_weight_name="weight" if weights is not None else None
        )
    )
    boundaries = (0, 2, 2, 5, len(labels))
    states = [
        algorithm.map_partition(
            (
                _batch(
                    features[start:stop],
                    labels[start:stop],
                    weights[start:stop] if weights is not None else None,
                    sparse_columns=sparse_columns,
                ),
            ),
            _context(),
        )
        for start, stop in zip(boundaries, boundaries[1:])
    ]
    left = algorithm.merge_states(states[0], states[1])
    right = algorithm.merge_states(states[2], states[3])
    merged = algorithm.merge_states(left, right)
    return algorithm, algorithm.finalize_model(merged).estimator, merged


@pytest.mark.parametrize(
    "config",
    [
        {"alpha": 1.0, "fit_prior": True},
        {"alpha": 0.25, "fit_prior": False},
        {"alpha": 0.5, "class_prior": [0.2, 0.3, 0.5]},
    ],
)
def test_tree_reduction_matches_central_sklearn(config: dict[str, object]) -> None:
    features = np.asarray(
        [
            [2, 1, 0],
            [0, 1, 3],
            [4, 0, 1],
            [1, 2, 1],
            [0, 0, 5],
            [3, 1, 2],
            [1, 4, 0],
            [2, 0, 2],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 2, 1, 2, 2, 0], dtype=np.int64)
    algorithm, distributed, merged = _distributed_fit(
        features,
        labels,
        config=config,
    )
    central = MultinomialNB(**config).fit(features, labels)

    np.testing.assert_array_equal(distributed.classes_, central.classes_)
    np.testing.assert_allclose(distributed.class_count_, central.class_count_, atol=0)
    np.testing.assert_allclose(
        distributed.feature_count_, central.feature_count_, atol=0
    )
    np.testing.assert_allclose(
        distributed.feature_log_prob_, central.feature_log_prob_, atol=1e-12
    )
    np.testing.assert_allclose(
        distributed.class_log_prior_, central.class_log_prior_, atol=1e-12
    )
    np.testing.assert_array_equal(
        distributed.predict(features), central.predict(features)
    )
    np.testing.assert_allclose(
        distributed.predict_proba(features), central.predict_proba(features), atol=1e-12
    )

    empty = algorithm.empty_partition()
    left_identity = algorithm.merge_states(empty, merged)
    right_identity = algorithm.merge_states(merged, empty)
    for name in ("classes", "class_count", "feature_count", "row_count"):
        np.testing.assert_array_equal(left_identity[name], merged[name])
        np.testing.assert_array_equal(right_identity[name], merged[name])


def test_weighted_sparse_batches_match_central_sklearn() -> None:
    features = np.asarray(
        [
            [0, 2, 1],
            [3, 0, 1],
            [1, 1, 0],
            [0, 4, 2],
            [2, 0, 3],
            [1, 2, 0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 2, 1, 2], dtype=np.int64)
    weights = np.asarray([0.5, 2.0, 1.5, 0.0, 3.0, 0.75], dtype=np.float64)
    _, distributed, merged = _distributed_fit(
        features,
        labels,
        config={"alpha": 0.75},
        weights=weights,
        sparse_columns=True,
    )
    central = MultinomialNB(alpha=0.75).fit(
        sparse.csr_matrix(features),
        labels,
        sample_weight=weights,
    )

    assert int(merged["row_count"]) == len(labels)
    np.testing.assert_allclose(distributed.class_count_, central.class_count_)
    np.testing.assert_allclose(distributed.feature_count_, central.feature_count_)
    np.testing.assert_allclose(
        distributed.predict_proba(features), central.predict_proba(features), atol=1e-12
    )


def test_sparse_batch_stays_sparse_until_bounded_statistics() -> None:
    features = np.asarray([[0, 2, 1], [3, 0, 1]], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    algorithm = DistributedMultinomialNB(_plan())

    batch_features, _, _ = algorithm._batch_arrays(
        _batch(features, labels, sparse_columns=True)
    )

    assert sparse.isspmatrix_csr(batch_features)


def _runtime_worker(rank: int, rows: int) -> WorkerExecutionEvidence:
    return WorkerExecutionEvidence(
        worker_id=f"worker-{rank}",
        node_id="node-a",
        rank=rank,
        world_size=2,
        shard_id=f"shard-{rank}",
        resources=WorkerResources(),
        rows_processed=rows,
    )


def test_runtime_rejects_empty_distributed_map_shard() -> None:
    plan = cast(
        ResolvedAlgorithmPlan,
        SimpleNamespace(
            runtime=SimpleNamespace(worker_count=2),
            distribution_spec=SimpleNamespace(distributed_min_workers=2),
        ),
    )

    with pytest.raises(AlgorithmInputError, match="non-empty input shard"):
        _validate_partition_row_count(plan, 0)


def test_runtime_cross_checks_observed_rows_against_driver_count() -> None:
    stage = _MapReduceStageResult(
        state={},
        state_digest="a" * 64,
        state_size_bytes=1,
        actual_versions={},
        workers=(_runtime_worker(0, 2), _runtime_worker(1, 1)),
        expected_total_rows=4,
    )

    with pytest.raises(AlgorithmExecutionError, match="coverage mismatch"):
        _validate_input_coverage(stage)

    complete = _MapReduceStageResult(
        state=stage.state,
        state_digest=stage.state_digest,
        state_size_bytes=stage.state_size_bytes,
        actual_versions=stage.actual_versions,
        workers=(_runtime_worker(0, 2), _runtime_worker(1, 2)),
        expected_total_rows=4,
    )
    assert _validate_input_coverage(complete) == 4


def test_state_schema_and_size_gate_fail_closed() -> None:
    algorithm = DistributedMultinomialNB(_plan())
    state = algorithm.empty_partition()
    _validate_state(state, algorithm.state_schema(), 4096)

    invalid = dict(state)
    invalid["classes"] = np.empty((0,), dtype=np.float64)
    with pytest.raises(Exception, match="dtype"):
        _validate_state(invalid, algorithm.state_schema(), 4096)
    with pytest.raises(Exception, match="exceeds"):
        _validate_state(state, algorithm.state_schema(), 1)


def test_public_api_finalizer_exports_runnable_onnx() -> None:
    import onnxruntime as ort

    features = np.asarray(
        [[2, 0, 1], [0, 3, 1], [1, 0, 4], [3, 1, 0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    algorithm, estimator, merged = _distributed_fit(features, labels)
    model = algorithm.finalize_model(merged)
    result = export_model(model=model, plan=_plan())

    assert result.status == "succeeded"
    assert result.artifacts[0].format == "application/onnx"
    session = ort.InferenceSession(result.artifacts[0].payload)
    outputs = session.run(None, {"float_input": features.astype(np.float32)})
    predicted = np.asarray(outputs[0]).reshape(-1)
    np.testing.assert_array_equal(predicted, estimator.predict(features))


def test_finalizer_publishes_a_valid_bundle_when_requested(tmp_path: Any) -> None:
    features = np.asarray(
        [[2, 0, 1], [0, 3, 1], [1, 0, 4], [3, 1, 0]],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    algorithm, _, merged = _distributed_fit(features, labels)
    bundle_root = tmp_path / "bundles"

    result = export_model(
        model=algorithm.finalize_model(merged),
        plan=_plan(config={"output": {"bundle_uri": str(bundle_root)}}),
    )

    assert result.status == "succeeded"
    assert result.outputs["bundle_id"]
    assert result.outputs["execution_id"]
    assert result.outputs["manifest_sha256"]
    bundle_uri = result.outputs["bundle_uri"]
    assert isinstance(bundle_uri, str)
    assert (bundle_root / bundle_uri.rsplit("/", 1)[-1] / "manifest.json").is_file()
