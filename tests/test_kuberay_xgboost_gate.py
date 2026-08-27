"""Unit tests for the formal KubeRay XGBoost workload envelope."""

from __future__ import annotations

import pytest

from tributo.kuberay_submission import KubeRayWorkerResources
from tributo.kuberay_xgboost_gate import (
    _execution_config,
    _validate_ray_resource_evidence,
)


def test_execution_config_uses_formal_xgboost_and_declared_cpu() -> None:
    config = _execution_config(
        KubeRayWorkerResources(num_cpus=2, memory_bytes=2 * 1024**3)
    )

    assert config["algorithm"] == "xgboost"
    assert config["implementation_id"] == "tributo.official.boosting.xgboost"
    assert config["profile"] == "cluster"
    assert config["input"]["ingestion"]["source"]["path"] == (
        "/opt/tributo-kuberay/xgboost-data-parts"
    )
    assert config["worker_count"] == 2
    assert config["resources_per_worker"] == {"num_cpus": 2.0, "num_gpus": 0.0}
    assert config["input"]["features"] == ["x0", "x1"]
    assert config["input"]["label"] == "label"
    assert config["input"]["ingestion"]["read_options"] == {"target_parallelism": 2}
    assert config["algorithm_config"]["data"] == {
        "label_col": "label",
        "feature_columns": ["x0", "x1"],
    }


def test_execution_config_preserves_requested_worker_count() -> None:
    config = _execution_config(
        KubeRayWorkerResources(num_cpus=1, memory_bytes=1024),
        worker_count=3,
    )

    assert config["worker_count"] == 3
    assert config["input"]["ingestion"]["read_options"] == {"target_parallelism": 3}


def test_ray_resource_evidence_validates_total_and_per_node_capacity() -> None:
    resources = KubeRayWorkerResources(num_cpus=2, memory_bytes=2 * 1024**3)

    evidence = _validate_ray_resource_evidence(
        worker_count=3,
        resources=resources,
        cluster_resources={"CPU": 7, "GPU": 0, "memory": 10 * 1024**3},
        nodes=[
            {"Alive": True, "Resources": {"CPU": 1, "memory": 4 * 1024**3}},
            {"Alive": True, "Resources": {"CPU": 2, "memory": 2 * 1024**3}},
            {"Alive": True, "Resources": {"CPU": 2, "memory": 2 * 1024**3}},
            {"Alive": True, "Resources": {"CPU": 2, "memory": 2 * 1024**3}},
        ],
    )

    assert evidence["eligible_node_count"] == 3
    assert evidence["required_total_resources"] == {
        "CPU": 6.0,
        "GPU": 0.0,
        "memory": float(6 * 1024**3),
    }


def test_ray_resource_evidence_rejects_insufficient_eligible_nodes() -> None:
    resources = KubeRayWorkerResources(num_cpus=2, memory_bytes=2 * 1024**3)

    with pytest.raises(RuntimeError, match="eligible nodes"):
        _validate_ray_resource_evidence(
            worker_count=3,
            resources=resources,
            cluster_resources={"CPU": 7, "GPU": 0, "memory": 10 * 1024**3},
            nodes=[
                {"Alive": True, "Resources": {"CPU": 1, "memory": 4 * 1024**3}},
                {"Alive": True, "Resources": {"CPU": 2, "memory": 2 * 1024**3}},
                {"Alive": True, "Resources": {"CPU": 1, "memory": 2 * 1024**3}},
                {"Alive": True, "Resources": {"CPU": 1, "memory": 2 * 1024**3}},
            ],
        )
