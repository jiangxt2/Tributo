"""Formal XGBoost workload used by the KubeRay integration gate."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ray

from tributo.kuberay_submission import KubeRayWorkerResources

_DATA_PATH = Path("/opt/tributo-kuberay/xgboost-data-parts")
_CONFIG_PATH = Path("/tmp/tributo-kuberay-xgboost-execution.json")
_STORAGE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--num-cpus", type=float, required=True)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--storage-key", default="unit")
    return parser


def _execution_config(
    resources: KubeRayWorkerResources,
    *,
    worker_count: int = 2,
    storage_key: str = "unit",
) -> dict[str, Any]:
    return {
        "algorithm": "xgboost",
        "implementation_id": "tributo.official.boosting.xgboost",
        "profile": "cluster",
        "worker_count": worker_count,
        "input": {
            "ingestion": {
                "source": {"type": "csv", "path": str(_DATA_PATH)},
                "engine": "ray",
                "read_options": {"target_parallelism": worker_count},
            },
            "features": ["x0", "x1"],
            "label": "label",
        },
        "algorithm_config": {
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
            "ray": {"storage_path": f"/tmp/tributo-kuberay-shared/{storage_key}"},
            "output": {
                "bundle_uri": f"/tmp/tributo-kuberay-xgboost-bundle/{storage_key}"
            },
        },
        "resources_per_worker": {
            "num_cpus": resources.num_cpus,
            "num_gpus": resources.num_gpus,
        },
    }


def _resource_map(value: Mapping[Any, Any], field_name: str) -> dict[str, float]:
    """Normalize Ray's resource maps without exposing node identities."""
    normalized: dict[str, float] = {}
    for name, amount in value.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{field_name} contains an invalid resource name")
        if (
            not isinstance(amount, (int, float))
            or isinstance(amount, bool)
            or not math.isfinite(float(amount))
            or float(amount) < 0
        ):
            raise RuntimeError(f"{field_name}.{name} is not a finite resource amount")
        normalized[name] = float(amount)
    return dict(sorted(normalized.items()))


def _resource_satisfies(
    available: Mapping[str, float], required: Mapping[str, float]
) -> bool:
    """Return whether one Ray node can satisfy one worker resource request."""
    for name, amount in required.items():
        actual = float(available.get(name, 0.0))
        tolerance = max(1e-6, abs(amount) * 1e-9)
        if actual + tolerance < amount:
            return False
    return True


def _validate_ray_resource_evidence(
    *,
    worker_count: int,
    resources: KubeRayWorkerResources,
    cluster_resources: Mapping[Any, Any],
    nodes: list[Mapping[Any, Any]],
) -> dict[str, Any]:
    """Validate and return credential-free Ray resource evidence."""
    if worker_count < 1:
        raise RuntimeError("worker_count must be positive")
    cluster = _resource_map(cluster_resources, "cluster_resources")
    required_per_worker = {
        "CPU": resources.num_cpus,
        "GPU": resources.num_gpus,
        "memory": float(resources.memory_bytes),
        **dict(resources.custom),
    }
    required_total = {
        name: amount * worker_count for name, amount in required_per_worker.items()
    }
    if not _resource_satisfies(cluster, required_total):
        raise RuntimeError(
            "Ray cluster resources do not satisfy the requested worker total"
        )
    alive_nodes: list[dict[str, float]] = []
    for node in nodes:
        if node.get("Alive") is True:
            raw_resources = node.get("Resources", {})
            if not isinstance(raw_resources, Mapping):
                raise RuntimeError("Ray node Resources must be a mapping")
            alive_nodes.append(_resource_map(raw_resources, "node_resources"))
    if len(alive_nodes) < worker_count:
        raise RuntimeError("Ray has fewer alive nodes than requested workers")
    eligible_nodes = [
        node for node in alive_nodes if _resource_satisfies(node, required_per_worker)
    ]
    if len(eligible_nodes) < worker_count:
        raise RuntimeError(
            "Ray has fewer eligible nodes than requested worker replicas"
        )
    return {
        "cluster_resources": cluster,
        "alive_node_resources": alive_nodes,
        "required_resources_per_worker": required_per_worker,
        "required_total_resources": required_total,
        "eligible_node_count": len(eligible_nodes),
    }


def _ray_resource_evidence(
    worker_count: int, resources: KubeRayWorkerResources
) -> dict[str, Any]:
    """Read and validate the actual Ray resource view after training."""
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    return _validate_ray_resource_evidence(
        worker_count=worker_count,
        resources=resources,
        cluster_resources=ray.cluster_resources(),
        nodes=ray.nodes(),
    )


def _run_formal_algorithm() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tributo.cli",
            "algo",
            "run",
            "--config",
            str(_CONFIG_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "formal XGBoost execution failed:\n" + completed.stdout + completed.stderr
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("formal XGBoost CLI did not return JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("formal XGBoost CLI returned a non-object result")
    return value


def main(argv: list[str] | None = None) -> int:
    """Run XGBoost through the formal Tributo CLI and print a receipt."""
    args = _parser().parse_args(argv)
    if args.worker_count < 2:
        raise ValueError("the formal XGBoost gate requires at least two workers")
    if _STORAGE_KEY_RE.fullmatch(args.storage_key) is None:
        raise ValueError("storage_key must be a safe path component")
    resources = KubeRayWorkerResources(
        num_cpus=args.num_cpus,
        memory_bytes=args.memory_bytes,
    )
    _CONFIG_PATH.write_text(
        json.dumps(
            _execution_config(
                resources,
                worker_count=args.worker_count,
                storage_key=args.storage_key,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result = _run_formal_algorithm()
    receipt = result.get("execution_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("formal XGBoost result has no ExecutionReceipt")
    if (
        receipt.get("execution_profile") != "cluster"
        or receipt.get("requested_worker_count") != args.worker_count
        or receipt.get("requested_resources_per_worker", {}).get("num_cpus")
        != args.num_cpus
        or receipt.get("distributed") is not True
        or receipt.get("cluster_distributed") is not True
    ):
        raise RuntimeError("formal XGBoost ExecutionReceipt is not distributed")
    if not isinstance(result.get("outputs"), dict):
        raise RuntimeError("formal XGBoost result has no outputs")
    ray_resources = _ray_resource_evidence(args.worker_count, resources)
    print(
        "RESULT: "
        + json.dumps(
            {
                "status": "succeeded",
                "algorithm": "xgboost",
                "ray_version": ray.__version__,
                "node_count": len(ray.nodes()),
                "requested_worker_count": args.worker_count,
                "requested_resources_per_worker": resources.model_dump(mode="json"),
                "ray_resources": ray_resources,
                "execution_receipt": receipt,
                "outputs": result["outputs"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
