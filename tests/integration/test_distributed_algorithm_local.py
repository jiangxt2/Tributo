"""Owned ``local[*]`` runtime Gate executed inside one isolated Linux container."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _result_from_output(output: str) -> list[dict[str, Any]]:
    for line in output.splitlines():
        if line.startswith("RESULT: "):
            return json.loads(line.removeprefix("RESULT: "))
    raise AssertionError(f"RESULT line not found in local Gate output:\n{output}")


def test_owned_local_runtime_executes_without_ray_jobs() -> None:
    if os.environ.get("TRIBUTO_DOCKER_ALGORITHM_LOCAL_IT") != "1":
        pytest.fail("local[*] IT must run inside its isolated Docker container")
    if os.environ.get("RAY_ADDRESS"):
        pytest.fail("local[*] IT must not inherit an external Ray address")
    command = [
        "/opt/tributo/.venv/bin/python",
        "tests/training/jobs/distributed_algorithm_gate_job.py",
    ]
    process = subprocess.run(
        command,
        cwd=Path(__file__).parents[2],
        env={
            **os.environ,
            "TRIBUTO_DISTRIBUTED_GATE_PROFILE": "local",
            "TRIBUTO_DISTRIBUTED_GATE_ROOT": (
                "/workspace/tributo-work/tributo-distributed-gate-local"
            ),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, output
    results = _result_from_output(output)
    expected = {
        ("dnn", 1),
        ("pu", 1),
        ("xgboost", 1),
        ("multinomial_nb", 1),
        ("dnn", 2),
        ("multinomial_nb", 2),
        ("third_party_mean_regressor", 1),
        ("third_party_mean_regressor", 2),
        ("third_party_binary_linear", 1),
        ("third_party_binary_linear", 2),
        ("x_learner", 1),
        ("x_learner", 2),
    }
    if os.environ.get("TRIBUTO_ALGORITHM_LOCAL_ONLY") == "x_learner":
        expected = {("x_learner", 1), ("x_learner", 2)}
    assert {
        (result["algorithm"], result["worker_count"]) for result in results
    } == expected
    for result in results:
        receipt = result["receipt"]
        assert result["status"] == "succeeded"
        assert receipt["execution_profile"] == "local"
        assert receipt["runtime_owned"] is True
        assert receipt["cross_node"] is False
        assert receipt["cluster_distributed"] is False
        assert receipt["driver_materialized_training_rows"] == 0
        assert receipt["distributed"] is (result["worker_count"] >= 2)
        if result["algorithm"] == "third_party_mean_regressor":
            assert receipt["result_policy"] == "fit_only"
            assert receipt["artifact_ids"] == []
        else:
            assert receipt["result_policy"] == "bundle_required"
            assert receipt["artifact_ids"]
