"""Static contract checks for the KubeRay IT lifecycle boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.ci_safe

_ROOT = Path(__file__).parents[2]


def test_kuberay_runner_owns_only_infrastructure_and_cleanup() -> None:
    runner = (_ROOT / "scripts" / "run_kuberay_rayjob_it.sh").read_text(
        encoding="utf-8"
    )

    assert "create cluster" in runner
    assert "delete cluster" in runner
    assert "Heavyweight manual external IT" in runner
    assert "unless the user explicitly requests the KubeRay IT" in runner
    assert "upgrade --install kuberay-operator" in runner
    assert "load docker-image" in runner
    assert "show chart" in runner
    assert "CHART_VERSION" in runner
    assert '--version "${KUBERAY_VERSION}"' in runner
    assert "kubectl apply" not in runner
    assert "docker system prune" not in runner
    assert "docker image prune" not in runner
    assert "docker volume prune" not in runner
    assert "test_kuberay_rayjob_resources.py" in runner
    build_script = (_ROOT / "scripts" / "build_kuberay_xgboost_image.sh").read_text(
        encoding="utf-8"
    )
    assert "tributo-algorithms-boosting" in build_script
    workload = (
        _ROOT / "tests" / "integration" / "test_kuberay_rayjob_resources.py"
    ).read_text(encoding="utf-8")
    assert "tributo.kuberay_xgboost_gate" in workload
    assert '(3, 2, 2 * _BYTES_PER_GIB, "2Gi")' in workload
    assert 'result["ray_resources"]' in workload


def test_kind_baseline_is_one_control_plane_and_three_workers() -> None:
    config = yaml.safe_load(
        (_ROOT / "tests" / "integrations" / "kind-kuberay-it.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config == {
        "kind": "Cluster",
        "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [
            {
                "role": "control-plane",
                "extraMounts": [
                    {
                        "hostPath": "/tmp/tributo-kuberay-shared",
                        "containerPath": "/var/local/tributo-kuberay-shared",
                    }
                ],
            },
            {
                "role": "worker",
                "extraMounts": [
                    {
                        "hostPath": "/tmp/tributo-kuberay-shared",
                        "containerPath": "/var/local/tributo-kuberay-shared",
                    }
                ],
            },
            {
                "role": "worker",
                "extraMounts": [
                    {
                        "hostPath": "/tmp/tributo-kuberay-shared",
                        "containerPath": "/var/local/tributo-kuberay-shared",
                    }
                ],
            },
            {
                "role": "worker",
                "extraMounts": [
                    {
                        "hostPath": "/tmp/tributo-kuberay-shared",
                        "containerPath": "/var/local/tributo-kuberay-shared",
                    }
                ],
            },
        ],
    }


def test_ci_manifest_registers_a_manual_kuberay_suite() -> None:
    manifest = json.loads(
        (_ROOT / "ci" / "test-suites.json").read_text(encoding="utf-8")
    )
    suite = next(item for item in manifest["suites"] if item["id"] == "kuberay-rayjob")

    assert suite["tier"] == "manual_external"
    assert suite["ci_allowed"] is False
    assert suite["entrypoint"] == ["bash", "scripts/run_kuberay_rayjob_it.sh"]
    assert "tests/integration/test_kuberay_rayjob_resources.py" in suite["test_paths"]
    assert (
        "heavyweight manual gate must run only after an explicit request"
        in suite["rationale"]
    )
    for part in range(3):
        path = f"tests/integrations/kuberay_xgboost_data_parts/part-{part}.csv"
        assert path in suite["test_paths"]
        assert path in suite["trigger_paths"]
    contract_suite = next(
        item
        for item in manifest["suites"]
        if item["id"] == "unit-integration-contracts"
    )
    assert "tests/integration/test_kuberay_it_contract.py" in contract_suite["args"]
