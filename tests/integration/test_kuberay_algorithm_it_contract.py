"""Static safety contract for the Ray-native KubeRay provision Gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "run_kuberay_algorithm_it.sh"
_MANIFEST = _ROOT / "tests" / "integration" / "kuberay" / "provision-gate.yaml"


def test_kuberay_gate_script_is_syntax_valid_and_scoped() -> None:
    source = _SCRIPT.read_text(encoding="utf-8")

    assert (
        subprocess.run(
            ["bash", "-n", str(_SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    assert 'CLUSTER_NAME="tributo-ray-native-gate-${RUN_ID}"' in source
    assert 'kind delete cluster --name "${CLUSTER_NAME}"' in source
    assert "docker system prune" not in source
    assert "docker network prune" not in source
    assert "docker volume prune" not in source
    assert "kubectl delete" not in source
    assert 'KUBERAY_VERSION="1.6.0"' in source
    assert "kindest/node:v1.32.11@sha256:" in source
    assert "tools/build_tributo_image.py" in source
    assert "shutdownAfterJobFinishes" not in source


def test_kuberay_gate_manifest_runs_the_portable_cluster_workload() -> None:
    documents = list(yaml.safe_load_all(_MANIFEST.read_text(encoding="utf-8")))
    by_kind = {document["kind"]: document for document in documents}
    config_map = by_kind["ConfigMap"]
    ray_job = by_kind["RayJob"]
    execution = json.loads(config_map["data"]["execution.json"])

    assert execution["profile"] == "cluster"
    assert execution["algorithm"] == "multinomial_nb"
    assert ray_job["spec"]["entrypoint"] == (
        "tributo algo run --config /opt/tributo-gate/execution.json"
    )
    assert ray_job["spec"]["shutdownAfterJobFinishes"] is True
    assert ray_job["spec"]["ttlSecondsAfterFinished"] == 60
    assert ray_job["spec"]["rayClusterSpec"]["rayVersion"] == "2.55.1"
    head = ray_job["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]
    worker = ray_job["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"][
        "spec"
    ]["containers"][0]
    assert head["image"] == "tributo-runtime-full:local"
    assert worker["image"] == "tributo-runtime-full:local"
    assert head["imagePullPolicy"] == "Never"
    assert worker["imagePullPolicy"] == "Never"
