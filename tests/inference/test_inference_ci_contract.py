"""Static guardrails for the external isolated inference validation."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_MANIFEST = _ROOT / "ci" / "test-suites.json"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.inference.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.inference"
_VERSIONS = _ROOT / "tests" / "integrations" / "inference-it-versions.conf"
_RUNNER = _ROOT / "scripts" / "run_inference_it.sh"


def _inference_suite() -> dict[str, object]:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return next(
        suite for suite in payload["suites"] if suite["id"] == "inference-cluster"
    )


def test_inference_it_versions_are_explicit_and_immutable() -> None:
    versions = _VERSIONS.read_text(encoding="utf-8")

    assert "rayproject/ray:2.55.1-py312@sha256:" in versions
    assert "ghcr.io/astral-sh/uv:0.11.23@sha256:" in versions
    assert "minio/minio:RELEASE.2025-09-07T16-13-09Z@sha256:" in versions
    for expected in (
        "TRIBUTO_EXPECTED_RAY_VERSION=2.55.1",
        "TRIBUTO_EXPECTED_MLFLOW_VERSION=2.22.5",
        "TRIBUTO_EXPECTED_ONNX_VERSION=1.22.0",
        "TRIBUTO_EXPECTED_ONNXRUNTIME_VERSION=1.28.0",
        "TRIBUTO_EXPECTED_PYARROW_VERSION=19.0.1",
        "TRIBUTO_EXPECTED_XGBOOST_VERSION=3.3.0",
    ):
        assert expected in versions


def test_inference_image_uses_one_locked_project_environment() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG BASE_IMAGE" in dockerfile
    assert "ARG UV_IMAGE" in dockerfile
    assert "uv sync" in dockerfile
    assert "--extra dev" in dockerfile
    assert "--extra test-integration" in dockerfile
    assert "--no-default-groups" in dockerfile
    assert "--locked" in dockerfile


def test_inference_compose_is_project_scoped_without_host_ports() -> None:
    compose = _COMPOSE.read_text(encoding="utf-8")

    for service in ("ray-head:", "ray-worker:", "mlflow:", "minio:"):
        assert service in compose
    assert "--num-cpus=0" in compose
    assert "--num-cpus=4" in compose
    assert "TRIBUTO_MLFLOW_TRACKING_URI: http://mlflow:5000" in compose
    assert "TRIBUTO_MINIO_ENDPOINT: http://minio:9000" in compose
    assert "inference-workspace:/workspace" in compose
    assert "container_name:" not in compose
    assert "ports:" not in compose


def test_inference_runner_always_performs_exact_scoped_cleanup() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")

    assert "trap cleanup EXIT" in runner
    assert '--project-name "$COMPOSE_PROJECT_NAME"' in runner
    assert "down --volumes --remove-orphans" in runner
    assert "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" in runner
    assert "Compose project already owns Docker resources" in runner
    assert "ray.cluster_resources().get('CPU', 0) >= 4" in runner
    assert "verify_existing_containers" in runner
    assert "docker prune" not in runner
    assert "docker system prune" not in runner
    assert "docker rm" not in runner


def test_inference_cluster_is_external_and_absent_from_ci() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    suite = _inference_suite()

    assert suite["tier"] == "manual_external"
    assert suite["workflow"] == "external"
    assert suite["entrypoint"] == ["./scripts/run_inference_it.sh"]
    assert "tests/integration/test_inference_ray_jobs.py" in suite["test_paths"]
    assert "inference-distributed:" not in workflow
    assert "run_inference_it.sh" not in workflow


def test_inference_impact_rule_includes_all_direct_domain_dependencies() -> None:
    trigger_paths = _inference_suite()["trigger_paths"]

    for path in (
        "src/tributo/inference/**",
        "src/tributo/data/**",
        "src/tributo/_common/**",
        "src/tributo/exporting/**",
        "src/tributo/integrations/model_importers/**",
        "src/tributo/integrations/sinks/**",
    ):
        assert path in trigger_paths
