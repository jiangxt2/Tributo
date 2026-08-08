"""Static guardrails for the required distributed ingestion CI gate."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.data-ingestion"


def test_distributed_ingestion_is_a_required_core_gate() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert "data-ingestion-distributed:" in workflow
    assert "Data Ingestion Distributed (Docker Ray + MinIO)" in workflow
    assert "test_data_ingestion_dual_engine.py" in workflow
    assert "DATA_INGESTION_DISTRIBUTED_RESULT" in workflow
    assert (
        "require_success data-ingestion-distributed "
        '"$DATA_INGESTION_DISTRIBUTED_RESULT"'
    ) in workflow
    assert (
        "require_skipped data-ingestion-distributed "
        '"$DATA_INGESTION_DISTRIBUTED_RESULT"'
    ) in workflow


def test_distributed_ingestion_uses_isolated_ray_and_minio_services() -> None:
    compose = _COMPOSE.read_text(encoding="utf-8")

    assert "ray-head:" in compose
    assert "ray-worker:" in compose
    assert "minio:" in compose
    assert "workspace-init:" in compose
    assert "ingestion-workspace:/workspace" in compose
    assert "TRIBUTO_MINIO_ENDPOINT: http://minio:9000" in compose
    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose
    assert "user: root" in compose
    assert "- chown" in compose
    assert "- ray:users" in compose
    assert "container_name:" not in compose


def test_distributed_ingestion_runs_from_the_project_import_root() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert "python -m tests.integrations.test_data_ingestion_dual_engine" in workflow
    assert (
        "python /opt/tributo/tests/integrations/test_data_ingestion_dual_engine.py"
    ) not in workflow


def test_distributed_ingestion_image_uses_locked_project_dependencies() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")

    assert "rayproject/ray:2.55.1-py312" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.23" in dockerfile
    assert "uv sync" in dockerfile
    assert "--extra data-daft" in dockerfile
    assert "--no-default-groups" in dockerfile
    assert "--locked" in dockerfile
