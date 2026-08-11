"""Static guardrails for the required distributed ingestion CI gate."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.data-ingestion"
_PROFILE = _ROOT / "tests" / "integrations" / "runtime-profiles.json"
_RUNNER = _ROOT / "tools" / "tributo_it.py"
_SCRIPT = _ROOT / "scripts" / "run_data_ingestion_it.sh"
_RUNTIME_WORKFLOW = _ROOT / ".github" / "workflows" / "it-runtime-image.yml"


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
    assert "source-init:" in compose
    assert "workspace-init:" in compose
    assert "ingestion-source:/workspace/tributo-src:ro" in compose
    assert "ingestion-workspace:/workspace/tributo-work" in compose
    assert "TRIBUTO_MINIO_ENDPOINT: http://minio:9000" in compose
    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose
    assert 'user: "0:0"' in compose
    assert 'restart: "no"' in compose
    assert "pull_policy: never" in compose
    assert "build:" not in compose
    assert "container_name:" not in compose


def test_distributed_ingestion_runs_from_the_project_import_root() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    script = _SCRIPT.read_text(encoding="utf-8")

    assert "./scripts/run_data_ingestion_it.sh" in workflow
    assert "tools/tributo_it.py run-data-ingestion" in script
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
    assert "--no-install-project" in dockerfile
    assert "--locked" in dockerfile
    assert "COPY --chown=1000:100 pyproject.toml uv.lock" in dockerfile
    assert "COPY --chown=ray:users ." not in dockerfile
    assert "src/" not in dockerfile
    assert "tests/" not in dockerfile


def test_runtime_prepare_is_content_addressed_and_never_uses_compose_build() -> None:
    runner = _RUNNER.read_text(encoding="utf-8")
    profile = _PROFILE.read_text(encoding="utf-8")

    assert 'docker",\n            "buildx",\n            "build' in runner
    assert '"--load"' in runner
    assert '"--tag"' in runner
    assert "runtime_lock(identity)" in runner
    assert "fcntl.flock" in runner
    assert "TRIBUTO_IT_RUNTIME_LOCK_TIMEOUT_SECONDS" in runner
    assert "dangling=true" in runner
    assert '"<none>"' in runner
    assert '"--no-build"' in runner
    assert '"--pull",\n                "never"' in runner
    assert "runtime-gc-dry-run" in runner
    assert "prune" not in runner
    assert '"--rmi"' not in runner
    assert '"runtime_repository": "tributo-it-runtime"' in profile
    assert '"minio_image": "minio/minio:' in profile
    assert '"infrastructure_images"' not in profile


def test_integration_readme_has_only_lifecycle_owned_cluster_entries() -> None:
    readme = (_ROOT / "tests" / "integrations" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "./scripts/run_data_ingestion_it.sh" in readme
    assert "rayDocker" not in readme
    assert "docker exec ray-head" not in readme
    assert "ray-worker-[1-3]" not in readme


def test_runtime_publish_workflow_is_trusted_and_immutable() -> None:
    workflow = _RUNTIME_WORKFLOW.read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "branches: [master]" in workflow
    assert "packages: write" in workflow
    assert (
        "docker/setup-buildx-action@bb05f3f5519dd87d3ba754cc423b652a5edd6d2c"
        in workflow
    )
    assert "publish-runtime" in workflow
    assert "linux/amd64" in workflow
    assert ":latest" not in workflow
