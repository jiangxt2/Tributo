"""Static guardrails for the external distributed ingestion validation."""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.it_versions import load_it_component_versions

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_MANIFEST = _ROOT / "ci" / "test-suites.json"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.data-ingestion"
_PROFILE = _ROOT / "tests" / "integrations" / "runtime-profiles.json"
_RUNNER = _ROOT / "tools" / "tributo_it.py"
_SCRIPT = _ROOT / "scripts" / "run_data_ingestion_it.sh"
_RUNTIME_WORKFLOW = _ROOT / ".github" / "workflows" / "it-runtime-image.yml"
_MODEL_EXPORT_RUNNER = _ROOT / "scripts" / "run_model_export_it.sh"


def _suite(suite_id: str) -> dict[str, object]:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return next(suite for suite in payload["suites"] if suite["id"] == suite_id)


def test_distributed_ingestion_is_external_and_absent_from_ci() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    suite = _suite("data-ingestion-cluster")

    assert suite["tier"] == "manual_external"
    assert suite["workflow"] == "external"
    assert suite["entrypoint"] == ["./scripts/run_data_ingestion_it.sh"]
    assert (
        "tests/integrations/test_data_ingestion_dual_engine.py" in suite["test_paths"]
    )
    assert "tests/integrations/setup_hive_fixture.py" in suite["trigger_paths"]
    assert "tests/integrations/hive-setup.sql" in suite["trigger_paths"]
    assert "hive" in suite["requires"]
    assert "data-ingestion-distributed:" not in workflow
    assert "run_data_ingestion_it.sh" not in workflow
    assert "docker compose" not in workflow


def test_distributed_ingestion_uses_isolated_ray_and_minio_services() -> None:
    compose = _COMPOSE.read_text(encoding="utf-8")

    assert "ray-head:" in compose
    assert "ray-worker:" in compose
    assert "minio:" in compose
    assert "postgres:" in compose
    assert "postgres-test:" in compose
    assert "tests/integration/test_postgresql_ingestion.py" in compose
    assert "postgres-ingestion" in compose
    assert "source-init:" in compose
    assert "workspace-init:" in compose
    assert "ingestion-source:/workspace/tributo-src:ro" in compose
    assert "ingestion-workspace:/workspace/tributo-work" in compose
    assert "TRIBUTO_MINIO_ENDPOINT: http://minio:9000" in compose
    assert "TRIBUTO_POSTGRESQL_HOST: postgres" in compose
    assert 'TRIBUTO_POSTGRESQL_PORT: "5432"' in compose
    runner = _RUNNER.read_text(encoding="utf-8")
    assert "enable_postgresql=True" in runner
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

    assert "./scripts/run_data_ingestion_it.sh" not in workflow
    assert "tools/tributo_it.py run-data-ingestion" in script
    assert (
        "python /opt/tributo/tests/integrations/test_data_ingestion_dual_engine.py"
    ) not in workflow


def test_distributed_ingestion_image_uses_locked_project_dependencies() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    versions = load_it_component_versions()
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))["profiles"][
        "data-ingestion"
    ]

    assert versions["RAY_IMAGE"].startswith(
        f"rayproject/ray:{versions['RAY_VERSION']}-py312@sha256:"
    )
    assert profile["base_image"] == versions["RAY_IMAGE"]
    assert profile["dependency_mode"] == "host-uv-export"
    assert profile["uv_version"] == versions["UV_VERSION"]
    assert profile["minio_image"] == versions["MINIO_IMAGE"]
    assert profile["hive_image"] == versions["HIVE_IMAGE"]
    assert "uv_image" not in profile
    assert "tool_image" not in profile
    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "ARG UV_IMAGE" not in dockerfile
    assert "FROM ${UV_IMAGE}" not in dockerfile
    assert "COPY --from=uv" not in dockerfile
    assert "uv sync" not in dockerfile
    assert "COPY --from=locked-requirements" in dockerfile
    assert "python -m pip install --require-hashes" in dockerfile
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
    assert "_assert_project_absent(project)" in runner
    assert "ownership-scoped audit" in runner
    assert "_report_new_image_artifacts(project, images_before)" in runner
    assert "detected and ignored" in runner
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
    assert "./scripts/run_model_export_it.sh" in readme
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
    assert "astral-sh/setup-uv@d269b9917d8e33f7165632f15cc38a13726f64b5" in workflow
    assert 'version: "0.11.23"' in workflow
    assert "publish-runtime" in workflow
    assert "linux/amd64" in workflow
    assert ":latest" not in workflow


def test_pr_workflow_never_pulls_or_executes_external_runtime() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert "packages: read" not in workflow
    assert "docker login ghcr.io" not in workflow
    assert "TRIBUTO_IT_RUNTIME_REGISTRY" not in workflow
    assert "prepare-runtime" not in workflow


def test_model_export_external_suite_owns_mlflow_contract_and_minio_lifecycle() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    suite = _suite("model-export-cluster")
    compose = _COMPOSE.read_text(encoding="utf-8")

    runner = _MODEL_EXPORT_RUNNER.read_text(encoding="utf-8")
    full_condition = 'if [[ "${SUITE}" == "full" ]]; then'
    common_block, separator, full_tail = runner.partition(full_condition)
    assert separator
    full_block, separator, _ = full_tail.partition("\nfi")
    assert separator

    assert suite["tier"] == "manual_external"
    assert suite["entrypoint"] == ["./scripts/run_model_export_it.sh"]
    assert "tests/integrations/test_e2e_mlflow.py" in suite["test_paths"]
    assert "run_model_export_it.sh" not in workflow
    assert "tests/integrations/test_e2e_mlflow.py" in runner
    for full_only_path in (
        "tests/training/exporters/test_trainer_bundle_contract.py",
        "tests/integration/test_export_s3.py",
        "tests/integration/test_minio_compat.py",
    ):
        assert full_only_path not in common_block
        assert full_block.count(full_only_path) == 1
        assert runner.count(full_only_path) == 1
    assert '-m "s3_contract or minio_compat"' in full_block
    assert "http://minio:9000/minio/health/live" in compose


def test_postgresql_external_contract_keeps_an_immutable_component_pin() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    versions = load_it_component_versions()

    assert versions["POSTGRES_IMAGE"].startswith("postgres:16.14@sha256:")
    assert versions["POSTGRES_IMAGE"] not in workflow
    assert (
        "tests/integration/test_postgresql_ingestion.py"
        in _suite("data-ingestion-cluster")["test_paths"]
    )


def test_model_export_runner_is_isolated_and_cleans_its_own_project() -> None:
    runner = _MODEL_EXPORT_RUNNER.read_text(encoding="utf-8")

    assert "tributo-model-export-it-$(date +%Y%m%d%H%M%S)-$$" in runner
    assert "prepare-runtime --profile data-ingestion" in runner
    assert "--detach --no-build --pull never --wait" in runner
    assert "cache_dir=/workspace/tributo-work/cache/pytest-model-export" in runner
    assert "BASELINE_CAPTURED=0" in runner
    assert "COMPOSE_TOUCHED=0" in runner
    assert 'if [[ "${COMPOSE_TOUCHED}" -eq 1 ]]' in runner
    assert 'if [[ "${BASELINE_CAPTURED}" -eq 1 ]]' in runner
    assert "Concurrent container change:" in runner
    assert "Other Docker activity changed pre-existing containers" in runner
    assert "trap cleanup EXIT" in runner
    assert "down --volumes --remove-orphans" in runner
    assert "label=com.docker.compose.project=${PROJECT_NAME}" in runner
    assert "TRIBUTO_IT_TOOL_IMAGE" not in runner
    assert 'TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE%%@*}"' in runner
    assert "prepare-infrastructure --profile data-ingestion" in runner
    assert "create-source-snapshot" in runner
    assert "test_it_component_versions.py" in runner
    assert "--collect-only" in runner
    assert "--preflight-only" in runner
    assert "walking-skeleton" in runner
    assert 'TRIBUTO_IT_SOURCE_ROOT="${PREFLIGHT_SOURCE}"' in runner
    assert runner.index("create-source-snapshot") < runner.index("command -v docker")
    assert "-u HTTP_PROXY" in runner
    assert "-u http_proxy" in runner
    assert "test_e2e_mlflow.py::" in runner
    assert "tests/fixtures/preflight_stubs" in runner
    assert "docker system prune" not in runner
    assert "docker container prune" not in runner
