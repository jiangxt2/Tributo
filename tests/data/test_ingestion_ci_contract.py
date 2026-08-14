"""Static guardrails for the required distributed ingestion CI gate."""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.it_versions import load_it_component_versions

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.data-ingestion"
_PROFILE = _ROOT / "tests" / "integrations" / "runtime-profiles.json"
_RUNNER = _ROOT / "tools" / "tributo_it.py"
_SCRIPT = _ROOT / "scripts" / "run_data_ingestion_it.sh"
_RUNTIME_WORKFLOW = _ROOT / ".github" / "workflows" / "it-runtime-image.yml"
_MODEL_EXPORT_RUNNER = _ROOT / "scripts" / "run_model_export_it.sh"


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
    versions = load_it_component_versions()
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))["profiles"][
        "data-ingestion"
    ]

    assert versions["RAY_IMAGE"].startswith(
        f"rayproject/ray:{versions['RAY_VERSION']}-py312@sha256:"
    )
    assert versions["UV_IMAGE"].startswith(
        f"ghcr.io/astral-sh/uv:{versions['UV_VERSION']}@sha256:"
    )
    assert profile["base_image"] == versions["RAY_IMAGE"]
    assert profile["uv_image"] == versions["UV_IMAGE"]
    assert profile["minio_image"] == versions["MINIO_IMAGE"]
    assert profile["tool_image"] == versions["TOOL_IMAGE"]
    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "ARG UV_IMAGE" in dockerfile
    assert "FROM ${UV_IMAGE} AS uv" in dockerfile
    assert "COPY --from=uv" in dockerfile
    assert "uv sync" in dockerfile
    assert "--extra embeddings" in dockerfile
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
    assert "publish-runtime" in workflow
    assert "linux/amd64" in workflow
    assert ":latest" not in workflow


def test_registry_runtime_consumers_authenticate_only_on_trusted_pushes() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    job_boundaries = (
        ("  data-ingestion-distributed:", "  inference-distributed:", 80),
        ("  core-walking-skeleton:", "  core-gate:", 65),
    )

    for start, end, timeout_minutes in job_boundaries:
        job = workflow.split(start, 1)[1].split(end, 1)[0]
        assert "packages: read" in job
        assert "Authenticate to GHCR for the published runtime" in job
        assert "if: github.event_name == 'push'" in job
        assert "secrets.GITHUB_TOKEN" in job
        assert "docker login ghcr.io" in job
        assert f"timeout-minutes: {timeout_minutes}" in job
        assert (
            "TRIBUTO_IT_RUNTIME_REGISTRY_WAIT_SECONDS: "
            "${{ github.event_name == 'push' && '2100' || '0' }}"
        ) in job


def test_model_export_ci_executes_mlflow_contract_and_waits_for_minio() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    core_job = workflow.split("  core-walking-skeleton:", 1)[1].split(
        "  core-gate:", 1
    )[0]
    trainer_bundle_filter = workflow.split("            trainer_bundle:", 1)[1].split(
        "            docs:", 1
    )[0]
    compose = _COMPOSE.read_text(encoding="utf-8")

    runner = _MODEL_EXPORT_RUNNER.read_text(encoding="utf-8")
    full_condition = 'if [[ "${SUITE}" == "full" ]]; then'
    common_block, separator, full_tail = runner.partition(full_condition)
    assert separator
    full_block, separator, _ = full_tail.partition("\nfi")
    assert separator

    assert "./scripts/run_model_export_it.sh --suite ci" in core_job
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
    assert "tests/integrations/test_e2e_mlflow.py" not in trainer_bundle_filter
    assert "tests/integration/test_walking_skeleton.py" not in trainer_bundle_filter
    assert "tests/integrations/component-versions.env" not in trainer_bundle_filter
    assert "http://minio:9000/minio/health/live" in compose


def test_postgresql_ci_service_uses_the_pinned_component_image() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    versions = load_it_component_versions()

    assert f"image: {versions['POSTGRES_IMAGE']}" in workflow


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
    assert "docker system prune" not in runner
    assert "docker container prune" not in runner
