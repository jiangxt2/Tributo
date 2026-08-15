"""Static safety contract for the owned local and Docker Gate runners."""

from __future__ import annotations

import shlex
import subprocess
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL = _ROOT / "scripts" / "run_distributed_algorithm_local_it.sh"
_DISTRIBUTED = _ROOT / "scripts" / "run_distributed_algorithm_it.sh"
_DISTRIBUTED_COMPOSE = (
    _ROOT / "tests" / "integrations" / "docker-compose.distributed-algorithm.yml"
)
_PLUGIN_PYPROJECT = (
    _ROOT / "tests" / "fixtures" / "distributed_algorithm_plugin" / "pyproject.toml"
)


def _shell_function(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def test_local_gate_owns_only_its_named_container_and_volumes() -> None:
    runner = _LOCAL.read_text(encoding="utf-8")

    assert "TRIBUTO_ALGORITHM_LOCAL_IT_RUN_ID" in runner
    assert 'docker rm --force "${CONTAINER_NAME}"' in runner
    assert 'docker volume rm "${SOURCE_VOLUME_NAME}"' in runner
    assert 'docker volume rm "${WORK_VOLUME_NAME}"' in runner
    assert "create-source-snapshot" in runner
    assert "uv build" in runner
    assert "tests/fixtures/distributed_algorithm_plugin" in runner
    assert "tributo-plugin.whl" in runner
    assert '"${SOURCE_VOLUME_NAME}:/workspace/tributo-src:ro"' in runner
    assert '"${PLUGIN_WHEEL}:/workspace/tributo-plugin.whl:ro"' in runner
    assert "TRIBUTO_DOCKER_ALGORITHM_LOCAL_IT=1" in runner
    assert "test_distributed_algorithm_local.py" in runner
    assert "dangling=true" in runner
    assert '"<none>"' in runner
    assert "verify_required_runtime_image || cleanup_status=1" in runner
    assert "Global Docker image changes are diagnostic only" in runner
    assert "report_global_image_changes || cleanup_status=1" not in runner
    assert "report_existing_container_changes || cleanup_status=1" not in runner
    assert "docker system prune" not in runner
    assert "docker image prune" not in runner
    assert "docker volume prune" not in runner


def test_distributed_gate_uses_cached_images_and_scoped_compose_cleanup() -> None:
    runner = _DISTRIBUTED.read_text(encoding="utf-8")
    compose_override = _DISTRIBUTED_COMPOSE.read_text(encoding="utf-8")

    assert "tributo-distributed-algorithm-it-" in runner
    assert "prepare-runtime --profile data-ingestion" in runner
    assert "runtime-key --profile data-ingestion" in runner
    assert "--scale ray-worker=2" in runner
    assert "docker-compose.distributed-algorithm.yml" in runner
    assert "--no-build" in runner
    assert "--pull never" in runner
    assert "uv build" in runner
    assert "TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL" in runner
    assert "docker compose" in runner
    assert "down --volumes --remove-orphans" in runner
    assert "com.docker.compose.project=${PROJECT_NAME}" in runner
    assert "dangling=true" in runner
    assert '"<none>"' in runner
    assert "verify_required_runtime_image || cleanup_status=1" in runner
    assert "Global Docker image changes are diagnostic only" in runner
    assert "report_global_image_changes || cleanup_status=1" not in runner
    assert "report_existing_container_changes || cleanup_status=1" not in runner
    assert "test_formal_distributed_algorithms_complete_on_ray_cluster" in runner
    assert '--temp-dir="/workspace/tributo-work/tmp/ray-$${HOSTNAME}"' in (
        compose_override
    )
    assert "docker system prune" not in runner
    assert "docker image prune" not in runner
    assert "docker volume prune" not in runner
    assert "rm -rf" not in runner


def test_distributed_fixture_is_a_code_only_wheel() -> None:
    pyproject = tomllib.loads(_PLUGIN_PYPROJECT.read_text(encoding="utf-8"))
    fixture_source = (
        _PLUGIN_PYPROJECT.parent
        / "src"
        / "tributo_test_distributed_algorithm"
        / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "dependencies" not in pyproject["project"]
    assert "tributo.algorithms" in pyproject["project"]["entry-points"]
    assert "AlgorithmBuilder.from_distributed_algorithm" in fixture_source
    assert "MapReduceAlgorithm" in fixture_source
    assert "ResultPolicy.FIT_ONLY" in fixture_source
    assert "tributo.algorithms.builtin" not in fixture_source
    assert "exporter=" not in fixture_source


def test_concurrent_daemon_images_are_diagnostic_only(tmp_path: Path) -> None:
    for index, runner_path in enumerate((_LOCAL, _DISTRIBUTED)):
        runner = runner_path.read_text(encoding="utf-8")
        baseline = tmp_path / f"baseline-{index}.txt"
        final = tmp_path / f"final-{index}.txt"
        baseline.write_text("", encoding="utf-8")
        command = "\n".join(
            (
                "set -Eeuo pipefail",
                f"BASELINE_IMAGES={shlex.quote(str(baseline))}",
                f"FINAL_IMAGES={shlex.quote(str(final))}",
                "snapshot_images() { printf '%s\\n' 'sha256:external' >\"$1\"; }",
                "docker() { printf '%s\\n' 'concurrent-project'; }",
                _shell_function(runner, "report_global_image_changes"),
                "report_global_image_changes",
            )
        )

        completed = subprocess.run(
            ["bash", "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert "compose_project=concurrent-project" in completed.stderr
        assert "diagnostic only" in completed.stderr
