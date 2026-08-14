"""Static guardrails for the distributed Lance vector integration gate."""

from __future__ import annotations

import json
from pathlib import Path

from tools import tributo_it

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pr-test-suite.yml"
_DOCKERFILE = _ROOT / "tests" / "integrations" / "Dockerfile.data-ingestion"
_PROFILE = _ROOT / "tests" / "integrations" / "runtime-profiles.json"
_COMPOSE = _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
_SCRIPT = _ROOT / "scripts" / "run_lance_vector_index_it.sh"
_RUNNER = _ROOT / "tools" / "tributo_it.py"


def test_vector_gate_is_required_when_vector_paths_change() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert "lance_vector:" in workflow
    for path in (
        "src/tributo/vector_index/**",
        "src/tributo/_common/storage.py",
        "tests/vector_index/**",
        "tests/integrations/test_lance_vector_index.py",
        "tests/integrations/Dockerfile.data-ingestion",
        "tests/integrations/Dockerfile.data-ingestion.dockerignore",
        "scripts/run_lance_vector_index_it.sh",
        "tools/tributo_it.py",
    ):
        assert f'"{path}"' in workflow
    assert "lance-vector-distributed:" in workflow
    assert "Lance Vector Index Distributed (Docker Ray + MinIO)" in workflow
    assert "timeout-minutes: 80" in workflow
    assert "if: needs.paths-filter.outputs.lance_vector == 'true'" in workflow
    assert "LANCE_VECTOR_RESULT" in workflow
    assert 'require_success lance-vector-distributed "$LANCE_VECTOR_RESULT"' in workflow
    assert 'require_skipped lance-vector-distributed "$LANCE_VECTOR_RESULT"' in workflow


def test_vector_gate_reuses_the_owned_ray_minio_lifecycle() -> None:
    script = _SCRIPT.read_text(encoding="utf-8")
    runner = _RUNNER.read_text(encoding="utf-8")
    compose = _COMPOSE.read_text(encoding="utf-8")
    assert "tools/tributo_it.py run-lance-vector-index" in script
    assert "_run_docker_ray_suite(" in runner
    assert 'test_module="tests.integrations.test_lance_vector_index"' in runner
    assert "docker-compose.data-ingestion.yml" in runner
    assert "ray-head:" in compose
    assert "ray-worker:" in compose
    assert "minio:" in compose
    assert "build:" not in compose
    assert "container_name:" not in compose
    assert "prune" not in runner
    assert "_assert_project_absent(project)" in runner
    assert "Concurrent external Docker activity detected and ignored" in runner
    assert "_report_new_image_artifacts(project, images_before)" in runner
    assert "New dangling or untagged Docker image artifacts detected" in runner


def test_vector_runtime_is_exactly_versioned_in_the_shared_profile() -> None:
    dockerfile = _DOCKERFILE.read_text(encoding="utf-8")
    profile = json.loads(_PROFILE.read_text(encoding="utf-8"))["profiles"][
        "data-ingestion"
    ]
    assert "vector-index" in profile["extras"]
    assert profile["version_contract"] == {
        "daft_prefix": "0.7.",
        "lance_ray": "0.5.0",
        "pyarrow": "19.0.1",
        "pylance": "9.0.0",
        "ray": "2.55.1",
    }
    assert "--extra vector-index" in dockerfile
    assert "m.version('pylance') == '9.0.0'" in dockerfile
    assert "m.version('lance-ray') == '0.5.0'" in dockerfile
    assert "m.version('pyarrow') == '19.0.1'" in dockerfile


def test_vector_project_names_are_scoped_and_cli_dispatches(monkeypatch) -> None:
    tributo_it._validate_project(
        "tributo-lance-vector-ci-1",
        "tributo-lance-vector",
    )
    called: list[tributo_it.RuntimeProfile] = []
    profile = tributo_it.load_profile("data-ingestion")
    monkeypatch.setattr(tributo_it, "load_profile", lambda _name: profile)
    monkeypatch.setattr(tributo_it, "run_lance_vector_index", called.append)
    assert tributo_it.main(["run-lance-vector-index"]) == 0
    assert called == [profile]


def test_production_adapter_does_not_reimplement_ray_data_indexing() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_ROOT / "src" / "tributo" / "vector_index").glob("*.py")
    )
    assert "ray.data" not in production
    assert "create_index_uncommitted" not in production
    assert "commit_existing_index_segments" not in production
    assert "merge_existing_index_segments" not in production
