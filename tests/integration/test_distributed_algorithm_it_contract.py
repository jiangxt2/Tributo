"""Static safety contract for the owned local and Docker Gate runners."""

from __future__ import annotations

import ast
import shlex
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from tests.training.jobs.official_algorithm_matrix import (
    ALL_ENTRY_POINTS,
    CATEGORY_ENTRY_POINTS,
    DISTRIBUTION_ENTRY_POINTS,
    ENTRY_POINT_DISTRIBUTIONS,
    OFFICIAL_ALGORITHM_IDENTITIES,
    entry_point_for,
    entry_points_for_gate,
    parse_entry_point_selection,
)
from tests.training.jobs.official_algorithm_output_contract import (
    build_output_expectation,
    validate_output_dtype,
    validate_output_value,
)
from tools.algorithm_gate_provenance import (
    SourceRevision,
    build_preflight_provenance,
    build_provenance,
    load_preflight_provenance,
    write_provenance,
)

_ROOT = Path(__file__).resolve().parents[2]
_LOCAL = _ROOT / "scripts" / "run_distributed_algorithm_local_it.sh"
_DISTRIBUTED = _ROOT / "scripts" / "run_distributed_algorithm_it.sh"
_DISTRIBUTED_COMPOSE = (
    _ROOT / "tests" / "integrations" / "docker-compose.distributed-algorithm.yml"
)
_PLUGIN_PYPROJECT = (
    _ROOT / "tests" / "fixtures" / "distributed_algorithm_plugin" / "pyproject.toml"
)
_TORCH_RECIPE_PLUGIN_PYPROJECT = (
    _ROOT / "tests" / "fixtures" / "torch_recipe_algorithm_plugin" / "pyproject.toml"
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
    assert "tests/fixtures/torch_recipe_algorithm_plugin" in runner
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
    assert "TRIBUTO_ALGORITHMS_ROOT is required" in runner
    assert "OFFICIAL_ALGORITHMS_COMMIT" in runner
    assert "prepare-runtime --profile data-ingestion" in runner
    assert "runtime-key --profile data-ingestion" in runner
    assert "--scale ray-worker=2" in runner
    assert "docker-compose.distributed-algorithm.yml" in runner
    assert "--no-build" in runner
    assert "--pull never" in runner
    assert "uv build" in runner
    assert "TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL" in runner
    assert "tests/fixtures/offline_algorithm_dependency" in runner
    assert "tests/fixtures/torch_recipe_algorithm_plugin" in runner
    assert "tests/fixtures/offline_algorithm_plugin" in runner
    assert "tools/build_algorithm_bundle.py" in runner
    assert "offline-bundle.zip" in runner
    assert "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_URI" in runner
    assert "AWS_ENDPOINT_URL" in (
        (
            _ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
        ).read_text(encoding="utf-8")
    )
    assert "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE" in runner
    assert (
        "test_offline_wheelhouse_installs_unique_dependency_on_driver_and_workers"
        in runner
    )
    assert "docker compose" in runner
    assert "prepare-infrastructure" in runner
    assert "TRIBUTO_IT_TOOL_IMAGE" not in runner
    assert 'TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE%%@*}"' in runner
    assert "down --volumes --remove-orphans" in runner
    assert "com.docker.compose.project=${PROJECT_NAME}" in runner
    assert "dangling=true" in runner
    assert '"<none>"' in runner
    assert "verify_required_runtime_image || cleanup_status=1" in runner
    assert "Global Docker image changes are diagnostic only" in runner
    assert "report_global_image_changes || cleanup_status=1" not in runner
    assert "report_existing_container_changes || cleanup_status=1" not in runner
    assert "test_formal_distributed_algorithms_complete_on_ray_cluster" not in runner
    assert "test_official_algorithm_wheels_complete_on_ray_cluster" in runner
    assert "test_out_of_tree_torch_recipe_completes_on_ray_cluster" in runner
    assert "TRIBUTO_DISTRIBUTED_ALGORITHM_RERUN_FAILED_ONLY" in runner
    assert "TRIBUTO_OFFICIAL_CATBOOST_WHEEL" in runner
    assert "TRIBUTO_OFFICIAL_GATE_CATEGORIES" in runner
    assert "TRIBUTO_OFFICIAL_GATE_ENTRY_POINTS" in runner
    assert "TRIBUTO_DISTRIBUTED_ALGORITHM_EVIDENCE_MODE" in runner
    assert "Certification requires clean Core and algorithm worktrees" in runner
    assert "tools/algorithm_gate_provenance.py" in runner
    assert "Preflight provenance:" in runner
    assert "Final provenance:" in runner
    assert 'if [[ "${test_status}" -eq 0 ]]; then' in runner
    assert runner.index("  preflight \\\n") < runner.index("docker info")
    assert "  final \\\n  --preflight" in runner
    assert (
        'uv run --locked python "${PROJECT_ROOT}/tools/build_algorithm_bundle.py"'
        in runner
    )
    assert (
        'uv run --locked --no-sync python "${PROJECT_ROOT}/tools/build_algorithm_bundle.py"'
        not in runner
    )
    assert '"${DISTRIBUTED_ALGORITHM_SCOPE}" == "official"' in runner
    assert "--timeout=14400" in runner
    assert "CASE_RESULT: " in (
        _ROOT / "tests" / "training" / "jobs" / "priority_algorithm_gate_job.py"
    ).read_text(encoding="utf-8")
    assert (
        "test_remote_offline_wheelhouse_archive_installs_on_driver_and_workers"
        in runner
    )
    assert '--temp-dir="/workspace/tributo-work/tmp/ray-$${HOSTNAME}"' in (
        compose_override
    )
    assert "docker system prune" not in runner
    assert "docker image prune" not in runner
    assert "docker volume prune" not in runner
    assert "rm -rf" not in runner


def test_algorithm_gate_provenance_binds_clean_sources_wheels_and_runtime(
    tmp_path: Path,
) -> None:
    wheel_paths = tuple(
        tmp_path / f"tributo_gate_package_{index}-1.0.0-py3-none-any.whl"
        for index in range(16)
    )
    for index, wheel in enumerate(wheel_paths):
        wheel.write_bytes(f"package-{index}".encode("ascii"))
    core = SourceRevision(root="/core", commit="a" * 40, dirty=False)
    algorithms = SourceRevision(root="/algorithms", commit="b" * 40, dirty=False)
    preflight = build_preflight_provenance(
        project_name="tributo-distributed-algorithm-it-test",
        scope="official",
        evidence_mode="certification",
        core=core,
        algorithms=algorithms,
    )
    preflight_output = tmp_path / "preflight-provenance.json"
    write_provenance(preflight_output, preflight)
    loaded_preflight, preflight_sha256 = load_preflight_provenance(preflight_output)
    provenance = build_provenance(
        project_name="tributo-distributed-algorithm-it-test",
        scope="official",
        evidence_mode="certification",
        core=core,
        algorithms=algorithms,
        wheel_paths=wheel_paths,
        runtime_image="tributo-it-runtime:test",
        runtime_image_id="sha256:" + "c" * 64,
        runtime_python_version="3.12.12",
        runtime_ray_version="2.55.1",
        preflight=loaded_preflight,
        preflight_sha256=preflight_sha256,
    )

    assert provenance["certifying"] is True
    assert provenance["document_type"] == "final"
    assert provenance["preflight"]["document_sha256"] == preflight_sha256
    assert len(provenance["configuration_sha256"]) == 64
    assert [record["filename"] for record in provenance["wheels"]] == sorted(
        wheel.name for wheel in wheel_paths
    )
    assert all(len(record["sha256"]) == 64 for record in provenance["wheels"])
    another_preflight = build_preflight_provenance(
        project_name="tributo-distributed-algorithm-it-another-run",
        scope="official",
        evidence_mode="certification",
        core=SourceRevision(root="/different/core", commit="a" * 40, dirty=False),
        algorithms=SourceRevision(
            root="/different/algorithms", commit="b" * 40, dirty=False
        ),
    )
    another_preflight_output = tmp_path / "another-preflight-provenance.json"
    write_provenance(another_preflight_output, another_preflight)
    loaded_another_preflight, another_preflight_sha256 = load_preflight_provenance(
        another_preflight_output
    )
    same_configuration = build_provenance(
        project_name="tributo-distributed-algorithm-it-another-run",
        scope="official",
        evidence_mode="certification",
        core=SourceRevision(root="/different/core", commit="a" * 40, dirty=False),
        algorithms=SourceRevision(
            root="/different/algorithms", commit="b" * 40, dirty=False
        ),
        wheel_paths=wheel_paths,
        runtime_image="tributo-it-runtime:test",
        runtime_image_id="sha256:" + "c" * 64,
        runtime_python_version="3.12.12",
        runtime_ray_version="2.55.1",
        preflight=loaded_another_preflight,
        preflight_sha256=another_preflight_sha256,
    )
    assert (
        same_configuration["configuration_sha256"] == provenance["configuration_sha256"]
    )
    output = tmp_path / "provenance.json"
    write_provenance(output, provenance)
    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_provenance(output, provenance)


def test_algorithm_gate_provenance_rejects_dirty_or_filtered_certification(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "tributo-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"core")
    clean = SourceRevision(root="/core", commit="a" * 40, dirty=False)
    dirty = SourceRevision(root="/algorithms", commit="b" * 40, dirty=True)

    def build_preflight(
        algorithms: SourceRevision,
        *,
        categories: str = "",
    ) -> dict[str, object]:
        return build_preflight_provenance(
            project_name="tributo-distributed-algorithm-it-test",
            scope="official",
            evidence_mode="certification",
            core=clean,
            algorithms=algorithms,
            categories=categories,
        )

    with pytest.raises(ValueError, match="clean Core and algorithm"):
        build_preflight(dirty)
    with pytest.raises(ValueError, match="unfiltered official"):
        build_preflight(clean, categories="graph")
    preflight = build_preflight(clean)
    with pytest.raises(ValueError, match="Core plus 15 algorithm Wheels"):
        build_provenance(
            project_name="tributo-distributed-algorithm-it-test",
            scope="official",
            evidence_mode="certification",
            core=clean,
            algorithms=clean,
            wheel_paths=(wheel,),
            runtime_image="tributo-it-runtime:test",
            runtime_image_id="sha256:" + "c" * 64,
            runtime_python_version="3.12.12",
            runtime_ray_version="2.55.1",
            preflight=preflight,
            preflight_sha256="d" * 64,
        )


def test_algorithm_gate_final_provenance_rejects_preflight_source_drift() -> None:
    core = SourceRevision(root="/core", commit="a" * 40, dirty=False)
    algorithms = SourceRevision(root="/algorithms", commit="b" * 40, dirty=True)
    preflight = build_preflight_provenance(
        project_name="tributo-distributed-algorithm-it-test",
        scope="official",
        evidence_mode="diagnostic",
        core=core,
        algorithms=algorithms,
    )

    with pytest.raises(ValueError, match="does not match final source configuration"):
        build_provenance(
            project_name="tributo-distributed-algorithm-it-test",
            scope="official",
            evidence_mode="diagnostic",
            core=core,
            algorithms=SourceRevision(root="/algorithms", commit="c" * 40, dirty=True),
            wheel_paths=(),
            runtime_image="tributo-it-runtime:test",
            runtime_image_id="sha256:" + "d" * 64,
            runtime_python_version="3.12.12",
            runtime_ray_version="2.55.1",
            preflight=preflight,
            preflight_sha256="e" * 64,
        )


def test_official_gate_matrix_covers_current_37_entry_points_once() -> None:
    assert set(CATEGORY_ENTRY_POINTS) == {
        "boosting",
        "causal",
        "classical",
        "graph",
        "recsys_multistage_nlp",
        "torch",
    }
    assert sum(len(values) for values in CATEGORY_ENTRY_POINTS.values()) == 37
    assert len(ALL_ENTRY_POINTS) == 37
    assert len(ALL_ENTRY_POINTS) == sum(
        len(set(values)) for values in CATEGORY_ENTRY_POINTS.values()
    )
    assert type(CATEGORY_ENTRY_POINTS).__name__ == "mappingproxy"
    assert type(DISTRIBUTION_ENTRY_POINTS).__name__ == "mappingproxy"
    assert type(ENTRY_POINT_DISTRIBUTIONS).__name__ == "mappingproxy"
    assert type(OFFICIAL_ALGORITHM_IDENTITIES).__name__ == "mappingproxy"
    assert len(DISTRIBUTION_ENTRY_POINTS) == 15
    assert len(OFFICIAL_ALGORITHM_IDENTITIES) == 37
    assert set(OFFICIAL_ALGORITHM_IDENTITIES) == ALL_ENTRY_POINTS
    assert all(
        identity.distribution and identity.algorithm_id and identity.implementation_id
        for identity in OFFICIAL_ALGORITHM_IDENTITIES.values()
    )
    assert set(ENTRY_POINT_DISTRIBUTIONS) == ALL_ENTRY_POINTS
    assert {
        (entry_point, distribution)
        for distribution, entry_points in DISTRIBUTION_ENTRY_POINTS.items()
        for entry_point in entry_points
    } == set(ENTRY_POINT_DISTRIBUTIONS.items())

    gate = (
        _ROOT / "tests" / "training" / "jobs" / "official_algorithm_gate_job.py"
    ).read_text(encoding="utf-8")
    test = (_ROOT / "tests" / "training" / "test_dnn_pu_training.py").read_text(
        encoding="utf-8"
    )
    assert "_validate_installed_official_entry_points" in gate
    assert "validate_installed_algorithm_package" in gate
    assert "INSTALLATION_RESULT: " in gate
    assert "report.algorithm_id != expected_identity.algorithm_id" in gate
    assert "report.implementation_id != expected_identity.implementation_id" in gate
    assert "_execute_bundle_inference" in gate
    assert "run_inference(" in gate
    assert "ray.data.read_parquet" in gate
    assert "_MemoryResultSink" not in gate
    assert "sink=sink" not in gate
    assert "RayExecutionPolicy(" in gate
    assert "TRIBUTO_OFFICIAL_EXTENDED_CONTROLS" in gate
    assert '"inference_history_width": 8' in gate
    assert "for category in categories" in test
    assert "expected_selected = selected_entry_points or frozenset(" in test

    tree = ast.parse(gate)
    exercised: set[str] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name):
            continue
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        if call.func.id == "_execute":
            algorithm_node = keywords.get("algorithm")
            implementation_node = keywords.get("implementation_id")
            if not isinstance(algorithm_node, ast.Constant) or not isinstance(
                algorithm_node.value, str
            ):
                continue
            implementation = (
                implementation_node.value
                if isinstance(implementation_node, ast.Constant)
                and isinstance(implementation_node.value, str)
                else None
            )
            exercised.add(entry_point_for(algorithm_node.value, implementation))
        elif call.func.id == "_execute_graph":
            algorithm_node = keywords.get("algorithm")
            algorithm = (
                algorithm_node.value
                if isinstance(algorithm_node, ast.Constant)
                and isinstance(algorithm_node.value, str)
                else "graphsage_node_classifier"
            )
            exercised.add(entry_point_for(algorithm, None))
        elif call.func.id == "_execute_gcm":
            exercised.add(entry_point_for("gcm_root_cause", None))
    assert exercised == ALL_ENTRY_POINTS


def test_official_gate_validates_materialized_output_dtype_and_shape() -> None:
    scalar = build_output_expectation(
        tensor_name="label",
        column="result__label",
        dtype="int64",
        shape=("batch",),
    )
    vector = build_output_expectation(
        tensor_name="probabilities",
        column="result__probabilities",
        dtype="float32",
        shape=("batch", 2),
    )
    dynamic = build_output_expectation(
        tensor_name="embedding",
        column="result__embedding",
        dtype="float32",
        shape=("batch", "embedding"),
    )

    validate_output_value(scalar, np.int64(1))
    validate_output_value(vector, np.asarray([0.25, 0.75], dtype=np.float32))
    validate_output_value(dynamic, np.asarray([0.5, 0.25, 0.125], dtype=np.float32))
    validate_output_dtype(scalar, pa.int64())
    validate_output_dtype(vector, pa.fixed_shape_tensor(pa.float32(), (2,)))


@pytest.mark.parametrize(
    ("shape", "dtype", "value", "message"),
    [
        (("batch",), "float32", np.asarray([1.0], dtype=np.float32), "row shape"),
        (("batch", 1), "float32", np.float32(1.0), "row shape"),
        (
            ("batch", 2),
            "float32",
            np.asarray([1.0], dtype=np.float32),
            "requires 2",
        ),
        (("batch",), "float32", np.float32(np.nan), "non-finite"),
    ],
)
def test_official_gate_rejects_materialized_output_contract_drift(
    shape: tuple[int | str, ...],
    dtype: str,
    value: object,
    message: str,
) -> None:
    expectation = build_output_expectation(
        tensor_name="prediction",
        column="result__prediction",
        dtype=dtype,
        shape=shape,
    )

    with pytest.raises(AssertionError, match=message):
        validate_output_value(expectation, value)


def test_official_gate_rejects_persisted_output_dtype_drift() -> None:
    expectation = build_output_expectation(
        tensor_name="prediction",
        column="result__prediction",
        dtype="float32",
        shape=("batch",),
    )

    with pytest.raises(AssertionError, match="persisted dtype"):
        validate_output_dtype(expectation, pa.float64())


def test_official_gate_requires_dynamic_batch_output_axis() -> None:
    with pytest.raises(AssertionError, match="dynamic batch first"):
        build_output_expectation(
            tensor_name="prediction",
            column="result__prediction",
            dtype="float32",
            shape=(1,),
        )


def test_official_gate_entry_point_selection_is_exact_and_fail_closed() -> None:
    assert parse_entry_point_selection("") is None
    assert parse_entry_point_selection(
        "teacher_student_distillation,graphsage_node_classifier"
    ) == {
        "teacher_student_distillation",
        "graphsage_node_classifier",
    }
    with pytest.raises(ValueError, match="unique names"):
        parse_entry_point_selection(
            "graphsage_node_classifier,graphsage_node_classifier"
        )
    with pytest.raises(ValueError, match="unknown"):
        parse_entry_point_selection("not-an-official-algorithm")
    assert entry_points_for_gate(
        "recsys_multistage_nlp",
        ("teacher_student_distillation,graphsage_node_classifier,rgcn_node_classifier"),
    ) == {"teacher_student_distillation"}
    assert entry_points_for_gate(
        "graph",
        ("teacher_student_distillation,graphsage_node_classifier,rgcn_node_classifier"),
    ) == {"graphsage_node_classifier", "rgcn_node_classifier"}
    assert (
        entry_point_for(
            "random_forest", "tributo.official.random_forest.native_ensemble"
        )
        == "random_forest.native"
    )
    with pytest.raises(KeyError, match="no Entry Point identity"):
        entry_point_for("random_forest", None)
    with pytest.raises(KeyError, match="identity mismatch"):
        entry_point_for("dnn", "tributo.official.graph_pyg.graphsage")


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
    assert set(pyproject["project"]["entry-points"]["tributo.algorithms"]) == {
        "third_party_mean_regressor"
    }
    assert "AlgorithmBuilder.from_distributed_algorithm" in fixture_source
    assert "MapReduceAlgorithm" in fixture_source
    assert "ResultPolicy.FIT_ONLY" in fixture_source
    assert "tributo.algorithms.builtin" not in fixture_source
    assert "exporter=" not in fixture_source


def test_torch_recipe_fixture_is_a_code_only_low_code_wheel() -> None:
    pyproject = tomllib.loads(
        _TORCH_RECIPE_PLUGIN_PYPROJECT.read_text(encoding="utf-8")
    )
    fixture_source = (
        _TORCH_RECIPE_PLUGIN_PYPROJECT.parent
        / "src"
        / "tributo_test_torch_recipe_algorithm"
        / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "dependencies" not in pyproject["project"]
    assert "tributo.algorithms" in pyproject["project"]["entry-points"]
    assert set(pyproject["project"]["entry-points"]["tributo.algorithms"]) == {
        "third_party_binary_linear"
    }
    assert "AlgorithmBuilder.from_torch" in fixture_source
    assert "TorchRecipe" in fixture_source
    assert "train_loop_per_worker" not in fixture_source
    assert "tributo.algorithms.builtin" not in fixture_source


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
