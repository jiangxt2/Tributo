"""Execution tests for repository-backed documentation examples."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import tributo.vector_index as vector_index
from tributo.config import AlgorithmExecutionConfig
from tributo.training.pu_trainer import PUTrainingConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_data_and_request_example_executes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(
        str(
            REPOSITORY_ROOT
            / "docs"
            / "examples"
            / "doc_code"
            / "create_quickstart_data.py"
        ),
        run_name="__main__",
    )

    workspace = tmp_path / "tributo-quickstart"
    table = pq.read_table(workspace / "training.parquet")
    request = AlgorithmExecutionConfig.model_validate_json(
        (workspace / "execution.json").read_text(encoding="utf-8")
    )

    assert table.num_rows == 8
    assert request.algorithm == "multinomial_nb"
    assert request.worker_count == 2
    assert json.loads((workspace / "execution.json").read_text())["profile"] == (
        "local"
    )


def test_pu_execution_request_is_valid() -> None:
    request = AlgorithmExecutionConfig.model_validate_json(
        (
            REPOSITORY_ROOT / "docs" / "examples" / "doc_code" / "pu_execution.json"
        ).read_text(encoding="utf-8")
    )
    training = PUTrainingConfig.model_validate(request.algorithm_config)

    assert request.algorithm == "pu"
    assert request.worker_count == training.ray.num_workers == 2
    assert training.pu.class_prior == 0.3
    assert training.output.bundle_uri == "/shared/models/pu-fraud"


def test_local_data_example_executes(tmp_path: Path) -> None:
    source = tmp_path / "input.parquet"
    target = tmp_path / "output"
    expected = pa.table({"entity_id": [1, 2], "value": [0.5, 1.5]})
    pq.write_table(expected, source)
    script = REPOSITORY_ROOT / "docs" / "examples" / "doc_code" / "local_data.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    environment["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    completed = subprocess.run(
        [sys.executable, str(script), str(source), str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )

    assert "committed=True" in completed.stdout
    assert ds.dataset(target, format="parquet").to_table().equals(expected)


def test_vector_index_example_builds_valid_requests(monkeypatch) -> None:
    captured: dict[str, object] = {}
    build_receipt = SimpleNamespace(coverage=SimpleNamespace(status="complete"))
    search_receipt = SimpleNamespace(inline_rows=())

    def fake_build(request):
        captured["build"] = request
        return build_receipt

    def fake_search(request):
        captured["search"] = request
        return search_receipt

    monkeypatch.setattr(vector_index, "build_vector_index", fake_build)
    monkeypatch.setattr(vector_index, "search_vectors", fake_search)
    example = runpy.run_path(
        str(
            REPOSITORY_ROOT
            / "docs"
            / "examples"
            / "doc_code"
            / "vector_index_requests.py"
        ),
        run_name="tributo_docs_vector_index",
    )

    dataset_uri = "/data/vectors.lance"
    assert example["build"](dataset_uri) is build_receipt
    assert example["search"](dataset_uri, 7) is search_receipt
    assert captured["build"].dataset.uri == dataset_uri
    assert captured["build"].num_workers == 4
    assert captured["search"].dataset.version == 7
    assert captured["search"].query_vector == (0.2, 0.1, 0.4, 0.3)
