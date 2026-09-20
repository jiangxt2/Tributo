"""Execution tests for repository-backed documentation examples."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.dataset as ds
import pyarrow.parquet as pq

import tributo.vector_index as vector_index
from tributo.config import AlgorithmExecutionConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_quickstart_examples_execute(
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
    table = pq.read_table(workspace / "input.parquet")

    assert table.num_rows == 8
    assert table.column_names == ["message_count", "call_duration", "label"]
    assert table.column("label").to_pylist() == [0, 0, 0, 1, 1, 1, 0, 1]

    target = workspace / "output"
    script = REPOSITORY_ROOT / "docs" / "examples" / "doc_code" / "local_data.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    environment["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    environment["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    completed = subprocess.run(
        [sys.executable, str(script), str(workspace / "input.parquet"), str(target)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )

    assert "committed=True" in completed.stdout
    assert ds.dataset(target, format="parquet").to_table().equals(table)


def test_pu_execution_request_is_valid() -> None:
    request = AlgorithmExecutionConfig.model_validate_json(
        (
            REPOSITORY_ROOT / "docs" / "examples" / "doc_code" / "pu_execution.json"
        ).read_text(encoding="utf-8")
    )
    training = request.algorithm_config

    assert request.algorithm == "pu"
    assert request.worker_count == 2
    assert training["loss"]["class_prior"] == 0.3
    assert training["output"]["bundle_uri"] == "/shared/models/pu-fraud"


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
