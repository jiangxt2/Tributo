"""Ray Jobs transport and CLI tests for vector operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from tributo.cli import main as cli_main
from tributo.vector_index.contracts import (
    CoverageStatus,
    FragmentSetEvidence,
    IndexCoverageEvidence,
    LanceDatasetRef,
    RuntimeVersionEvidence,
    VectorCompactRequest,
    VectorIndexBuildReceipt,
    VectorIndexBuildRequest,
    VectorOptimizeRequest,
    VectorSearchRequest,
)
from tributo.vector_index.errors import VectorIndexConfigurationError
from tributo.vector_index.job import (
    FAILURE_MARKER,
    REQUEST_ENV,
    RESULT_MARKER,
    VectorBuildJobRequest,
    VectorBuildJobResult,
    VectorCompactJobRequest,
    VectorOptimizeJobRequest,
    VectorSearchJobRequest,
    _failure_payload,
    decode_job_request,
    encode_job_request,
    parse_job_result,
    submit_vector_job,
)


def _ref(tmp_path) -> LanceDatasetRef:
    return LanceDatasetRef(uri=str(tmp_path / "vectors.lance"))


def _build_job(tmp_path) -> VectorBuildJobRequest:
    return VectorBuildJobRequest(
        request=VectorIndexBuildRequest(
            dataset=_ref(tmp_path),
            column="vector",
            index_name="vector_idx",
            index_type="IVF_FLAT",
            num_partitions=2,
            sample_rate=2,
        )
    )


def _runtime() -> RuntimeVersionEvidence:
    return RuntimeVersionEvidence(
        ray="2.55.1",
        pylance="9.0.0",
        lance_ray="0.5.0",
        pyarrow="19.0.1",
        worker_count=0,
        worker_validation_complete=True,
    )


def _build_result(tmp_path) -> VectorBuildJobResult:
    fragments = FragmentSetEvidence.from_ids({1, 2})
    empty = FragmentSetEvidence.from_ids(set())
    return VectorBuildJobResult(
        receipt=VectorIndexBuildReceipt(
            request_digest=_build_job(tmp_path).request.request_digest,
            dataset_ref=_ref(tmp_path).identity_digest,
            planning_base_version=1,
            output_dataset_version=2,
            index_name="vector_idx",
            index_type="IVF_FLAT",
            metric="l2",
            coverage=IndexCoverageEvidence(
                status=CoverageStatus.COMPLETE,
                planning=fragments,
                current=fragments,
                indexed=fragments,
                unindexed=empty,
                stale=empty,
                segment_count=2,
            ),
            num_workers=2,
            worker_resources={},
            runtime=_runtime(),
            elapsed_seconds=1.25,
        )
    )


def test_job_request_round_trip_is_bounded(tmp_path) -> None:
    request = _build_job(tmp_path)
    encoded = encode_job_request(request)
    decoded = decode_job_request(encoded)
    assert decoded == request
    with pytest.raises(VectorIndexConfigurationError, match="base64"):
        decode_job_request("not base64")


def test_job_request_rejects_oversized_query_payload(tmp_path) -> None:
    request = VectorSearchJobRequest(
        request=VectorSearchRequest(
            dataset=_ref(tmp_path),
            column="vector",
            query_vector=tuple(float(item) for item in range(20_000)),
            index_name="vector_idx",
        )
    )
    with pytest.raises(VectorIndexConfigurationError, match="64 KiB"):
        encode_job_request(request)


class _FakeClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def submit(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return "vector-job-123"


def test_submission_reuses_tributo_client_and_logs_only_digest(
    tmp_path, monkeypatch
) -> None:
    from tributo.vector_index import job as module

    monkeypatch.setattr(
        module,
        "build_runtime_env",
        lambda **kwargs: {"env_vars": kwargs["env_vars"]},
    )
    request = _build_job(tmp_path)
    client = _FakeClient()
    job_id = submit_vector_job(
        address="http://ray.test:8265",
        job_request=request,
        project_root=Path(tmp_path),
        client=client,
    )
    assert job_id == "vector-job-123"
    assert client.kwargs is not None
    assert client.kwargs["entrypoint"] == "python -m tributo.vector_index.job"
    assert REQUEST_ENV in client.kwargs["runtime_env"]["env_vars"]
    metadata_text = json.dumps(client.kwargs["metadata"])
    assert str(request.request.dataset.uri) not in metadata_text
    assert request.request.request_digest in metadata_text


def test_result_marker_round_trip(tmp_path) -> None:
    result = _build_result(tmp_path)
    logs = "worker output\n" + RESULT_MARKER + result.model_dump_json() + "\n"
    assert parse_job_result(logs) == result
    with pytest.raises(VectorIndexConfigurationError, match="no vector result"):
        parse_job_result("ordinary worker output")


def test_failure_payload_suppresses_arbitrary_exception_text() -> None:
    payload = _failure_payload(RuntimeError("secret_access_key=do-not-log"))
    assert payload.startswith("{")
    assert "RuntimeError" in payload
    assert "do-not-log" not in payload


def test_failure_payload_includes_safe_domain_diagnostic_and_cause_type() -> None:
    try:
        raise RuntimeError("secret_access_key=do-not-log")
    except RuntimeError as cause:
        error = VectorIndexConfigurationError("dataset version is unsupported")
        error.__cause__ = cause
    payload = json.loads(_failure_payload(error))
    assert payload["diagnostic"] == "dataset version is unsupported"
    assert payload["cause_types"] == ["RuntimeError"]
    assert "do-not-log" not in json.dumps(payload)


def test_job_main_reports_missing_request_without_traceback(
    monkeypatch, capsys
) -> None:
    from tributo.vector_index import job as module

    monkeypatch.delenv(REQUEST_ENV, raising=False)
    assert module.main([]) == 1
    output = capsys.readouterr().out
    assert output.startswith(FAILURE_MARKER)
    assert REQUEST_ENV in output


def test_job_main_succeeds_after_reporting_stale_build_receipt(
    tmp_path, monkeypatch, capsys
) -> None:
    from tributo.vector_index import job as module

    result = _build_result(tmp_path)
    receipt = result.receipt
    stale_coverage = receipt.coverage.model_copy(
        update={
            "status": CoverageStatus.STALE,
            "stale": FragmentSetEvidence.from_ids({1}),
        }
    )
    stale_result = result.model_copy(
        update={
            "receipt": receipt.model_copy(
                update={
                    "coverage": stale_coverage,
                    "warnings": ("rebuild before querying",),
                }
            )
        }
    )
    request = _build_job(tmp_path)
    monkeypatch.setenv(REQUEST_ENV, encode_job_request(request))
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=lambda **kwargs: None),
    )
    monkeypatch.setattr(module, "run_job_request", lambda job_request: stale_result)
    assert module.main([]) == 0
    output = capsys.readouterr().out
    assert output.startswith(RESULT_MARKER)
    assert '"status":"stale"' in output


def test_vector_build_cli_submits_validated_config(tmp_path, monkeypatch) -> None:
    from tributo.vector_index import cli as vector_cli

    config = tmp_path / "build.json"
    config.write_text(_build_job(tmp_path).request.model_dump_json(), encoding="utf-8")
    captured: dict[str, Any] = {}

    def submit(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "job-build"

    monkeypatch.setattr(vector_cli, "submit_vector_job", submit)
    result = CliRunner().invoke(
        cli_main,
        [
            "vector",
            "build",
            "--address",
            "http://127.0.0.1:8265",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "job-build" in result.output
    assert isinstance(captured["job_request"], VectorBuildJobRequest)


@pytest.mark.parametrize(
    ("command", "request_type", "job_type"),
    [
        (
            "optimize",
            VectorOptimizeRequest,
            VectorOptimizeJobRequest,
        ),
        (
            "compact",
            VectorCompactRequest,
            VectorCompactJobRequest,
        ),
    ],
)
def test_vector_maintenance_cli_submits_validated_config(
    command: str,
    request_type: type[VectorOptimizeRequest] | type[VectorCompactRequest],
    job_type: type[VectorOptimizeJobRequest] | type[VectorCompactJobRequest],
    tmp_path,
    monkeypatch,
) -> None:
    from tributo.vector_index import cli as vector_cli

    config = tmp_path / f"{command}.json"
    config.write_text(
        request_type(dataset=_ref(tmp_path)).model_dump_json(),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        vector_cli,
        "submit_vector_job",
        lambda **kwargs: captured.update(kwargs) or f"job-{command}",
    )
    result = CliRunner().invoke(
        cli_main,
        [
            "vector",
            command,
            "--address",
            "http://127.0.0.1:8265",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["job_request"], job_type)


def test_vector_search_cli_hides_invalid_query_input(tmp_path) -> None:
    config = tmp_path / "search.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {"uri": str(tmp_path / "vectors.lance")},
                "column": "vector",
                "query_vector": [0.0, "private-query-value"],
                "index_name": "vector_idx",
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli_main,
        [
            "vector",
            "search",
            "--address",
            "http://127.0.0.1:8265",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 1
    assert "private-query-value" not in result.output


def test_vector_result_cli_prints_validated_receipt(tmp_path, monkeypatch) -> None:
    from tributo.vector_index import cli as vector_cli

    structured = _build_result(tmp_path)

    class _LogsClient:
        def __init__(self, address: str) -> None:
            self.address = address

        def get_logs(self, job_id: str) -> str:
            return RESULT_MARKER + structured.model_dump_json()

    monkeypatch.setattr(vector_cli, "TributoClient", _LogsClient)
    result = CliRunner().invoke(
        cli_main,
        [
            "vector",
            "result",
            "--address",
            "http://127.0.0.1:8265",
            "job-build",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"operation": "build"' in result.output
    assert '"status": "complete"' in result.output
