"""Conformance harness for framework-neutral InferenceExecutor adapters."""

from __future__ import annotations

from tests.inference.test_executor import _plan, _receipt
from tributo.inference.contracts import (
    InferenceExecutor,
    InferenceResult,
    ResultSinkReceipt,
)


class _FakeExecutor:
    api_version = 1
    executor_id = "fake-conformance-v1"

    def execute(self, plan, sink):
        receipt = sink.write(
            object(),
            plan.result_sink,
            run_id=plan.run_id,
            plan_digest=plan.plan_digest,
        )
        return InferenceResult(
            run_id=plan.run_id,
            attempt_id=plan.attempt_id,
            submission_id=plan.submission_id,
            parent_run_id=plan.parent_run_id,
            plan_digest=plan.plan_digest,
            bundle_id=plan.model.bundle_ref.bundle_id,
            manifest_sha256=plan.model.bundle_ref.manifest_sha256,
            role=plan.model.role,
            flavor_id=plan.model.flavor_id,
            source_ref_id=plan.input.descriptor.source_ref,
            ingestion_receipt=_receipt(),
            sink_receipt=receipt,
            output_rows=receipt.rows_written,
            status="succeeded",
        )


class _FakeSink:
    api_version = 1
    sink_id = "parquet-v1"

    def write(self, dataset, request, *, run_id, plan_digest):
        del dataset, run_id, plan_digest
        return ResultSinkReceipt(
            sink_id=self.sink_id,
            result_id="a" * 64,
            uri=request.uri,
            rows_written=2,
        )


def assert_executor_conformance(executor: InferenceExecutor) -> None:
    plan = _plan()
    assert executor.api_version == 1
    assert executor.executor_id
    first = executor.execute(plan, _FakeSink())
    second = executor.execute(plan, _FakeSink())
    assert first == second
    assert first.status == "succeeded"
    assert first.output_rows == 2


def test_fake_executor_runs_without_initializing_ray() -> None:
    assert_executor_conformance(_FakeExecutor())
