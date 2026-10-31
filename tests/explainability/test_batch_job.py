"""The Job boundary retains the actual explainability receipt status."""

import argparse
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tributo.explainability.batch_job import main


@pytest.mark.parametrize("status", ["succeeded", "partial"])
def test_driver_retains_cached_receipt_status_and_identity(tmp_path, caplog, status):
    path = tmp_path / "request.json"
    path.write_text("{}")
    request = SimpleNamespace(
        input=SimpleNamespace(engine="tributo.ray_data"),
        operation_store_uri="file:///store",
    )
    receipt = SimpleNamespace(
        status=status,
        operation_id="operation-1",
        request_id="request-1",
        bundle_id="bundle-1",
    )
    with (
        patch("tributo.explainability.batch_job.bootstrap_logging"),
        patch("tributo.explainability.batch_job.ray.init"),
        patch(
            "tributo.explainability.batch_job.ExplainabilityRequest.model_validate",
            return_value=request,
        ),
        patch(
            "tributo.explainability.executor.run_batch_explainability",
            return_value=receipt,
        ) as execute,
        caplog.at_level(logging.INFO),
    ):
        assert main(argparse.Namespace(config=str(path))) == 0
    execute.assert_called_once_with(request)
    terminal = next(
        r
        for r in caplog.records
        if getattr(r, "phase", None) == "explainability_driver"
    )
    assert terminal.status == status
    assert (
        terminal.operation_id == receipt.operation_id
        and terminal.request_id == receipt.request_id
    )
