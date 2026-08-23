"""Data-persistence adapter for explainability result materialization."""

from __future__ import annotations

import logging
from typing import ClassVar

from tributo.data.persistence import default_object_store, inspect_parquet_output
from tributo.explainability.contracts import ExplainabilityReceipt
from tributo.explainability.protocols import ExplainabilityMaterialization
from tributo.inference.contracts import ParquetResultSinkRequest
from tributo.integrations.sinks.parquet import ParquetResultSink
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="alpha")
class ParquetExplainabilityResultStore:
    """Delegate result writes to ResultSink and storage operations to data APIs."""

    provider_id: ClassVar[str] = "tributo.parquet-explainability-results-v1"

    def materialize(
        self,
        dataset: object,
        *,
        uri: str,
        storage_profile: str | None,
        max_bytes: int | None,
        run_id: str,
        plan_digest: str,
    ) -> ExplainabilityMaterialization:
        ParquetResultSink().write(
            dataset,
            ParquetResultSinkRequest(
                uri=uri,
                storage_profile=storage_profile,
                max_bytes=max_bytes,
            ),
            run_id=run_id,
            plan_digest=plan_digest,
        )
        inspection = inspect_parquet_output(uri, storage_profile=storage_profile)
        return ExplainabilityMaterialization(
            digest=inspection.digest,
            total_bytes=inspection.total_bytes,
            rows=inspection.rows,
        )

    def write_receipt(
        self,
        uri: str,
        receipt: ExplainabilityReceipt,
        *,
        storage_profile: str | None,
    ) -> None:
        default_object_store().write_bytes(
            _receipt_uri(uri),
            receipt.model_dump_json().encode(),
            storage_profile=storage_profile,
            content_type="application/json",
        )

    def read_receipt(
        self,
        uri: str,
        *,
        storage_profile: str | None,
    ) -> ExplainabilityReceipt | None:
        receipt_uri = (
            uri if uri.rstrip("/").endswith("/receipt.json") else _receipt_uri(uri)
        )
        try:
            payload = default_object_store().read_bytes(
                receipt_uri,
                storage_profile=storage_profile,
            )
            return ExplainabilityReceipt.model_validate_json(payload)
        except (FileNotFoundError, OSError, ValueError):
            return None

    def cleanup(self, uri: str, *, storage_profile: str | None) -> None:
        try:
            default_object_store().delete_tree(uri, storage_profile=storage_profile)
        except (FileNotFoundError, OSError, ValueError):
            logger.warning(
                "Failed to clean explainability result %s", uri, exc_info=True
            )


def _receipt_uri(result_uri: str) -> str:
    return result_uri.rstrip("/") + "/receipt.json"


__all__ = ["ParquetExplainabilityResultStore"]
