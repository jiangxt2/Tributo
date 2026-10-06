"""Persisted v1 source shapes and identities survive API stabilization."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from pydantic import TypeAdapter

from tributo.data.base import S3Config as ReexportedS3Config
from tributo.data.contracts.storage import S3Config
from tributo.data.refs import DatasetRef, compute_ref_id, schema_fingerprint
from tributo.data.source_config import CanonicalSourceInput

_FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "data-source-contract-v1.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize("record", _FIXTURE["sources"])
def test_persisted_canonical_source_shape(record: dict[str, Any]) -> None:
    adapter: TypeAdapter[CanonicalSourceInput] = TypeAdapter(CanonicalSourceInput)
    model = adapter.validate_json(json.dumps(record["input"]))
    assert model.model_dump(mode="json") == record["expected"]
    reloaded = adapter.validate_json(model.model_dump_json())
    assert reloaded.model_dump(mode="json") == record["expected"]


def test_persisted_reference_identity_v1() -> None:
    identity = _FIXTURE["identity"]
    schema = pa.schema(
        [pa.field("x", pa.float64()), pa.field("y", pa.int64(), nullable=False)],
        metadata={"version": "1", "owner": "tributo"},
    )
    assert compute_ref_id(**identity) == _FIXTURE["ref_id"]
    assert schema_fingerprint(schema) == _FIXTURE["schema_fingerprint"]
    serialized = {
        "ref_id": _FIXTURE["ref_id"],
        "provider_id": identity["provider_id"],
        "uri": identity["canonical_uri"],
        "schema_fingerprint": _FIXTURE["schema_fingerprint"],
        "row_count": 3,
        "provenance": "snapshot-7",
    }
    assert asdict(DatasetRef(**json.loads(json.dumps(serialized)))) == serialized


def test_storage_config_compatibility_reexport() -> None:
    assert ReexportedS3Config is S3Config
