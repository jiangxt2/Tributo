"""Credential boundary tests for the writing control plane."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tributo.data.writing import WriteMode, WriteReceipt, WriteRequest

SECRET = "credential-secret-42"


def test_request_rejects_inline_credentials() -> None:
    with pytest.raises((ValueError, ValidationError), match="credentials"):
        WriteRequest(
            engine="ray",
            target_kind="parquet",
            target="s3://bucket/output",
            mode=WriteMode.OVERWRITE,
            options={"secret_access_key": SECRET},
        )


def test_request_digest_and_repr_do_not_include_secret() -> None:
    request = WriteRequest(
        engine="ray",
        target_kind="parquet",
        target="s3://bucket/output",
        mode=WriteMode.OVERWRITE,
        options={"profile": "analytics"},
        runtime_options={"secret_ref": "env://AWS_SECRET_ACCESS_KEY"},
    )

    assert SECRET not in request.request_digest
    assert SECRET not in repr(request)


def test_receipt_rejects_credential_bearing_diagnostics_and_metadata() -> None:
    common = {
        "request_digest": "0" * 64,
        "engine_id": "ray",
        "binding_id": "test.ray.parquet",
        "target_kind": "parquet",
        "target_ref": "s3://bucket/output",
        "mode": WriteMode.OVERWRITE,
        "committed": True,
    }

    with pytest.raises((ValueError, ValidationError), match="credential"):
        WriteReceipt(**common, diagnostics=(f"secret={SECRET}",))
    with pytest.raises((ValueError, ValidationError), match="credential"):
        WriteReceipt(**common, metadata={"token": SECRET})


def test_receipt_repr_hides_target_reference_and_metadata() -> None:
    receipt = WriteReceipt(
        request_digest="0" * 64,
        engine_id="ray",
        binding_id="test.ray.parquet",
        target_kind="parquet",
        target_ref="s3://bucket/private-output",
        mode=WriteMode.OVERWRITE,
        committed=True,
        metadata={"partition": "2026-08-13"},
    )

    rendered = repr(receipt)
    assert "private-output" not in rendered
    assert "2026-08-13" not in rendered


def test_runtime_options_reject_nested_resolved_credentials() -> None:
    with pytest.raises((ValueError, ValidationError), match="inline credentials"):
        WriteRequest(
            engine="ray",
            target_kind="parquet",
            target="s3://bucket/output",
            mode=WriteMode.OVERWRITE,
            runtime_options={"catalog": {"password": SECRET}},
        )


def test_runtime_options_accept_nested_credential_references() -> None:
    request = WriteRequest(
        engine="ray",
        target_kind="iceberg",
        target="catalog.table",
        mode=WriteMode.OVERWRITE,
        runtime_options={
            "catalog": {"credential_ref": "secret://catalog/prod"},
            "profiles": [{"secret_ref": "env://AWS_SECRET_ACCESS_KEY"}],
        },
    )

    assert request.runtime_options["catalog"]["credential_ref"] == (
        "secret://catalog/prod"
    )


def test_runtime_options_reject_nested_invalid_credential_references() -> None:
    with pytest.raises((ValueError, ValidationError), match="approved URI"):
        WriteRequest(
            engine="ray",
            target_kind="iceberg",
            target="catalog.table",
            mode=WriteMode.OVERWRITE,
            runtime_options={"catalog": {"secret_ref": "plaintext-secret"}},
        )
