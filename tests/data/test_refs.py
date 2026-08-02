"""Tests for DatasetRef identity: determinism, versioning, credential safety."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pyarrow as pa
import pytest

from tributo.data.refs import (
    CREDENTIAL_KEYS,
    DatasetRef,
    compute_ref_id,
    digest,
    schema_fingerprint,
)


class TestDigest:
    """Canonical JSON determinism."""

    def test_same_input_same_digest(self) -> None:
        assert digest({"a": 1, "b": [1, 2]}) == digest({"a": 1, "b": [1, 2]})

    def test_key_order_insensitive(self) -> None:
        assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})

    def test_hex_length(self) -> None:
        assert len(digest({})) == 64

    def test_ascii_only(self) -> None:
        # ensure_ascii=True: non-ASCII content still yields the same hex string.
        assert digest({"col": "中文"}) == digest({"col": "中文"})

    def test_datetime_and_decimal_supported(self) -> None:
        # Database bound parameters may carry datetime/Decimal — they get a
        # deterministic string representation, never a raw TypeError.
        from datetime import datetime
        from decimal import Decimal

        ts = datetime(2026, 1, 1, 8, 30)
        assert digest({"p": ts}) == digest({"p": ts})
        assert digest({"p": Decimal("1.50")}) == digest({"p": Decimal("1.50")})
        assert len(digest({"p": ts})) == 64
        assert len(digest({"p": Decimal("1.50")})) == 64


class TestComputeRefId:
    """Versioned SHA-256 identity of a bounded data source."""

    def test_stable(self) -> None:
        ref1 = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/data.parquet",
            result_affecting_options={"columns": ["a", "b"]},
        )
        ref2 = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/data.parquet",
            result_affecting_options={"columns": ["a", "b"]},
        )
        assert ref1 == ref2

    def test_option_order_insensitive(self) -> None:
        a = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/data.parquet",
            result_affecting_options={"columns": ["a"], "snapshot": 7},
        )
        b = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/data.parquet",
            result_affecting_options={"snapshot": 7, "columns": ["a"]},
        )
        assert a == b

    def test_uri_changes_identity(self) -> None:
        a = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={},
        )
        b = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/b.parquet",
            result_affecting_options={},
        )
        assert a != b

    def test_columns_change_identity(self) -> None:
        a = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={"columns": ["a"]},
        )
        b = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={"columns": ["a", "b"]},
        )
        assert a != b

    def test_version_changes_identity(self) -> None:
        a = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={},
        )
        b = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={},
            version=2,
        )
        assert a != b

    def test_nested_options_stable(self) -> None:
        opts = {
            "sql_digest": digest("SELECT * FROM t WHERE x = %(p)s"),
            "params": {"p": 1},
        }
        ref1 = compute_ref_id(
            provider_id="tributo.clickhouse",
            canonical_uri="ch://db.example:8443/analytics",
            result_affecting_options=opts,
        )
        ref2 = compute_ref_id(
            provider_id="tributo.clickhouse",
            canonical_uri="ch://db.example:8443/analytics",
            result_affecting_options={
                "params": {"p": 1},
                "sql_digest": opts["sql_digest"],
            },
        )
        assert ref1 == ref2

    def test_credential_keys_are_known(self) -> None:
        # The redaction contract: credential keys are enumerated so Providers
        # strip them before building result_affecting_options.
        assert "password" in CREDENTIAL_KEYS
        assert "secret_access_key" in CREDENTIAL_KEYS

    @pytest.mark.parametrize("key", sorted(CREDENTIAL_KEYS))
    def test_rejects_credential_keys_in_options(self, key: str) -> None:
        # Defensive backstop: a Provider that leaks a credential into the
        # identity is a bug — fail loudly instead of hashing it in.
        with pytest.raises(ValueError, match="credential key"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                result_affecting_options={key: "s3cr3t"},
            )

    def test_rejects_credential_uri(self) -> None:
        # canonical_uri must be credential-free too — userinfo and sensitive
        # query parameters are the canonical_uri backstop check.
        with pytest.raises(ValueError, match="canonical_uri must not contain"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://user:pass@host/a.parquet",
                result_affecting_options={},
            )
        with pytest.raises(ValueError, match="canonical_uri must not contain"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet?token=abc",
                result_affecting_options={},
            )
        # Non-credential query keys are fine — they can be result-affecting.
        ref = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet?versionId=42",
            result_affecting_options={},
        )
        assert len(ref) == 64

    def test_nested_credential_keys_are_rejected(self) -> None:
        # The defensive guard applies recursively, including nested provider
        # option mappings and lists.
        with pytest.raises(ValueError, match="credential key"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                result_affecting_options={
                    "s3": {"endpoint": "http://x", "secret": "s3cr3t"}
                },
            )
        with pytest.raises(ValueError, match="credential key"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                result_affecting_options={"parts": [{"rest.token": "tok"}]},
            )
        with pytest.raises(ValueError, match="credential key/field"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                result_affecting_options={"catalog_uri": "https://user:pass@host"},
            )
        # A clean identity with a nested s3 endpoint computes fine.
        ref = compute_ref_id(
            provider_id="tributo.parquet",
            canonical_uri="s3://bkt/a.parquet",
            result_affecting_options={"s3": {"endpoint": "http://x", "region": "r"}},
        )
        assert len(ref) == 64

    def test_aws_style_credential_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="credential key"):
            compute_ref_id(
                provider_id="tributo.parquet",
                canonical_uri="s3://bkt/a.parquet",
                result_affecting_options={"AWSAccessKeyId": "access-secret"},
            )


class TestSchemaFingerprint:
    """Arrow schema identity."""

    def test_same_schema_same_fingerprint(self) -> None:
        s1 = pa.schema([("a", pa.int64()), ("b", pa.string())])
        s2 = pa.schema([("a", pa.int64()), ("b", pa.string())])
        assert schema_fingerprint(s1) == schema_fingerprint(s2)

    def test_field_order_changes_fingerprint(self) -> None:
        s1 = pa.schema([("a", pa.int64()), ("b", pa.string())])
        s2 = pa.schema([("b", pa.string()), ("a", pa.int64())])
        assert schema_fingerprint(s1) != schema_fingerprint(s2)

    def test_type_change_changes_fingerprint(self) -> None:
        s1 = pa.schema([("a", pa.int64())])
        s2 = pa.schema([("a", pa.float64())])
        assert schema_fingerprint(s1) != schema_fingerprint(s2)

    def test_nullable_change_changes_fingerprint(self) -> None:
        s1 = pa.schema([("a", pa.int64())])
        s2 = pa.schema([pa.field("a", pa.int64(), nullable=False)])
        assert schema_fingerprint(s1) != schema_fingerprint(s2)

    def test_metadata_key_order_insensitive(self) -> None:
        s1 = pa.schema([("a", pa.int64())], metadata={b"k1": b"v1", b"k2": b"v2"})
        s2 = pa.schema([("a", pa.int64())], metadata={b"k2": b"v2", b"k1": b"v1"})
        assert schema_fingerprint(s1) == schema_fingerprint(s2)

    def test_rejects_non_schema(self) -> None:
        with pytest.raises(TypeError, match="pa.Schema"):
            schema_fingerprint(cast(Any, "not-a-schema"))


class TestDatasetRef:
    """Credential-free record shape."""

    def test_defaults(self) -> None:
        ref = DatasetRef(
            ref_id="a" * 64,
            provider_id="tributo.parquet",
            uri="s3://bkt/a.parquet",
            schema_fingerprint="b" * 64,
        )
        assert ref.row_count is None
        assert ref.provenance == ""

    @pytest.mark.parametrize("field", ["ref_id", "schema_fingerprint"])
    def test_digest_fields_require_sha256(self, field: str) -> None:
        kwargs = {
            "ref_id": "a" * 64,
            "provider_id": "tributo.parquet",
            "uri": "s3://bkt/a.parquet",
            "schema_fingerprint": "b" * 64,
        }
        kwargs[field] = "password=secret"
        with pytest.raises(ValueError, match="64-character"):
            DatasetRef(**kwargs)

    def test_frozen(self) -> None:
        ref = DatasetRef(
            ref_id="a" * 64,
            provider_id="tributo.parquet",
            uri="s3://bkt/a.parquet",
            schema_fingerprint="b" * 64,
        )
        with pytest.raises(FrozenInstanceError):
            ref.__setattr__("provider_id", "tributo.csv")

    def test_repr_contains_no_credentials(self) -> None:
        # DatasetRef carries no credential fields by construction — repr must
        # not accidentally surface any.
        ref = DatasetRef(
            ref_id="a" * 64,
            provider_id="tributo.clickhouse",
            uri="ch://db.example/analytics",
            schema_fingerprint="b" * 64,
        )
        text = repr(ref)
        assert "password" not in text.lower()
        assert "secret" not in text.lower()

    def test_credential_uri_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="canonical_uri must not contain"):
            DatasetRef(
                ref_id="a" * 64,
                provider_id="tributo.parquet",
                uri="s3://user:pass@host/a.parquet",
                schema_fingerprint="b" * 64,
            )
        with pytest.raises(ValueError, match="provenance must not contain"):
            DatasetRef(
                ref_id="a" * 64,
                provider_id="tributo.parquet",
                uri="s3://bkt/a.parquet",
                schema_fingerprint="b" * 64,
                provenance="https://user:pass@host/run",
            )

        with pytest.raises(ValueError, match="provenance must not contain"):
            DatasetRef(
                ref_id="a" * 64,
                provider_id="tributo.parquet",
                uri="s3://bkt/a.parquet",
                schema_fingerprint="b" * 64,
                provenance="run=42 password=supersecret",
            )
