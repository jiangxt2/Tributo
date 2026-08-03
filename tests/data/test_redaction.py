"""Redaction contract: credentials never reach repr, logs, errors, ref_id."""

from __future__ import annotations

import logging

import pytest

from tributo.data.provider import ResolvedSource
from tributo.data.provider_builtins import ClickHouseProvider, ParquetProvider
from tributo.data.refs import compute_ref_id, digest
from tributo.data.source_config import ProviderSourceConfig

SECRET = "s3cr3t-password-42"


def resolved_with_credentials() -> ResolvedSource:
    provider = ClickHouseProvider()
    return provider.normalize(
        ProviderSourceConfig(
            provider="tributo.clickhouse",
            uri="clickhouse://db.example/analytics",
            options={
                "sql": "SELECT * FROM t",
                "host": "db.example",
                "password": SECRET,
                "params": {"p": 1},
            },
        )
    )


class TestReprRedaction:
    def test_repr_hides_password(self) -> None:
        resolved = resolved_with_credentials()
        assert SECRET not in repr(resolved)

    def test_repr_hides_s3_secrets(self) -> None:
        provider = ParquetProvider()
        resolved = provider.normalize(
            ProviderSourceConfig(
                provider="tributo.parquet",
                uri="s3://bkt/data.parquet",
                options={
                    "s3": {
                        "access_key_id": SECRET,
                        "secret_access_key": SECRET,
                    }
                },
            )
        )
        assert SECRET not in repr(resolved)
        # Runtime option keys may appear; values never do.
        assert "s3" in repr(resolved)


class TestLogRedaction:
    def test_logging_resolved_source_is_safe(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        resolved = resolved_with_credentials()
        with caplog.at_level(logging.INFO):
            logging.getLogger("tributo.data").info("opened %r", resolved)
        assert SECRET not in caplog.text


class TestErrorRedaction:
    def test_validation_error_message_is_safe(self) -> None:
        # An invalid provider/uri input carrying a password must not leak it.
        provider = ClickHouseProvider()
        try:
            provider.normalize(
                ProviderSourceConfig(
                    provider="tributo.parquet",  # mismatch
                    uri="clickhouse://db.example",
                    options={"sql": "SELECT 1", "password": SECRET},
                )
            )
            raise AssertionError("expected JobConfigurationError")
        except Exception as exc:  # noqa: BLE001 - asserting redaction
            assert SECRET not in str(exc)


class TestRefIdRedaction:
    def test_ref_id_uses_identity_options_only(self) -> None:
        resolved = resolved_with_credentials()
        ref_id = compute_ref_id(
            provider_id=resolved.provider_id,
            canonical_uri=resolved.canonical_uri,
            result_affecting_options=dict(resolved.identity_options),
        )
        # The identity contains digests — never the raw credential.
        assert SECRET not in ref_id
        assert SECRET not in str(dict(resolved.identity_options))

    def test_identity_options_have_no_credential_values(self) -> None:
        resolved = resolved_with_credentials()
        text = str(dict(resolved.identity_options))
        assert SECRET not in text
        assert "password" not in text
        assert "sql_digest" in text

    def test_sql_digest_stable_and_raw_free(self) -> None:
        sql = "SELECT * FROM t WHERE secret = %(s)s"
        assert digest(sql) != sql
        assert len(digest(sql)) == 64
