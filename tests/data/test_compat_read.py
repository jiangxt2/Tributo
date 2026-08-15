"""Canonical planning boundary for Provider implementations."""

from __future__ import annotations

import warnings

import pytest

from tributo.data.ingestion import IngestionRequest, open_ingestion
from tributo.data.provider import DatasetHandle, DataSourceProvider, ResolvedSource
from tributo.data.provider_registry import register_provider, unregister_provider
from tributo.data.source_config import CanonicalSourceInput, ProviderSourceConfig
from tributo.exceptions import JobConfigurationError


class _LegacyProvider(DataSourceProvider):
    provider_id = "myorg.legacy"
    aliases = frozenset({"legacy"})

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        assert isinstance(source, ProviderSourceConfig)
        return ResolvedSource(self.provider_id, source.uri)

    def open(self, resolved: ResolvedSource) -> DatasetHandle:
        raise AssertionError(f"canonical ingestion must not call open({resolved!r})")


def test_open_only_provider_fails_closed_at_canonical_plan() -> None:
    register_provider(_LegacyProvider)
    try:
        source = ProviderSourceConfig(provider="myorg.legacy", uri="legacy://events")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(JobConfigurationError, match="planned ingestion"):
                open_ingestion(IngestionRequest(source=source, engine="ray"))
    finally:
        unregister_provider(_LegacyProvider.provider_id)
