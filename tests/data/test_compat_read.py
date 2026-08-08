"""Compatibility adapter tests for planned and legacy Provider contracts."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from tributo.data._compat_read import open_ray_compat
from tributo.data.ingestion import IngestionRequest, RayDataHandle, open_ingestion
from tributo.data.provider import DatasetHandle, DataSourceProvider, ResolvedSource
from tributo.data.provider_registry import register_provider, unregister_provider
from tributo.data.source_config import CanonicalSourceInput, ProviderSourceConfig
from tributo.exceptions import JobConfigurationError


class _LegacyHandle(DatasetHandle):
    def __init__(self, dataset: Any) -> None:
        super().__init__()
        self.dataset = dataset
        self.release_count = 0

    def _read(self) -> Any:
        return self.dataset

    def _release(self) -> None:
        self.release_count += 1


class _LegacyProvider(DataSourceProvider):
    provider_id = "myorg.legacy"
    aliases = frozenset({"legacy"})
    last_handle: _LegacyHandle | None = None

    def normalize(self, source: CanonicalSourceInput) -> ResolvedSource:
        assert isinstance(source, ProviderSourceConfig)
        return ResolvedSource(self.provider_id, source.uri)

    def open(self, resolved: ResolvedSource) -> DatasetHandle:
        del resolved
        handle = _LegacyHandle(cast(Any, object()))
        type(self).last_handle = handle
        return handle


def test_open_only_provider_is_preserved_only_by_ray_compatibility_adapter() -> None:
    register_provider(_LegacyProvider)
    try:
        source = ProviderSourceConfig(provider="myorg.legacy", uri="legacy://events")
        with pytest.raises(JobConfigurationError, match="planned ingestion"):
            open_ingestion(IngestionRequest(source=source, engine="ray"))
        with pytest.warns(FutureWarning, match=r"normalize\(\)\+open\(\)"):
            dataset = open_ray_compat(source)

        handle = _LegacyProvider.last_handle
        assert handle is not None
        assert dataset is handle.dataset
        assert handle.release_count == 1
    finally:
        unregister_provider(_LegacyProvider.provider_id)
        _LegacyProvider.last_handle = None


def test_planned_provider_still_uses_gateway_without_legacy_fallback() -> None:
    dataset = MagicMock()
    result = MagicMock(handle=RayDataHandle(dataset))
    source = ProviderSourceConfig(provider="tributo.parquet", uri="s3://b/data")

    with patch(
        "tributo.data._compat_read.open_ingestion", return_value=result
    ) as open_:
        actual = open_ray_compat(source)

    request = open_.call_args.args[0]
    assert request.engine == "tributo.ray_data"
    assert actual is dataset
    result.close.assert_called_once_with()
