"""tributo.data — Data connector abstraction layer.

Unified data read/write interface supporting Parquet, Lance, Iceberg, and other formats.

Usage example::

    from tributo.data import get_connector, S3Config

    connector = get_connector("parquet")
    ds = connector.read(path="s3://bucket/data.parquet", s3=S3Config(...))
    connector.write(ds, path="s3://bucket/output")
"""

from __future__ import annotations

import importlib
import logging

from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.graph import GraphDataBundle, GraphSchema
from tributo.data.provider import DatasetHandle, DataSourceProvider, ResolvedSource
from tributo.data.provider_registry import (
    list_providers,
    register_provider,
    resolve_provider,
    unregister_provider,
)
from tributo.data.refs import (
    DatasetRef,
    compute_ref_id,
    digest,
    schema_fingerprint,
)
from tributo.data.registry import get_connector, list_connectors, register_connector
from tributo.data.source_config import (
    CanonicalSourceInput,
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    LegacySourceInput,
    ParquetSourceConfig,
    ProviderSourceConfig,
    RawSourceConfig,
    SourceConfig,
    SourceInput,
    SqlPartitioning,
    SqlSourceConfig,
)
from tributo.exceptions import JobConfigurationError
from tributo.plugin import discover_connector_plugins

logger = logging.getLogger(__name__)

# Trigger built-in connector registration (must be after registry import):
# connectors register themselves via module-level register_connector()
# calls (parquet.py/lance.py/iceberg.py), so the registry must be loaded
# before the importlib.import_module() calls below. Never move connector
# registration into registry/source_config module init — that would
# create an import cycle with data/__init__.py.
# iceberg/lance are optional dependencies — skip registration when not installed.
try:
    importlib.import_module("tributo.data.iceberg")
except ImportError:
    pass
try:
    importlib.import_module("tributo.data.lance")
except ImportError:
    pass
importlib.import_module("tributo.data.parquet")
importlib.import_module("tributo.data.csv")
# Built-in providers register themselves via module-level register_provider()
# calls in provider_builtins.py.
importlib.import_module("tributo.data.provider_builtins")

__all__ = [
    # Abstract base classes and config
    "DataConnector",
    "WriteMode",
    "S3Config",
    # Graph data abstraction
    "GraphSchema",
    "GraphDataBundle",
    # Connector registry
    "get_connector",
    "register_connector",
    "list_connectors",
    # Provider contract (D1+D2)
    "DataSourceProvider",
    "ResolvedSource",
    "DatasetHandle",
    # Provider registry
    "register_provider",
    "resolve_provider",
    "unregister_provider",
    "list_providers",
    # Dataset identity
    "DatasetRef",
    "compute_ref_id",
    "digest",
    "schema_fingerprint",
    # Source configuration
    "SourceConfig",
    "ParquetSourceConfig",
    "CsvSourceConfig",
    "SqlSourceConfig",
    "IcebergSourceConfig",
    "ProviderSourceConfig",
    "CanonicalSourceInput",
    "LegacySourceInput",
    "LegacyConfigNormalizer",
    "RawSourceConfig",
    "SourceInput",
    "SqlPartitioning",
]

# Auto-discover third-party connector plugins via entry_points
for _ep_cls in discover_connector_plugins():
    from tributo.data.registry import register_connector as _reg_conn

    _conn_name = _ep_cls.__name__.lower()
    try:
        _reg_conn(_conn_name, _ep_cls)
    except JobConfigurationError:
        logger.debug(
            "Connector %r from plugin already registered; skipping.",
            _conn_name,
        )
