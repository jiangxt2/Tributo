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
from tributo.data.registry import get_connector, list_connectors, register_connector
from tributo.data.source_config import (
    CsvSourceConfig,
    IcebergSourceConfig,
    LegacyConfigNormalizer,
    ParquetSourceConfig,
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

__all__ = [
    # Abstract base classes and config
    "DataConnector",
    "WriteMode",
    "S3Config",
    # Graph data abstraction
    "GraphSchema",
    "GraphDataBundle",
    # Registry
    "get_connector",
    "register_connector",
    "list_connectors",
    # Source configuration
    "SourceConfig",
    "ParquetSourceConfig",
    "CsvSourceConfig",
    "SqlSourceConfig",
    "IcebergSourceConfig",
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
