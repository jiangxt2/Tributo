"""tributo.data — Data connector abstraction layer.

Unified data read/write interface supporting Parquet, Lance, Iceberg, and other formats.

Usage example::

    from tributo.data import get_connector, S3Config

    connector = get_connector("parquet")
    ds = connector.read(path="s3://bucket/data.parquet", s3=S3Config(...))
    connector.write(ds, path="s3://bucket/output")
"""

from __future__ import annotations

import logging

from tributo.exceptions import JobConfigurationError

logger = logging.getLogger(__name__)

# Trigger built-in connector registration (must be after registry import).
# iceberg/lance are optional dependencies — skip registration when not installed.
try:
    import tributo.data.iceberg  # noqa: F401, I001
except ImportError:
    pass
try:
    import tributo.data.lance  # noqa: F401, I001
except ImportError:
    pass
import tributo.data.parquet  # noqa: E402, F401, I001
from tributo.data.base import DataConnector, S3Config, WriteMode  # noqa: E402
from tributo.data.graph import GraphDataBundle, GraphSchema  # noqa: E402
from tributo.data.registry import get_connector, list_connectors, register_connector  # noqa: E402
from tributo.data.source_config import (  # noqa: E402
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
from tributo.plugin import discover_connector_plugins  # noqa: E402

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
