"""Tributo bounded-ingestion contracts and compatibility adapters.

Ray Data, Daft, or an installed connector owns physical reads. Tributo owns
typed requests, logical plans, engine bindings, ETL translation, errors, and
provenance. Historical ``DataConnector`` exports remain for compatibility and
write paths.

Usage example::

    from tributo.data import IngestionRequest, ParquetSourceConfig, open_ingestion

    result = open_ingestion(
        IngestionRequest(
            source=ParquetSourceConfig(path="s3://bucket/data.parquet"),
            engine="ray",
        )
    )
    dataset = result.handle.dataset
"""

from __future__ import annotations

import importlib
import logging

from tributo.data.base import DataConnector, S3Config, WriteMode
from tributo.data.engine_binding import (
    BindingCompilation,
    BindingCompileRequest,
    BindingDescriptor,
    BindingKey,
    BindingPlanConstraints,
    EngineBinding,
)
from tributo.data.graph import GraphDataBundle, GraphSchema
from tributo.data.handle_adapters import (
    HandleConversionReceipt,
    RayHandleAdaptation,
    adapt_daft_result_to_ray,
)
from tributo.data.ingestion import (
    DaftDataFrameHandle,
    DistributionVersionEvidence,
    HandleOwnership,
    IngestionDescriptor,
    IngestionGateway,
    IngestionOpenResult,
    IngestionPlanReceipt,
    IngestionRequest,
    IngestionRuntimeContext,
    PhysicalSplitSummary,
    RayDataHandle,
    ReadHint,
    ReadOptions,
    SchemaContract,
    TransformDecision,
    describe_ingestion,
    open_ingestion,
    ray_worker_distribution_probe,
)
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
    apply_source_projection,
    source_projection,
)
from tributo.data.transform_ir import (
    CastColumn,
    ColumnRename,
    DropColumns,
    FillNull,
    FilterComparison,
    FilterEq,
    FilterIsIn,
    FilterNotEq,
    FilterNotNull,
    FilterNull,
    FilterRange,
    Limit,
    RenameColumns,
    SelectColumns,
    TransformPipeline,
    transform_ir_digest,
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
    # Explicit native-handle adapters
    "HandleConversionReceipt",
    "RayHandleAdaptation",
    "adapt_daft_result_to_ray",
    # Candidate bounded-ingestion contract
    "IngestionRequest",
    "IngestionRuntimeContext",
    "ReadOptions",
    "ReadHint",
    "SchemaContract",
    "IngestionGateway",
    "IngestionDescriptor",
    "IngestionOpenResult",
    "IngestionPlanReceipt",
    "HandleOwnership",
    "TransformDecision",
    "DistributionVersionEvidence",
    "PhysicalSplitSummary",
    "RayDataHandle",
    "DaftDataFrameHandle",
    "describe_ingestion",
    "open_ingestion",
    "ray_worker_distribution_probe",
    # Third-party ingestion Binding SPI
    "BindingKey",
    "BindingPlanConstraints",
    "BindingCompileRequest",
    "BindingCompilation",
    "BindingDescriptor",
    "EngineBinding",
    # Engine-neutral Transform IR
    "TransformPipeline",
    "transform_ir_digest",
    "FilterEq",
    "FilterNotEq",
    "FilterComparison",
    "FilterRange",
    "FilterIsIn",
    "FilterNull",
    "FilterNotNull",
    "SelectColumns",
    "DropColumns",
    "ColumnRename",
    "RenameColumns",
    "CastColumn",
    "FillNull",
    "Limit",
    # Connector registry
    "get_connector",
    "register_connector",
    "list_connectors",
    # Provider contract
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
    "source_projection",
    "apply_source_projection",
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
