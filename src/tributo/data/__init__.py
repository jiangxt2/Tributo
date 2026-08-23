"""Tributo bounded-ingestion, neutral contract, and native-write APIs.

Ray Data, Daft, or an installed binding owns physical reads and writes.
Tributo owns typed requests, logical plans, engine bindings, ETL translation,
errors, and provenance.  The former ``DataConnector`` facade and registry are
intentionally not exported.

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

from importlib import import_module

from tributo.data.contracts import S3Config, WriteMode
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
from tributo.data.persistence import (
    CheckpointStore,
    LanceRayBinding,
    LocalS3ObjectStore,
    ObjectFile,
    ObjectStore,
    ParquetInspection,
    RayCheckpointStore,
    ResolvedLanceDataset,
    default_checkpoint_store,
    default_object_store,
    inspect_parquet_output,
    write_parquet_table,
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
    normalize_legacy_inference_json_source,
    normalize_legacy_inference_source,
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
from tributo.data.writing import (
    DataWriteTargetRequest,
    GenericWriteTargetProvider,
    LogicalWritePlan,
    WriteBinding,
    WriteBindingError,
    WriteBindingRegistry,
    WriteCapability,
    WriteCapabilityError,
    WriteDescriptor,
    WriteError,
    WriteExecutionContext,
    WriteGateway,
    WriteHandle,
    WriteReceipt,
    WriteRequest,
    WriteTargetProvider,
    WriteTargetRegistry,
    default_write_gateway,
)

__all__ = [
    # Neutral shared contracts
    "S3Config",
    "WriteMode",
    # Graph data abstraction
    "GraphSchema",
    "GraphDataBundle",
    # Explicit native-handle adapters
    "HandleConversionReceipt",
    "RayHandleAdaptation",
    "adapt_daft_result_to_ray",
    # Bounded-ingestion contract
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
    # Provider contract and registry
    "DataSourceProvider",
    "ResolvedSource",
    "DatasetHandle",
    "register_provider",
    "resolve_provider",
    "unregister_provider",
    "list_providers",
    # Checkpoint persistence boundary
    "CheckpointStore",
    "LanceRayBinding",
    "RayCheckpointStore",
    "ResolvedLanceDataset",
    "default_checkpoint_store",
    "LocalS3ObjectStore",
    "ObjectFile",
    "ObjectStore",
    "default_object_store",
    "ParquetInspection",
    "inspect_parquet_output",
    "write_parquet_table",
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
    "normalize_legacy_inference_source",
    "normalize_legacy_inference_json_source",
    # Native-engine writing control plane
    "WriteBinding",
    "DataWriteTargetRequest",
    "WriteBindingError",
    "WriteCapability",
    "WriteBindingRegistry",
    "WriteCapabilityError",
    "WriteDescriptor",
    "WriteError",
    "WriteExecutionContext",
    "WriteGateway",
    "default_write_gateway",
    "WriteHandle",
    "WriteReceipt",
    "WriteRequest",
    "GenericWriteTargetProvider",
    "LogicalWritePlan",
    "WriteTargetProvider",
    "WriteTargetRegistry",
]

# Importing the canonical provider catalog is the only data-module
# initialization required here. It registers Provider classes, not legacy
# Connector classes, and has no optional Connector side effects.
import_module("tributo.data.provider_builtins")
