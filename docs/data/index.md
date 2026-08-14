# Data

Tributo has one bounded-ingestion control path and two explicit execution
engines. Tributo validates requests, creates an engine-neutral logical scan,
translates the portable ETL subset, and records provenance. Ray Data, Daft, or
an installed third-party Connector owns the physical read.

```text
IngestionRequest
    -> ProviderRegistry (built-ins + installed Provider plugins)
    -> LogicalScanPlan
    -> EngineBinding
    -> Ray Data or Daft native handle
    -> IngestionPlanReceipt
```

The Gateway never silently selects or falls back to another engine. Existing
Ray-only loaders are compatibility adapters over this same path.

`tributo.data` is the consumer facade: callers use `IngestionRequest`,
`IngestionGateway`, typed handles, receipts, and explicit handle adapters.
`scan_plan`, `engine_binding`, Provider registries, and native Connector plans
are Developer SPI for ingestion extensions and must not be imported by
Training, Inference, or graph algorithms. Historical beta Provider
exports remain only for their documented compatibility window.

New bounded sources use `ProviderSourceConfig(provider, uri, options)` and may
publish a versioned Provider descriptor through
`tributo.ingestion_providers`. Physical Ray/Daft implementations publish
Binding descriptors through `tributo.ingestion_bindings`. Provider metadata
declares projection-option and relative-file-URI semantics, so consumer modules
never add source-name branches. `TableScan.storage_format_id` and declarative
Binding constraints cover catalog tables whose storage resolves to Parquet,
ORC, or Iceberg. Multiple matching Bindings are an error unless the request
selects `binding_id` explicitly.

## Input status

| Input | Ray Data | Daft | Status |
| --- | --- | --- | --- |
| Local/S3 Parquet | Native reader | Native reader | Verified |
| Local/S3 CSV | Native reader | Native reader | Verified |
| Local/S3 Iceberg | Native reader | Native reader | Verified |
| Local/S3 Lance | Native reader | Native reader | Verified |
| PostgreSQL structured table | Native SQL reader | Native SQL reader | Verified |
| HDFS Parquet/CSV | Native reader with PyArrow HDFS | No locked public reader | Adapted; cluster gate pending |
| ClickHouse | No selected Connector | `daft-olap-connectors` | Adapter only; external package and database gates pending |
| Doris | `ray-doris` | `daft-olap-connectors` | Adapter only; external packages and database gates pending |
| ORC or Hive external table | No locked public reader | No locked public reader | Unsupported, fail-closed |

“Verified” means the current combination has semantic Conformance and real
storage or database evidence. It does not turn every engine/source combination
into a supported path. See the [support matrix](../reference/support-matrix.md)
for the exact boundary.

Credentials belong to runtime configuration. They must not appear in dataset
identifiers, logical plans, receipts, logs, or public errors. Bounded providers
and unbounded stream sources remain separate contracts, and an input Connector
does not imply an inference output sink.

## Lance output compatibility

`LanceDataConnector` is a beta compatibility adapter over the same distributed
Lance writer used by inference. It always writes Lance and never infers an
output format from vector-like columns. Older releases wrote Parquet with ZSTD
when no floating-point list column was present; callers that require Parquet
must now select `ParquetDataConnector` or a Parquet ResultSink explicitly.

Lance save modes are fail-closed: `create` is an atomic create-only operation,
`append` requires an existing target, and `overwrite` creates or replaces the
target. Empty create/overwrite inputs still materialize the declared schema;
an empty append preserves the existing dataset version.
