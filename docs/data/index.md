# Data

Tributo has one bounded-ingestion control path, an engine-neutral transform
contract, and a native-write control path. Ray Data, Daft, or an installed
binding owns physical data movement.

## Start with a task

```{toctree}
:maxdepth: 1

key-concepts
user-guides/read-data
user-guides/write-data
```

See the generated [Data API reference](../reference/api/data.md) for every
annotated public object.

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

Third-party bounded sources use `ProviderSourceConfig(provider, uri, options)` and may
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

`LanceDataConnector` is a beta compatibility adapter over `WriteGateway`, the
same control-plane boundary used by inference. It always writes Lance and never
infers an output format from vector-like columns. Older releases wrote Parquet
with ZSTD when no floating-point list column was present; callers that require
Parquet must now select `ParquetDataConnector` or a Parquet ResultSink
explicitly.

The stable Ray Binding currently delegates to the official Lance-Ray
`write_lance(..., stream=False)` API; the Daft Binding delegates to
`DataFrame.write_lance`. Tributo forwards `create`, `append`, and `overwrite`
without rewriting target state and does not add guarantees for empty writes,
schema compatibility or evolution, exclusive creation, fragment counts, or a
post-write dataset version. Both Bindings remain data-plane replaceable: once
the locked Ray/PyLance combination is compatible, adopting
`Dataset.write_lance` changes only the Ray Binding.
