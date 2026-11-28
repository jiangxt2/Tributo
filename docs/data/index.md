# Data

The development runtime baseline in the
[support matrix](../reference/support-matrix.md) has passed focused local/S3
read/write and Hive worker-projection gates on Linux arm64. The statuses below
retain the implemented routes' validation boundaries; these focused checks do
not certify every data source or platform on the new runtime combination.

Tributo has one bounded-ingestion control path, an engine-neutral transform
contract, and a native-write control path. Ray Data or an installed
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
    -> Ray Data native handle
    -> IngestionPlanReceipt
```

The Gateway never silently selects or falls back to another engine. Existing
Ray-only loaders are compatibility adapters over this same path.

`tributo.data` is the consumer facade: callers use `IngestionRequest`,
`IngestionGateway`, typed handles, receipts, and runtime configuration.
`scan_plan`, `engine_binding`, Provider registries, and native Binding plans
are Developer SPI for ingestion extensions and must not be imported by
Training, Inference, or graph algorithms. Historical beta Provider
exports remain only for their documented compatibility window.

Third-party bounded sources use `ProviderSourceConfig(provider, uri, options)` and may
publish a versioned Provider descriptor through
`tributo.ingestion_providers`. Physical Ray implementations publish
Binding descriptors through `tributo.ingestion_bindings`. Provider metadata
declares projection-option and relative-file-URI semantics, so consumer modules
never add source-name branches. `TableScan.storage_format_id` and declarative
Binding constraints cover catalog tables whose storage resolves to Parquet,
ORC, or Iceberg. Multiple matching Bindings are an error unless the request
selects `binding_id` explicitly.

## Input status

| Input | Ray Data | Status |
| --- | --- | --- |
| Local/S3 Parquet, CSV, Iceberg and Lance | Native readers | Verified |
| PostgreSQL structured table | Native SQL reader | Verified |
| HDFS Parquet/CSV | Native reader with PyArrow HDFS | Adapted; cluster gate pending |
| ClickHouse | `ray-clickhouse` | Adapter only; database conformance pending |
| Doris | `ray-doris` | Adapter only |
| HiveServer2 structured table | `ray-hive` through `tributo.ray.hive` | Alpha, verified with Hive 4.2.0 |
| Native ORC file | No built-in route | Unsupported |

See the [support matrix](../reference/support-matrix.md) for exact validation boundaries.

Credentials belong to runtime configuration. They must not appear in dataset
identifiers, logical plans, receipts, logs, or public errors. Bounded input
providers only describe reads; inference output sinks are selected separately.

## Lance output compatibility

Lance output is written through `WriteGateway`, the same control-plane boundary
used by inference. The Lance ResultSink always writes Lance and never infers an
output format from vector-like columns. Older releases exposed format-specific
compatibility facades; callers now select a `WriteRequest` /
native Binding or a ResultSink explicitly.

The stable Ray Binding currently delegates to the official Lance-Ray
`write_lance(..., stream=False)` API. Tributo forwards `create`, `append`, and `overwrite`
without rewriting target state and does not add guarantees for empty writes,
schema compatibility or evolution, exclusive creation, fragment counts, or a
post-write dataset version. The Binding remains data-plane replaceable: once
the locked Ray/PyLance combination is compatible, adopting
`Dataset.write_lance` changes only the Ray Binding.
