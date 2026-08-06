# Data

Tributo has one bounded-ingestion control path and two explicit execution
engines. Tributo validates requests, creates an engine-neutral logical scan,
translates the portable ETL subset, and records provenance. Ray Data, Daft, or
an installed third-party Connector owns the physical read.

```text
IngestionRequest
    -> ProviderRegistry
    -> LogicalScanPlan
    -> EngineBinding
    -> Ray Data or Daft native handle
    -> IngestionPlanReceipt
```

The Gateway never silently selects or falls back to another engine. Existing
Ray-only loaders are compatibility adapters over this same path.

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
