# Support matrix

This page distinguishes implemented paths from extension contracts and
prototypes.

## Data

| Capability | Status | Boundary |
| --- | --- | --- |
| Local/S3 Parquet and CSV reads | Verified | Native Ray Data or Daft handle through one Gateway |
| Local/S3 Iceberg reads | Verified | Native Ray Data or Daft table reader; broader Catalog/delete-file matrix remains gated |
| Local/S3 Lance reads | Verified | Native Ray Data or Daft table reader |
| PostgreSQL structured table reads | Verified | Ray Data or Daft public SQL reader; no arbitrary SQL in the new path |
| HDFS Parquet/CSV reads | Adapter only | Ray binding exists; real HDFS/JVM/worker gate is pending |
| ClickHouse reads | Adapter only | Requires unpublished `daft-olap-connectors` and real-database Conformance |
| Doris reads | Adapter only | Requires unpublished `ray-doris` or `daft-olap-connectors` and real-database Conformance |
| ORC and Hive external-table reads | Not implemented | Locked Ray/Daft versions expose no validated public reader |
| Lance output | Implemented for embedding workflows | Not a generic inference sink |
| Database inference sinks | Extension point | No built-in ClickHouse or Doris sink |
| Ray/Daft transform compiler | Alpha | Portable bounded ETL subset with dual-engine Conformance |

## Training

| Capability | Status | Boundary |
| --- | --- | --- |
| XGBoost | Beta | Multi-worker Ray Train path |
| DNN | Beta | PyTorch and Ray Train |
| PU learning | Beta | Current trainer requires one worker |
| Ray Tune | Beta | Capability-gated algorithms only |
| Graph training | Alpha skeleton | No built-in PyG/DGL trainer |
| Causal estimation | Extension contract | No concrete estimator is bundled |

## Bundle and inference

| Capability | Status | Boundary |
| --- | --- | --- |
| Local and `file://` bundle publication | Beta | Manifest and digest validation |
| S3 bundle publication | Beta | Manifest-last and alias compare-and-set |
| HDFS bundle publication | Not implemented | Storage backend extension |
| Ray Data batch inference | Beta | Actor-based model reuse |
| Batch output to local/S3 Parquet | Implemented | Database sinks are separate |
| ONNX Runtime flavor | Implemented | Other artifacts require matching loaders |

## Serving and streaming

| Capability | Status | Boundary |
| --- | --- | --- |
| ONNX HTTP serving | Beta | Ray Serve |
| gRPC serving | Beta | Install the `grpc` extra |
| LLM SSE serving | Alpha | Streaming service contract |
| Kafka source | Alpha | Fail-closed microbatch source |
| Kafka-to-inference service loop | Not built in | Requires explicit orchestration and sink |

## Configuration and compatibility

| Contract | Status |
| --- | --- |
| Python | `>=3.12,<3.14` |
| Configuration files | JSON |
| YAML configuration | Rejected |
| Public stability source | `@PublicAPI` and the stability inventory |
| Versioning | Semantic Versioning |

For symbol-level compatibility promises, consult the
[API stability inventory](../STABILITY.md).
