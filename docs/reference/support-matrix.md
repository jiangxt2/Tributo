# Support matrix

This page distinguishes implemented paths from extension contracts and
prototypes.

## Data

| Capability | Status | Boundary |
| --- | --- | --- |
| Local/S3 Parquet and CSV reads | Implemented | Bounded Ray Dataset input |
| Iceberg reads | Implemented | PyIceberg catalog and planned files |
| ClickHouse reads | Implemented | Native clickhouse-connect client |
| Doris reads | Implemented | MySQL protocol through PyMySQL |
| Lance output | Implemented for embedding workflows | Not a generic inference sink |
| Database inference sinks | Extension point | No built-in ClickHouse or Doris sink |
| Daft transform compiler | Prototype | Validation-only; not the stable data path |

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
