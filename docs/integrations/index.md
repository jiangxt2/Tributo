# Integrations

Tributo integrates with external systems through explicit provider, callback,
exporter, registry, and stream-source contracts.

| Area | Supported integrations |
| --- | --- |
| Data | Verified Parquet, CSV, Iceberg, Lance, and PostgreSQL paths; gated external ClickHouse/Doris adapters |
| Training | Ray Train, XGBoost, and PyTorch |
| Tracking and registry | MLflow |
| Model runtime | ONNX Runtime and Ray Serve |
| Object storage | Local files and S3 |
| Streaming input | Kafka |

An installed dependency does not by itself imply an end-to-end supported
workflow. Consult the [support matrix](../reference/support-matrix.md) for the
current maturity and limitations of each integration.

## Guides

- [Track published bundles with MLflow](../how-to/mlflow.md)
