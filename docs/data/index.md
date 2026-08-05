# Data

The stable bounded-data path resolves canonical source configuration through
the provider registry and produces a Ray Dataset.

```text
CanonicalSourceInput -> ProviderRegistry -> DatasetHandle -> Ray Dataset
```

Current built-in providers cover:

- local and S3 Parquet or CSV files;
- Iceberg tables;
- ClickHouse queries;
- Doris queries.

Credentials belong to runtime configuration. They must not appear in dataset
identifiers, logs, persisted references, or error messages.

Bounded providers and unbounded stream sources are separate contracts. A data
provider does not imply that the same system is available as an inference
output sink.
