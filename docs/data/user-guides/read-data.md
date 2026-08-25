# Read bounded data

Use the ingestion gateway when a workflow needs Ray Data or Daft input.

## Read local Parquet with Ray Data

```{literalinclude} ../../examples/doc_code/local_data.py
:language: python
:pyobject: read_local_parquet
```

The caller owns the open result and closes it. The receipt describes the
selected provider, binding, engine, source identity, and transform decision
without containing credentials.

## Select Daft explicitly

Set `engine="daft"` and install the `data-daft` extra. The result is a
`DaftDataFrameHandle`. Tributo does not convert that handle to Ray Data unless
the caller invokes an explicit documented adapter.

## Choose a source

Use built-in source models for Parquet, CSV, Iceberg, and structured SQL paths.
Use `ProviderSourceConfig` for a versioned third-party provider. Check the
[support matrix](../../reference/support-matrix.md) because adapter discovery
alone is not verification evidence.

## Read a HiveServer2 table with Ray Data

Install `tributo[hive-ray]`, then configure a structured database/table URI and
select Ray explicitly:

```python
from tributo.data import IngestionRequest, ProviderSourceConfig, open_ingestion

source = ProviderSourceConfig(
    provider="tributo.hive",
    uri="hive://hiveserver2.example:10000/analytics/events",
    options={"columns": ["id", "category"]},
)

with open_ingestion(IngestionRequest(source=source, engine="ray")) as result:
    dataset = result.handle.dataset
```

The built-in route accepts only structured table reads over binary
HiveServer2. Passwords use an environment-variable reference through
`password_env`; raw SQL, Daft Hive, native ORC/HDFS files, HTTP transport, TLS,
Kerberos, and writes are not part of this contract.

```{warning}
Do not place passwords, tokens, signed query strings, or URI user information
in a source that can appear in a receipt or log. Use the source's environment,
IAM, or storage-profile mechanism.
```
