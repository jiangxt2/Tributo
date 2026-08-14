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

```{warning}
Do not place passwords, tokens, signed query strings, or URI user information
in a source that can appear in a receipt or log. Use the source's environment,
IAM, or storage-profile mechanism.
```
