# Write bounded data

Use `WriteGateway` to validate a write and delegate it to the selected native
engine. This Alpha API requires a typed handle produced by ingestion or another
documented engine boundary.

```{literalinclude} ../../examples/doc_code/local_data.py
:language: python
:pyobject: write_local_parquet
```

## Select a verified combination

Ray and Daft bindings cover different target and dependency combinations.
Parquet, CSV, Iceberg, and Lance also expose different mode and option sets.
Call `WriteGateway.plan()` when you need to inspect the selected descriptor
before execution.

The explicit gateway can support `APPEND` where the native engine contract is
verified. Legacy Parquet and CSV `DataConnector` facades reject append and do
not define the explicit gateway's capability boundary.

## Interpret the receipt

`WriteReceipt.committed` reports the binding's terminal result. Row and byte
counts can be absent when the native API does not return them. The receipt does
not promise a fixed fragment count, exclusive create, or an engine-specific
snapshot unless the binding contract states it.
