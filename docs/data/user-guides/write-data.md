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
verified. Mode support is defined by the selected native Binding's capability
descriptor; there is no format-specific compatibility facade that changes that
boundary.

## Append a Ray Dataset to ClickHouse

Install the external `ray-clickhouse==0.1.0` wheel, then use the same
`WriteGateway` boundary as other native writers. The target URI must contain no
credentials. Pass the password as an environment-variable reference so it does
not enter plans, digests, receipts, or logs.

```python
from tributo.data import RayDataHandle
from tributo.data.writing import WriteMode, WriteRequest, default_write_gateway

request = WriteRequest(
    engine="ray",
    target_kind="clickhouse",
    target="clickhouse://ch.example:8123/analytics/inference_results",
    binding_id="tributo.ray.clickhouse",
    mode=WriteMode.APPEND,
    runtime_options={
        "user": "writer",
        "credential_ref": "env://CLICKHOUSE_PASSWORD",
    },
)
receipt = default_write_gateway().execute(request, RayDataHandle(dataset))
```

This adapter supports append to an existing table only. ClickHouse writes have
no Tributo exactly-once guarantee; an ambiguous native write fails closed.

## Interpret the receipt

`WriteReceipt.committed` reports the binding's terminal result. Row and byte
counts can be absent when the native API does not return them. The receipt does
not promise a fixed fragment count, exclusive create, or an engine-specific
snapshot unless the binding contract states it.
