# Write bounded data

Use `WriteGateway` to validate a write and delegate it to the selected native
engine. This Alpha API requires a typed handle produced by ingestion or another
documented engine boundary.

```{literalinclude} ../../examples/doc_code/local_data.py
:language: python
:pyobject: write_local_parquet
```

## Select a verified combination

Ray bindings cover different target and dependency combinations.
Parquet, CSV, Iceberg, and Lance also expose different mode and option sets.
Call `WriteGateway.plan()` when you need to inspect the selected descriptor
before execution.

The explicit gateway can support `APPEND` where the native engine contract is
verified. Mode support is defined by the selected native Binding's capability
descriptor; there is no format-specific compatibility facade that changes that
boundary.

## Append a Ray Dataset to ClickHouse

Install the connector selected by the lock with
`uv sync --locked --no-dev --extra clickhouse`, then use the same
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

## Write a Ray Dataset to Doris

Install `ray-doris` with `uv sync --locked --no-dev --extra mysql`. Use a
credential-free Doris target URI whose port is the FE HTTP port. The password
reference names an environment variable available to the driver and Ray workers.

```python
from tributo.data import RayDataHandle
from tributo.data.writing import WriteMode, WriteRequest, default_write_gateway

request = WriteRequest(
    engine="ray",
    target_kind="doris",
    target="doris://fe.example:8030/analytics/results",
    binding_id="tributo.ray.doris",
    mode=WriteMode.APPEND,
    options={"operation": "upsert", "mysql_port": 9030},
    runtime_options={"user": "writer", "credential_ref": "env://DORIS_PASSWORD"},
)
receipt = default_write_gateway().execute(request, RayDataHandle(dataset))
```

`operation` defaults to `load`; `upsert` requires a Unique Key table and
`partial_update` requires a Merge-on-Write Unique Key table. The connector
validates the table and schema and chooses the required format. Tables must
already exist. The binding forwards its declared batching, connection, TLS,
redirect, and Stream Load options to `ray-doris`; inspect `plan()` for the
supported option names. Redirect hosts and ports must be included in the allowed
lists when the FE redirects to a different endpoint.

For a confirmed success, `rows_written` and `bytes_written` contain Doris
loaded rows and uploaded bytes; receipt metadata includes batch, attempted and
filtered row counts. A publish timeout means Doris completed the load, but
visibility may be delayed. Wait or check visibility; do not resubmit the write.
The same applies when some batches succeeded and others returned publish
timeouts. For an uncertain batch outcome or a Ray worker crash,
`WriteAmbiguousError` requires callers to reconcile what was written before
deciding whether to retry. Known failures raise `WriteBindingError`. Multiple
batches can partially succeed even when a failure is known, so callers must
reconcile the target before retrying the dataset. The connector disables Ray
write-task retries; no whole-dataset atomicity or exactly-once guarantee is
provided.

## Interpret the receipt

`WriteReceipt.committed` reports the binding's terminal result. Row and byte
counts can be absent when the native API does not return them. The receipt does
not promise a fixed fragment count, exclusive create, or an engine-specific
snapshot unless the binding contract states it.
