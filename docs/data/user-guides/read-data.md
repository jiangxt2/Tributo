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

## Bridge Daft input to Ray algorithms

Configure Daft's official Ray runner once during Ray Job/Application startup,
before creating or opening the Daft source:

```python
import daft

daft.set_runner_ray(address="auto", noop_if_initialized=True)
```

Then request the conversion explicitly at the algorithm input boundary:

```python
from tributo.data import IngestionRequest, ParquetSourceConfig
from tributo.integrations.algorithm_inputs import (
    IngestionInputInvocation,
)

invocation = IngestionInputInvocation(
    request=IngestionRequest(
        source=ParquetSourceConfig(path="s3://bucket/train.parquet"),
        engine="daft",
    ),
    handle_adapter_id="tributo.daft_to_ray",
)
```

The adapter uses Daft's `DataFrame.to_ray_dataset()` API. It may execute the
Daft lazy plan, does not preserve row order, and refuses the Native runner or
an unconfigured runner. The adapter does not configure a runner or silently
fall back between engines. The conversion receipt records the source and
target engines and its execution semantics.

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

## Read a Doris table with Ray Data

Install the `mysql` extra for the Ray Doris Binding, then use the Doris-only
typed read options:

```python
from tributo.data import (
    IngestionRequest,
    RayReadTaskOptions,
    SqlSourceConfig,
    open_ingestion,
)

source = SqlSourceConfig(
    dialect="doris",
    host="doris-fe.example",
    database="analytics",
    table="events",
    tablet_size=128,
    on_query_plan_error="error",
    ray_remote_args=RayReadTaskOptions(
        num_cpus=1,
        scheduling_strategy="SPREAD",
        max_retries=3,
    ),
)

with open_ingestion(IngestionRequest(source=source, engine="ray")) as result:
    dataset = result.handle.dataset
```

`tablet_size` and `on_query_plan_error` are delegated to `ray-doris`. Tributo
validates the Doris dialect and forwards only the typed Ray task subset:
`num_cpus`, `scheduling_strategy="SPREAD"`, and non-negative `max_retries`.
Other SQL dialects and arbitrary Ray resource dictionaries are rejected. The
Ray Doris route remains adapter-only until its real-database conformance gate
is satisfied.

```{warning}
Do not place passwords, tokens, signed query strings, or URI user information
in a source that can appear in a receipt or log. Use the source's environment,
IAM, or storage-profile mechanism.
```
