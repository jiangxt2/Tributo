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

## Tune a Lance scan with Ray Data

Lance scanner controls are available only with `engine="ray"`. Pass the
bounded `scanner_options` object through the `tributo.lance` provider:

```python
from tributo.data import IngestionRequest, ProviderSourceConfig, open_ingestion

source = ProviderSourceConfig(
    provider="tributo.lance",
    uri="/data/images.lance",
    options={
        "columns": ["id", "image_bytes"],
        "filter": "id > 0",
        "scanner_options": {
            "batch_size": 128,
            "batch_readahead": 1,
            "fragment_readahead": 0,
            "io_buffer_size": 32 * 1024 * 1024,
            "late_materialization": ["image_bytes"],
        },
    },
)

with open_ingestion(IngestionRequest(source=source, engine="ray")) as result:
    dataset = result.handle.dataset
```

The accepted scanner fields are `batch_size`, `batch_readahead`,
`fragment_readahead`, `io_buffer_size`, and `late_materialization`. They tune
scan execution and do not change the source identity. The Daft binding rejects
non-empty `scanner_options`. These settings are not a hard process-memory
limit; `batch_readahead` must be positive, while `fragment_readahead` can be
zero. Keep `io_buffer_size` large enough for the largest scanned batch.

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

## Read a ClickHouse table with Ray Data

Run `uv sync --locked --no-dev --extra clickhouse` in the source checkout,
then select Ray explicitly. The lock selects the stable version
`ray-clickhouse==1.1`; see the installation requirements in the
[installation guide](../../getting-started/installation.md).

```python
from tributo.data import (
    IngestionRequest,
    ReadOptions,
    SqlPartitioning,
    SqlSourceConfig,
    open_ingestion,
)

source = SqlSourceConfig(
    dialect="clickhouse",
    host="clickhouse.example",
    database="analytics",
    table="events",
    columns=["id", "category"],
    partitioning=SqlPartitioning(mode="auto", num_partitions=8),
)

request = IngestionRequest(
    source=source,
    engine="ray",
    read_options=ReadOptions(
        batch_size=32_768,
        target_split_size_bytes=32 * 1024 * 1024,
        concurrency=4,
    ),
)
with open_ingestion(request) as result:
    dataset = result.handle.dataset
```

`single` maps to one streaming ClickHouse query, `auto` requests physical
partition splitting, and `parallel` maps its one declared integer column to
range splitting. Views and Distributed tables must use `single`; physical
partition and range splitting are limited to eligible direct MergeTree-family
tables. `batch_size` and `target_split_size_bytes` become the connector's row
and byte block targets. Raw SQL remains outside Tributo's ingestion contract.

## Filter Doris and ClickHouse reads at the source

Use the existing structured Transform pipeline. The Ray bindings forward a
supported leading filter prefix to `ray-doris` or `ray-clickhouse`; callers do
not provide database-specific SQL.

```python
from tributo.data import FilterComparison, FilterIsIn, TransformPipeline

request = IngestionRequest(
    source=source,
    engine="ray",
    transforms=TransformPipeline(
        steps=(
            FilterComparison(column="id", operator="gte", value=9900),
            FilterIsIn(column="category", values=("east", "west")),
        )
    ),
)
with open_ingestion(request) as result:
    decisions = result.receipt.transform_decisions
    rows = result.handle.dataset.take_all()
```

The complete pipeline is validated against the source-projected schema before
filtered reader creation. Schema discovery uses metadata without fetching a
sample; the final filtered reader must expose the same input schema.

| Input and predicate | Source compilation |
| --- | --- |
| Integer column and integer scalar | Equality, inequality, ordered comparison, inclusive range, and IN |
| Float64 column with finite float scalars or integer scalars within the exact Float64 integer range | Equality, inequality, ordered comparison, inclusive range, and IN |
| Boolean column and Boolean scalar | Equality, inequality, and IN |
| UTF-8 string column and string scalar | Equality, inequality, and IN; compare the textual output while preserving NULL |
| NULL / non-NULL and empty IN | IS NULL / IS NOT NULL; empty IN is always false |
| Other type/operator combinations | Ray residual execution |

The first unsupported filter or non-filter transform ends the pushdown prefix.
Later filters remain in their original position. This includes filters after
projection, rename, cast, null filling, or limit. Float32, Decimal, date/time
comparisons, string ordering, and implicit scalar coercion remain residual.
Existing source column projections are not widened to admit a missing field.

```{warning}
Ray 2.59.0 can fail with `UnicodeEncodeError` when a residual expression
such as `text == "中文"` puts a Unicode literal in an operator's metric
tag. [Ray issue #67230](https://github.com/ray-project/ray/issues/67230) tracks
this limitation. A supported leading filter executed by the database avoids
that particular Ray Filter, while residual filters after projection, cast,
or limit can still fail. Keep the original transform order when planning reads.
```

Each receipt decision retains its original ordinal. `exact` means the source
enforces that step and Ray does not repeat it; `none` means Ray still executes
it. A credential-free diagnostic explains the boundary. Receipts do not contain
generated SQL or parameter values. Query planning, authentication, and read
errors propagate; a failed filtered read is not retried with its filter removed.

The owned `database-filter` CI suite uses one Docker Ray node with a real
ClickHouse instance or Doris FE/BE pair. It observes successful real driver
execution calls and checks the native Dataset output without a Ray residual
filter. SQL drivers, connections, and data responses are not mocked. Evidence
includes the source snapshot, versions, SQL execution trace, exact ordinals,
and results; skips or missing evidence fail validation. Index pruning and
elapsed-time improvements are separate performance observations.

Run the same controlled validation locally:

```bash
./scripts/run_database_filter_it.sh --database clickhouse
./scripts/run_database_filter_it.sh --database doris
```

The lifecycle uses a unique Compose project, locked dependencies and pinned
images. It retains evidence and removes only its own service resources.

```{warning}
Do not place passwords, tokens, signed query strings, or URI user information
in a source that can appear in a receipt or log. Use the source's environment,
IAM, or storage-profile mechanism.
```
