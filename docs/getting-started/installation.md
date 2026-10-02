# Install Tributo

Tributo supports Python 3.12 and 3.13. Install the smallest dependency set for
your workload.

Tributo 1.0.0 is distributed as a source-only GitHub Release. Clone
the versioned tag and use the committed lock file:

```bash
git clone --branch tributo-1.0.0 --depth 1 https://github.com/jiangxt2/Tributo.git
cd Tributo
```

## Install the core package

```bash
uv sync --locked --no-dev
uv run --locked --no-sync tributo --help
```

The core package includes the Ray Jobs client, Ray Data, Ray Serve, Ray Tune,
Pydantic, ONNX Runtime, PyArrow, pandas, and S3 filesystem support.

## Select optional capabilities

| Workload | Installation |
| --- | --- |
| Ray Data table formats | `uv sync --locked --no-dev --extra data` |
| Daft ingestion | `uv sync --locked --no-dev --extra data --extra data-daft` |
| HiveServer2 via Ray Data connector package | `uv sync --locked --no-dev --extra hive-ray` |
| PostgreSQL ingestion | `uv sync --locked --no-dev --extra postgresql` |
| ClickHouse via Daft | `uv sync --locked --no-dev --extra clickhouse` |
| ClickHouse via Ray Data | Sync `clickhouse`, then install the external `ray-clickhouse==0.1.0` wheel into `.venv` |
| Doris via Daft/Ray Data | `uv sync --locked --no-dev --extra mysql` |
| Doris Flight via Daft/Ray Data | `uv sync --locked --no-dev --extra doris-flight` |
| Training data, export, and storage profile | `uv sync --locked --no-dev --extra training` |
| BayesOpt search for Ray Tune | `uv sync --locked --no-dev --extra tune` |
| Explainability | `uv sync --locked --no-dev --extra explainability` |
| Lance vector indexing | `uv sync --locked --no-dev --extra vector-index` |
| Torch model export | `uv sync --locked --no-dev --extra model-export-torch` |
| Hugging Face sources/exporters | `uv sync --locked --no-dev --extra model-export-hf` |
| MLflow registry | `uv sync --locked --no-dev --extra registry` |
| gRPC serving | `uv sync --locked --no-dev --extra grpc` |

Each row is an installation profile. `uv sync` reconciles the environment to
the selected extras, so combine multiple workloads in one command by providing
all required `--extra` flags together.

An extra installs dependencies. It does not turn a protocol, adapter, or
reserved problem type into a verified implementation. Check the
[support matrix](../reference/support-matrix.md) before deployment. Ray Tune
itself is included by the core Ray dependency; the `tune` extra adds the
optional BayesOpt search implementation. The `clickhouse` extra installs
`daft-clickhouse==1.0` and the shared ClickHouse driver dependencies. The Ray
route additionally requires the `ray-clickhouse==0.1.0` GitHub Release wheel
until its PyPI publication. `mysql` installs `daft-doris==1.0` and
`ray-doris==1.0` for their explicit engine routes, while `doris-flight` adds
their Flight dependencies. The `hive-ray` extra installs `ray-hive==1.0` for
the built-in Ray-only HiveServer2 Provider/Binding route. This does not add
Daft Hive, native ORC/HDFS access, raw SQL, or Hive writes.

The `training` extra aggregates Core's optional data, export, and storage
dependencies for training workloads. It does not install XGBoost or any
official algorithm implementation. Install a compatible, independently
versioned algorithm Wheel before using `tributo algo run`, then confirm
discovery with `uv run --locked --no-sync tributo algo list --json`.

## Add development dependencies

```bash
uv sync --locked --extra dev
uv run --locked --no-sync tributo --help
```

Use `uv.lock` for development and runtime tests. Documentation uses the
separate `requirements-doc.lock` for the lightweight Read the Docs build.

## Use JSON configuration

Tributo rejects `.yaml` and `.yml` configuration files. Use JSON for persisted
job, algorithm, inference, explainability, and vector-index requests.
