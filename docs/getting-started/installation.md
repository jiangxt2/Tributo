# Install Tributo

Tributo supports Python 3.12 and 3.13. Install the smallest dependency set for
your workload.

## Install the core package

```bash
python -m pip install tributo
```

The core package includes the Ray Jobs client, Ray Data, Ray Serve, Ray Tune,
Pydantic, ONNX Runtime, PyArrow, pandas, and S3 filesystem support.

## Select optional capabilities

| Workload | Installation |
| --- | --- |
| Ray Data table formats | `python -m pip install "tributo[data]"` |
| Daft ingestion | `python -m pip install "tributo[data,data-daft]"` |
| HiveServer2 via Ray Data connector package | `python -m pip install "tributo[hive-ray]"` |
| PostgreSQL ingestion | `python -m pip install "tributo[postgresql]"` |
| ClickHouse via Daft | `python -m pip install "tributo[clickhouse]"` |
| Doris via Daft/Ray Data | `python -m pip install "tributo[mysql]"` |
| Doris Flight via Daft/Ray Data | `python -m pip install "tributo[doris-flight]"` |
| Distributed training | `python -m pip install "tributo[training]"` |
| BayesOpt search for Ray Tune | `python -m pip install "tributo[tune]"` |
| Explainability | `python -m pip install "tributo[explainability]"` |
| Lance vector indexing | `python -m pip install "tributo[vector-index]"` |
| Torch model export | `python -m pip install "tributo[model-export-torch]"` |
| Hugging Face sources/exporters | `python -m pip install "tributo[model-export-hf]"` |
| MLflow registry | `python -m pip install "tributo[registry]"` |
| gRPC serving | `python -m pip install "tributo[grpc]"` |
| Kafka stream source | `python -m pip install "tributo[streaming-inference]"` |

An extra installs dependencies. It does not turn a protocol, adapter, or
reserved problem type into a verified implementation. Check the
[support matrix](../reference/support-matrix.md) before deployment. Ray Tune
itself is included by the core Ray dependency; the `tune` extra adds the
optional BayesOpt search implementation. The `clickhouse` extra installs
`daft-clickhouse==1.0`; `mysql` installs `daft-doris==1.0` and
`ray-doris==1.0` for their explicit engine routes, while `doris-flight` adds
their Flight dependencies. The `hive-ray` extra installs the external
`ray-hive==1.0` HiveServer2 connector package; it does not register a Tributo
Hive Provider or Binding. The equivalent uv commands are
`uv sync --extra clickhouse`, `uv sync --extra mysql`,
`uv sync --extra doris-flight`, and `uv sync --extra hive-ray`.

## Prepare a source checkout

```bash
git clone https://github.com/jiangxt2/tributo.git
cd tributo
uv sync --extra dev --locked
uv run --locked --no-sync tributo --help
```

Use `uv.lock` for development and runtime tests. Documentation uses the
separate `requirements-doc.lock` for the lightweight Read the Docs build.

## Use JSON configuration

Tributo rejects `.yaml` and `.yml` configuration files. Use JSON for persisted
job, algorithm, inference, explainability, and vector-index requests.
