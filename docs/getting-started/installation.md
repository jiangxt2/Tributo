# Install Tributo

Tributo supports Python 3.12 and 3.13. Install the smallest dependency set for
your workload.

## Choose a source revision

Tributo 1.0.0 is distributed as a source-only GitHub Release. To install that
release, clone its tag and use its committed lock file:

```bash
git clone --branch tributo-1.0.0 --depth 1 https://github.com/jiangxt2/Tributo.git
cd Tributo
```

The tag and a development checkout can declare the same package version while
containing different APIs. Select the checkout before preparing dependencies,
and use the documentation, metadata, and lock file from that revision.
The optional profiles below describe this documentation's source tree.

For an independently built algorithm package, use a reviewed Core/Wheel
combination. The [fixed-source boosting recipe](../algorithms/getting-started.md#build-and-install-the-example-package)
builds Core and the algorithm Wheel from explicit public revisions. It does
not establish that the algorithm Wheel works with the historical release tag.

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
| ClickHouse via Ray Data | `uv sync --locked --no-dev --extra clickhouse`, then `uv pip install --python .venv/bin/python ray-clickhouse==1.0` |
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
optional BayesOpt search implementation. The `clickhouse` extra installs the
shared ClickHouse driver dependency. ClickHouse reads use the Ray route, which
additionally requires the separately installed `ray-clickhouse==1.0` package
from PyPI; this connector remains outside the Tributo lockfile. `mysql`
installs `daft-doris==1.0` and
`ray-doris==1.0` for their explicit engine routes, while `doris-flight` adds
their Flight dependencies. The `hive-ray` extra installs `ray-hive==1.0` for
the built-in Ray-only HiveServer2 Provider/Binding route. This does not add
Daft Hive, native ORC/HDFS access, raw SQL, or Hive writes.

The `training` extra aggregates Core's optional data, export, and storage
dependencies for training workloads. It does not install XGBoost or any
official algorithm implementation. Install a compatible, independently
versioned algorithm Wheel before using `tributo algo run`, then confirm
discovery with `uv run --locked --no-sync tributo algo list --json`.

Complete all required `uv sync` steps before installing an external algorithm
Wheel into the checkout environment. A later exact sync removes packages
outside the project's selected dependency set. Use `uv run --locked --no-sync`
for subsequent repository commands, or repeat the reviewed Wheel installation
after changing extras. For a separate Wheel environment, use that environment's
Python and CLI directly, as shown in the algorithm recipe.

## Add development dependencies

```bash
uv sync --locked --extra dev
uv run --locked --no-sync tributo --help
```

Use `uv.lock` for development and runtime tests. Documentation uses the
separate `requirements-doc.lock` for the lightweight Read the Docs build.
Follow [CONTRIBUTING](https://github.com/jiangxt2/Tributo/blob/master/CONTRIBUTING.md) for hooks, controlled test
selection, and the repository PR check. Core's development environment does
not install official algorithm implementations.

## Use JSON configuration

Tributo rejects `.yaml` and `.yml` configuration files. Use JSON for persisted
job, algorithm, inference, explainability, and vector-index requests.
