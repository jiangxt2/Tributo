# Install Tributo

Tributo supports Python 3.12 and 3.13. Install the smallest dependency set for
your workload.

## Choose a source revision

Tributo 1.1.0 is distributed through PyPI and GitHub Releases. To install
the source release, clone its tag and use its committed lock file:

```bash
git clone --branch tributo-1.1.0 --depth 1 https://github.com/jiangxt2/Tributo.git
cd Tributo
```

The tag and a development checkout can declare the same package version while
containing different APIs in historical snapshots. Current `master` declares
`1.3.0.dev0`, while release branches use their selected formal version. Select
the checkout before preparing dependencies, and use the documentation,
metadata, and lock file from that revision. The development version does not
determine which patch or minor release will be published next.
The optional profiles below describe this documentation's source tree.

The development source targets Ray 2.59.0 and Daft 0.7.26. Connector extras
declare minimum stable versions: `ray-doris>=1.3`, `ray-hive>=1.2`,
`ray-clickhouse>=1.1`, and `daft-doris>=1.1`. These packages are available on
PyPI. The committed `uv.lock` selects their exact versions and hashes for the
development, CI, and image environment. Use the locked `uv sync` commands
below to reproduce that environment.

Bindings record installed versions and retain driver/worker consistency checks.
Each connector maintains its own Ray or Daft compatibility requirements. The
minimums describe supported starting versions, not an automatic promise that
every newer connector release has been validated. New releases do not
automatically raise the minimums or update the lock.

For an independently built algorithm package, use a reviewed Core/Wheel
combination. The [fixed-source boosting recipe](../algorithms/getting-started.md#build-and-install-the-example-package)
builds Core and the algorithm Wheel from explicit public revisions. It does
not establish that the algorithm Wheel works with the historical release tag.

## Install the core package

For an isolated PyPI installation, use a separate directory and virtual
environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install "tributo==1.1.0"
.venv/bin/tributo --help
```

For the selected source checkout, use its committed lockfile:

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
| ClickHouse via Ray Data | `uv sync --locked --no-dev --extra clickhouse` |
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
optional BayesOpt search implementation. In this locked checkout, the `clickhouse` extra installs the
shared ClickHouse driver and locked `ray-clickhouse==1.1` for the
explicit Ray read/write route. `mysql`
installs `daft-doris==1.1` and
`ray-doris==1.3` for their explicit engine routes, while `doris-flight` adds
their Flight dependencies. The `hive-ray` extra installs `ray-hive==1.2` for
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

Core's `graph` and `causal` extras install dependencies for Core-level
integration APIs. They do not install the separate `tributo-algorithms`
distribution. To install official algorithms, choose the matching aggregate
extra, such as `graph-pyg` or `causal-dowhy`; see the
[aggregate algorithm installation](../algorithms/getting-started.md#install-the-aggregate-distribution).

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
