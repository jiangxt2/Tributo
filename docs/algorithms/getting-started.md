# Install and inspect a formal algorithm

Tributo Core provides the execution framework. The
[official algorithm repository](https://github.com/jiangxt2/tributo-algorithms)
publishes one independently versioned distribution, `tributo-algorithms`, with
optional domain extras. Other trusted providers may publish their own Wheels.
Registry discovery does not install packages.
The Core `training` extra prepares framework dependencies; it does not
install an algorithm implementation.

Use the [Core data quickstart](../getting-started/quickstart.md) for local
Parquet read/write, or follow the installation-only recipe below to prepare
an algorithm package.

## Install the aggregate distribution

The aggregate Wheel contains all 37 official algorithm entry points and their
lightweight descriptors. The catalog groups implementations of the same
algorithm under one logical name. Extras
install the runtime dependencies needed for selected domains, for example
`tributo-algorithms[boosting]` or `tributo-algorithms[causal-xlearner]`.
Installing one extra does not remove the other descriptors from the catalog;
executing an algorithm requires its matching extra. Current algorithms require
Core 1.1 or newer; choose Core within the selected Wheel's declared range.

Use Python 3.12 or 3.13. Once the selected aggregate version is published,
install it from PyPI with the domain extra:

```bash
python -m pip install "tributo-algorithms[boosting]==0.1.0"
python -m pip check
```

Before publication, build a Wheel from a reviewed aggregate source checkout.
Set `TRIBUTO_ALGORITHMS_ROOT` to its absolute path, and run the following in
an empty directory outside the Core and algorithm checkouts:

```bash
git -C "${TRIBUTO_ALGORITHMS_ROOT}" rev-parse HEAD
uv build --no-sources --wheel --python 3.12 --out-dir artifacts \
  --project "${TRIBUTO_ALGORITHMS_ROOT}"
shasum -a 256 artifacts/*.whl
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "artifacts/tributo_algorithms-0.1.0-py3-none-any.whl[boosting]"
uv pip check --python .venv/bin/python
```

This example uses aggregate version `0.1.0`; match the Wheel filename and
version to the selected checkout. Build from the repository root:
`packages/boosting` contains a source domain, not an independent distribution.
Keep the source revision, Wheel digest, and resolved environment with the
deployment. Follow the algorithm repository's platform-specific dependency
instructions when selecting other extras, including its SciPy Wheel guidance
for macOS arm64/Python 3.12 with `causal-dowhy` or `all`.

Use `tributo-algorithms[graph-pyg]` or the appropriate `causal-*` extra for
official graph and causal algorithms. Core's `graph` and `causal` extras are
separate Core integration profiles; they do not install the aggregate
`tributo-algorithms` distribution.

These commands install packages without starting Ray. Installation and
descriptor discovery do not award a validated cluster profile; use the
[support matrix](../reference/support-matrix.md) for execution scope.

## Historical source snapshot

The recipe below retains the independently packaged boosting source snapshot.
Its commands apply to the pinned revisions, rather than the aggregate layout
above.

### Choose the source and package combination

This recipe binds explicit source revisions because the Core package version
alone does not identify its APIs. It does not certify the same algorithm
Wheel against the historical `tributo-1.0.0` tag.

| Component | Source revision | Package version |
| --- | --- | --- |
| Tributo Core | `ff0ee58cd37d7288e19747b6de22f48bcd4a49b0` | `1.0.0` |
| Official boosting package | `71a6a16c3fd3676aba2225a0c528817634927fc7` | `tributo-algorithms-boosting==0.1.0` |

Installation and descriptor conformance were checked with Python 3.12.12 on
macOS arm64, Ray 2.55.1, XGBoost 3.4.1, LightGBM 4.7.0, and onnxmltools 1.16.0.
The package installs both `xgboost` and `lightgbm` catalog entries. These
checks cover packaging and discovery, not training, model export, inference,
or a validated cluster profile. See the
[support matrix](../reference/support-matrix.md) and
[stability inventory](../STABILITY.md) for execution and compatibility scope.

### Build and install the example package

Install uv and an available Python 3.12 interpreter. Run these commands in an
empty directory outside either source checkout. Fetch both public snapshots:

```bash
TRIBUTO_CORE_REV=ff0ee58cd37d7288e19747b6de22f48bcd4a49b0
TRIBUTO_ALGORITHMS_REV=71a6a16c3fd3676aba2225a0c528817634927fc7

curl --fail --location \
  "https://github.com/jiangxt2/Tributo/archive/$TRIBUTO_CORE_REV.tar.gz" \
  --output core-source.tar.gz
curl --fail --location \
  "https://github.com/jiangxt2/tributo-algorithms/archive/$TRIBUTO_ALGORITHMS_REV.tar.gz" \
  --output algorithms-source.tar.gz
tar -xzf core-source.tar.gz
tar -xzf algorithms-source.tar.gz
```

Build just Core and the boosting package. `--no-sources` avoids local build
dependency overrides; this path does not require the algorithm workspace's
editable Core checkout.

```bash
uv build --no-sources --wheel --python 3.12 --out-dir artifacts \
  "Tributo-$TRIBUTO_CORE_REV"
uv build --no-sources --wheel --python 3.12 --out-dir artifacts \
  "tributo-algorithms-$TRIBUTO_ALGORITHMS_REV/packages/boosting"
shasum -a 256 artifacts/*.whl
```

Keep the Wheel digests with the source revisions. A package version does not
identify the exact built bytes. See the
[uv package build guide](https://docs.astral.sh/uv/guides/package/) for build
options.

Export the selected package's locked third-party dependencies. Omitting
workspace and local packages prevents a developer-only path from entering
the installation requirements:

```bash
(
  cd "tributo-algorithms-$TRIBUTO_ALGORITHMS_REV"
  uv export --frozen --no-dev --package tributo-algorithms-boosting \
    --no-emit-workspace --no-emit-local --no-hashes \
    --output-file ../boosting-requirements.txt
)
```

Create an isolated environment and install the reviewed Wheels with those
constraints:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --constraint boosting-requirements.txt \
  "artifacts/tributo-1.0.0-py3-none-any.whl[model-export]"
uv pip install --python .venv/bin/python --constraint boosting-requirements.txt \
  artifacts/tributo_algorithms_boosting-0.1.0-py3-none-any.whl
uv pip check --python .venv/bin/python
```

These commands install dependencies and packages without starting Ray or
submitting a task. Other platforms need their own dependency and execution
validation; installation on one platform is not a portability Gate.

## Inspect the installed package

Use the isolated environment directly:

```bash
.venv/bin/tributo --help
.venv/bin/tributo algo list --json
.venv/bin/tributo algo info xgboost
```

Before installing an algorithm Wheel, an empty catalog is expected. The
aggregate distribution exposes 37 entry points, including implementations of `xgboost` and
`lightgbm`; the historical boosting snapshot exposes only its own entries.
An available entry means discovery succeeded. The `tested`, `supported`, and validated
profile fields depend on the separate execution evidence model; installing
or inspecting a package does not update them.

In a Core source environment, use
`uv run --locked --no-sync tributo algo list --json` instead. Complete all
required extras syncing before installing external Wheels. A later exact
sync can remove them; see the [installation guide](../getting-started/installation.md).

## Prepare for execution

A formal request declares the algorithm, operation, profile, worker count,
bounded input, algorithm configuration, and reviewed resource settings.
The algorithm configuration supplies its explicit Bundle destination.
Honor the package's worker range; this XGBoost descriptor requires at least
two workers. The profile must match the selected runtime and its validation
evidence.

You can inspect the outer execution request's JSON Schema without executing
a task:

```bash
.venv/bin/python -I -c \
  'import json; from tributo.config import AlgorithmExecutionConfig; print(json.dumps(AlgorithmExecutionConfig.model_json_schema(), indent=2))'
```

This describes the Core request envelope. Algorithm-specific configuration
uses the package's declared contracts. The `algo config-schema` and
`algo validate` commands require an algorithm with a `config_model`; the
boosting descriptors above do not declare one. For example,
`algo config-schema xgboost` rejects the request with
`does not declare a config_model`. Do not treat those commands as a generic
formal-request validation path.

`tributo algo run --config execution.json` validates while planning and then
runs or submits work. It is not a dry run. Submit cluster workloads through
Ray Jobs using compatible algorithm artifacts and the declared runtime
profile. The [inference guide](../how-to/inference.md) explains the separate
Bundle consumption path.
