# Developer guide

Tributo keeps planning, data movement, execution, and publication contracts
separate. New integrations should implement the relevant protocol instead of
adding technology-specific branches to the CLI or pipelines.

Review the [architecture guide](../architecture/index.md) before changing
framework boundaries, and the [security guide](../security/index.md) before
changing credentials, artifacts, streaming, or publication behavior.
The [test execution policy](testing.md) defines CI-safe, scheduled, external,
and quarantined suites.

```{toctree}
:maxdepth: 1

documentation
../architecture/index
../integrations/index
testing
```

## Local checks

Install the project development environment:

```bash
uv sync --extra dev --locked
```

Run the normal unit suite:

```bash
python3 scripts/ci_test_plan.py run --suite unit
```

## Documentation environment

Use the independent documentation lock for the lightweight build:

```bash
uv venv .docs-venv --python 3.12
uv pip install --python .docs-venv/bin/python -r requirements-doc.lock
make strict SPHINXBUILD=.docs-venv/bin/sphinx-build
make spelling SPHINXBUILD=.docs-venv/bin/sphinx-build
```

Run the real-import integration build before publishing:

```bash
uv sync \
  --extra dev \
  --extra grpc \
  --extra training \
  --extra vector-index \
  --locked
uv pip install --python .venv/bin/python -r requirements-doc.lock
SPHINX_REAL_IMPORTS=1 make strict \
  SPHINXBUILD=.venv/bin/sphinx-build \
  BUILDDIR=docs/_build-real \
  HTMLDIR=docs/_build-real/html
make api-smoke PYTHON=.venv/bin/python
```

External link checking is separate because remote sites can fail
transiently:

```bash
make linkcheck SPHINXBUILD=.docs-venv/bin/sphinx-build
```

Python examples are syntax-checked in CI. Examples that submit work require a
reachable Ray cluster and are covered by the existing project integration
procedures rather than executed during an unprivileged RTD build.

Repository-backed local examples execute in CI. Examples that require Ray
Jobs, Docker, databases, S3, or another external service name the corresponding
integration evidence instead of contacting that infrastructure during Read the
Docs builds.
