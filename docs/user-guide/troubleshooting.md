# Troubleshooting

## The CLI cannot reach Ray

Confirm that the dashboard endpoint is reachable from the CLI host:

```bash
curl http://127.0.0.1:8265/api/version
```

The HTTP dashboard address is different from a Ray Client address. Tributo job
management uses the Ray Jobs API.

## A configuration file is rejected

Tributo accepts JSON and rejects files ending in `.yaml` or `.yml`. Convert the
configuration to JSON and validate that field names match the relevant
Pydantic model.

## An optional module cannot be imported

Install the extra associated with the feature:

```bash
python -m pip install "tributo[training]"
```

Missing optional dependencies should fail with an installation hint. Do not
install every extra by default in a production image.

## A job works locally but fails on the cluster

The Ray worker environment is distinct from the submitting process. Include
the package and runtime dependencies in the Ray `runtime_env`, or build them
into the cluster image. Tributo's shared runtime environment builder does not
derive `site-packages` paths from the submitting host or implicitly combine
multiple Python environments through `PYTHONPATH`. An explicitly configured,
cluster-visible `PYTHONPATH` remains supported for compatibility. Avoid passing
credentials in URIs or log messages.

## S3 access fails

Prefer workload identity or environment-based credentials. If a storage
profile is required, ensure it is available to the Ray workers. Dataset
references and bundle manifests must remain credential-free.

## PU training rejects a distributed configuration

The formal `pu` algorithm uses the same Ray Train/PyTorch DDP kernel as DNN and
accepts multiple workers. It fails closed when a shard cannot provide both
positive and unlabeled observations for globally coordinated PU batches, when
the requested resources are unavailable, or when multi-worker BatchNorm is
enabled. Reduce the worker count, provide enough examples of both observed
classes, or disable `model.use_batch_norm`.

## A bundle cannot be loaded

Check:

- the manifest exists at the canonical bundle URI;
- every artifact digest matches the manifest;
- the requested role points to an existing artifact;
- the artifact flavor has a registered loader;
- the process has access to the matching storage profile.

Safetensors, PT2, native XGBoost, and quantized artifacts require matching
flavor implementations before they can enter the unified serving path.

## Documentation fails only without mocks

Run the real-import build in the project environment:

```bash
uv sync --extra dev --extra training --locked
uv pip install --python .venv/bin/python -r requirements-doc.lock
SPHINX_REAL_IMPORTS=1 make strict \
  SPHINXBUILD=.venv/bin/sphinx-build \
  BUILDDIR=docs/_build-real \
  HTMLDIR=docs/_build-real/html
```

The lightweight build may mock third-party packages. It never mocks
`tributo.*`; the real-import build is the guard against a mock hiding a package
import regression.
