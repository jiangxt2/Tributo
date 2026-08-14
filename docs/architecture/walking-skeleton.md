# Walking Skeleton

The walking skeleton is a real distributed external validation for the
model-export architecture:

```text
isolated Docker Compose project
  → Ray Jobs API
  → Ray Data Parquet
  → XGBoostTrainerImpl
  → S3 Bundle (ONNX + UBJ)
  → BundleReader
  → Ray Data map_batches
  → Ray Serve HTTP
```

The same job records committed-Bundle provenance through the required MLflow
Hook. It verifies that the Hook records the Bundle URI, manifest digest, tags,
and artifacts without creating an MLflow Model Version.

## Runtime Topology

`tests/integrations/docker-compose.data-ingestion.yml` starts one Ray head, one
independent Ray worker, MinIO, and MLflow under the `model-export` profile. The
test submits `tests/integration/jobs/model_export_architecture_job.py` through
the Ray Jobs HTTP API; it does not start a host-local Ray runtime or execute the
job directly in the head container.

No service publishes a fixed host port and no service has a fixed container
name. The Compose project name is unique per run, so the gate cannot attach to,
restart, or remove an existing container.

## Contracts Exercised

| Boundary | Evidence |
|----------|----------|
| Version contract | The job checks Python, Ray, MLflow, boto3, botocore, XGBoost, ONNX, ONNX Runtime, onnxmltools, Torch, Transformers, PyArrow, and pandas against `component-versions.env`; image references are digest-pinned. |
| Training lifecycle | A first-party XGBoost trainer receives an explicit Bundle URI, uses its default ONNX opset 12 and UBJ targets, and returns the stable `TrainingResult` fields. |
| Bundle publication | Both required artifacts commit to MinIO in one immutable Bundle and expose `inference → onnx-model`. |
| Integrity | `BundleReader` verifies the exact committed manifest bytes and artifact digests before materialization. |
| Batch inference | The ONNX artifact is loaded through the Bundle role and used by Ray Data `map_batches`. |
| Online inference | Ray Serve loads the same Bundle and returns predictions through HTTP. |
| Provenance | The explicit required MLflow Hook records one run and zero Model Versions. |
| Distributed execution | Ray reports both the head and the independent worker alive while the Ray Job runs. |

The first-party conformance suite runs in the same pinned Linux image before
the golden path. It covers the XGBoost ONNX/UBJ/JSON exporters, Torch
ONNX/Safetensors/PT2 exporters, Hugging Face ONNX exporter, ONNX quantizer,
Structure and ONNX Runtime validators, and Ray checkpoint source providers.

## Authoritative Component Versions

All model-export IT components are defined in
`tests/integrations/component-versions.env`. Package values must match
`uv.lock`; the Ray, uv, MinIO, and PostgreSQL images must also include a
SHA-256 digest. The Ray Job compares its actual Python major/minor version with
`PYTHON_VERSION`, while the static contract requires the same version tag in
the pinned Ray image.
`tests/integration/test_it_component_versions.py` fails when Docker build or
Compose configuration bypasses that contract.

The IT environment never installs a floating dependency and never inherits a
host package version as test evidence.

## External Validation Gate

`ci/test-suites.json` reports this validation when relevant code changes. No
GitHub Actions event runs it. From an approved Docker environment, the
lifecycle-owned runner creates a unique Compose project, executes the pinned
component check and first-party conformance suites, submits the Ray Job, and
saves service logs:

```bash
./scripts/run_model_export_it.sh --suite ci
```

The historically named `ci` subset includes the component-version contract,
first-party conformance, real MLflow Hook contract, and distributed walking
skeleton. The name describes its bounded contents and does not make it a CI
job. The runner always performs exact-project cleanup equivalent to:

```bash
docker compose \
  --env-file tests/integrations/component-versions.env \
  --file tests/integrations/docker-compose.data-ingestion.yml \
  --profile model-export \
  down --volumes --remove-orphans
```

The cleanup command is scoped by the run-specific `COMPOSE_PROJECT_NAME`.

## Full-Flow External Verification

The same lifecycle-owned runner exposes a release-oriented full suite that is
also excluded from every GitHub Actions workflow:

```bash
./scripts/run_model_export_it.sh --suite full
```

It performs prerequisites checks, snapshots the ID and state of every existing
container, uses a unique Compose project and the validated content-addressed
data-ingestion runtime image, writes pytest and service logs to
`/tmp/<project-name>/`, and installs an `EXIT` trap before any container is
started. In addition to the bounded subset, full mode runs the trainer Bundle contract
and complete S3/MinIO contract suites. The trap always removes only that
project's containers, volumes, network, and orphans. It then fails if an owned
container remains or reports if concurrent host activity changed a pre-existing
container.

## Deliberate Exclusions

| Excluded | Boundary |
|----------|----------|
| GPU training | The external validation is CPU-only. |
| DNN/PU full training | Their Bundle vertical slices and exporter/source conformance are separate integration tests. |
| Streaming | Streaming has a separate lifecycle and is not part of model-export publication. |
| MLflow Model Version/Alias | This cycle provides Bundle provenance only. |
| Cross-process Hook recovery | PostgreSQL, Outbox, and asynchronous workers require a separate scope amendment. |

<!-- END -->
