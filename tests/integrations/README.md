# Tributo Integration Tests

> Most end-to-end scripts run inside a Docker Ray cluster. The MLflow Hook suite
> is collected by pytest and runs against a real MLflow Tracking Server.

---

## Test List

| Test | File | Coverage | Prerequisites |
|------|------|----------|---------------|
| MLflow Hook | `test_e2e_mlflow.py` | Committed Bundle upload, replay deduplication, explicit run reuse, and failure semantics | MLflow + `registry` extra |
| ClickHouse E2E | `test_e2e_clickhouse.py` | ClickHouse table → Daft OLAP Binding → explicit Daft-to-Ray adapter → XGBoost distributed training → MLflow → ONNX | Ray + Daft + `daft-olap-connectors` + ClickHouse + MLflow |
| Dual-engine Docker | `test_data_ingestion_dual_engine.py` | Local Parquet, full ETL chain, typed handles, worker-version evidence | Docker Ray cluster + Daft |
| File conformance | `../integration/test_data_ingestion_conformance.py` | Local/MinIO Parquet and CSV through Ray Data and Daft | Local Ray runtime + MinIO |
| Table conformance | `../integration/test_table_format_ingestion.py` | Local/MinIO Iceberg and Lance through Ray Data and Daft | Local Ray runtime + MinIO |
| PostgreSQL conformance | `../integration/test_postgresql_ingestion.py` | Structured table read through Ray Data and Daft | Local Ray runtime + PostgreSQL |
| Inference Ray Jobs | `../integration/test_inference_ray_jobs.py` | Bundle, real post-training inline/detached inference, MLflow import, external artifacts, retry identity, credential domains, empty/NaN behavior | Isolated version-locked Docker Ray + MLflow + MinIO |
| Streaming | `test_e2e_streaming.py` | Streaming inference service | TBD |
| Tune | `test_e2e_tune.py` | Hyperparameter search | TBD |

---

## Inference Ray Jobs Suite

Use only the lifecycle-owned runner:

```bash
./scripts/run_inference_it.sh
```

Do not invoke `test_inference_ray_jobs.py` against a developer Ray cluster.
The test module fails when the Compose ownership marker is absent. The runner
uses `inference-it-versions.conf`, creates a unique Compose project, exposes no
host ports, and runs pytest inside `ray-head`; the head has zero Ray CPUs so
model actors execute on the independent worker.

An EXIT/INT/TERM trap captures logs and executes project-scoped `down
--volumes --remove-orphans`. CI repeats the same exact-project cleanup with an
`always()` step. Both paths verify that project-labelled resources are gone;
no prune, global deletion, or shared-image cleanup is permitted. Test and
service logs remain under `/tmp/<compose-project>-*.log` after cleanup.

---

## MLflow Hook Suite

The Hook suite validates the committed-bundle integration rather than automatic
Model Registry or Stage behavior. Missing infrastructure is a test failure, not
a reason to skip the suite.

Start and verify the existing local service:

```bash
docker start pista-mlflow-server
curl --fail 'http://127.0.0.1:8050/api/2.0/mlflow/experiments/search?max_results=1'
```

Run the dedicated suites:

```bash
uv run --extra registry pytest tests/integrations/test_e2e_mlflow.py \
  -m integration -vv
uv run --extra registry pytest tests/registry/test_integration.py \
  -m integration -vv
```

Set `MLFLOW_TRACKING_URI` to use another real server. The default is
`http://127.0.0.1:8050`.

---

## Required Data Ingestion Gate

Run the complete Data Ingestion Docker gate from the repository root:

```bash
./scripts/run_data_ingestion_it.sh
```

This is the only supported lifecycle entry for
`test_data_ingestion_dual_engine.py`. It computes a dependency-only runtime
key from the runtime profile, Dockerfile, `pyproject.toml`, `uv.lock`, version
contract, and Docker platform. A matching
`tributo-it-runtime:data-ingestion-<runtime-key>` image is validated and reused;
when missing, exactly one process builds it through a profile/key/platform/
daemon-scoped file lock and a single `docker buildx build --load` output.

Compose never builds or implicitly pulls an image. The runner explicitly
prepares digest-pinned infrastructure images, copies the checkout once through
`source-init` into a run-scoped source volume, and mounts that snapshot read-only
on the Ray head and worker. The snapshot also projects deterministic
`importlib.metadata` and entry-point metadata from `pyproject.toml`, so Tributo
remains source-delivered without installing it into the dependency runtime.
Test data, caches, and temporary files use a separate writable volume. The suite
records the long-running container IDs and source digest, reuses that same
cluster for all test groups, and then removes only its unique Compose project's
containers, network, and volumes.

The stable runtime and third-party images remain available for later runs.
Before preparation and after scoped cleanup, the runner compares Docker image
IDs and fails if the run added a dangling or `<none>` image; it never invokes a
global prune or deletes an unattributed image. Runtime retention can be audited
without deletion:

```bash
python3 tools/tributo_it.py runtime-gc-dry-run --profile data-ingestion
```

For a source-only change, the runtime key remains unchanged and the existing
runtime is reused. Set `TRIBUTO_IT_RUNTIME_REGISTRY` to an immutable GHCR
repository to prefer a published runtime before the local Buildx fallback.

The PR workflow runs two required ingestion gates. The semantic gate executes
the file, table-format, and PostgreSQL Conformance files with locked `data`,
`data-daft`, and `postgresql` extras. The distributed gate obtains one
content-addressed runtime, starts one Ray head, one Ray worker, and MinIO with
`docker-compose.data-ingestion.yml`, freezes the checked-out source in a
run-scoped read-only volume, then runs this module with the Daft Ray runner and
mandatory Driver/Worker version and snapshot evidence. Both jobs feed
`core-gate`; infrastructure absence is a failure rather than a passing skip.

## Redis Stream Message Format (Protocol v2.0)

**Training Task** (downstream → Tributo):

```
XADD tributo:training:tasks * \
  job_id "train-job-001" \
  payload '{"job_id":"train-job-001","algorithm":{...},...}'
```

**Training Event** (Tributo → downstream):

```
XADD tributo:aimodel:training:events:train-job-001 * \
  job_id "train-job-001" \
  payload '{"protocol_version":"2.0","event_type":"COMPLETED",...}'
```

Each stream entry contains exactly two fields: `{job_id, payload}`.
`payload` is the full JSON string of the business object — field flattening is forbidden.

---

## Troubleshooting

The Data Ingestion runner prints every lifecycle command and preserves the test
and service logs under `/tmp/tributo-it-logs/<compose-project>-*.log`. Do not
start, rebuild, or clean its Compose file manually: rerun the lifecycle-owned
entry after inspecting those logs. A failed run still performs exact-project
cleanup and reports any remaining labelled resource or new dangling image.
