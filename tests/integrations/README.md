# Tributo Integration Tests

Model-export integration tests run in an isolated Docker Compose project and
submit work through the Ray Jobs API. They never reuse a host Ray runtime, an
existing MLflow server, or a pre-existing container.

---

## Test List

| Test | File | Coverage | Prerequisites |
|------|------|----------|---------------|
| Model-export golden path | `../integration/test_walking_skeleton.py` | Ray Data Parquet → XGBoost → ONNX + UBJ S3 Bundle → BundleReader → batch inference → Ray Serve HTTP, plus MLflow provenance | Docker only; the runner creates all infrastructure |
| First-party export conformance | `../training/exporters/test_first_party_conformance.py` | XGBoost, Torch, Hugging Face, quantizer, validator, and checkpoint-source contracts in the pinned Linux image | Docker only; executed by the model-export runner |
| S3/MinIO contract | `../integration/test_export_s3.py`, `../integration/test_minio_compat.py` | Manifest-last publication, Lease/CAS, alias, GC, path-style access, and conditional writes against the run-owned MinIO service | Isolated model-export runner |
| MLflow Hook | `test_e2e_mlflow.py` | Committed Bundle upload, replay deduplication, explicit run reuse, and failure semantics | Isolated model-export runner |
| ClickHouse E2E | `test_e2e_clickhouse.py` | ClickHouse table → Daft OLAP Binding → explicit Daft-to-Ray adapter → XGBoost distributed training → MLflow → ONNX | Ray + Daft + `daft-olap-connectors` + ClickHouse + MLflow |
| Dual-engine Docker | `test_data_ingestion_dual_engine.py` | Local Parquet, full ETL chain, typed handles, worker-version evidence | Docker Ray cluster + Daft |
| Lance vector index | `test_lance_vector_index.py` | Distributed IVF_FLAT/IVF_PQ build, append coverage, global Top-K, fallback, optimization, compaction, Ray Jobs, and S3 result delivery | Docker Ray cluster + Lance-Ray + MinIO |
| File conformance | `../integration/test_data_ingestion_conformance.py` | Local/MinIO Parquet and CSV through Ray Data and Daft | Local Ray runtime + MinIO |
| Table conformance | `../integration/test_table_format_ingestion.py` | Local/MinIO Iceberg and Lance through Ray Data and Daft | Local Ray runtime + MinIO |
| PostgreSQL conformance | `../integration/test_postgresql_ingestion.py` | Structured table read through Ray Data and Daft | Local Ray runtime + PostgreSQL |
| Inference Ray Jobs | `../integration/test_inference_ray_jobs.py` | Bundle, real post-training inline/detached inference, MLflow import, external artifacts, retry identity, credential domains, empty/NaN behavior | Isolated version-locked Docker Ray + MLflow + MinIO |
| Streaming | `test_e2e_streaming.py` | Streaming inference service | TBD |
| Tune trial correctness | `../integration/test_tune_ray_cluster.py` | Ray Jobs → two concurrent XGBoost Tune trials, strict target metric, isolated checkpoints, ResultGrid, and zero Bundle publication | Isolated version-locked Docker Ray cluster via `run_tune_it.sh` |

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

## Tune Integration Gate

Run the Tune-only Gate from the repository root:

```bash
./scripts/run_tune_it.sh
```

The runner creates a unique `tributo-tune-it-<timestamp>-<pid>` Compose
project, submits the workload through the Ray Jobs API, and stores test,
cluster-readiness, and service logs under `/tmp/<project-name>/`. It removes
only that project's containers, network, and volumes. The Gate scales the Ray
worker service from one to three replicas, adding one worker at a time. After
each step, it requires the Ray State API to report the exact head-plus-worker
node count for three consecutive samples before continuing, avoiding concurrent
worker registration pressure. The final one-CPU head and three two-CPU workers
provide seven logical CPUs. Two outer Tune trial actors and their two inner Ray
Train placement groups require six CPUs in total, leaving one CPU of scheduling
headroom.

The `trials` directory is a Tributo-reserved namespace passed to Ray Tune as
its storage root, so Tune owns recovery state below `output_path/trials` and
Tributo does not delete it. With local storage, the inner Ray Train run starts
under the trial driver staging directory and Ray's final experiment sync
persists it below the corresponding `Result.path`. With remote storage, the
inner run writes to its isolated persistent URI directly. The Docker Gate uses
a run-scoped volume, so its isolated test artifacts are removed with the
Compose project after logs have been captured.

---

## Model-Export Integration Gate

Run the required CI subset from the repository root:

```bash
./scripts/run_model_export_it.sh --suite ci
```

It covers the pinned component contract, first-party conformance, real MLflow
Hook contract, and distributed walking skeleton. Run the release-oriented
superset explicitly when the trainer Bundle and full S3/MinIO contracts are
also required:

```bash
./scripts/run_model_export_it.sh --suite full
```

Calling the runner without `--suite` remains equivalent to `--suite full` for
backward compatibility; CI always names its suite explicitly.

The script creates a unique project named
`tributo-model-export-it-<timestamp>-<pid>` (or accepts a unique
`tributo-model-export-*` project from CI), prepares the same content-addressed
dependency runtime used by the ingestion gate, starts one Ray head, one
independent Ray worker, MinIO, and MLflow, and stores traceable logs under
`/tmp/<project-name>/`. The checkout is copied into a run-scoped source volume
and mounted read-only; Compose neither builds nor implicitly pulls images.

Compose readiness is transitive: the Ray head becomes healthy only after both
the Ray control plane and MinIO health endpoint respond, while `up --wait`
also waits for MLflow's own health check. Tests therefore never race a merely
started object-storage container.

An `EXIT` trap is installed before startup. On success, test failure, or
interruption it runs project-scoped Compose cleanup with volumes and orphans,
then checks that:

- no container carrying that exact Compose project label remains.
- no network or volume carrying that exact Compose project label remains.

The runner also snapshots pre-existing container IDs, names, projects, and
states. Any drift is reported as concurrent host activity for diagnosis, but
does not override the project-scoped result: another integration-test project
may legitimately restart or remove its own containers while this gate runs.

The runner never calls `docker stop`, `docker restart`, `docker rm`, or a prune
command against an unscoped target. It does not publish host ports or assign
fixed container names.

The default workflow invokes the same runner with a run-specific
`COMPOSE_PROJECT_NAME`, uploads its test and service logs, and performs a
defensive repeat of the same scoped `down --volumes --remove-orphans` operation
under `if: always()`.

## Pinned Component Contract

`component-versions.env` is authoritative for the shared integration-test
components used by the model-export and ingestion CI gates:

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| Ray | 2.55.1 |
| uv | 0.11.23 |
| MinIO | RELEASE.2025-09-07T16-13-09Z |
| PostgreSQL | 17.6 |
| MLflow | 2.22.5 |
| boto3 / botocore | 1.43.56 / 1.43.56 |
| XGBoost | 3.3.0 |
| ONNX / ONNX Runtime / onnxmltools | 1.22.0 / 1.28.0 / 1.16.0 |
| Torch | 2.13.0 |
| Transformers | 4.57.6 |
| PyArrow / pandas | 19.0.1 / 2.3.3 |

Ray, uv, the source-snapshot tool image, MinIO, and PostgreSQL image references
also include immutable SHA-256 digests.
`test_it_component_versions.py` ties all Python versions to `uv.lock` and
fails if the Dockerfile or Compose file bypasses the version contract.

## MLflow Hook Suite

The Hook suite validates committed-Bundle provenance against the MLflow server
created by the isolated model-export profile. It verifies Bundle URI, digest,
tags, artifact upload, replay behavior, and required/optional failure semantics.
It also asserts that no Model Version is created.

The supported entry point is the complete runner above so cleanup remains
automatic. Do not invoke the suite against a shared or pre-existing MLflow
service.

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

## Required Lance Vector Index Gate

Run the complete distributed Lance vector gate from the repository root:

```bash
./scripts/run_lance_vector_index_it.sh
```

The runner reuses the same content-addressed dependency runtime and isolated
Ray head, worker, and MinIO lifecycle as the ingestion gate. It invokes only
the public Lance-Ray build, search, optimize, and compaction entry points from
production code. The test-only coordinator appends a real Lance fragment after
Lance-Ray freezes its fragment batches, proving that Tributo reports partial
coverage instead of claiming a fixed build snapshot. The suite also verifies
global Top-K against a brute-force baseline, IVF_PQ recall, unindexed fallback,
fast search, S3 Parquet delivery, Ray Jobs receipts, and post-compaction
fail-closed coverage evidence.

Ray Job requests use the Ray control plane as a trusted administrative
boundary. The complete serialized request is limited to 64 KiB; this is
separate from the 65,536-dimension limit for direct in-process search calls.
The query vector is excluded from ordinary Tributo logs and receipts, but Ray
administrators who can inspect Job runtime environments can view the encoded
request and therefore must be treated as trusted operators.

Inline search delivery is bounded by both `inline_max_rows` and
`inline_max_bytes` (1 MiB by default). Larger results must use materialized
Parquet delivery. S3 materialization uses a conditional create and refuses to
replace an existing object, including when two jobs race for the same output
URI.

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
