# Tributo Integration Tests

External integration tests run through lifecycle-owned entry points and, when
required, submit work through the Ray Jobs API in an isolated Docker Compose
project. They never reuse a host Ray runtime, an existing MLflow server, or a
pre-existing container. `ci/test-suites.json` owns their impact rules and
reports required evidence, but no GitHub Actions event executes these suites.

The data-ingestion image installs database drivers and the v1.0 database
connectors through Tributo extras. `daft-clickhouse==1.0`,
`daft-doris==1.0`, `ray-doris==1.0`, and `ray-hive==1.0` are resolved by
`uv.lock`. The Data Ingestion Gate validates the Tributo Ray HiveServer2
Provider/Binding separately from the canonical full-runtime package-presence
gate. A custom external wheelhouse is only for packages outside that locked
set.

---

## Test List

| Test | File | Tier | Coverage | Prerequisites |
|------|------|------|----------|---------------|
| Model-export golden path | `../integration/test_walking_skeleton.py` | `manual_external` | Ray Data Parquet → XGBoost → ONNX + UBJ S3 Bundle → BundleReader → batch inference → Ray Serve HTTP, plus MLflow provenance | Docker only; the runner creates all infrastructure |
| First-party export conformance | `../training/exporters/test_first_party_conformance.py` | `manual_external` | XGBoost, Torch, Hugging Face, quantizer, validator, and checkpoint-source contracts in the pinned Linux image | Docker only; executed by the model-export runner |
| S3/MinIO contract | `../integration/test_export_s3.py`, `../integration/test_minio_compat.py` | `ci_fast` Moto / `manual_external` MinIO | Manifest-last publication, Lease/CAS, alias, GC, path-style access, and conditional writes | Ephemeral Moto in CI; run-owned MinIO externally |
| Runtime image gate | `../../scripts/run_runtime_image_it.sh`, `jobs/runtime_image_gate_job.py` | `manual_external` | Pinned full image, first-party and Alpha imports on driver/worker, Ray Data, Ray Jobs, and image attestations | Docker Buildx + two-node Ray cluster on a matching native host architecture |
| MLflow Hook | `test_e2e_mlflow.py` | `manual_external` | Committed Bundle upload, replay deduplication, explicit run reuse, and failure semantics | Isolated model-export runner |
| ClickHouse E2E | `test_e2e_clickhouse.py` | `quarantine` | ClickHouse table → Daft ClickHouse Binding → explicit Daft-to-Ray adapter → XGBoost distributed training → MLflow → ONNX | Full image or `tributo[clickhouse]`; lifecycle and ownership contract pending |
| Data Ingestion Docker | `test_data_ingestion_dual_engine.py` | `manual_external` | Local/S3 Parquet, Iceberg-on-MinIO Ray/Daft reads, native writes, Ray HiveServer2 structured projection, typed handles, and worker-version evidence | Docker Ray cluster + Daft + MinIO + Hive 4.2.0 |
| Lance vector index | `test_lance_vector_index.py` | `manual_external` | Distributed IVF_FLAT/IVF_PQ build, append coverage, global Top-K, fallback, optimization, compaction, Ray Jobs, and S3 result delivery | Docker Ray cluster + Lance-Ray + MinIO |
| File conformance | `../integration/test_data_ingestion_conformance.py` | `manual_external` | Local/MinIO Parquet and CSV through Ray Data and Daft | Local Ray runtime + MinIO |
| Table conformance | `../integration/test_table_format_ingestion.py` | `manual_external` | Local/MinIO Iceberg and Lance through Ray Data and Daft | Local Ray runtime + MinIO |
| PostgreSQL conformance | `../integration/test_postgresql_ingestion.py` | `manual_external` | Structured table read through Ray Data and Daft | Data Ingestion Docker gate + PostgreSQL 16.14 |
| Inference Ray Jobs | `../integration/test_inference_ray_jobs.py`, `../integration/test_lance_result_sink_ray.py` | `manual_external` | Bundle, real post-training inline/detached inference, MLflow import, external artifacts, retry identity, credential domains, empty/NaN behavior, Lance-Ray local/S3 create/append/overwrite, and vector schemas | Isolated version-locked Docker Ray + Lance-Ray + MLflow + MinIO |
| Streaming | `test_e2e_streaming.py` | `quarantine` | Streaming inference service | Model and service lifecycle pending |
| Tune trial correctness | `../integration/test_tune_ray_cluster.py` | `manual_external` | Ray Jobs → two concurrent XGBoost Tune trials, strict target metric, isolated checkpoints, ResultGrid, and zero Bundle publication | Isolated version-locked Docker Ray cluster via `run_tune_it.sh` |
| Explainability Ray Jobs | `../integration/test_explainability_ray_jobs.py` | `manual_external` | Explanation submission, distributed execution, and output evidence | Lifecycle-owned Ray Jobs runner |
| Distributed algorithms | `../integration/test_distributed_algorithm_local.py` | `manual_external` | Collective and MapReduce execution, sharding, receipts, and Bundle atomicity | Isolated multi-worker Ray environment |
| KubeRay RayJob resource profiles | `../integration/test_kuberay_rayjob_resources.py` | `manual_external` | Tributo KubeRay submission adapter, multiple worker CPU/memory profiles, formal XGBoost ExecutionReceipt/Bundle, and cleanup | Kind 1 control-plane + 3 workers, fixed KubeRay Operator, and local XGBoost Tributo image |
| Ray runtime environment | `../integration/test_ray_runtime_env.py` | `manual_external` | uv runtime-environment propagation into Ray workers | Host capable of starting the required Ray runtime |

### KubeRay RayJob resource profiles

This is a heavyweight manual external gate. Do not run the runner or invoke
the test module directly during normal development, pre-checks, or routine
reviews unless the user explicitly requests this KubeRay IT. One run creates
a disposable four-node Kind cluster, loads a large runtime image into every
node, installs KubeRay, and executes two distributed XGBoost jobs.

The KubeRay gate uses a small XGBoost image so it can reuse a locally cached Ray
runtime without building the full algorithm image. It installs the current
Tributo wheel and the official boosting wheel from the sibling
`tributo-algorithms` checkout:

```bash
TRIBUTO_ALGORITHMS_ROOT=/path/to/tributo-algorithms \
TRIBUTO_KUBERAY_RUNTIME_IMAGE=tributo-kuberay-xgboost:it-local \
  bash scripts/build_kuberay_xgboost_image.sh
TRIBUTO_KIND_NODE_IMAGE='kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5' \
TRIBUTO_KUBERAY_RUNTIME_IMAGE=tributo-kuberay-xgboost:it-local \
  ./scripts/run_kuberay_rayjob_it.sh
```

The runner creates and destroys one uniquely named Kind cluster. The Python IT
submits each business `RayJob` through `KubeRayJobSubmitter`; the shell runner
only owns Kind, Operator setup, diagnostics, and scoped cleanup. The fixed node
image must match the local Docker architecture.

The IT runs two positive profiles: two workers with 1 CPU/1 GiB each, and
three workers with 2 CPU/2 GiB each. The image contains three small CSV parts
so the three-worker case has deterministic non-empty Ray Data input shards.
The workload prints a credential-free Ray resource snapshot; the test checks
aggregate cluster capacity and that enough alive Ray nodes can satisfy one
requested worker.

For an internal or offline environment, set
`TRIBUTO_KUBERAY_CHART_ARCHIVE` to a locally available archive for the same
`TRIBUTO_KUBERAY_VERSION`; the runner then skips the public Helm repository.

---

## Inference Ray Jobs Suite

Use only the lifecycle-owned runner:

```bash
./scripts/run_inference_it.sh
```

Do not invoke `test_inference_ray_jobs.py` against a developer Ray cluster.
The test module fails when the Compose ownership marker is absent. The runner
uses `inference-it-versions.conf`, creates a unique Compose project, exposes no
host ports, and runs pytest inside `ray-head`. The locked `lance-ray` and
`pylance` versions are verified before the suite. The head has zero Ray CPUs,
so model actors execute on the independent worker.

An EXIT/INT/TERM trap captures logs and executes project-scoped `down
--volumes --remove-orphans`. The runner verifies that project-labelled
resources are gone; no prune, global deletion, or shared-image cleanup is
permitted. Test and service logs remain under
`/tmp/<compose-project>-*.log` after cleanup.

---

## Tune External Validation Gate

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

## Model-Export External Validation Gate

Run the bounded named subset from the repository root:

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
backward compatibility. The historical `ci` suite name describes its bounded
contents and does not authorize execution in GitHub Actions.

The script creates a unique project named
`tributo-model-export-it-<timestamp>-<pid>` (or accepts a unique
`tributo-model-export-*` project from an external operator), prepares the same content-addressed
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

The CI impact planner reports this suite when relevant paths change. Execution,
log retention, and cleanup remain the responsibility of the external
lifecycle-owned runner.

## Pinned Component Contract

`component-versions.env` is authoritative for the shared integration-test
components used by the model-export and ingestion external validations:

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| Ray | 2.55.1 |
| uv | 0.11.23 |
| MinIO | RELEASE.2025-09-07T16-13-09Z |
| PostgreSQL | 16.14 |
| MLflow | 2.22.5 |
| boto3 / botocore | 1.43.56 / 1.43.56 |
| XGBoost | 3.3.0 |
| ONNX / ONNX Runtime / onnxmltools | 1.22.0 / 1.28.0 / 1.16.0 |
| Torch | 2.13.0 |
| Transformers | 4.57.6 |
| PyArrow / pandas | 19.0.1 / 2.3.3 |

Ray, Hive, MinIO, and PostgreSQL image references include immutable SHA-256
digests. All Docker IT runtime profiles use host uv with `0.11.23` as the CI
baseline to validate the lock and export hashed requirements. A different
local uv version warns and proceeds only when locked validation and export
succeed. The inference and full-runtime images install the host-built Tributo
wheel and do not pull uv or standalone Python tool images.
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

## Data Ingestion External Validation Gate

Run the complete Data Ingestion Docker gate from the repository root:

```bash
./scripts/run_data_ingestion_it.sh
```

This is the only supported lifecycle entry for
`test_data_ingestion_dual_engine.py`. Host uv first checks `uv.lock` and exports
hashed requirements without the Tributo project. It computes a dependency-only
runtime key from the actual uv version, exported-requirements digest, runtime
profile, Dockerfile, effective Docker ignore file, `pyproject.toml`, `uv.lock`,
version contract, and Docker platform. A matching
`tributo-it-runtime:data-ingestion-<runtime-key>` image is validated and reused;
when missing, exactly one process builds it through a profile/key/platform/
daemon-scoped file lock and a single `docker buildx build --load` output.

Compose never builds or implicitly pulls an image. The runner explicitly
prepares digest-pinned Ray, MinIO, and Hive images. `source-init` and
`workspace-init` reuse the already prepared Tributo Runtime; there is no
standalone Python tool image. The checkout is copied once into a run-scoped
source volume and mounted read-only on the Ray head and worker. The snapshot
also projects deterministic
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

The CI impact planner reports the ingestion suite when its source, contracts,
runtime definition, or runner changes. The external runner obtains one
content-addressed runtime, starts one zero-CPU Ray head, one Ray worker, MinIO,
PostgreSQL 16.14, and HiveServer2 4.2.0 with
`docker-compose.data-ingestion.yml`, freezes the checkout in a run-scoped
read-only volume, and requires Driver/Worker version, snapshot, Hive
projection, PostgreSQL dual-engine conformance, and Iceberg-on-MinIO evidence.
Infrastructure absence cannot be represented as passing evidence.

## Lance Vector Index External Validation Gate

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
