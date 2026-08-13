# Migration Safety Rules

Rules for safely migrating between old and new code paths during architecture
convergence. These apply to all migration PRs (D1+D2, E1, E2, E3, E4, T1, T3).

## Core Principle

**New path + typed compat adapter + observable stop-loss gate.** Compatibility
inputs may remain after a legacy execution backend is removed; an adapter must
normalize into the new path rather than preserve two reader implementations.

## Migration Strategy Per Domain

### Data Provider (D1+D2 / D3)

```
Old input: legacy type/path/dialect config
  ↓
Compat: LegacyConfigNormalizer (conversion only)
  ↓
New: DataSourceProvider / IngestionGateway (one execution path)
```

Current exit-gate status:
- [x] All built-in logical sources (Parquet, CSV, Iceberg, Lance, ClickHouse,
  Doris, and PostgreSQL) have Provider implementations with contract tests.
  Adapter presence is not a runtime-support claim; each physical combination
  still needs its own infrastructure gate.
- [x] Representative compatibility comparisons show that canonical and legacy
  inputs enter the same Gateway and produce equivalent Ray Data results. The
  new public path returns an `IngestionOpenResult` with a typed native handle.
- [x] Training, inference, and embeddings all route through the new Provider
  (verified by code audit, not just test coverage).
- [ ] Legacy input adapters have completed their documented compatibility
  window. They remain available and keep their `FutureWarning`; only the
  duplicate execution backend has been removed.

The maintainer-approved architecture convergence removed the duplicate
direct-dispatch execution backend without claiming that the legacy *input*
deprecation window had elapsed. `TRIBUTO_DATA_BACKEND=legacy` therefore remains
accepted with a `FutureWarning`, but it selects the same conversion and Gateway
path as the default. Rollback uses normal release rollback or disables adoption
of the alpha Gateway; the compatibility selector does not reactivate duplicate
reader code.

### D3 Delivery Record

Migration impact:

- `InferenceConfig.source` is the canonical input for new callers.
- Legacy `input_uri`, flat ClickHouse fields, `data.uri`, `data.input`, and
  embedding `--input`/`s3_input_path` remain compatibility inputs.
- Feature and text column selection is expressed through provider-native
  projection.
- Output sinks are unchanged.

Compatibility:

- Legacy inference and embedding input forms remain accepted during their
  deprecation windows.
- Legacy local-runner `val_path` and `test_path` remain relative to the caller
  working directory during that window; canonical source objects use the
  shared project-root path policy.
- They normalize to the same Provider path and cannot select a legacy backend.
- The internal training loader normalizes legacy input and constructs an explicit
  `IngestionRequest` for the Gateway with `engine="ray"`; it is a Ray Dataset
  shape adapter, not an alternate reader backend.
- `DataConnector.read()` and other old public Ray compatibility callers continue
  to use the Ray compatibility adapter during the deprecation window.
  Third-party Providers that shipped only the beta `normalize()+open()` SPI
  remain callable from that adapter with a `FutureWarning`.
  The alpha Gateway never catches a planning or Binding failure and falls back
  to `open()`; external Providers must migrate to `plan()` plus an
  `EngineBinding` before the next major release.
- Legacy ClickHouse and Doris raw-SQL shapes remain parseable only to produce a
  credential-free migration error. Built-in execution supports structured
  table/projection/partitioning requests; Tributo does not restore an arbitrary
  SQL reader.
- Embedding submission validates and transports a credential-free source but
  does not resolve Providers, Bindings, or optional dependencies on the submit
  host. The Ray job performs that validation in its cluster image. Its engine
  remains explicit; Daft input reaches the Ray-only embedding consumer through
  the recorded Daft-to-Ray adapter.

Deprecation window:

Legacy inputs remain supported for at least two minor versions or six months,
whichever is longer. Warning emission and adapter removal are separate follow-up
work after the data migration exit gates are satisfied.

Exit gate:

The independent runtime backend is removed. Do not remove legacy *input
normalizers* until their compatibility windows and route audits are satisfied.

Provider IDs identify logical data sources (`tributo.parquet`,
`tributo.clickhouse`, etc.); engine selectors are not persisted as provider
IDs. The unused `SourcePlan` / `SourceRouter` prototype has been replaced by
the explicit `IngestionRequest` → `IngestionGateway` → `LogicalScanPlan` →
`EngineBinding` path. Gateway `describe()` performs static validation without
metadata I/O; `open()` creates the lazy typed handle and receipt. Built-in
`DataConnector.read()` remains a one-way Ray adapter over this path. Training
loader surfaces normalize compatibility input and open the explicit Gateway
request directly; neither surface may restore an independent reader. Provider
`open()` is a temporary external beta-SPI compatibility exception described
above, not a Gateway fallback. File
providers accept local paths and S3 paths with an explicit `S3Config`; S3 URI
userinfo, query parameters, and fragments are rejected because accepting them
would change object-key or credential semantics.

New third-party sources do not use the deprecated `open()` exception. An
installed package publishes a versioned logical Provider through
`tributo.ingestion_providers` and physical Ray/Daft Bindings through
`tributo.ingestion_bindings`. Provider-declared projection and relative-path
metadata replace consumer-side provider maps. Catalog and storage-format
Binding constraints allow Hive tables backed by Parquet, ORC, or Iceberg
without changing the Gateway or any algorithm consumer. Duplicate or ambiguous
registrations fail closed; discovery never overrides an existing route.

### Bundle Export (E1 / E2 / E4)

```
Old: training/exporters/ (per-trainer export logic)
  ↓
New: exporting.BundleExportService (unified kernel)
  ↓
Compat: training/exporters/__init__.py re-exports from exporting
```

Exit gates before removing old exporters:
- [x] DNN, PU, XGBoost all produce valid Bundles through the new kernel
  (verified by Bundle vertical slice test per trainer type).
- [x] `BundleReader` can load Bundles from all three trainer types.
- [x] Required artifact failure propagates correctly (E0 fix verified).
- [ ] Old exporter classes are accessible via compat re-exports for ≥ 2 minor
  versions or 6 months (whichever is longer).
- [ ] `DeprecationWarning` is emitted when old exporter classes are imported.
- [ ] Migration guide is published and linked from the warning message.

First-party trainers now require an explicit Bundle destination and fill in
their standard targets when `BundleOutputConfig.targets` is omitted. There is
no process-wide export backend selector and no automatic fallback. A caller
that still needs a raw artifact must opt in for that invocation with
`legacy_export=True`; the call emits `DeprecationWarning` and returns the
`legacy_artifact_uri` field in its `TrainingResult` projection.

Bundle compatibility also requires the following rules:

- `BundleRef.canonical_uri`, the manifest URI, and the published URI identify
  the same immutable Bundle for local paths, `file://`, and S3.
- Event and Hook identity is derived from the exact manifest bytes returned by
  the winning repository commit; the service does not re-read a mutable path.
- A caller-provided manifest is accepted only with its exact bytes, and a
  supplied `BundleRef` is checked against both digest and `bundle_id`.
- Ray checkpoint sources finish conversion inside `Checkpoint.as_directory()`;
  success, consumer failure, repeated opens, and cleanup are conformance-tested.
- Orphan GC rechecks the manifest after acquiring its lease and releases the
  lease on every exit path.
- Orphan GC must receive the exact Bundle store root used by Publisher. A
  parent or nested prefix selects a different lease namespace and is not a
  safe substitute for that root. A bucket-root scan emits an additional
  warning because its deletion scope is unusually broad; the warning cannot
  prove that an arbitrary nested prefix matches the Publisher configuration.
- Real IT uses a unique Compose project, digest-pinned infrastructure, and an
  unconditional project-scoped cleanup trap. Existing containers must keep
  their original state.

### Manifest Schema (E1)

```
v1 (current) → v2 (future, when breaking change is needed)
```

Exit gates before introducing v2:
- [ ] ADR documents the exact schema change and why v1 is insufficient.
- [ ] v2 Reader implements read support for v1 manifests.
- [ ] v1 Reader fails-fast on v2 manifests with a clear error message
  (including the minimum Tributo version needed).
- [ ] All existing v1 manifests in test fixtures are still readable.

Rollback: v2 is NOT rolled back once published Bundles reference it. Instead,
v3 is introduced to fix any v2 issues. The compatibility window rule (reader
supports N-1 and N-2) ensures v1 Bundles remain readable through v3.

## Stop-Loss Thresholds

Defined in `docs/architecture/benchmark-protocol.md`. Summary:

| Scenario | Primary Metric | Stop Threshold | Consecutive Runs |
|----------|---------------|----------------|-----------------|
| Training (no semantic change) | Throughput | > 10% drop | 2 |
| Training (no semantic change) | Peak worker RSS | > 20% increase | 2 |
| Inference/Serving | p95 latency | > 20% increase | 2 |
| Any path | Required artifact / data results | Regression | 1 (immediate) |

Thresholds apply to **unchanged business semantics** only. If a migration
intentionally changes behavior (e.g., fixes a bug that caused incorrect
results), the threshold is waived for that specific metric — but the change
must be documented in the PR and the `decision-log.md`.

## Stop-Loss Process

1. **Detect**: Benchmark comparison (baseline commit vs candidate commit)
   exceeds a threshold on two consecutive runs.
2. **Pause**: Migration PR is blocked. A `decision-log.md` entry records:
   - Which threshold was exceeded
   - Raw numbers (baseline mean, candidate mean, change%)
   - Hardware, dataset, and commit SHAs
3. **Diagnose**: Profile to identify root cause. Do NOT guess.
4. **Fix**: Separate PR (not the migration PR) addresses the regression.
5. **Re-benchmark**: After fix, two consecutive passes on both baseline and
   candidate are required.
6. **Resume**: Migration PR is unblocked.

## Feature Flag Convention

Migration paths may retain environment selectors while two supported
implementations intentionally coexist, or for a documented compatibility
window after a selector stops choosing a distinct implementation:

| Flag | Default | Effect |
|------|---------|--------|
| `TRIBUTO_DATA_BACKEND` | `provider` | Deprecated `legacy` value warns and enters the same Provider/Gateway path; retained for the compatibility window only |

Active flags are read once at import time. Changing a flag at runtime has no effect
(by design — to prevent inconsistent state within a single process).

`TRIBUTO_CONFIG_FORMAT` is a reserved name only; no environment selector is
implemented. Persisted configuration entry points accept JSON and reject other
formats directly, without consulting an environment variable.

## Compatibility Adapter Lifecycle

```
Introduce  →  Warn  →  Remove
    |            |         |
    v            v         v
  New path   Deprecation   Exit gates
  + adapter  Warning        satisfied
  coexist    emitted
```

- **Introduce**: Adapter is added alongside the new path. No warning.
- **Warn**: `DeprecationWarning` is emitted when the adapter is used. Must
  include a migration guide URL. Duration: ≥ 2 minor versions or 6 months.
- **Remove**: Adapter is deleted (E4 for exporters; D3 completion for data).
  Removal is a separate PR, not bundled with the migration.

<!-- END -->
