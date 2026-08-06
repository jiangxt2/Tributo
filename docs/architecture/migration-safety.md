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
deprecation window had elapsed. `TRIBUTO_DATA_BACKEND` no longer selects an
execution backend. Rollback uses normal release rollback or disables adoption
of the alpha Gateway; it does not reactivate duplicate reader code.

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
- They normalize to the same Provider path and cannot select a legacy backend.
- Existing downstream callers use the Ray compatibility adapter, which now
  delegates the same Gateway with `engine="ray"`; it is an API-shape adapter,
  not an alternate reader backend.

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
`DataConnector.read()`, Provider `open()`, and training loader surfaces are
one-way Ray adapters over this path and must never restore a reader. File
providers accept local paths and S3 paths with an explicit `S3Config`; S3 URI
userinfo, query parameters, and fragments are rejected because accepting them
would change object-key or credential semantics.

### Bundle Export (E1 / E2 / E4)

```
Old: training/exporters/ (per-trainer export logic)
  ↓
New: exporting.BundleExportService (unified kernel)
  ↓
Compat: training/exporters/__init__.py re-exports from exporting
```

Exit gates before removing old exporters:
- [ ] DNN, PU, XGBoost all produce valid Bundles through the new kernel
  (verified by Bundle vertical slice test per trainer type).
- [ ] `BundleReader` can load Bundles from all three trainer types.
- [ ] Required artifact failure propagates correctly (E0 fix verified).
- [ ] Old exporter classes are accessible via compat re-exports for ≥ 2 minor
  versions or 6 months (whichever is longer).
- [ ] `DeprecationWarning` is emitted when old exporter classes are imported.
- [ ] Migration guide is published and linked from the warning message.

Rollback: Set `TRIBUTO_EXPORT_BACKEND=legacy` to use old per-trainer exporters.
The compat adapter wraps old exporter calls in the new `ModelExporter` protocol.

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

Migration paths may use environment variables only while two supported
implementations intentionally coexist:

| Flag | Default | Effect |
|------|---------|--------|
| `TRIBUTO_EXPORT_BACKEND` | `bundle` | `legacy` to use old per-trainer exporters |
| `TRIBUTO_CONFIG_FORMAT` | `json` | Reserved; rejects non-JSON input regardless |

Flags are read once at import time. Changing a flag at runtime has no effect
(by design — to prevent inconsistent state within a single process).

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
