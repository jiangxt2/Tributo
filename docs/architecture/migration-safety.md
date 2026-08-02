# Migration Safety Rules

Rules for safely migrating between old and new code paths during architecture
convergence. These apply to all migration PRs (D1+D2, E1, E2, E3, E4, T1, T3).

## Core Principle

**New path + compat adapter + rollback switch — never delete the old path until
exit gates are satisfied.**

## Migration Strategy Per Domain

### Data Provider (D1+D2 / D3)

```
Old: training.data_loader direct dispatch (legacy config dict)
  ↓
New: DataSourceProvider + SourceConfig (JSON, provider ID)
  ↓
Compat: LegacyConfigNormalizer wraps old config dicts → SourceConfig
```

Exit gates before removing legacy adapter:
- [ ] All built-in data sources (Parquet, Lance, Iceberg, S3, ClickHouse) have
  Provider implementations with contract tests.
- [ ] Contract/golden comparison: old path vs new path produce identical
  `DatasetHandle` for the same input on a fixed benchmark dataset.
- [ ] Training, inference, and embeddings all route through the new Provider
  (verified by code audit, not just test coverage).
- [ ] Legacy adapter has been in place for ≥ 1 minor version with
  `DeprecationWarning`.

Rollback: Set `TRIBUTO_DATA_BACKEND=legacy` environment variable (or equivalent
feature flag) to bypass the Provider router and use the old dispatch.

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

All migration paths use environment variables for rollback:

| Flag | Default | Effect |
|------|---------|--------|
| `TRIBUTO_DATA_BACKEND` | `provider` | `legacy` to use old data_loader dispatch |
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
