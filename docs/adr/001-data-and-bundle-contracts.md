# ADR 001: Data and Bundle Contract Namespaces

**Status**: Accepted
**Date**: 2026-08-02
**Deciders**: Tributo maintainers
**Supersedes**: None (initial architecture baseline)

## Context

Tributo currently has multiple abstractions that overlap in name and partial
semantics:

| Domain | Protocol / Class | Location |
|--------|-----------------|----------|
| Export source resolution | `SourceProvider` (protocol) | `exporting/protocols.py` |
| Data source ingestion | `DataSourceProvider` (prototype) | `data/provider.py` |
| Data connector | `DataConnector` (base class) | `data/base.py` |
| Inference data routing | `InferenceConfig.data_type` branches | `inference/pipeline.py` |
| Model export | `ModelExporter` (protocol) | `exporting/protocols.py` |
| Legacy training export | `TrainingExporter` abstract | `training/exporters/` |
| Bundle manifest | `ExportManifest` | `exporting/manifest.py` |

Without a frozen namespace, the Data (D1+D2) and Bundle (E1) tracks risk
introducing incompatible ID formats, overlapping names, or conflicting
evolution rules. This ADR defines the shared contract before either track
writes implementation code.

## Decision

### Identifier Namespaces

Each domain gets an independent, non-overlapping ID namespace.

#### 1. Provider IDs (Data Domain)

```
Format:      "<domain>.<name>"
Domain:      "tributo" for built-in providers
Examples:    "tributo.parquet", "tributo.lance", "tributo.iceberg", "tributo.clickhouse"
Third-party: "<package>.<name>"  (e.g. "myorg.mysql")
```

Rules:
- Lowercase, dot-separated. Each segment matches `[a-z][a-z0-9_]*`.
- Built-in providers use the `tributo.` prefix. Third-party providers use their
  package name as prefix.
- The provider ID is the **canonical key** in `SourceConfig.provider`. User-facing
  short aliases (e.g. `engine="daft"`) are mapped to full provider IDs by a
  **selector table** that lives in the data registry — not in the provider itself.
- Provider IDs are **immutable once released**. Renaming a provider requires a
  new ID and a deprecation window for the old one.

#### 2. Exporter IDs (Bundle Domain)

```
Format:      "<framework>-<format>[-<variant>]"
Examples:    "xgboost-onnx-v1", "torch-onnx", "torch-safetensors", "hf-onnx"
```

Rules:
- Already defined by `ModelExporter.exporter_id` and enforced by plugin discovery
  (`entry_point.name == exporter_id`). This ADR ratifies the existing format.
- Exporter IDs are **immutable** once any Bundle Manifest references them. New
  variants get new IDs (e.g. `xgboost-onnx-v2`).
- The exporter ID appears in `ManifestExecutionNode.exporter_id`.

#### 3. Validator IDs (Bundle Domain)

```
Format:      "<scope>-<check>"
Examples:    "onnx-runtime", "schema-conformance", "roundtrip-parity"
```

Rules:
- Already defined by `ExportValidator.validator_id`. Ratified as-is.
- Immutable once referenced in a validator binding chain.

#### 4. Flavor IDs (Bundle Domain)

```
Format:      "<framework>[-<variant>]"
Examples:    "xgboost", "torch", "onnx", "safetensors"
```

Rules:
- Already defined by `ModelFlavor.flavor_id`. Ratified as-is.
- A flavor represents a serialized model format that a `BundleReader` can
  deserialize.

#### 5. Architecture IDs (Model Factory Domain)

```
Format:      "<model-family>[-<variant>]"
Examples:    "dnn-mlp", "dnn-tabnet", "pu-nnpu"
```

Rules:
- Already defined by `ModelFactory.architecture_id`. Ratified as-is.
- Used by `BundleReader` to reconstruct model skeletons before loading weights.

### DatasetRef (Shared Across Domains)

A `DatasetRef` is a lightweight, credential-free record of what data was used:

```python
@dataclass(frozen=True)
class DatasetRef:
    ref_id: str            # SHA-256 of (provider_id + canonical_uri + partition_filter)
    provider_id: str       # e.g. "tributo.parquet"
    uri: str               # Canonical URI (s3://, file://, etc.)
    schema_fingerprint: str  # SHA-256 of canonical Arrow schema JSON
    row_count: int | None  # None if not computed
    provenance: str        # Free-form version/timestamp string (not parsed)
```

Rules:
- `DatasetRef` lives in `tributo.data.refs` (to be created by D1+D2; this module
  does not exist yet as of A0) and is importable by both `tributo.data`
  and `tributo.exporting`.
- **Credentials must never appear** in a `DatasetRef` or Bundle Manifest.
- `DatasetRef` is stored in `ManifestSourceInfo` (via `source_fingerprint`).
- Future evolution: if richer provenance is needed, add optional `parent_refs:
  tuple[DatasetRef, ...]` — never change the existing fields.

### SourceConfig (Data Domain)

`SourceConfig` is the user-facing configuration that describes a data source:

- Serialization format: **JSON only**. YAML input is rejected at the parser level.
- Unknown fields: **fail-fast** (Pydantic `extra="forbid"`).
- Compatibility: New optional fields with defaults must not break existing configs.
  Removing or renaming a field requires a deprecation window.
- `SourceConfig.provider` stores the **full provider ID** (not the short alias).

### Manifest Schema Version (Bundle Domain)

- `ExportManifest.schema_version` is an **integer** that increments on breaking changes.
- Current version: **1**.
- v1 → v2 trigger: adding a required field, changing field semantics, or changing
  the digest algorithm.
- **Reader compatibility rule**: a vN reader MUST read v(N-1) and v(N-2) manifests.
  A v1 reader encountering v3+ MUST fail-fast with a clear error message.
- **Writer compatibility rule**: a new writer MAY write the latest schema version
  only. It MUST NOT silently write an older version to paper over reader gaps.
- The compatibility window is **2 schema versions or 2 Tributo minor versions,
  whichever is longer**.

#### Manifest v1 Field-Evolution Compatibility Matrix (E1)

The E1 contract keeps `schema_version = 1` and adds **optional fields only**
(`SignatureField`, `ManifestSignature.input_fields/output_fields`).  The
resulting matrix is:

| Direction | Behavior |
|-----------|----------|
| New reader → old v1 manifest | Supported — optional fields default to empty; v1 artifacts without `artifact_kind` get `"model"` injected |
| Old reader → new v1 manifest | Fails fast via `extra="forbid"` — expected, not a bug; old versions never promise to read new manifests |
| Digest verification | `manifest_sha256` is always computed over the **raw published bytes**, so field additions never invalidate old bundles |
| Rollback | Reverting E1 restores the pre-E1 reader; already-published E1 bundles remain readable by the pre-E1 reader only if they predate the new fields |
| Removal conditions | `SourceProvider` (protocol name) is removed 2 minor versions after E1 (STABILITY.md), gated on migration telemetry. `input_names`/`dynamic_axes` are v1 compatibility fields and are **retained indefinitely** — new readers parse old manifests through them; they are never removal candidates |


### Bounded Data vs Streaming Protocol Boundary

```
                    ┌─────────────────────────┐
                    │     SourceConfig          │
                    │  (JSON, provider="...")    │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
     ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
     │ DataSourceProvider │  │DataSourceProvider│  │DataSourceProvider│
     │  (parquet)         │  │  (lance)        │  │  (clickhouse)    │
     └───────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                  │
             ▼                 ▼                  ▼
     ┌───────────────────────────────────────────────┐
     │              DatasetHandle                      │
     │  (Ray Dataset / Daft DataFrame — bounded batch) │
     └───────────────────────────────────────────────┘

     ═══════════════════════════════════════════════════
     Protocol boundary: bounded ≠ unbounded streaming
     ═══════════════════════════════════════════════════

     ┌──────────────┐
     │ StreamConfig  │  (JSON, source="kafka", ...)
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ StreamSource  │  (protocol — independent lifecycle)
     └──────┬───────┘
            │
            ▼
     ┌──────────────────────────────────────────────┐
     │           Unbounded micro-batch stream         │
     │  (offset tracking, commit, partition ownership) │
     └──────────────────────────────────────────────┘
```

- `DataSourceProvider` handles **bounded** reads with finite lifecycle:
  open → read → return `DatasetHandle` → close. No offset tracking, no commit,
  no partition ownership.
- `StreamSource` handles **unbounded** streaming with persistent offset/commit
  lifecycle. Kafka enters through `StreamSource`, never through `DataSourceProvider`.
- Both share: schema representation, error model, credential resolution, and
  capability description format.
- They do **not** share: offset lifecycle, commit semantics, partition ownership,
  watermark, or triggering.

### Error Model

All domain errors inherit from `TributoError` and carry structured context:

```python
class DataProviderError(TributoError):
    provider_id: str
    uri: str
    reason: str

class ExportError(TributoError):
    exporter_id: str
    phase: str  # "plan" | "execute" | "validate" | "publish"
    reason: str
```

- Provider/Exporter implementations raise domain-specific subtypes.
- The framework catches and wraps unexpected exceptions from third-party code.
- Stack traces from third-party code are logged at DEBUG level, not exposed to users.

### Artifact Status Model

Every exported artifact has one of these statuses:

| Status | Meaning |
|--------|---------|
| `SUCCESS` | Artifact produced and validated |
| `PARTIAL_SUCCESS` | Required artifacts OK, optional artifacts missing |
| `FAILED` | Required artifact could not be produced |
| `SKIPPED` | Artifact was not requested (mutually exclusive with `FAILED`) |

- The Bundle as a whole succeeds if **all required artifacts** are `SUCCESS`.
- If any required artifact is `FAILED`, the Bundle is `FAILED` — this is the
  fix from E0.
- `PARTIAL_SUCCESS` only applies when optional artifacts are missing.
- `FAILED` is an **execution-time state**: a failed bundle is never published
  (`BundleExportService` raises `BundleExportError` before publish), so no
  committed Manifest ever carries `status="failed"`.  Published Manifests
  expose only `succeeded` / `partial` (lowercase), while per-node states in
  `ManifestExecutionNode.status` retain the lowercase `failed` / `blocked` /
  `cancelled` values.
- Exception boundary: an ordinary optional-node failure produces a
  `partial` bundle; session-fatal integrity failures (path traversal,
  undeclared files, missing files) fail the whole execution even when the
  failing node is optional, and are never published.

## Consequences

### What this enables

- D1+D2 can implement `DataSourceProvider` with confidence that provider IDs won't
  collide with exporter or validator IDs.
- E1 can evolve the Manifest schema with clear compatibility rules.
- The `DatasetRef` type can be implemented once in a shared location and consumed
  by both data and bundle code.
- Kafka streaming enters through a separate protocol without polluting the bounded
  data provider contract.

### What this constrains

- Any new identifier namespace must be registered here before use.
- `SourceConfig` YAML support is permanently removed; restoring it requires a new ADR.
- The Manifest schema version compatibility window is a hard constraint on E1/E2/E3.

### Migration impact

- Existing `SourceProvider` in `exporting/protocols.py` is renamed to
  `ExportSourceProvider` (E1). The old name is deprecated with a re-export shim.
- Existing provider IDs in prototype code (`data/provider.py`) are aligned to
  the `tributo.<name>` format.
- No existing Manifest data is affected (v1 remains v1).

### Rollback

- Identifier format changes are one-way. Rollback means reverting the rename
  before any downstream code depends on the new name.
- Manifest schema v1 is read-only from the moment v2 is introduced.

### Deprecation window

- `SourceProvider` → `ExportSourceProvider`: 2 minor versions with
  `DeprecationWarning`.
- Removed YAML support: effective immediately (was never documented as
  supported in JSON-first config model).

## Related Documents

- `docs/architecture/product-scope.md` — Framework/SDK positioning and Non-Goals
- `docs/architecture/version-policy.md` — SemVer, API stability tiers, Manifest/Plugin versioning
- `docs/architecture/migration-safety.md` — Migration stop-loss rules and rollback strategy
- `docs/architecture/benchmark-protocol.md` — Reproducible benchmark methodology
- `docs/architecture/walking-skeleton.md` — End-to-end contract boundary design
- `docs/architecture/call-chain-inventory.md` — Current data/export call chains
- `docs/architecture/decision-log.md` — Go/No-Go decisions for condition tracks
- `docs/STABILITY.md` — Module-level stability inventory

<!-- END -->
