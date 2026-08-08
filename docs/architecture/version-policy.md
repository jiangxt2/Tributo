# Version Policy

How Tributo handles versioning across the Python package, API stability tiers,
Bundle Manifests, and Plugin contracts.

## Package Version (SemVer)

Tributo follows [Semantic Versioning 2.0.0](https://semver.org/).

| Version component | When to increment |
|------------------|-------------------|
| **Major** (X.0.0) | Breaking change to `stable` public API; removal of deprecated API after compatibility window |
| **Minor** (0.X.0) | New `stable` or `beta` feature; `beta` API change with deprecation notice; new optional dependency |
| **Patch** (0.0.X) | Bug fix; internal refactor with no API change; dependency version bump within declared range |

Current version: **1.0.0** (as declared in `pyproject.toml`).

`pyproject.toml` is the authoritative source. README, Docker, CI, and lock
files must be verified against it whenever the version changes.

## API Stability Tiers

Every public symbol falls into one of these tiers. The canonical inventory is
`docs/STABILITY.md`.

| Tier | Annotation | Compatibility Promise | Change Rules |
|------|-----------|----------------------|-------------|
| `stable` | `@PublicAPI(stability="stable")` | Backward-compatible within the same major version | Breaking change → major version bump + migration guide |
| `beta` | `@PublicAPI(stability="beta")` | May change with notice | Change → deprecation warning + ≥ 2 minor versions before removal |
| `alpha` | `@PublicAPI(stability="alpha")` | No compatibility promise | May change or be removed without notice |
| `deprecated` | (module docstring or decorator) | Will be removed | Must state replacement and removal timeline |
| `prototype` | (module docstring) | Not for production use | May be rewritten or deleted at any time |
| `developer` | `@DeveloperAPI` or unannotated | Internal; not part of public API | May change in any release |

### When to Change Stability

- `alpha` → `beta`: The API has been used in at least one real workload and
  the interface is unlikely to change significantly.
- `beta` → `stable`: The API has been stable for ≥ 2 minor versions with no
  reported interface issues.
- `stable` → `deprecated`: A replacement exists, the deprecation window has
  started, and the migration guide is published.

Stability changes must be recorded in `STABILITY.md` and the PR description
must include a `Migration impact` section.

## Manifest Schema Version

`ExportManifest.schema_version` is an **integer** independent of the Tributo
package version.

| Rule | Detail |
|------|--------|
| Increment trigger | Adding a required field, changing field semantics, or changing the digest algorithm |
| Reader compatibility | A vN reader MUST read v(N-1) and v(N-2) manifests |
| Unknown version | A reader encountering v(N+2)+ MUST fail-fast with a clear error message stating the minimum Tributo version required |
| Writer compatibility | A writer MAY write only the latest schema version. It MUST NOT silently write an older version to paper over reader gaps |
| Compatibility window | 2 schema versions or 2 Tributo minor versions, whichever is longer |

Current schema version: **1**.

v1 → v2 trigger examples:
- Adding a required `framework_version` field to `ManifestSourceInfo`.
- Changing the digest algorithm from SHA-256 to SHA-512.
- Adding a `parent_manifest_sha256` field for lineage tracking.

Not a v2 trigger:
- Adding an optional field with a default value.
- Adding a new `LogicalArtifact` format (handled by format negotiation, not
  schema version).
- Changing internal implementation details of the Reader.

## Plugin API Version

Plugin contracts use `api_version` (integer), independent of both the package
version and the Manifest schema version.

| Contract | Current `api_version` | Defined In |
|----------|----------------------|-----------|
| `ModelExporter` | 1 | `exporting/protocols.py` |
| `ExportValidator` | 1 | `exporting/protocols.py` |
| `ExportSourceProvider` (renamed from `SourceProvider` in E1) | 1 | `exporting/protocols.py` |
| `ModelFactory` | 1 | `exporting/protocols.py` |
| `ModelFlavor` | 1 | `integrations/` |
| Bounded-ingestion Provider descriptor | 1 | `data/provider_plugins.py` |
| Bounded-ingestion Binding descriptor | 1 | `data/binding_plugins.py` |
| Trainer entry points | N/A (not versioned) | `plugin.py` |
| Connector entry points | N/A (not versioned) | `plugin.py` |

### Plugin Compatibility Rules

- `api_version` is **exact-match** — a plugin declaring `api_version=1` is
  only loaded by a framework that expects `api_version=1`.
- When a protocol changes incompatibly, the framework increments its expected
  `api_version`. Old plugins are rejected with a diagnostic message stating
  the required version.
- The framework MAY support multiple `api_version` values concurrently during
  a transition period (e.g., accept both v1 and v2 plugins). This requires
  explicit code — the default is single-version.
- Bounded-ingestion Provider and Binding descriptors declare an exact plugin
  distribution version plus a Tributo version specifier. Discovery rejects a
  descriptor when either installed version disagrees, isolates the bad entry
  point, and never replaces an already registered route.
- Other plugin authors declare compatibility range via `min_tributo_version`
  and `max_tributo_version` (optional, for future PL1+PL2). Current non-data
  plugins do not declare this.

### Plugin Lifecycle (Current — Pre-PL1+PL2)

- Discovery: independent `importlib.metadata.entry_points()` loaders. Bounded
  ingestion discovers Provider and Binding descriptors lazily on first use;
  older plugin groups retain their existing import-time behavior.
- Validation: Structural check (`api_version`, required classvars), then
  `api_version == expected`.
- Filtering: `TRIBUTO_PLUGINS` env var for selective loading.
- No `PluginManager` class exists yet — each `discover_*` function is
  independent.

### Plugin Lifecycle (Future — PL1+PL2, Go-required)

- `PluginManager.discover()` — centralized discovery with caching.
- `PluginManager.load(group)` — load plugins for a specific group.
- `PluginManager.validate()` — run contract test suite.
- `PluginManager.reset()` — clear caches for test isolation.
- Structured diagnostics for version conflicts, missing capabilities, and
  import failures.

## Dependency Version Policy

- **Ray**: Pinned to exact version (`==2.55.1`). Ray's own SemVer does not
  guarantee Train/Data/Serve API stability across minor versions.
- **PyIceberg**: Constrained to `>=0.11.1,<0.12.0`. Built-in Iceberg bindings
  select `PyArrowFileIO`; named profiles are materialized into its public S3
  properties and alternative FileIO implementations fail closed.
- **Core dependencies** (pydantic, click): Lower-bound only (`>=X.Y.Z`),
  trusting their SemVer.
- **Optional dependencies** (onnxruntime, xgboost, torch, transformers):
  Lower-bound with upper-bound where known incompatibilities exist.
- **Lock file**: `uv.lock` is committed to the repository and regenerated on
  any dependency change.
- **Transitive-only dependencies** (PyMySQL, ConnectorX): Declared as optional
  extras; imported at runtime with `ModuleNotFoundError` → install hint.

### Data dependency upgrade gates

- Ray 2.55.1 still accepts Iceberg `row_filter` and `selected_fields`, but both
  reader arguments are deprecated. A Ray upgrade must first move filtering and
  projection to the replacement public Dataset APIs, then rerun local/S3,
  empty-table, and dual-engine Iceberg Conformance before widening the pin.
- A PyIceberg range change must rerun named-profile, session-token, path-style,
  FileIO rejection, empty-table, and MinIO Iceberg tests. Tributo must not
  support a second FileIO by inheriting backend-specific string semantics;
  that requires a separately declared and tested Connector/Binding.

<!-- END -->
