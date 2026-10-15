# Version policy

How Tributo handles versioning across the Python package, API stability tiers,
Bundle Manifests, and Plugin contracts.

## Package Version (SemVer)

Tributo follows [Semantic Versioning 2.0.0](https://semver.org/).

| Version component | When to increment |
|------------------|-------------------|
| **Major** (X.0.0) | Breaking change to `stable` public API; removal of deprecated API after compatibility window |
| **Minor** (0.X.0) | New `stable` or `beta` feature; `beta` API change with deprecation notice; new optional dependency |
| **Patch** (0.0.X) | Bug fix; internal refactor with no API change; dependency version bump within declared range |

Current version: **1.2.0** (as declared in `pyproject.toml`).

`pyproject.toml` is the authoritative source. README, Docker, CI, and lock
files must be verified against it whenever the version changes.

### Development and release branches

Formal releases use SemVer numbers. Python development versions use the
[PEP 440 development-release format](https://packaging.python.org/en/latest/specifications/version-specifiers/#developmental-releases).
`master` declares `1.3.0.dev0` for ongoing development. The `release-1.2.0`
branch declares `1.2.0`, and the `tributo-1.2.0` tag must identify its exact
validated commit. A development version does not determine whether the next
published release is a patch or a minor version.

Release branches start from an explicitly reviewed commit. Their formal-version
changes are not merged back into `master`. Patch releases start from the
appropriate release snapshot and receive selected fixes rather than all new
master features. Beta compatibility windows count published minor releases;
development versions do not consume those windows. Manifest schema and Plugin
API versions keep their independent identities.

README and installation examples distinguish the published source tag from
the development checkout. Update the mainline's published installation entry
only after the corresponding GitHub source release and PyPI package exist.

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
- `beta` → `stable`: The public contract, supported scope, and limitations are
  documented; compatibility and relevant functional validation pass; no known
  unresolved interface issues affect that scope. There is no minimum release
  count or waiting period.
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
| `ModelExporter` | 2 | `exporting/protocols.py` |
| `ExportValidator` | 1 | `exporting/protocols.py` |
| `ExportSourceProvider` (renamed from `SourceProvider` in E1) | 1 | `exporting/protocols.py` |
| `ModelFactory` | 1 | `exporting/protocols.py` |
| `ModelFlavor` | 1 | `integrations/` |
| Bounded-ingestion Provider descriptor | 1 | `data/provider_plugins.py` |
| Bounded-ingestion Binding descriptor | 1 | `data/binding_plugins.py` |
| Distributed algorithm descriptor | 1, 2 | `algorithms/api/descriptor.py`; v2 requires explicit contract bindings |
| DistributionSpec | 1 | `algorithms/api/distribution.py` |
| ExecutionReceipt | 1 | `algorithms/api/execution.py` |
| Trainer entry points | N/A (not versioned) | `plugin.py` |

### Plugin Compatibility Rules

- `api_version` is **exact-match** — a plugin declaring `api_version=1` is
  only loaded by a framework that expects `api_version=1`.
- When a protocol changes incompatibly, the framework increments its expected
  `api_version`. Old plugins are rejected with a diagnostic message stating
  the required version.
- The framework MAY support multiple `api_version` values concurrently during
  a transition period (e.g., accept both v1 and v2 plugins). This requires
  explicit code — the default is single-version.
- Bounded-ingestion Provider descriptors declare an exact plugin distribution
  version plus a Tributo version specifier. Binding descriptors declare an exact
  plugin distribution version and an engine version specifier. Discovery
  rejects incompatible descriptors, isolates the bad entry point, and never
  replaces an already registered route.
- Distributed algorithm descriptors declare an exact installed package version,
  a Tributo package compatibility range, and an exact descriptor API version.
  Discovery also verifies inheritance from the strategy named by
  `DistributionSpec`; a failing package remains diagnostic-only and is not
  partially registered.
- Older unversioned plugin groups retain their existing behavior. They do not
  inherit the stronger guarantees of bounded-ingestion or distributed-algorithm
  descriptor discovery.

### Core development-version matching

Provider registration, distributed algorithm descriptor validation, offline
algorithm Bundle validation, and support-evidence matching share one Core
version rule. An explicitly selected Core development version may match its
declared range. The full version remains subject
to PEP 440 bounds, exact pins, and exclusions; it is not converted into the
corresponding formal release.

For `1.3.0.dev0`, `>=1,<2` matches, while `<1.3`, `>=1.3`, `==1.2.0`, and
`!=1.3.*` do not. Formal versions retain the default matching behavior.
Release candidates and beta versions require permission in their version
specifier; this rule does not automatically admit them.

Plugin package versions, descriptor API versions, Wheel digests, and support
evidence identity remain strict. Matching a development version does not award
execution support, change the version recorded in evidence, or certify a
different Core/Wheel combination. Ray, Daft, and Python version checks retain
their independent policies. Development installations must select Core source
or a Core Wheel explicitly; this runtime rule does not change package-index
resolution.

### Plugin lifecycle before PL1 and PL2

- Discovery: independent `importlib.metadata.entry_points()` loaders. Bounded
  ingestion discovers Provider and Binding descriptors lazily on first use;
  older plugin groups retain their existing import-time behavior.
- Validation: Structural check (`api_version`, required classvars), then
  `api_version == expected`.
- Filtering: `TRIBUTO_PLUGINS` env var for selective loading.
- No `PluginManager` class exists yet — each `discover_*` function is
  independent.

### Plugin lifecycle after PL1 and PL2 approval

- `PluginManager.discover()` — centralized discovery with caching.
- `PluginManager.load(group)` — load plugins for a specific group.
- `PluginManager.validate()` — run contract test suite.
- `PluginManager.reset()` — clear caches for test isolation.
- Structured diagnostics for version conflicts, missing capabilities, and
  import failures.

## Dependency Version Policy

- **Ray**: Pinned to the exact version declared in `pyproject.toml`. Ray's own SemVer does not
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

### Maintaining the runtime baseline

SDK defaults, runtime checks, and installation hints share
`src/tributo/_common/runtime_versions.json`. Ordinary test fixtures read
`uv.lock` independently; tests compare that lock and the dependency declarations
with the SDK baseline. Compatibility boundary tests and source audit notes keep
their explicit versions and require review during an upgrade.

Image builders share pinned base image inputs from
`tools/tributo-runtime-full.json`. Managed IT profiles select their package
checks from `uv.lock`; they do not repeat package version numbers. Use the
repository builders, which supply the required Docker build arguments
and named contexts. Shell runner version files remain explicit inputs for
their separate integration suites.

For an upgrade, update the SDK baseline, dependency declarations and lock,
image inputs, and affected shell runner inputs. Review the
[installation requirements](../getting-started/installation.md),
[support matrix](../reference/support-matrix.md), compatibility tests, and
source audit notes against the new validation evidence. General guides link
to those references; historical recipes retain the versions actually tested.

### Data dependency upgrade gates

- Ray 2.59.0 still accepts Iceberg `row_filter` and `selected_fields`, but both
  reader arguments are deprecated. Before adopting a Ray release that removes these arguments, move filtering and
  projection to the replacement public Dataset APIs and rerun local/S3,
  empty-table, and dual-engine Iceberg Conformance before widening the pin.
- A PyIceberg range change must rerun named-profile, session-token, path-style,
  FileIO rejection, empty-table, and MinIO Iceberg tests. Tributo must not
  support a second FileIO by inheriting backend-specific string semantics;
  that requires a separately declared and tested Connector/Binding.
- Connector extras declare minimum stable versions: `daft-doris>=1.1`,
  `ray-hive>=1.2`, `ray-doris>=1.3`, and `ray-clickhouse>=1.1`. These minimums
  apply to master and release branches and increase only when required APIs or
  fixes justify a change. SDK bindings record installed versions and retain
  driver/worker consistency checks; exact versions and hashes belong to the
  locked test/image environment. A lower bound does not certify every future
  release. Known incompatible versions require an evidence-based exclusion or
  upper bound. Changes to the Hive adapter or connector behavior require
  affected Provider/Binding tests and the
  Hive Data Ingestion Gate for worker execution, projection, schema, receipt,
  and credential boundaries.

<!-- END -->
