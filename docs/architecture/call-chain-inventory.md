# Call Chain Inventory

Documenting every data entry point and model export entry point in the current
architecture candidate. Support claims are governed by Product Scope and the
real-infrastructure gates, not by this inventory alone.

## Data Entry Points

### 1. Training Data Loading

**Primary entry**: `training/data_loader.py`

```
User code
  ↓
load_ray_dataset_from_config(config: dict)        ← main public entry
  ↓
LegacySourceInput(raw=config, mode="legacy")
  ↓
load_ray_dataset_from_source(source: CanonicalSourceInput)
  ↓
IngestionRequest(source, engine="ray")
  ↓
IngestionGateway.open()
  ↓
ProviderRegistry.resolve() → normalize() → plan()
  ↓
EngineBindings.compile() → RayDataHandle.dataset
```

**Compatibility aliases**: `load_ray_dataset_from_config()` converts legacy
flat dictionaries through `LegacyConfigNormalizer`; `load_dataframe_from_config()`
materializes its Ray result for small historical callers. Neither selects a
separate reader backend.

**Canonical ingestion implementation**:
- `data/transform_ir.py` — versioned engine-neutral ETL contract.
- `data/scan_plan.py` — credential-free `FileScan` / `SqlScan` / `TableScan`.
- `data/engine_binding.py` — four-part Binding identity, constraint matching,
  capability negotiation, and deterministic selection.
- `data/provider_plugins.py` — versioned
  `tributo.ingestion_providers` descriptor discovery.
- `data/binding_plugins.py` — narrow
  `tributo.ingestion_bindings` descriptor discovery.
- `data/transform_compiler.py` — internal Ray/Daft expression translation.
- The unused `SourcePlan` / `SourceRouter` auto-routing prototype was removed;
  engine selection is explicit in `IngestionRequest`.
- `data/provider.py` owns logical normalization and planning;
  `DatasetHandle` remains a legacy Ray compatibility type, not a second reader.

**Canonical Gateway path**:

```
CanonicalSourceInput (provider/uri or type/path/dialect shapes)
  → ProviderRegistry.resolve()           exact ID → alias → built-in mapping
                                         after lazy Provider-plugin discovery
  → provider.normalize() → ResolvedSource(provider_id, canonical_uri, options)
  → provider.plan() → LogicalScanPlan
  → EngineBindings.resolve(engine, scan_kind, connector, binding_id/constraints)
  → thin Binding → public Ray Data / Daft / installed connector API
  → IngestionOpenResult(Typed Handle, IngestionPlanReceipt, ownership)

IngestionGateway.describe(IngestionRequest)
  → resolve relative built-in file paths against project_root_path
  → ProviderRegistry.resolve() → provider.normalize() → provider.plan()
  → EngineBindings.describe(engine_id, scan_kind, connector_id, constraints)
  → credential-free IngestionDescriptor (no engine plan or metadata I/O)

IngestionGateway.open(IngestionRequest)
  → the same Provider → LogicalScanPlan route
  → EngineBindings.compile(engine_id, scan_kind, connector_id, binding_id)
  → thin Binding → public Ray Data / Daft / installed connector reader API
  → IngestionOpenResult(Typed Handle, IngestionPlanReceipt, ownership)
```

The relative-path resolver is shared with the beta compatibility entrypoint.
The resolved path reaches `ResolvedSource` before source and plan digests are
computed, so migrating entrypoints cannot silently change either the file read
or its identity. URI sources and absolute paths are unchanged.

Built-ins register explicitly. Independent packages contribute the logical
Provider through `tributo.ingestion_providers` and the physical Ray/Daft
Binding through `tributo.ingestion_bindings`. Both descriptor-only SPIs are
versioned, discovered lazily, isolate bad plugins, and cannot replace an
existing route. Provider-declared projection/path semantics plus declarative
Binding filesystem/catalog/storage-format constraints prevent consumer
modules from adding source-name branches. These SPIs do not add a plugin
lifecycle or permit a third ingestion engine.

### Inference Data Loading

**Primary bundle-aware entry**: `inference/api.py`

```
User code
  ↓
InferenceRequest(model, input, named bindings, result sink, execution policy)
  ↓
InferenceResolver.resolve()
  ↓
BundleReader.read_manifest_with_bytes() → pinned BundleRef
  ↓
IngestionGatewayInputResolver.describe()
  → IngestionGateway.describe(engine="ray")
  → pinned binding ID + credential-free descriptor
  ↓
ResolvedInference (credential-free, immutable)
  ↓
RayMapBatchesExecutor
  ├── IngestionGatewayInputResolver.open()
  │     → worker-local describe() + open()
  │     → RayDataHandle.dataset + IngestionPlanReceipt
  ├── BundleBatchPredictor → BundleModelLoader → named tensor predict
  └── ParquetResultSink → Dataset.write_parquet()
```

The Executor never imports `training.data_loader`, Provider, LogicalScanPlan,
or EngineBinding. Feature plus passthrough projection is appended to the
bounded-ingestion Transform IR before the Gateway opens the source. Source,
model, and sink storage profiles are resolved independently. Row counts remain
optional and no extra `Dataset.count()` job is launched for metrics.

External model references are normalized before this chain executes:

```
RegistryModelReference / ArtifactModelReference
  → explicit ModelImporter selected by provider ID
  → immutable version and content verification
  → verified Bundle publication
  → BundleRef + internal ResolvedModelSelection
```

The first-party importers support MLflow numeric versions and Aliases, explicit
ONNX artifacts, and explicit native XGBoost JSON/UBJ artifacts. Alias resolution
is frozen to a numeric version before Ray Job submission. No registry SDK object
or framework model object crosses into `ResolvedInference`.

**Compatibility entry**: `inference/pipeline.py` retains `InferenceConfig`, raw
ONNX, and strict JSON parsing. Its read path now uses
`IngestionGatewayInputResolver`, and its output path uses `ParquetResultSink`.
The flat `s3_config` field warns and remains only inside this adapter.

**Post-training entry**: a published `BundleRef` plus parent run ID is bound by
`PostTrainingInferenceAction`. Inline and detached modes both create an
ordinary `InferenceRequest`; no Trainer or in-memory model crosses the domain
boundary.

### 3. Embedding Data Loading

**Primary entry**: `embeddings/job_runner.py`

```
CLI / Python API
  ↓
submit_embedding_job(source=... or s3_input_path=...)
  ↓
IngestionRequest(source, explicit engine)
  ↓
credential-safe source serialization + deterministic submission identity
  (no Provider/Binding resolution on the submit host)
  ↓
Ray Job submission (entrypoint script runs inside Ray cluster)
  ↓
embeddings.batch_job._resolve_embedding_source()
  ↓
IngestionGateway.open(explicit engine)
  ↓
RayDataHandle.dataset, or explicit DaftDataFrameHandle → Ray adapter
  ↓
Ray Data → model inference → output writer
```

Ray Job entrypoints carry only source configuration accepted by
`IngestionRequest.source_json_for_remote_transport()`; credentials are resolved
from the cluster environment, IAM, or a storage profile. Optional Provider,
Binding, and Connector dependencies are resolved inside the cluster, never on
the submit host. Embeddings never calls Provider normalization or a registry
directly and never performs automatic engine fallback.

### 4. CLI Data Entry

**Primary entry**: `cli.py`

```
tributo submit --config <json>
  ↓
CLI parses JSON → builds JobConfig / TrainingConfig
  ↓
submits to Ray Job API or calls local runner
```

### 5. Plugin and Optional Data Connectors

**Discovery**: `plugin.py::discover_connector_plugins()`

```
importlib.metadata.entry_points(group="tributo.connectors")
  ↓
DataConnector subclass validation
  ↓
Registered in connector registry (data/registry.py)
```

**Currently**: No third-party connector plugins exist. All built-in connectors
are in `data/` module. This historical SPI serves `DataConnector` compatibility
and write paths.

**Bounded-ingestion Binding discovery**:

```
first default EngineBindings use
  ↓
importlib.metadata.entry_points(group="tributo.ingestion_bindings")
  ↓
credential-safe descriptor validation and atomic registration
  ↓
four-part Binding selection; native dependency import occurs at factory/compile
```

Selected optional integrations (`ray-doris`, `daft-olap-connectors`) also have
thin built-in descriptors and explicit install diagnostics. Their adapters are
not support claims until their external packages and infrastructure gates pass.

---

## Model Export Entry Points

### First-Party Trainer Bundle Path

**Entry**: `training/base.py::BaseTrainer.run`

```
XGBoostTrainerImpl / DNNTrainerImpl / PUTrainerImpl
  ↓
BaseTrainer.run(bundle_config=BundleOutputConfig(bundle_uri=...))
  ↓
TrainingLifecycle
  ├── preflight explicit Bundle URI before setup
  ├── fill trainer-owned default targets and inference role
  ├── setup → training_loop → ExportCheckpointV1
  ├── ExportSourceProvider.open_source(checkpoint)
  │     └── Ray Checkpoint conversion stays inside as_directory()
  └── BundleExportService.export_bundle(source, config)
        ├── Hook preflight before planning or staging
        ├── Planner.plan(source, targets)
        ├── ExportManager.execute(plan)
     ├── ModelExporter.export(context, source, upstream, target)
     │     └── integrations/exporters/
     │           ├── xgboost_onnx.py        ← XGBoost → ONNX
     │           ├── xgboost_native.py      ← distinct UBJ and JSON exporters
     │           ├── torch_onnx.py          ← torch → ONNX
     │           ├── torch_export.py        ← torch.export
     │           ├── torch_safetensors.py   ← torch → safetensors
     │           ├── hf_onnx.py             ← HuggingFace → ONNX
     │           └── onnx_quantizer.py      ← ONNX FP32 → quantized
     │
     └── ExportValidator.validate(artifact)
           └── integrations/validators/
                 └── onnx_runtime.py        ← ONNX Runtime inference check
        ├── Publisher → BundleRepository commit
        │     └── PublishedBundle(BundleResult + exact manifest bytes)
        ├── OperationEvent(bundle.published)
        │     └── derived from the exact bytes that won the commit
        └── InlineHookDispatcher
              ├── OperationStore claim/complete
              ├── BundleArtifactAccessor (committed Bundle only)
              └── MLflowPostPublishHook when explicitly configured
  ↓
TrainingResult projection
  (model_uri, bundle_uri, metrics, legacy_artifact_uri,
   training_status, bundle_status, hook_status, execution_id)
```

XGBoost defaults to ONNX opset 12 plus UBJ in the same Bundle. DNN and PU
default to ONNX opset 18. All three bind the `inference` role to
`onnx-model`. An omitted destination fails before trainer setup; there is no
implicit current-directory or temporary output.

Hook execution is synchronous. `OperationEvent` is an in-process delivery
contract derived from the exact committed manifest bytes; the service does not
perform a second repository GET. `ExecutionRecord` and delivery records remain
the persisted operation facts. Outbox and asynchronous Hook workers are not
part of this call chain.

Hook plugins are resolved only when listed in `BundleOutputConfig.hooks`.
Plugin resolution and option validation happen before planning or staging. A
required Hook failure raises `PostPublishCallbackError` with the committed
`BundleResult` and receipts; it never rolls back the Bundle commit. The same
error exposes the terminal `TrainingResult` as `error.training_result`.

### Bundle Consumption Path

**Entry**: `exporting/bundle_reader.py::BundleReader`

```
BundleRef or direct immutable Bundle URI
  ↓
BundleRepositoryRouter
  ├── optional storage-alias resolution → immutable BundleRef
  └── repository.read_manifest() → ExportManifest + exact bytes
        ├── verify BundleRef manifest_sha256 and bundle_id
        └── resolve exactly one artifact by role or name
              ↓
        repository.materialize_artifact()
              ├── descriptor equality and resource bounds
              ├── digest verification
              └── context-managed ResolvedArtifact
```

Runtime, inference, serving, and Hook artifact access propagate the parsed
manifest together with its exact bytes. A supplied manifest without matching
bytes is rejected, and an already verified snapshot is never paired with a
second alias read.

`BundleRef` and storage aliases provide an external identity anchor through
the expected manifest digest and `bundle_id`. A direct immutable Bundle URI
provides self-consistency checks only: the Reader still validates the schema,
paths, sizes, and every artifact digest, but cannot detect replacement by a
different internally consistent Manifest without an independently retained
digest. Consumers that require replacement detection must retain a
`BundleRef` or publication event rather than reconstructing a raw URI.

### Deprecated Raw-Artifact Path

**Entry**: `training/exporters/` (re-exports from `exporting`)

```
BaseTrainer.run(output_path=..., legacy_export=True)
  ↓
export_artifacts() / export_model()
  ├── torch_onnx_exporter.py   ← old torch→ONNX (uses exporting internally)
  ├── safetensors.py           ← old safetensors export
  ├── torchscript.py           ← TorchScript export
  └── causal_report.py         ← Causal inference report
```

The compatibility path is never selected by missing targets or an environment
flag. It requires an explicit per-call switch and emits `DeprecationWarning`.
New code must consume Bundle URI and role rather than `legacy_artifact_uri`.

### Plugin Exporters

**Discovery**: `plugin.py::discover_exporter_plugins()`

```
importlib.metadata.entry_points(group="tributo.exporters")
  ↓
ModelExporter validation (api_version==2, exporter_id matches entry-point name)
  ↓
Registered in ExportRegistry (exporting/registries.py)
```

Plugin groups also discovered:
- `tributo.source_providers` → `discover_source_provider_plugins()`
- `tributo.validators` → `discover_validator_plugins()`
- `tributo.model_flavors` → `discover_flavor_plugins()`
- `tributo.model_factories` → `discover_model_factory_plugins()`

---

## Key Observations

| Issue | Location | Impact |
|-------|----------|--------|
| Existing downstream consumers still expect Ray Dataset values | Training / local runner / inference / embeddings | Their compatibility functions explicitly select Ray and delegate the same Gateway; native Daft consumption requires a later consumer capability change |
| A Ray-only consumer intentionally selects a Daft-only source | Consumer boundary | `adapt_daft_result_to_ray()` calls Daft's public adapter and records conversion evidence; the Gateway never performs this conversion implicitly |
| External model importers use explicit IDs and normalize to BundleRef | `inference/importers.py`, `integrations/model_importers/` | MLflow and typed ONNX/XGBoost artifacts pass Conformance and real-service IT |
| Raw-artifact compatibility still exists | `BaseTrainer.run(..., legacy_export=True)` | Removal waits for the documented compatibility window and a separate E4 change |
| ExportSourceProvider ≠ DataSourceProvider | `exporting/protocols.py` vs `data/provider.py` | The distinct names and ownership prevent model-checkpoint conversion from leaking into data ingestion |
| Only ONNX has a first-party runtime flavor | `exporting/runtime.py` | UBJ, JSON, Safetensors, and PT2 are exportable/readable artifacts, not automatically batch- or serve-capable models |
| Hook delivery is in-process | `exporting/dispatch.py` | A committed Bundle survives Hook failure, but cross-process retry/recovery needs the separately scoped Outbox design |
| Transform pushdown optimization has no benchmark evidence; alpha Bindings classify current ETL as residual | `data/transform_compiler.py` | D4 remains NO-GO for pushdown claims |
| HDFS, Hive, ClickHouse, and Doris have incomplete delivery evidence | Bindings and external packages | Keep them at adapted/unsupported status until their real-infrastructure gates pass |

<!-- END -->
