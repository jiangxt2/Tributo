# Call chain inventory

Documenting every data entry point and model export entry point in the current
architecture candidate. Support claims are governed by Product Scope and the
real-infrastructure gates, not by this inventory alone.

## Data entry points

### Training data loading

**Primary entry**: `training/data_loader.py`

```
User code
  ├─ load_ray_dataset_from_source(source: CanonicalSourceInput)
  │    └─ resolve_file_source_path()
  │         └─ _load_via_ingestion(CanonicalSourceInput)
  │
  └─ load_ray_dataset_from_config(config: dict)        ← legacy compatibility entry
       └─ LegacySourceInput(raw=config, mode="legacy")
            └─ _load_via_ingestion(LegacySourceInput)
                 ├─ LegacyConfigNormalizer.normalize()
                 └─ resolve_file_source_path()

Both `_load_via_ingestion()` branches then perform:
  └─ require_local_file_source_exists()
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
flat dictionaries through `LegacyConfigNormalizer`; the training loader then
constructs the explicit `IngestionRequest` shown above. `load_dataframe_from_config()`
materializes its Ray result for small historical callers. Neither selects a
separate reader backend or invokes a legacy execution adapter.

### Bounded data writing

**Shared control-plane entry**: `data.writing.WriteGateway`

```
ParquetResultSink / LanceResultSink / bounded-write caller
  └─ explicit WriteRequest

Any entry above
  ↓
WriteTargetRegistry → LogicalWritePlan
  ↓
WriteBindingRegistry → credential-free capability negotiation
  ↓
WriteGateway.execute(typed RayDataHandle or DaftDataFrameHandle)
  ↓
RayWriteBinding → ray.data.Dataset.write_*
DaftWriteBinding → daft.DataFrame.write_*
  ↓
WriteReceipt
```

The Gateway owns request validation, target planning, capability checks,
credential-free receipts, and error redaction. Ray Data or Daft owns the
distributed data plane and native file/table commit. Iceberg catalog loading or
table creation in a binding is control-plane preflight only. No Tributo
consumer may bypass the Gateway with `to_arrow()`, a PyIceberg data mutation,
or a Lance fragment/commit helper.

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
  → EngineBindings.compile(engine_id, plan, binding_id, ...)
                                        descriptor selection and capability validation
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

`EngineBindings.resolve(key)` is the registry's exact descriptor lookup API; the
Gateway's `open()` path uses `EngineBindings.compile()`, which performs binding
selection and capability validation internally. `describe()` exposes the same
validation without compiling an engine-native plan.

Built-ins register explicitly. Independent packages contribute the logical
Provider through `tributo.ingestion_providers` and the physical Ray/Daft
Binding through `tributo.ingestion_bindings`. Both descriptor-only SPIs are
versioned, discovered lazily, isolate bad plugins, and cannot replace an
existing route. Provider-declared projection/path semantics plus declarative
Binding filesystem/catalog/storage-format constraints prevent consumer
modules from adding source-name branches. These SPIs do not add a plugin
lifecycle or permit a third ingestion engine.

### Inference data loading

**Primary bundle-aware entry**: `inference/api.py`

```
User code
  ↓
InferenceRequest(model, input, named bindings, result sink, execution policy)
  ↓
InferenceResolver.resolve()
  ↓
BundleModelReferenceResolver → pinned BundleRef + verified signatures
  ↓
IngestionGatewayInputResolver.describe()
  → IngestionGateway.describe(engine="ray")
  → pinned binding ID + credential-free descriptor
  ↓
ResolvedInference (credential-free, immutable)
  ↓
Top-level runtime composition
  ├── BundleModelKernelProvider → serializable PredictionKernelFactory
  └── ResultSinkProvider.bind() → BoundResultSink
  ↓
Compatibility facade
  ├── IngestionGatewayInputResolver.open()
  │     → RayDataHandle.dataset + IngestionPlanReceipt
  └── ResolvedInference → PreparedInferencePlan
        (no input request, Bundle selection, or result-target request)
  ↓
RayMapBatchesExecutor.execute_prepared() / run_prepared_inference()
  ├── KernelBatchPredictor → injected PredictionKernelFactory
  │     → worker-local Runtime Flavor load → named tensor predict
  └── BoundResultSink → WriteGateway → WriteBinding
```

The core Executor never imports `training.data_loader`, Provider,
LogicalScanPlan, EngineBinding, BundleReader, FlavorRegistry, or a concrete
model runtime. Feature plus passthrough projection is appended to the
bounded-ingestion Transform IR before the Gateway opens the source. Source,
model, and sink storage profiles are resolved independently. Row counts remain
optional and no extra `Dataset.count()` job is launched for metrics.

`run_prepared_inference()` is the fully injected core entry: callers provide an
already-opened Ray input, stripped `PreparedInferencePlan`,
PredictionKernelFactory, and BoundResultSink. Existing `run_inference()` and
`run_resolved_inference()` remain compatibility facades that resolve those
ports through the top-level composition root.

External model references are normalized before this chain executes:

```
RegistryModelReference / ArtifactModelReference
  → explicit ModelImporter selected by provider ID
  → immutable version and content verification
  → verified Bundle publication
  → BundleRef + internal ResolvedModelSelection
```

The first-party importers support MLflow numeric versions and Aliases, explicit
ONNX artifacts, and canonical native XGBoost `ubj`/`xgboost-json` artifacts.
Both XGBoost formats route through the shared `xgboost-native-v1` flavor;
legacy first-party format-plus-variant references normalize at the contract
boundary. Alias resolution is frozen to a numeric version before Ray Job
submission. No registry SDK object or framework model object crosses into
`ResolvedInference`.

**Compatibility entry**: `inference/pipeline.py` retains `InferenceConfig`, raw
ONNX, and strict JSON parsing. Its read path now uses
`IngestionGatewayInputResolver`, and its output path uses `ParquetResultSink`.
The flat `s3_config` field warns and remains only inside this adapter.

**Post-training entry**: a published `BundleRef` plus parent run ID is bound by
`PostTrainingInferenceAction`. Inline and detached modes both create an
ordinary `InferenceRequest`; no Trainer or in-memory model crosses the domain
boundary.

### Batch explainability

**Primary entry**: `explainability/executor.py`

```
tributo explain / submit_explainability_job()
  → Ray Job driver
  → ExplainabilityRequest
  → run_batch_explainability()
       ├─ BundleExplainabilityModelProvider
       │    → verified ExplainabilityModelBinding
       │    → serializable ExplainabilityModelSessionFactory
       ├─ OperationStore → idempotency key, lease, and attempt state
       ├─ IngestionGatewayInputResolver → bounded Ray Data input
       └─ Ray Data map_batches(ExplainabilityBatchWorker)
            → ExplainabilityPlanner → ExplainerAdapter
            → injected ExplainabilityResultStore → ResultSink/data persistence
            → ExplainabilityReceipt + terminal operation record
```

The model-runtime adapter validates the request against the Bundle descriptor
and encapsulates exact manifest bytes inside its serializable worker factory;
the executor receives neither Bundle internals nor a concrete model provider.
Each attempt writes below a unique lease-token path. Output size and row limits
are checked before a successful receipt is recorded. Tree SHAP and explicitly
enabled model-agnostic SHAP are batch operations; model loading and dynamic
instance-capability checks remain inside the model-runtime adapter. Result
materialization, inspection, receipt bytes, and attempt cleanup are owned by
the injected data-persistence adapter rather than the Explainability executor.

### Vector-index operations

**Primary entries**: `vector_index/index_job.py`, `search.py`, and
`maintenance.py`

```
tributo vector <build|search|optimize|compact>
  → submit_vector_job() → Ray Job driver → run_job_request()
  → one validated operation request
       ├─ build_vector_index() → LanceRayAdapter.create_index()
       ├─ search_vectors() → fixed Lance version → global Top-K
       ├─ optimize_vector_indices() → index appended fragments
       └─ compact_vector_dataset() → compact files and recheck indices
  → coverage, runtime, search, or maintenance receipt
```

Direct Python calls use the active Ray context. CLI requests cross the Ray Jobs
control plane as a size-bounded, validated payload and import Lance dependencies
inside the job. Search opens one requested dataset version and returns bounded
inline rows or a Parquet result. Build and maintenance re-open the active
dataset after mutation and derive coverage evidence from Lance metadata.

### CLI data entry

**Primary entry**: `cli.py`

```
tributo submit --master <local|ray-jobs-endpoint> --config <json>
  ↓
CLI parses JSON → builds JobConfig / TrainingConfig
  ↓
RuntimeTarget selects Ray local runtime, Ray Jobs API, or an explicit managed provider
```

### Plugin and optional data providers

The removed `tributo.connectors` entry-point group has no runtime discovery
path. Bounded data extensions use the descriptor-only Provider and Binding
groups below; they are separate from the model/export plugin groups discovered
by `tributo.plugin`.

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

**Bounded-write Binding discovery**:

```
first default WriteGateway use
  ↓
importlib.metadata.entry_points(group="tributo.write_bindings")
  ↓
credential-safe descriptor validation and atomic registration
  ↓
WriteBinding selection; native dependency import occurs at factory/execute
```

Selected optional integrations (`ray-doris==1.0`, `daft-doris==1.0`,
`daft-clickhouse==1.0`) also have thin built-in descriptors and explicit install
diagnostics. The canonical full runtime locks these packages into the image,
but their adapters are not support claims until database infrastructure gates
pass. Ray and Daft routes remain explicit and are not interchangeable.

---

## Model export entry points

### First-party trainer Bundle path

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
     │           ├── xgboost_native.py      ← UBJ/JSON exporters, shared native flavor
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

XGBoost defaults to ONNX opset 12 plus UBJ in the same Bundle. Its canonical
UBJ and JSON exporters both declare the executable `xgboost-native-v1`
flavor. DNN and PU
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

### Bundle consumption path

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

### Deprecated raw-artifact path

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

### Plugin exporters

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

## Key observations

| Issue | Location | Impact |
|-------|----------|--------|
| Existing downstream consumers still expect Ray Dataset values | Training / local runner / inference | Their compatibility functions explicitly select Ray and delegate the same Gateway; native Daft consumption requires a later consumer capability change |
| A Ray-only consumer intentionally selects a Daft-only source | Consumer boundary | `adapt_daft_result_to_ray()` calls Daft's public adapter and records conversion evidence; the Gateway never performs this conversion implicitly |
| External model importers use explicit IDs and normalize to BundleRef | `inference/importers.py`, `integrations/model_importers/` | MLflow and typed ONNX/XGBoost artifacts pass Conformance and real-service IT |
| Raw-artifact compatibility still exists | `BaseTrainer.run(..., legacy_export=True)` | Removal waits for the documented compatibility window and a separate E4 change |
| ExportSourceProvider ≠ DataSourceProvider | `exporting/protocols.py` vs `data/provider.py` | The distinct names and ownership prevent model-checkpoint conversion from leaking into data ingestion |
| ONNX Runtime and native XGBoost have first-party runtime flavors | `exporting/runtime.py` | Safetensors and PT2 are readable Bundle artifacts without executable loaders; raw UBJ/JSON load only through the explicit `xgboost-native-v1` flavor |
| Hook delivery is in-process | `exporting/dispatch.py` | A committed Bundle survives Hook failure, but cross-process retry/recovery needs the separately scoped Outbox design |
| Transform pushdown optimization has no benchmark evidence; alpha Bindings classify current ETL as residual | `data/transform_compiler.py` | D4 remains NO-GO for pushdown claims |
| HDFS, Hive, ClickHouse, and Doris have incomplete delivery evidence | Bindings and external packages | Keep them at adapted/unsupported status until their real-infrastructure gates pass |

<!-- END -->
