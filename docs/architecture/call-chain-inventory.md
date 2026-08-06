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

Built-in and selected optional Bindings are registered explicitly. Independent
packages can contribute versioned descriptors through
`tributo.ingestion_bindings`; this descriptor-only SPI does not add a plugin
lifecycle or permit a third ingestion engine.

### 2. Inference Data Loading

**Primary entry**: `inference/pipeline.py`

```
User code
  ↓
InferenceConfig(source=... or legacy input fields)
  ↓
run_batch_inference(config: InferenceConfig)       ← main public entry
  ↓
_legacy_source(config) or canonical source
  ↓
load_ray_dataset_from_source(source.model_dump(mode="python"))
  ↓
IngestionGateway.open(engine="ray") → RayDataHandle.dataset
```

Legacy JSON enters through `_legacy_json_source()` and is normalized to the same
canonical source object before the provider loader is called. Feature columns
are applied through the provider's native projection option.

**Secondary entry**: `run_inference_from_json(config_path)` — reads JSON → builds
`InferenceConfig` → calls `run_batch_inference()`.

### 3. Embedding Data Loading

**Primary entry**: `embeddings/job_runner.py`

```
CLI / Python API
  ↓
submit_embedding_job(source=... or s3_input_path=...)
  ↓
Ray Job submission (entrypoint script runs inside Ray cluster)
  ↓
embeddings.batch_job._resolve_embedding_source()
  ↓
load_ray_dataset_from_source(source.model_dump(mode="python"))
  ↓
IngestionGateway.open(engine="ray") → RayDataHandle.dataset
  ↓
Ray Data → model inference → output writer
```

Ray Job entrypoints carry only credential-free source configuration; credentials
are resolved from the cluster environment or IAM.

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

### 1. New Bundle Export (Primary)

**Entry**: `exporting/service.py::BundleExportService`

```
Trainer.fit() result
  ↓
BundleExportService.export(result)
  ↓
  1. SourceProvider.open_source(result)     ← resolves ExportSource
  2. Planner.plan(source, targets)          ← builds export DAG
  3. Executor.execute(plan)                 ← runs exporters
     ├── ModelExporter.export(context, source, upstream, target)
     │     └── integrations/exporters/
     │           ├── xgboost_onnx.py        ← XGBoost → ONNX
     │           ├── xgboost_native.py      ← XGBoost native JSON/UBJ
     │           ├── torch_onnx.py          ← torch → ONNX
     │           ├── torch_export.py        ← torch.export
     │           ├── torch_safetensors.py   ← torch → safetensors
     │           ├── hf_onnx.py             ← HuggingFace → ONNX
     │           └── onnx_quantizer.py      ← ONNX FP32 → quantized
     │
     └── ExportValidator.validate(artifact)
           └── integrations/validators/
                 └── onnx_runtime.py        ← ONNX Runtime inference check
  4. Publisher.publish(bundle)              ← local / S3
  5. Hook callbacks (MLflow, etc.)
  ↓
BundleRef (manifest_sha256, uri, status)
```

### 2. Old Per-Trainer Export (Deprecated)

**Entry**: `training/exporters/` (re-exports from `exporting`)

```
Trainer.export_model()
  ↓
training/exporters/
  ├── torch_onnx_exporter.py   ← old torch→ONNX (uses exporting internally)
  ├── safetensors.py           ← old safetensors export
  ├── torchscript.py           ← TorchScript export
  └── causal_report.py         ← Causal inference report
  ↓
training/onnx_exporter.py      ← legacy XGBoost ONNX (try/except swallows errors)
```

**Key issue**: `training/onnx_exporter.py` catches exceptions and logs them
without failing the task — the E0 fix target.

### 3. Trainer-Level Export Integration

**XGBoost**: `training/xgboost_trainer.py`

```
run_training_from_json(config_path)
  ↓
build_trainer(config)
  ↓
XGBoostTrainerImpl.fit()
  ├── export_artifacts()      ← new path → BundleExportService
  └── export_model()           ← old path → training/onnx_exporter.py
```

**DNN**: `training/dnn_trainer.py`

```
DNNTrainer.fit()
  └── export_model()           ← calls training/exporters/
```

**PU**: `training/pu_trainer.py`

```
PUTrainer.fit()
  └── export_model()           ← calls training/exporters/
```

### 4. Plugin Exporters

**Discovery**: `plugin.py::discover_exporter_plugins()`

```
importlib.metadata.entry_points(group="tributo.exporters")
  ↓
ModelExporter validation (api_version==1, exporter_id matches entry-point name)
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
| Old XGBoost ONNX export swallows errors | `training/onnx_exporter.py` | E0 fix target |
| SourceProvider (export) ≠ DataSourceProvider (data) | `exporting/protocols.py` vs `data/provider.py` | D1+D2 / E1 rename target |
| Transform pushdown optimization has no benchmark evidence; alpha Bindings classify current ETL as residual | `data/transform_compiler.py` | D4 remains NO-GO for pushdown claims |
| HDFS, Hive, ClickHouse, and Doris have incomplete delivery evidence | Bindings and external packages | Keep them at adapted/unsupported status until their real-infrastructure gates pass |

<!-- END -->
