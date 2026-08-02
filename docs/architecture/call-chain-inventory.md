# Call Chain Inventory

Documenting every data entry point and model export entry point in the
current codebase (baseline: commit `5ed81a0`, 2026-08-02).

## Data Entry Points

### 1. Training Data Loading

**Primary entry**: `training/data_loader.py`

```
User code
  ↓
load_ray_dataset_from_config(config: dict)        ← main public entry
  ↓
load_ray_dataset_from_source(source: SourceConfig) ← canonical (D1+D2 target)
  ↓
  type dispatch:
    ├── "s3" / "csv" / "parquet" / "lance"
    │     → _load_file_dataset()
    │       → ray.data.read_parquet / read_csv / read_lance
    │
    ├── "clickhouse"
    │     → _load_clickhouse_dataset()  ← deprecated loader wrapper
    │       → _load_clickhouse()
    │         → clickhouse-connect → ray.data.from_arrow
    │
    ├── "doris" / "mysql"
    │     → _load_doris_mysql()
    │       → PyMySQL → ray.data.from_arrow
    │
    └── "postgresql" / ConnectorX path
          → _load_connectorx()  ← NotImplementedError (experimental)
```

**Legacy alias**: `load_dataframe_from_config()` — wraps `load_ray_dataset_from_config()` into Pandas.

**Prototype (not on main path)**:
- `data/provider.py` — `DataSourceProvider`, `SourceRouter`, `SourcePlan` (has
  unit tests; no production callers)
- `data/transform_compiler.py` — `TransformCompiler`, Daft/Ray pushdown
  (prototype; no production callers)

### 2. Inference Data Loading

**Primary entry**: `inference/pipeline.py`

```
User code
  ↓
InferenceConfig.model_validate_json(...)           ← Pydantic model
  ↓
run_batch_inference(config: InferenceConfig)       ← main public entry
  ↓
  data_type dispatch (lines 151-):
    ├── data_type == "clickhouse"
    │     → load_ray_dataset_from_config({"type": "clickhouse", ...})
    │       → _load_clickhouse()  ← deprecated loader wrapper
    │
    ├── input_uri.startswith("s3://")
    │     → S3 connector (boto3 → ray.data.read_parquet)
    │
    └── else (local)
          → ray.data.read_parquet(input_uri)
```

**Key issue**: ClickHouse branch goes through the deprecated loader wrapper;
S3 and local paths go directly to Ray Data or connector. These three paths are
not unified — `InferenceConfig.data_type` has its own routing, separate from
`training.data_loader`'s `SourceConfig`.

**Secondary entry**: `run_inference_from_json(config_path)` — reads JSON → builds
`InferenceConfig` → calls `run_batch_inference()`.

### 3. Embedding Data Loading

**Primary entry**: `embeddings/job_runner.py`

```
CLI / Python API
  ↓
submit_embedding_job(config_path, ...)
  ↓
Ray Job submission (entrypoint script runs inside Ray cluster)
  ↓
Embedding runner (daft → Ray inference → Lance)
  ↓
Daft reads source data → Ray Data → model inference → Lance write
```

### 4. CLI Data Entry

**Primary entry**: `cli.py`

```
tributo submit --config <json>
  ↓
CLI parses JSON → builds JobConfig / TrainingConfig
  ↓
submits to Ray Job API or calls local runner
```

### 5. Plugin Data Connectors

**Discovery**: `plugin.py::discover_connector_plugins()`

```
importlib.metadata.entry_points(group="tributo.connectors")
  ↓
DataConnector subclass validation
  ↓
Registered in connector registry (data/registry.py)
```

**Currently**: No third-party connector plugins exist. All built-in connectors
are in `data/` module.

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
| Training and inference have separate data routing | `data_loader.py` vs `pipeline.py` | D3 fix target |
| Inference ClickHouse branch goes through deprecated loader wrapper | `pipeline.py:152-162` | D3 fix target |
| Old XGBoost ONNX export swallows errors | `training/onnx_exporter.py` | E0 fix target |
| SourceProvider (export) ≠ DataSourceProvider (data) | `exporting/protocols.py` vs `data/provider.py` | D1+D2 / E1 rename target |
| Prototype Provider/TransformCompiler has no production callers | `data/provider.py`, `data/transform_compiler.py` | D1+D2 / D4 target |
| ConnectorX path is NotImplementedError | `data_loader.py:261-264` | Future; explicit error message added in A0 |

<!-- END -->
