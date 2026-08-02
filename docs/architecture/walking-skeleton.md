# Walking Skeleton

The walking skeleton is the minimum end-to-end path that validates the core
architecture contracts. It exercises every major subsystem boundary:

```
SourceConfig(parquet) → XGBoostTrainer → Bundle → BundleReader → batch predict
```

## Why This Path

| Reason | Detail |
|--------|--------|
| Touches all major subsystems | Data (D1+D2), Training (T1), Bundle Export (E1), Serving/Inference (E3) |
| Uses the most mature Trainer | XGBoost is the only Trainer with a complete Bundle vertical slice |
| Exercises the full Bundle lifecycle | Export → publish → read → serve |
| Simple data source | Parquet avoids database/streaming dependencies |
| Small enough for CI | Entire path completes in < 60 seconds on synthetic data |

## Contract Boundaries Exercised

| Step | Contract | What it validates |
|------|----------|-------------------|
| 1. `SourceConfig(provider="tributo.parquet", uri=...)` | `DataSourceProvider` | Provider ID resolution, JSON-only config, `extra="forbid"` |
| 2. `XGBoostTrainer.run(source_config, ...)` | `TrainingLifecycle` + `CallbackDispatcher` | Trainer receives canonical `SourceConfig`, not legacy config dict |
| 3. `BundleExportService.export(result)` | `ModelExporter` + `ExportSourceProvider` | Exporter ID resolution, required artifact status |
| 4. `BundleReader.open(bundle_path)` | `ExportManifest` + `ManifestSignature` | Manifest v1 read, schema version check, digest verification |
| 5. `BundleReader.load_model()` → `batch_predictor.predict(data)` | `ModelFlavor` + `ManifestSignature` | Model deserialization, input/output schema match |

## Step-by-Step Design

### Step 1: SourceConfig → DataSourceProvider

```python
from tributo.data import SourceConfig, resolve_provider

config = SourceConfig.model_validate_json("""
{
  "provider": "tributo.parquet",
  "uri": "synthetic_data/train.parquet",
  "format_options": {}
}
""")
provider = resolve_provider(config.provider)
handle = provider.open(config)
dataset = handle.to_ray_dataset()
```

Expected: `resolve_provider("tributo.parquet")` returns a `DataSourceProvider`.
Legacy `SourceInput` is NOT on this path.

### Step 2: XGBoostTrainer with SourceConfig

```python
from tributo.training import XGBoostTrainer, TrainingConfig

trainer_config = TrainingConfig.model_validate_json("""
{
  "objective": "binary:logistic",
  "num_rounds": 10,
  "source": {...},
  "export": {
    "targets": [{"exporter": "xgboost-onnx-v1", "required": true}]
  }
}
""")
trainer = XGBoostTrainer(trainer_config)
result = trainer.run()
```

Expected: `result` contains a `TrainingResult` with status. The callback
dispatcher is initialized from `TrainingLifecycle`, not embedded in `BaseTrainer`.

### Step 3: Bundle → Export

```python
from tributo.exporting import BundleExportService

service = BundleExportService()
bundle_ref = service.export(result)
assert bundle_ref.status == "SUCCESS"  # required artifact was produced
```

Expected: `bundle_ref.status == "SUCCESS"` because ONNX was declared `required:
true`. If export fails, status is `FAILED` (not silently `SUCCESS` with missing
ONNX — the E0 fix).

### Step 4: BundleReader → Manifest → Signature

```python
from tributo.exporting import BundleReader

reader = BundleReader(bundle_ref.uri)
manifest = reader.manifest
assert manifest.schema_version == 1
signature = manifest.signature
assert "float_input" in signature.input_names
assert "probability" in signature.output_names
```

Expected: `BundleReader` reads the manifest, verifies digest, and exposes
`ManifestSignature` without needing the original training code.

### Step 5: BundleReader → Model → Batch Predict

```python
model = reader.load_model()
predictor = BatchPredictor(model, signature=manifest.signature)

import numpy as np
test_data = np.random.randn(10, 10).astype(np.float32)
predictions = predictor.predict(test_data)
assert predictions.shape == (10,)
```

Expected: `BatchPredictor` uses `ManifestSignature` to validate input shape and
dtype. It does not infer the schema from the model file alone.

## What Is NOT Covered

| Excluded | Why |
|----------|-----|
| GPU training | All Ray clusters may not have GPUs; CPU-only path is the baseline |
| S3/MinIO storage | Local filesystem exercises the same `Publisher` contract; S3 is an O1 fixture |
| Streaming/Kafka | Separate protocol (`StreamSource`); not on the walking skeleton |
| Multi-worker / DDP | Covered by T3 when Data-volume Go passes |
| DNN / PU Trainers | Added in E2 after XGBoost vertical slice is complete |

## When The Walking Skeleton Becomes CI

O1 Core converts this design into a runnable integration test. The test:

- Creates synthetic Parquet data in a temp directory.
- Runs all 5 steps end-to-end.
- Fails if any boundary contract is violated.
- Runs on every PR that touches `data/`, `training/`, `exporting/`, or
  `inference/`.

Until O1, the walking skeleton is a **manual verification** performed after
D1+D2 and E1 are both merged, as the combined checkpoint before D3/E2
migration.

<!-- END -->
