# Training on Ray

Run XGBoost, DNN, and PU training jobs through Ray Train. XGBoost supports
multiple workers. The current DNN and PU implementations require exactly one
worker; they use Ray Train for lifecycle and cluster execution, not DDP.

## Default Bundle Publication

First-party XGBoost, DNN, and PU trainers publish an immutable Bundle by
default. The caller must provide an explicit destination; Tributo never writes
a default Bundle into the current directory or a temporary directory.

```python
from tributo.exporting.models import BundleOutputConfig

summary = trainer.run(
    bundle_config=BundleOutputConfig(
        bundle_uri="s3://your-bucket/models/fraud-detector",
        storage_profile="production",
    )
)

print(summary["training_status"])
print(summary["bundle_status"])
print(summary["hook_status"])
print(summary["bundle_uri"])
print(summary["execution_id"])
```

Omitting `targets` selects the trainer's standard artifacts:

| Trainer | Default artifacts | Inference role |
|---------|-------------------|----------------|
| XGBoost | ONNX opset 12 and native UBJ in the same Bundle | `onnx-model` |
| DNN | ONNX opset 18 | `onnx-model` |
| PU | ONNX opset 18 | `onnx-model` |

The returned mapping includes the stable `TrainingResult` fields
`model_uri`, `bundle_uri`, `metrics`, `legacy_artifact_uri`,
`training_status`, `bundle_status`, `hook_status`, and `execution_id`.
Required Bundle or Hook failures are raised as typed errors and expose the
same terminal contract as `error.training_result`.

The old raw-artifact hooks are available only through
`trainer.run(output_path=..., legacy_export=True)`. That opt-in emits a
`DeprecationWarning` and must not be used for new integrations.

## XGBoost Training

The following example reads Parquet through the canonical ingestion gateway,
trains XGBoost, and lets the first-party defaults publish ONNX and UBJ together:

```python
from tributo.data import (
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
    open_ingestion,
)
from tributo.exporting.models import BundleOutputConfig
from tributo.training.xgboost_trainer import XGBoostTrainerImpl

train_input = open_ingestion(
    IngestionRequest(
        source=ParquetSourceConfig(path="s3://your-bucket/train/*.parquet"),
        engine="ray",
    )
)
assert isinstance(train_input.handle, RayDataHandle)

try:
    trainer = XGBoostTrainerImpl(
        datasets={"train": train_input.handle.dataset},
        config={
            "data": {"label_col": "label"},
            "model": {
                "objective": "binary:logistic",
                "max_depth": 6,
                "eta": 0.1,
            },
            "training": {"num_rounds": 200, "val_size": 0.2},
            "ray": {"num_workers": 4, "use_gpu": False},
        },
    )
    summary = trainer.run(
        bundle_config=BundleOutputConfig(
            bundle_uri="s3://your-bucket/models/xgboost",
            storage_profile="production",
        )
    )
    print(summary["bundle_uri"])
finally:
    train_input.close()
```

Tributo validates in-memory configuration with strict Pydantic contracts. JSON
is the built-in persisted format; deployment-specific parsers may convert other
formats to a mapping before validation.

Dictionary/JSON trainer entry points use `output.bundle_uri` for the default
Bundle lifecycle. The older `output.onnx_path`, `output.onnx_opset`, and
`output.metrics_path` fields belong to the deprecated raw-artifact lifecycle;
supplying `onnx_path` without `bundle_uri` selects that legacy path and emits a
`DeprecationWarning`. Combining `bundle_uri` with explicitly configured legacy
output fields is rejected; the two destinations are never interpreted as
aliases for one another.

## DNN and PU Training

DNN and PU use the same Bundle call after their trainer has been constructed:

```python
from tributo.exporting.models import BundleOutputConfig

summary = dnn_trainer.run(
    bundle_config=BundleOutputConfig(bundle_uri="s3://your-bucket/models/dnn")
)
print(summary["bundle_uri"])
```

```{note}
`num_workers > 1` is rejected for DNN and PU until preprocessing state, model
updates, metrics, early stopping, and checkpoints have coordinated
multi-worker semantics. Use distributed XGBoost when that algorithm fits the
problem; do not interpret XGBoost as a deep-learning algorithm.
```

## Bundle Checkpoint Compatibility

DNN and PU checkpoints produced before the E2 export contract do not contain
the required `ExportCheckpointV1` metadata and cannot be exported through
`BaseTrainer.run(bundle_config=...)`. Regenerate those checkpoints with the
E2 trainer implementation before using Bundle export.

The legacy `export_model()` path remains available for existing checkpoints
only through the explicit `legacy_export=True` compatibility switch.

## S3 Authentication

Three methods, in order of preference:

1. **IAM Role** (recommended for production) — no config needed; workers assume the role.
2. **Environment variables** — set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.
3. **Explicit config** — pass `s3` dict in the data config with `access_key_id`, `secret_access_key`, `endpoint`.

## Resource Tuning

| Parameter | Guidance |
|---|---|
| `num_workers` | XGBoost: select the required Ray Train workers. DNN/PU: must be `1`. |
| `use_gpu` | Requires GPU workers and a GPU-capable framework build: XGBoost for tree training or PyTorch for DNN/PU. |
| `num_cpus_per_worker` | XGBoost option; default 1. Increase when each worker has spare CPU. |

## Resource Budget

Every training worker enforces an unconditional materialization budget.
Over-budget inputs **fail fast** (a `ResourceBudgetExceededError` is raised
before the unbounded concat) — data is never silently truncated.

| Config field | Default | Meaning |
|---|---|---|
| `resource.max_batch_bytes` | 64 MiB | Per-batch size guard. |
| `resource.max_worker_materialization_bytes` | 1 GiB | Total bytes materialized per worker across all splits (includes the concat-copy peak). |
| `resource.max_input_rows_per_worker` | `null` (disabled) | Optional per-worker row guard; exceeding it fails fast instead of slicing. |

```json
"resource": {
  "max_batch_bytes": 67108864,
  "max_worker_materialization_bytes": 1073741824,
  "max_input_rows_per_worker": 1000000
}
```

> **Note**: `external_memory` and `data_iter` are reserved XGBoost
> parameters — they would bypass this budget contract and are rejected
> by the config.

## See Also

- Example: `examples/xgboost_s3_training.py`
