# Training on Ray

Run XGBoost, DNN, and PU training jobs through Ray Train. XGBoost supports
multiple workers. The current DNN and PU implementations require exactly one
worker; they use Ray Train for lifecycle and cluster execution, not DDP.

## XGBoost Training

### Configuration File

Create a JSON config (`training.json`):

```json
{
  "data": {
    "type": "s3",
    "uri": "s3://your-bucket/train/*.parquet",
    "format": "parquet"
  },
  "model": {
    "label_col": "label",
    "xgb_params": {
      "objective": "binary:logistic",
      "eval_metric": ["logloss", "auc"],
      "max_depth": 6,
      "eta": 0.1,
      "subsample": 0.8,
      "colsample_bytree": 0.8
    },
    "num_rounds": 200,
    "early_stopping_rounds": 20
  },
  "export": {
    "onnx_output": "s3://your-bucket/models/xgboost_model.onnx"
  }
}
```

> **Note**: Tributo validates the in-memory Pydantic contract with a strict
> schema. JSON is the built-in persisted format; deployment-specific parsers
> may convert other formats to a mapping before validation.

### Python API

```python
from tributo.training import build_trainer
from tributo.data import IngestionRequest, ParquetSourceConfig, RayDataHandle, open_ingestion

train_input = open_ingestion(IngestionRequest(
    source=ParquetSourceConfig(path="s3://your-bucket/train/*.parquet"),
    engine="ray",
))
val_input = open_ingestion(IngestionRequest(
    source=ParquetSourceConfig(path="s3://your-bucket/val/*.parquet"),
    engine="ray",
))
assert isinstance(train_input.handle, RayDataHandle)
assert isinstance(val_input.handle, RayDataHandle)

try:
    trainer = build_trainer(
        ray_dataset=train_input.handle.dataset,
        train_config=train_config,
        val_dataset=val_input.handle.dataset,
        num_workers=4,
        use_gpu=False,
    )
    result = trainer.fit()
    print(f"Model: {result.metrics['onnx_path']}")
finally:
    val_input.close()
    train_input.close()
```

## DNN Training

```python
from tributo.training.dnn_trainer import run_dnn_training_with_config

result = run_dnn_training_with_config(
    {
        "data": {"type": "parquet", "path": "/data/train.parquet"},
        "features": [
            {"name": "account_age", "type": "dense"},
            {"name": "monthly_usage", "type": "dense"},
        ],
        "label_col": "label",
        "training": {"epochs": 10, "batch_size": 256},
        "ray": {"num_workers": 1},
        "output": {"onnx_path": "/models/dnn"},
    }
)
print(result["onnx_path"])
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
and is not affected by this contract requirement.

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
