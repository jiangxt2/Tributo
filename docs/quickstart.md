# Quickstart

This guide gets you from zero to a running Tributo job in under 5 minutes.

## Prerequisites

- Python 3.12–3.13
- A [Ray cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) (or use `ray.init()` for local mode)
- Optional: S3-compatible storage (MinIO, AWS S3, etc.)

## 1. Install

```bash
pip install tributo

# Or with extras
pip install tributo[training]      # XGBoost training + ONNX export
pip install tributo[hf]            # User-provided Hugging Face Predictor dependencies
pip install tributo[data]          # Lance / Iceberg connectors
```

## 2. Submit Your First Job

```python
import ray
from tributo import TributoClient

# Connect to Ray
ray.init(address="ray://127.0.0.1:10001")

# Submit a job
client = TributoClient("http://127.0.0.1:8265")
job_id = client.submit(
    entrypoint="python -c 'print(sum(range(100)))'",
    runtime_env={"pip": ["numpy"]},
)

# Track status
status = client.get_status(job_id)
print(f"Job {job_id}: {status}")
```

## 3. Run a Training Job

```bash
# From the command line
uv run tributo submit \
  --address http://127.0.0.1:8265 \
  --entrypoint "python examples/xgboost_s3_training.py" \
  --config training_config.json
```

```python
# Or from Python
from tributo.training import build_trainer
from tributo.data import (
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
    open_ingestion,
)

ingestion = open_ingestion(
    IngestionRequest(
        source=ParquetSourceConfig(path="s3://your-bucket/train/*.parquet"),
        engine="ray",
    )
)
assert isinstance(ingestion.handle, RayDataHandle)
train_ds = ingestion.handle.dataset

try:
    trainer = build_trainer(
        ray_dataset=train_ds,
        train_config={
            "label_col": "label",
            "xgb_params": {"objective": "binary:logistic", "eval_metric": ["auc"]},
            "num_rounds": 100,
        },
        num_workers=4,
    )
    result = trainer.fit()
    print(f"Model saved to: {result.metrics['onnx_path']}")
finally:
    ingestion.close()
```

## 4. Serve a Model for Inference

```bash
# Start an ONNX inference service
uv run tributo serve start --model-path /path/to/model.onnx

# Send a request
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'

# Stop the service
uv run tributo serve stop
```

## Next Steps

- [Distributed Training](how-to/training.md) — configure XGBoost/DNN training jobs
- [Batch inference](how-to/inference.md) — run user-provided Predictors over Ray Data
- [PU Learning](how-to/pu-learning.md) — train with positive + unlabeled data
- [CLI Reference](cli.md) — all available commands
