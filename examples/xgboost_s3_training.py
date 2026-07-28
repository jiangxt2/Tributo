"""XGBoost distributed training end-to-end example.

Usage:
    # Connect to an existing Ray cluster
    ray.init(address="auto")

    # Or local single-node testing
    ray.init()

S3 authentication — three options (pick one, see comments below).
"""

from __future__ import annotations

import ray

from tributo.training import build_trainer
from tributo.training.data_loader import load_ray_dataset_from_config

# ── Auth mode 1: IAM Role (recommended for production; workers need permissions) ──
train_ds = load_ray_dataset_from_config(
    {
        "type": "s3",
        "uri": "s3://your-bucket/train/*.parquet",
        "format": "parquet",
    }
)
val_ds = load_ray_dataset_from_config(
    {
        "type": "s3",
        "uri": "s3://your-bucket/val/*.parquet",
        "format": "parquet",
    }
)

# ── Auth mode 2: Explicit AK/SK (local dev; never hardcode credentials) ──────────
# train_ds = load_ray_dataset_from_config({
#     "type": "s3",
#     "uri": "s3://your-bucket/train/*.parquet",
#     "format": "parquet",
#     "s3": {
#         "region": "cn-north-1",
#         "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
#         "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
#     },
# })

# ── Auth mode 3: Custom endpoint (MinIO / private S3-compatible storage) ──────
# train_ds = load_ray_dataset_from_config({
#     "type": "s3",
#     "uri": "s3://your-bucket/train/*.parquet",
#     "format": "parquet",
#     "s3": {
#         "access_key_id": "<your-access-key>",
#         "secret_access_key": "<your-secret-key>",
#         "endpoint": "http://127.0.0.1:9000",
#     },
# })

train_config = {
    "label_col": "label",
    "xgb_params": {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "num_rounds": 200,
    "early_stopping_rounds": 20,
    "onnx_output": "/tmp/xgboost_model.onnx",
}

ray.init(address="ray://127.0.0.1:10001", ignore_reinit_error=True)

trainer = build_trainer(
    ray_dataset=train_ds,
    train_config=train_config,
    val_dataset=val_ds,
    num_workers=4,
    use_gpu=False,
)

result = trainer.fit()

onnx_path = result.metrics.get("onnx_path")
print(f"Training complete. ONNX model saved to: {onnx_path}")
