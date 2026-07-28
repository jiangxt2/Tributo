"""XGBoost + ONNX distributed batch inference example.

Three usage modes:
    1. Python API: construct InferenceConfig and call run_batch_inference()
    2. JSON config: call run_inference_from_json()
    3. Ray Jobs API: submit via submit_inference_job() (recommended for production)

S3 auth follows the same patterns as xgboost_s3_training.py; see comments below.
"""

from __future__ import annotations

import os

import ray

from tributo.inference import InferenceConfig, run_batch_inference

ray.init(address="ray://127.0.0.1:10001", ignore_reinit_error=True)

# ── Mode 1: Python API ────────────────────────────────────────────────────────
config = InferenceConfig(
    s3_input_path="s3://your-bucket/input/*.parquet",
    s3_output_path="s3://your-bucket/output/predictions/",
    model_uri="s3://your-bucket/models/xgboost_model.onnx",
    # feature_columns defaults to auto-detection from ONNX model metadata
    feature_columns=[],
    prediction_column="prediction",
    return_probs=True,
    batch_size=4096,
    concurrency=4,
    num_cpus_per_actor=1.0,
    # For MinIO / private S3-compatible storage. Omit s3_config when using IAM roles.
    s3_config={
        "access_key_id": os.environ.get("S3_ACCESS_KEY_ID", "<your-access-key>"),
        "secret_access_key": os.environ.get(
            "S3_SECRET_ACCESS_KEY", "<your-secret-key>"
        ),
        "endpoint": os.environ.get("S3_ENDPOINT", "http://127.0.0.1:9000"),
        "region": os.environ.get("S3_REGION", "us-east-1"),
    },
)

result = run_batch_inference(config)
print(f"Inference complete: {result['input_path']} -> {result['output_path']}")

# ── Mode 2: JSON config ────────────────────────────────────────────────────
# from tributo.inference import run_inference_from_json
# result = run_inference_from_json("examples/inference/inference_s3.json")

# ── Mode 3: Ray Jobs API (recommended for production) ─────────────────────────
# from tributo.inference import submit_inference_job
# job_id = submit_inference_job(
#     "examples/inference/inference_s3.json",
#     dashboard_url="http://127.0.0.1:8265",
# )
# print(f"Job submitted: {job_id}")
