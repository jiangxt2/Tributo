# Batch Inference

Run distributed batch inference with XGBoost + ONNX models across a Ray cluster.

## Quick Start

```python
from tributo.inference import InferenceConfig, run_batch_inference

config = InferenceConfig(
    s3_input_path="s3://your-bucket/input/*.parquet",
    s3_output_path="s3://your-bucket/output/predictions/",
    model_uri="s3://your-bucket/models/xgboost_model.onnx",
    prediction_column="prediction",
    return_probs=True,
    batch_size=4096,
    concurrency=4,
)

result = run_batch_inference(config)
print(f"Inference complete: {result['input_path']} -> {result['output_path']}")
```

## JSON Configuration

Create `inference.json`:

```json
{
  "s3_input_path": "s3://your-bucket/input/*.parquet",
  "s3_output_path": "s3://your-bucket/output/",
  "model_uri": "s3://your-bucket/models/xgboost_model.onnx",
  "prediction_column": "prediction",
  "return_probs": true,
  "batch_size": 4096,
  "concurrency": 4,
  "num_cpus_per_actor": 1.0
}
```

```python
from tributo.inference import run_inference_from_json
run_inference_from_json("inference.json")
```

## Submit via Ray Jobs API

```python
from tributo.inference import submit_inference_job

job_id = submit_inference_job(
    "inference.yaml",
    dashboard_url="http://127.0.0.1:8265",
)
print(f"Job submitted: {job_id}")
```

## Configuration Reference

| Field | Type | Description |
|---|---|---|
| `s3_input_path` | str | S3 glob pattern for input Parquet files. |
| `s3_output_path` | str | S3 directory for output predictions. |
| `model_uri` | str | S3 or local path to ONNX model. |
| `feature_columns` | list[str] | Feature column names. Empty = auto-detect from ONNX metadata. |
| `prediction_column` | str | Name of the output prediction column. Default: `prediction`. |
| `return_probs` | bool | Include probability scores. Default: `false`. |
| `batch_size` | int | Rows per inference batch. Default: `4096`. |
| `concurrency` | int | Number of parallel inference actors. Default: `4`. |
| `num_cpus_per_actor` | float | CPUs per actor. Default: `1.0`. |

## S3 Authentication

Same three methods as [training](training.md#s3-authentication): IAM Role (preferred), environment variables, or explicit `s3_config` dict with `access_key_id` / `secret_access_key` / `endpoint`.

## See Also

- Example: `examples/batch_inference.py`
