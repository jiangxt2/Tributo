# Batch Inference

Run distributed batch inference with XGBoost + ONNX models across a Ray cluster.

## Quick Start

```python
from tributo.data import ParquetSourceConfig
from tributo.inference import InferenceConfig, run_batch_inference

config = InferenceConfig(
    source=ParquetSourceConfig(
        path="s3://your-bucket/input/*.parquet",
        columns=["feature_0", "feature_1"],
    ),
    output_uri="s3://your-bucket/output/predictions/",
    model_uri="s3://your-bucket/models/xgboost_model.onnx",
    predictor_config={"prediction_column": "prediction", "return_probs": True},
    batch_size=4096,
    concurrency=4,
)

result = run_batch_inference(config)
print(f"Inference complete: {result['input_path']} -> {result['output_path']}")
```

`result["input_path"]` is a credential-free display identifier. For SQL
sources it includes the dialect, host, port when non-default, and database;
it is not a connection string for re-use.

## JSON Configuration

Create `inference.json`:

```json
{
  "source": {
    "type": "parquet",
    "path": "s3://your-bucket/input/*.parquet",
    "columns": ["feature_0", "feature_1"]
  },
  "model": {
    "uri": "s3://your-bucket/models/xgboost_model.onnx",
    "return_probs": true
  },
  "output": {
    "uri": "s3://your-bucket/output/",
    "prediction_column": "prediction"
  },
  "ray": {"batch_size": 4096, "concurrency": 4, "num_cpus_per_actor": 1.0}
}
```

`source` accepts the same canonical `SourceConfig` and `provider`/`uri`
shapes used by training and embeddings. The historical `data.uri`,
`data.input`, and ClickHouse fields remain compatible and are normalized
before execution; they must not be mixed with `source`.

```python
from tributo.inference import run_inference_from_json
run_inference_from_json("inference.json")
```

## Submit via Ray Jobs API

```python
from tributo.inference import submit_inference_job

job_id = submit_inference_job(
    "inference.json",
    dashboard_url="http://127.0.0.1:8265",
)
print(f"Job submitted: {job_id}")
```

## Configuration Reference

| Field | Type | Description |
|---|---|---|
| `source` | object | Canonical bounded input source. |
| `input_uri` | str | Legacy input URI accepted by direct Python callers. |
| `output_uri` | str | S3 or local directory for output predictions. |
| `model_uri` | str | S3 or local path to ONNX model. |
| `bundle_uri` | str | Published model bundle URI, mutually exclusive with `model_uri`. |
| `source.columns` | list[str] | Provider-native feature projection. Omit to auto-detect from ONNX metadata. |
| `prediction_column` | str | Name of the output prediction column. Default: `prediction`. |
| `return_probs` | bool | Include probability scores. Default: `true`. |
| `batch_size` | int | Rows per inference batch. Default: `4096`. |
| `concurrency` | int | Number of parallel inference actors. Default: `4`. |
| `num_cpus_per_actor` | float | CPUs per actor. Default: `1.0`. |

## S3 Authentication

Same three methods as [training](training.md#s3-authentication): IAM Role (preferred), environment variables, or explicit `source.s3` configuration with `access_key_id` / `secret_access_key` / `endpoint`. Legacy direct callers may continue to use `s3_config`.

For a canonical SQL input with an S3 output, the output sink remains separate
from the SQL connection. Configure output credentials through IAM or the AWS
environment; SQL connection credentials are not reused as S3 credentials.

## See Also

- Example: `examples/batch_inference.py`
