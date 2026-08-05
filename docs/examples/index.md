# Examples

The repository examples demonstrate supported Tributo execution paths.

| Workflow | Source |
| --- | --- |
| Submit a Ray job | {download}`basic_submission.py <../../examples/basic_submission.py>` |
| Train XGBoost from S3 | {download}`xgboost_s3_training.py <../../examples/xgboost_s3_training.py>` |
| Run batch inference | {download}`batch_inference.py <../../examples/batch_inference.py>` |
| Serve an ONNX model | {download}`serve_onnx_model.py <../../examples/serve_onnx_model.py>` |

Examples that submit distributed workloads require a reachable Ray cluster.
They are not executed inside an unprivileged documentation build.
