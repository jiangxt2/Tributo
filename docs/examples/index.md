# Examples

Repository-backed examples use one of three verification levels. Local setup
examples execute in the unit suite. Every root `examples/*.py` cluster example
is syntax-compiled in the docs gate and relies on the named integration gate
for runtime evidence. External-service examples state their prerequisite
instead of contacting infrastructure during a Read the Docs build.

| Workflow | Source | Verification |
| --- | --- | --- |
| Prepare the local formal-algorithm quickstart | {download}`create_quickstart_data.py <doc_code/create_quickstart_data.py>` | Executed by `tests/docs/test_doc_examples.py`; the algorithm runtime is covered by the distributed-algorithm local Gate |
| Read and write local Parquet | {download}`local_data.py <doc_code/local_data.py>` | Executed with Ray Data in a temporary directory by `tests/docs/test_doc_examples.py` |
| Build and search a Lance vector index | {download}`vector_index_requests.py <doc_code/vector_index_requests.py>` | Request builders and operation wiring execute in the docs test; the Lance-Ray integration Gate supplies runtime evidence |
| Submit a Ray job | {download}`basic_submission.py <../../examples/basic_submission.py>` | Requires a reachable Ray dashboard; job client contracts have unit and integration coverage |
| Train XGBoost from S3 | {download}`xgboost_s3_training.py <../../examples/xgboost_s3_training.py>` | Requires Ray and S3; trainer and Bundle paths use their existing integration Gates |
| Run batch inference | {download}`batch_inference.py <../../examples/batch_inference.py>` | Requires Ray, a model, and input data; the inference Gate supplies runtime evidence |
| Serve an ONNX model | {download}`serve_onnx_model.py <../../examples/serve_onnx_model.py>` | Requires Ray Serve and a model; serving contracts have targeted tests |

The documentation workflow does not rerun Docker, database, S3, or remote
cluster Gates for prose-only changes. It preserves their source-aligned example
paths and runs the existing static and real-import checks.
