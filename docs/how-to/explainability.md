# Batch Explainability

Tributo provides explainability as an optional batch operation over a published
Bundle. The default is disabled: training, Bundle export, inference, and their
digests remain unchanged until `ExplainabilityConfig(enabled=True)` is set.

## Enable it during Bundle export

The configuration belongs to `BundleOutputConfig` (and to the `output` section
of the DNN/PU/XGBoost trainer configuration). XGBoost uses the native UBJ
companion artifact for exact Tree SHAP. DNN and PU use their ONNX artifact with
model-agnostic SHAP, which is approximate and must be explicitly enabled.

```python
from tributo.explainability import ExplainabilityConfig
from tributo.exporting.models import BundleOutputConfig

bundle_config = BundleOutputConfig(
    bundle_uri="s3://models/fraud-detector",
    explainability=ExplainabilityConfig(
        enabled=True,
        backend="auto",
    ),
)
```

For an XGBoost source, `backend="auto"` adds a required native UBJ companion
when needed. The published manifest contains an explainability descriptor and
an `explainability_model` role. Existing model roles remain ordinary Bundle
roles; the descriptor is a capability promise, not a new universal model
format.

For DNN or PU, select the approximate path explicitly and provide a reference
dataset:

```python
ExplainabilityConfig(
    enabled=True,
    backend="model_agnostic",
    allow_approximate=True,
    reference={"uri": "/shared/reference.npy", "privacy_level": "restricted"},
)
```

The reference can be a local path, `file://` URI, or S3 URI containing a
two-dimensional NPY or Parquet matrix. Credentials, query parameters, and
fragments are rejected. The binding may include a SHA-256 digest, row count,
privacy level, and TTL. NPY loading always uses `allow_pickle=False`.

## Submit a request

Use the `tributo explain` command or `submit_explainability_job()` with a JSON
request. The request is batch-only and must use the Ray ingestion engine.

```json
{
  "bundle_uri": "/shared/bundles/fraud-detector",
  "input": {
    "source": {"type": "parquet", "path": "/shared/input/*.parquet"},
    "engine": "ray"
  },
  "feature_columns": ["feature_a", "feature_b"],
  "input_id_column": "entity_id",
  "backend": "auto",
  "result_uri": "/shared/explanations/fraud-detector",
  "operation_store_uri": "/shared/operations/fraud-detector",
  "request_id": "explain-2026-08-13-001"
}
```

Submit the JSON file through the CLI:

```bash
tributo explain --config explain.json --address http://ray-head:8265
```

`model_role` is optional. When omitted, the executor resolves the role from
the Bundle descriptor, so a tree Bundle selects `explainability_model` instead
of accidentally loading an ONNX inference artifact. An explicit role is
accepted only when it resolves to an artifact declared by the descriptor.

Ray Job execution requires `operation_store_uri`. The first-party local JSON
OperationStore records running, succeeded, partial, and failed snapshots and
persists a lease. Long-running jobs renew that lease automatically. A running
operation whose lease has expired can be reclaimed only with
`force_resume=true`, after confirming that the previous driver is no longer
active.

## Read results

Results are written as sharded Parquet in long format. `result_uri` in the
request is a stable output root; each execution attempt writes to
`result_uri/attempts/<lease-token>/`. The receipt and OperationStore record
the concrete attempt URI, so a replacement driver cannot overwrite or clean up
another attempt's files. Each row identifies an input, output, and feature and
includes the contribution, base value, output semantics, backend, exactness,
model digest, and optional preprocessor/feature map digests. `receipt.json`
records the result digest, row and byte counts, reference provenance,
dependency versions, and the declared access/privacy/retention policy.

Consumers should read the `result_uri` and `receipt_uri` from the persisted
operation record, or use the returned receipt, rather than assuming that the
request's result root contains a receipt directly.

Use `limits.top_k` or `limits.max_features` to bound the number of features per
input. `limits.max_explanation_rows` and `limits.max_explanation_bytes` are
enforced before a successful receipt is published; limit violations fail and
clean the materialized result. Other execution failures may leave a
`partial` result with a partial receipt and operation record for diagnosis.

Tree SHAP supports `model_output`/raw, probability, and log-loss semantics
subject to the model objective and reference requirements. Unsupported
objectives, mismatched feature order, missing required sidecars, and ambiguous
ONNX outputs fail closed.

## Extend the adapter

Adapters implement the `ExplainerAdapter` SPI and register through the
`tributo.explainers` entry-point group. They must declare `api_version=1`, a
unique `adapter_id`, an `adapter_version`, and the `supports`, `prepare`,
`explain_batch`, and `summarize` methods. Optional dependencies must be loaded
inside the adapter's runtime methods. Adapter ID conflicts and invalid
conformance fail closed; an optional dependency import failure is reported as
a discovery diagnostic until that adapter is used.

Reference loading is a separate `ReferenceProvider` SPI. The built-in file
provider covers local/file and S3 NPY/Parquet inputs; custom providers can be
injected into the Python execution boundary without changing the framework
contracts.

The v1 `summarize` adapter method is a provenance-preserving exactness filter;
it does not aggregate rows. Pass `exactness="exact"`, `"approximate"`, or
`"conditional"` to select a slice.
