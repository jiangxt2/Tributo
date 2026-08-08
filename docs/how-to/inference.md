# Batch Inference

Run Bundle-aware distributed batch inference across a Ray cluster. The current
production flavor matrix contains `onnx-runtime-v1` and the safe native
`xgboost-native-v1` JSON/UBJ flavor.

## Bundle-aware request

New code should use strict named bindings. Table columns and model tensor names
are separate, and outputs are selected by name rather than by position.

```python
from tributo.data import IngestionRequest, ParquetSourceConfig
from tributo.inference import (
    BundleModelReference,
    InferenceRequest,
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    TensorInputBinding,
    TensorOutputBinding,
    run_inference,
)

request = InferenceRequest(
    model=BundleModelReference(
        uri="s3://models/classifier/bundle",
        expected_manifest_sha256="0" * 64,
        storage_profile="model-store",
    ),
    input=IngestionRequest(
        source=ParquetSourceConfig(path="s3://data/input/*.parquet"),
        engine="ray",
        storage_profile="source-store",
    ),
    input_binding=InputBindingSpec(
        tensors=(
            TensorInputBinding(
                tensor_name="float_input",
                columns=("feature_0", "feature_1"),
                dtype="float32",
            ),
        ),
        passthrough_columns=("entity_id",),
        # Default "error" rejects NaN. Use "allow" only when the model
        # explicitly treats NaN as a missing value, such as XGBoost.
        nan_policy="error",
    ),
    output_binding=OutputBindingSpec(
        tensors=(
            TensorOutputBinding(
                tensor_name="label",
                column="prediction",
                semantic="label",
            ),
            TensorOutputBinding(
                tensor_name="probabilities",
                column="score",
                semantic="probability",
            ),
        )
    ),
    result_sink=ParquetResultSinkRequest(
        uri="s3://results/predictions",
        storage_profile="result-store",
    ),
)

result = run_inference(request)
```

`InferenceRequest` never accepts plaintext credentials. Configure each domain
with a storage-profile reference or the provider's environment/IAM chain.
Source credentials are not copied to model acquisition or result output.
For Ray Data sources and result sinks, a named profile must resolve inside the
cluster through `TRIBUTO_STORAGE_PROFILE_<NAME>`; an unresolved boto3 named
profile is rejected instead of silently falling back to another domain's
default credentials. The default IAM/environment chain remains available when
the request omits the profile name.

Source profile resolution and source-local S3 precedence are owned by the Data
Gateway. Inference does not merge provider options or copy source credentials
into its model and sink adapters.

Batch inference currently requires the explicit Ray ingestion engine. It does
not auto-select an engine, switch engines after a failure, or implicitly
convert a Daft handle to Ray. Input feature projection is appended through the
shared Data Transform IR. The final result includes the Data Gateway's
credential-free ingestion receipt alongside the sink receipt.

Submit the same request as a detached Ray Job with
`submit_inference_request(request)`. The request is resolved once before
submission, including manifest digest, role, flavor, data identity, run ID,
attempt ID, and deterministic Ray submission ID.

`null_policy="error"` rejects explicit Arrow/object NULL values.
`nan_policy="error"` is separately fail-closed because Ray's NumPy batch
boundary can represent nullable numeric Arrow values as NaN. Set
`nan_policy="allow"` only for a model whose runtime defines that missing-value
behavior. The same policies apply to passthrough and preserved feature columns,
so retained output data cannot bypass the binding contract. A non-finite
floating-point value is never cast to an integer dtype because that conversion
would irreversibly corrupt the value. Binding dtypes use the canonical names
supported by the runtime; invalid aliases fail during request validation
instead of inside a Ray actor.

## Post-training inference

After training has published a Bundle, pass only its immutable `BundleRef` and
the training run ID to `PostTrainingInferenceAction`. `mode="inline"` continues
inside the current Ray Job; `mode="detached"` submits another Ray Job. Both
modes reload and verify the published Bundle through the same request,
resolver, runtime, executor, and sink path as standalone inference.

Trainer objects, checkpoints, Boosters, and PyTorch modules are not accepted at
this boundary.

Detached post-training submission accepts the same `env_vars` and
`project_root` arguments as standalone submission. Use `env_vars` to transport
named storage-profile definitions into the Ray Job; credentials remain outside
`PostTrainingInferenceAction` and the serialized inference request.

## Legacy compatibility

### Python entry

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

### JSON configuration

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

### Ray Jobs submission

```python
from tributo.inference import submit_inference_job

job_id = submit_inference_job(
    "inference.json",
    dashboard_url="http://127.0.0.1:8265",
)
print(f"Job submitted: {job_id}")
```

### Configuration reference

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

Same three methods as [training](training.md#s3-authentication): IAM Role
(preferred), environment variables, or a storage profile. Legacy direct
callers may continue to use `s3_config` for one compatibility window. It emits
`DeprecationWarning`; migrate to independent source, model, and output profile
references.

Legacy JSON `data.s3` belongs only to the input source. It is never propagated
to an S3 model or result sink; configurations that relied on that behavior emit
a migration warning without logging credential values. Configure
`model.storage_profile` and `output.storage_profile` independently. Canonical
sources also cannot be combined with legacy `feature_columns`; built-in file
sources use `source.columns`, while reusable transformations use the shared
Transform IR.

For a canonical SQL input with an S3 output, the output sink remains separate
from the SQL connection. Configure output credentials through IAM or the AWS
environment; SQL connection credentials are not reused as S3 credentials.

## External model systems

`RegistryModelReference`, `ArtifactModelReference`, and the explicit
`ModelImporter` protocol are alpha contracts. Two first-party importers are
available:

- `mlflow.v2` resolves a numeric Model Version or Alias, immediately freezes
  the result to its numeric version, and accepts either an existing Tributo
  Bundle or an MLflow model with an ONNX flavor and named tensor signature.
- `tributo.artifact` acquires an explicit local, `file://`, or `s3://` artifact.
  It supports ONNX and native XGBoost JSON/UBJ when the caller provides the
  exact format, flavor, and typed input/output signature. An optional expected
  digest adds a caller-supplied precondition; the importer always computes and
  records the acquired content digest. The native XGBoost flavor has one
  canonical `float_input` tensor; the importer rejects any other input field
  name before acquiring or publishing the artifact.

Both paths publish and verify a Tributo Bundle before Ray execution. They never
put an MLflow PyFunc, XGBoost Booster, SDK client, or mutable alias in the
resolved plan. Safetensors remains fail-closed without a trusted architecture
and ModelFactory; it is not inferred from the file extension.

`import_bundle_uri` is required because Tributo does not yet configure a
default Artifact Repository. It must identify a durable location accessible to
the Ray workers; `import_storage_profile` authenticates only that destination.
External importers require typed signatures and do not expose the legacy
`unsafe` escape hatch. Explicit unsafe compatibility is limited to callers
that already hold a `BundleModelReference` for an old Bundle.

Ray Data executes the lazy input, model, and distributed write graph from the
terminal sink action. Failures that the public Ray API cannot attribute to one
operator are reported as `materialization`, rather than incorrectly claiming
that the model or Sink alone failed. Sink configuration/profile failures remain
`sink` failures. Executor retry hints remain conservative; detached retries
require an explicit classifier in the Ray Job adapter.

```python
from tributo.inference import ArtifactModelReference, RegistryModelReference

mlflow_model = RegistryModelReference(
    provider_id="mlflow.v2",
    model_name="fraud-classifier",
    alias="champion",
    import_bundle_uri="s3://models/tributo-imports",
    import_storage_profile="model-store",
    options={"tracking_uri": "https://mlflow.example.com"},
)

xgboost_model = ArtifactModelReference(
    provider_id="tributo.artifact",
    uri="s3://external-models/fraud.ubj",
    storage_profile="external-model-store",
    format_id="xgboost",
    flavor_id="xgboost-native-v1",
    architecture_id="xgboost",
    expected_sha256="0" * 64,
    import_bundle_uri="s3://models/tributo-imports",
    import_storage_profile="model-store",
    options={
        "variant": "ubj",
        "input_fields": [
            {"name": "float_input", "dtype": "float32", "shape": ["batch", 12]}
        ],
        "output_fields": [
            {"name": "label", "dtype": "int64", "shape": ["batch"]},
            {
                "name": "probabilities",
                "dtype": "float32",
                "shape": ["batch", 2],
            },
        ],
    },
)
```

Pass either reference as `InferenceRequest.model`. `resolve_inference()` performs
external acquisition and immutable version resolution once. Use
`submit_resolved_inference()` when the plan must be frozen before a later Ray
Job submission, for example before changing an MLflow Alias. MLflow access uses
the MLflow/AWS environment chain; Bundle publication uses only
`import_storage_profile`.

## See Also

- Example: `examples/batch_inference.py`
