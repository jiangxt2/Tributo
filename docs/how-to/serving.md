# ONNX Model Serving

Deploy ONNX models as HTTP inference services using Ray Serve.

## Start a Service

```bash
uv run tributo serve start \
  --model-path /path/to/model.onnx \
  --app-name my-model \
  --num-replicas 2
```

The service will be available at `http://127.0.0.1:8000/predict`.

## Serve a Bundle

Bundles are the stable Serving entry point. A serveable bundle must contain
non-empty, typed `input_signature` and `output_signature` fields in its
manifest. The loader also verifies the declared names, dtypes, and fixed
shape dimensions against the model before serving it.

```bash
uv run tributo serve start \
  --bundle-uri /path/to/published-bundle \
  --role inference \
  --app-name my-model
```

The E3 loader intentionally rejects bundles with an empty signature. The
current pre-E2 export path can publish such legacy-compatible bundles; the
trainer-to-manifest typed-signature producer is delivered by E2. Re-running
that same pre-E2 export path does not add a signature.

`--unsafe` is an explicit compatibility escape hatch for legacy bundles or
non-safe flavors. It skips signature validation and must not be used for a
production Serving deployment:

```bash
uv run tributo serve start \
  --bundle-uri /path/to/legacy-bundle \
  --unsafe
```

Use `--storage-profile` for an S3-compatible bundle configured through a
registered storage profile. The same `--bundle-uri`, `--role`, `--unsafe`, and
`--storage-profile` options are available on `serve grpc start`.

## Send Inference Requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "features": [[0.5, 1.2, 3.7, 0.1]],
        "return_probs": True,
    },
)
result = response.json()
print(result["predictions"])
```

## Python API

```python
from tributo.serving import start_serving, stop_serving, get_serving_status

# Start
start_serving(
    model_path="/path/to/model.onnx",
    app_name="my-model",
    route_prefix="/predict",
    num_replicas=2,
)

# Check status
status = get_serving_status("my-model")
print(f"Running: {status['running']}")

# Stop
stop_serving("my-model")
```

## Streaming LLM Inference

For LLM models with token streaming:

```bash
uv run tributo serve streaming start \
  --model-path /path/to/model \
  --tokenizer-path /path/to/tokenizer
```

The endpoint supports Server-Sent Events (SSE):

```python
import requests

response = requests.get(
    "http://127.0.0.1:8000/stream",
    params={"prompt": "Explain machine learning in one sentence."},
    stream=True,
)
for line in response.iter_lines():
    if line:
        print(line.decode())
```

## gRPC Inference (alpha)

gRPC-based inference is available for lower latency. The gRPC deployment provides a
protobuf-based interface for model inference.

The HTTP and gRPC output contracts are not yet identical: HTTP exposes the
`return_probs` selector, while gRPC currently returns the model's primary
output (`outputs[0]`). For classification probabilities, use HTTP until the
gRPC response contract adds an explicit output selector. The legacy gRPC
`confidence` field is not a calibrated probability when the primary output is
class labels.

Versioned gRPC `InputTensor` requests must set `schema_version` to `1`.
Proto3's default value `0` is intentionally rejected; clients that do not set
the version must use the legacy `features` field or upgrade their request
builder.

```bash
uv run tributo serve grpc start --model-path /path/to/model.onnx --port 50051
```

```{warning}
gRPC inference is under active development. The client API is not yet
stabilized; use ONNX HTTP serving when a stable transport is required.
```

## Scaling

| Parameter | Guidance |
|---|---|
| `num_replicas` | Number of ONNX model copies. Each replica runs on a separate Ray actor. |
| Route prefix | Each deployment gets a unique HTTP path. Use different `--app-name` values for multiple models. |
| GPU | ONNX Runtime will use CUDA if a GPU is available and `onnxruntime-gpu` is installed. |

## See Also

- Example: `examples/serve_onnx_model.py`
