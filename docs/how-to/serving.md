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

```bash
uv run tributo serve grpc start --model-path /path/to/model.onnx --port 50051
```

!!! warning "Alpha API"
    gRPC inference is under active development. The client API is not yet stabilized;
    use ONNX HTTP serving for production workloads.

## Scaling

| Parameter | Guidance |
|---|---|
| `num_replicas` | Number of ONNX model copies. Each replica runs on a separate Ray actor. |
| Route prefix | Each deployment gets a unique HTTP path. Use different `--app-name` values for multiple models. |
| GPU | ONNX Runtime will use CUDA if a GPU is available and `onnxruntime-gpu` is installed. |

## See Also

- Example: `examples/serve_onnx_model.py`
