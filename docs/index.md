# Tributo

Telecom-native ML framework for Ray clusters. Tributo provides distributed training, batch embedding, and online inference — with first-class support for PU Learning on Ray.

## Key Capabilities

| Capability | Description |
|---|---|
| **PU Learning** | Train classifiers from positive + unlabeled data. nnPU/uPU loss, auto class-prior estimation, ONNX export. Single-worker training on Ray. |
| **Distributed XGBoost** | Multi-worker XGBoost on Ray Train with S3 data sources and automatic ONNX export. |
| **Batch Embedding** | Distributed BGE text embedding via Ray Jobs API, output to Lance/Parquet. |
| **ONNX Inference Serving** | Ray Serve deployment of ONNX models with HTTP endpoints. |
| **Batch Inference** | XGBoost + ONNX distributed batch inference across Ray clusters. |
| **Hyperparameter Tuning** | Random/BayesOpt search with FIFO/ASHA/HyperBand schedulers via Ray Tune. |

## Roadmap

| Feature | Status |
|---|---|
| Sequence pre-training (Temporal Transformer) | Planned — see roadmap |
| Large-scale vector pipeline (Daft → Ray → Lance) | In development |
| gRPC inference serving | Alpha |
| Streaming LLM inference | Alpha |

## Getting Started

```bash
pip install tributo
```

```python
from tributo import TributoClient

client = TributoClient("http://127.0.0.1:8265")
job_id = client.submit(entrypoint="python my_script.py")
print(client.get_status(job_id))
```

→ Head to the [Quickstart](quickstart.md) for a 5-minute walkthrough.

## Project Status

Tributo is in active development. APIs marked `@PublicAPI(stability="beta")` are stable enough for production use; those marked `"alpha"` may change. See [API Reference](api.md) for stability levels per module.
