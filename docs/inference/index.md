# Inference

Tributo separates bounded batch inference from long-running online serving.
Both paths prefer validated model bundles over raw model paths.

```{toctree}
:maxdepth: 1

key-concepts
../how-to/inference
../how-to/serving
../how-to/explainability
```

Batch inference uses Ray Data batch transforms and actor reuse. Online
inference uses Ray Serve HTTP, gRPC, or Server-Sent Events. Explainability is a
separate batch operation with its own result and operation receipt.

See the generated
[Inference and serving API](../reference/api/inference-serving.md) for public
signatures and stability labels.
