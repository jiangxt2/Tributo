# Inference

Tributo separates bounded batch inference from long-running online serving.
Both paths prefer validated model bundles over raw model paths.

```{toctree}
:maxdepth: 1

../how-to/inference
../how-to/serving
../how-to/explainability
```

Batch inference uses Ray Data batch transforms and actor reuse. Online
inference uses Ray Serve HTTP, gRPC, or streaming endpoints.
