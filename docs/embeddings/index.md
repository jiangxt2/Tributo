# Embeddings

Tributo supports Ray Jobs-based batch embedding and Ray Serve-based online
embedding. Batch work must be submitted through the Ray Jobs API so execution
and logs remain observable at the cluster boundary.

```{toctree}
:maxdepth: 1

../how-to/embeddings
```

Daft-related transformation support is currently an explicit prototype unless
a workflow is documented as using the stable provider-to-Ray-Dataset path.
