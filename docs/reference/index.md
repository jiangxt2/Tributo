# Reference

Reference pages are generated from source where possible. Python signatures
come from autodoc, and the complete Click command tree comes from
sphinx-click.

```{toctree}
:maxdepth: 1

Core API <api/core>
Data API <api/data>
Algorithms and training API <api/algorithms-training>
Model lifecycle API <api/model-lifecycle>
Inference and serving API <api/inference-serving>
Vector-index API <api/vector-index>
Pipeline and extension API <api/extensions>
../cli
support-matrix
../STABILITY
../api
```

The component pages are generated from source annotations. CI checks the
complete inventory, imports every documented object in the real project
environment, compares runtime stability to the source inventory, and rejects
stale generated pages.
