# Model lifecycle

Formal trainer output flows through the bundle export service. A bundle
contains validated artifacts, manifests, digests, schema signatures, roles,
and source provenance.

Bundle publication in v1.0.0 supports local paths, `file://` URIs, and S3.
MLflow tracking and registry integration are optional consumers rather than
hard dependencies of the core export path.

The availability of an exporter does not imply that every runtime can load its
format. Unified serving requires a registered flavor and loader.

```{toctree}
:maxdepth: 1

key-concepts
../how-to/mlflow
```

See the [support matrix](../reference/support-matrix.md) for supported export,
publication, and runtime combinations.
See the generated
[Model lifecycle API](../reference/api/model-lifecycle.md) for public
signatures and stability labels.
