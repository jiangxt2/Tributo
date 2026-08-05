# Model lifecycle

All new trainer output flows through the bundle export service. A bundle
contains validated artifacts, manifests, digests, schema signatures, roles,
and source provenance.

Bundle publication currently supports local paths, `file://` URIs, and S3.
MLflow tracking and registry integration are optional consumers rather than
hard dependencies of the core export path.

The availability of an exporter does not imply that every runtime can load its
format. Unified serving requires a registered flavor and loader.

See the [support matrix](../reference/support-matrix.md) for supported export,
publication, and runtime combinations.
