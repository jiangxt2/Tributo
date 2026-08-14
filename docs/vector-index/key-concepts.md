# Vector-index key concepts

## Dataset identity and version

`LanceDatasetRef` identifies either an absolute local, `file://`, or S3 URI, or
a namespace implementation with a table identifier. A numeric version or tag
pins read operations. Build and search open one requested version. Optimize and
compact operate only on the active version and reject historical versions.

## Index type and metric

The verified index types are IVF_FLAT and IVF_PQ. IVF_PQ requires
`num_sub_vectors`. The verified metrics are l2, cosine, and dot. A request that
uses another index type, metric, or incompatible parameter fails validation.

## Coverage evidence

Receipts compare the planned, active, indexed, uncovered, and stale fragment
sets. They record bounded counts, digests, and sample identifiers rather than
placing an unbounded fragment list in logs or job results.

## Search delivery

Inline delivery has row and byte limits. Materialized delivery writes one
Parquet file to an absolute local, `file://`, or S3 URI. Search includes
fragments not covered by the index by default. `fast_search=true` requires
`include_unindexed=false` because it trades completeness for index-only work.

## Runtime evidence

The receipt records Ray, PyLance, Lance-Ray, and PyArrow versions from the
driver and workers. Installing the extra is necessary but not sufficient; a
successful job proves the actual runtime combination.
