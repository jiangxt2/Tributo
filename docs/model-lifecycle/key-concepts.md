# Model lifecycle key concepts

## Bundle as the exchange boundary

A Bundle contains artifacts, a manifest, digests, typed signatures, roles,
source provenance, and lineage. Training, batch inference, serving, and
explainability exchange immutable Bundle references rather than live framework
objects.

## Export planning

`ExportPlanner` builds a directed graph without cycles for requested targets
and required intermediate formats. Each node runs in an isolated staging
directory. Validators inspect structure, signatures, runtime compatibility,
or round-trip behavior before publication.

Required target failure fails the export. Optional target failure can produce a
partial result. A blocked target records why its dependency did not complete.

## Publication and reading

The publisher commits artifacts before the manifest. Local and `file://`
destinations use local publication; S3 uses manifest-last publication, leases,
idempotency, and alias compare-and-set. HDFS publication is not implemented.

`BundleReader` validates the manifest, artifact digests, resource limits,
flavor, safe-mode policy, and typed signatures before returning content.

## Runtime flavors

Exported formats and executable formats are separate capabilities. The safe
unified runtime supports `onnx-runtime-v1` and `xgboost-native-v1`. Other
exported formats remain readable but fail closed in batch or serving until a
matching loader passes its compatibility gate.

## MLflow boundaries

Training tracking, Bundle provenance, and Model Registry governance are three
different concerns. MLflow is an optional event consumer and external model
source. The Bundle manifest remains the source of truth for artifact
readability.
