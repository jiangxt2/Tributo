# Data key concepts

## Bounded ingestion

`IngestionRequest` selects a source and an engine. The gateway produces a
logical scan, resolves one compatible binding, and returns a typed
`RayDataHandle` or `DaftDataFrameHandle` with an `IngestionPlanReceipt`.

The gateway never silently falls back to a different engine. Consumers that
require Ray Data must request Ray explicitly or use a documented, explicit
handle adapter.

## Portable transforms

`TransformPipeline` describes the subset that Tributo can compile for both Ray
Data and Daft. The selected engine still owns expression execution. Provider
pushdown and a residual transform are different facts and appear separately in
planning evidence.

## Native writing

`WriteRequest` identifies an engine, target kind, destination, write mode, and
bounded options. `WriteGateway` selects a matching native binding and returns a
credential-free `WriteReceipt`. The binding delegates to an engine API such as
`ray.data.Dataset.write_parquet` or `DataFrame.write_lance`.

Tributo does not implement file, fragment, transaction, manifest, or commit
writers.

## Unbounded input

`StreamSource` represents an unbounded microbatch source. It does not return a
finite dataset and cannot replace bounded ingestion. A Kafka batch remains
pending until downstream inference and output succeed and the caller commits
its offsets.

## Credential ownership

Logical requests, references, receipts, logs, and errors remain
credential-free. Resolve credentials in the runtime domain that uses them. Do
not copy source credentials into a model repository or result sink.
