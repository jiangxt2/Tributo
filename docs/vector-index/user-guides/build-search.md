# Build and search a vector index

## Build through Python

```{literalinclude} ../../examples/doc_code/vector_index_requests.py
:language: python
:pyobject: build
```

Direct Python execution uses the active Ray context. Use the CLI to submit the
same request as an observable Ray Job.

## Search a pinned version

```{literalinclude} ../../examples/doc_code/vector_index_requests.py
:language: python
:pyobject: search
```

The query vector dimension must equal the stored vector dimension. The index
name, column, and metric must match Lance metadata. Search fails closed on
missing or incompatible metadata.

The documentation test validates both request builders and their operation
wiring. The real Lance-Ray integration Gate verifies distributed build,
search, optimization, compaction, Ray Jobs, and S3 result delivery.

## Maintain appended data

Use `tributo vector optimize` to index fragments appended after the initial
build. Use `tributo vector compact` for Lance file compaction. Both operations
return a maintenance receipt with input and output versions and post-operation
coverage.
