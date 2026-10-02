# Run the local bounded-data quickstart

This Core-only workflow creates a local Parquet dataset, reads it through the
Alpha bounded-ingestion Gateway on Ray Data, and writes a native Parquet copy
through the Alpha bounded-write Gateway. It does not require an external
algorithm Wheel, S3, Docker, or a remote cluster.

## Prepare the Core environment

```bash
uv sync --locked --no-dev
```

## Create the local input

Run the repository-backed input generator from the source checkout:

```bash
uv run --locked --no-sync python \
  docs/examples/doc_code/create_quickstart_data.py
```

The script creates `tributo-quickstart/input.parquet` with a deterministic
schema and eight rows:

```{literalinclude} ../examples/doc_code/create_quickstart_data.py
:language: python
:caption: create_quickstart_data.py
```

## Read and write through Tributo

```bash
uv run --locked --no-sync python \
  docs/examples/doc_code/local_data.py \
  tributo-quickstart/input.parquet \
  tributo-quickstart/output
```

The example owns a one-CPU local Ray runtime, prints the native schema and read
receipt, and copies the input through `WriteGateway` with overwrite semantics.

## Inspect the result

Confirm that the command prints `committed=True` and that
`tributo-quickstart/output` contains the copied Parquet data. Tributo records
the selected Provider, Binding, engine version, and write evidence without
duplicating the data plane owned by Ray Data.

## Continue to algorithms and clusters

Tributo Core does not bundle production algorithms. Install a compatible,
independently versioned algorithm Wheel before following the
[formal algorithm guide](../algorithms/getting-started.md). Use the
[Ray Jobs and cluster guide](../ray-jobs/index.md) when a cluster should own
execution; do not replace that boundary with Ray Client.

## Continue learning

- Read the [data concepts](../data/key-concepts.md).
- Learn how to [read bounded data](../data/user-guides/read-data.md).
- Learn how to [write bounded data](../data/user-guides/write-data.md).
