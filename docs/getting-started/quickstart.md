# Run the local quickstart

This workflow creates a local Parquet dataset, runs the formal distributed
Multinomial Naive Bayes implementation on a local Ray runtime, and publishes a
validated ONNX Bundle. It does not require S3, Docker, or a remote cluster.

## Install the training dependencies

```bash
python -m pip install "tributo[training]"
```

## Create the input and execution request

Download {download}`create_quickstart_data.py <../examples/doc_code/create_quickstart_data.py>`
and run it from a writable directory:

```bash
python create_quickstart_data.py
```

The script uses the same JSON envelope that the CLI validates:

```{literalinclude} ../examples/doc_code/create_quickstart_data.py
:language: python
:caption: create_quickstart_data.py
```

## Run the formal algorithm

```bash
tributo algo run --config tributo-quickstart/execution.json
```

The local profile owns the Ray runtime for this process. Two workers consume
disjoint Ray Data shards, reduce bounded sufficient statistics, and publish the
Bundle under `tributo-quickstart/bundle/<bundle-id>`.

## Inspect the result

The command prints a structured algorithm result. Confirm that the result
contains a completed execution receipt and an `outputs.bundle_uri` value. The
configured `bundle_uri` is the store root; the returned value identifies the
committed Bundle. Inspect its manifest at:

```bash
python -m json.tool <outputs.bundle_uri>/manifest.json
```

## Move to a cluster

Do not replace the local profile with a Ray Client connection. Use the
[Ray Jobs and cluster guide](../ray-jobs/index.md) when a cluster should own
execution. The `kubernetes` algorithm profile runs inside an existing KubeRay
RayJob and connects to that job's cluster with `address="auto"`; Tributo does
not create the Kubernetes control plane.

## Continue learning

- Read [algorithm key concepts](../algorithms/key-concepts.md).
- Configure [distributed training](../how-to/training.md).
- Learn how [Bundles](../model-lifecycle/key-concepts.md) become the model
  exchange boundary between training and inference.
