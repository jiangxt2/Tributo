# Start a vector-index job

You need a reachable Ray dashboard and an existing Lance dataset. The vector
column must use a fixed-size Arrow vector type and all workers must reach the
same dataset location.

## Create a build request

Save a JSON file such as:

```json
{
  "dataset": {"uri": "s3://models/vectors.lance", "storage_profile": "vector-store"},
  "column": "embedding",
  "index_name": "embedding_idx",
  "index_type": "IVF_FLAT",
  "metric": "cosine",
  "num_workers": 4,
  "num_partitions": 64,
  "sample_rate": 256
}
```

Submit it through Ray Jobs:

```bash
tributo vector build \
  --address http://ray-head:8265 \
  --config build.json
```

Retrieve the validated structured result after the Ray Job finishes:

```bash
tributo vector result \
  --address http://ray-head:8265 \
  <job-id>
```

The receipt includes the request digest, input and output dataset versions,
fragment coverage, worker resources, distribution versions, warnings, and
elapsed time.

```{warning}
Do not place S3 credentials in the dataset URI or JSON. Resolve the named
storage profile in the Ray worker environment.
```
