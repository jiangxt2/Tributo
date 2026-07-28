# Batch Text Embedding

Generate embeddings for large text datasets using BGE models, distributed across Ray workers.

## Quick Start

```bash
uv run tributo embed batch \
  --input s3://bucket/articles/*.parquet \
  --output s3://bucket/articles_embedded.lance \
  --model bge-small-zh \
  --text-column content \
  --batch-size 64 \
  --concurrency 4
```

## Python API

Use the individual components to build a custom embedding workflow:

```python
from tributo.embeddings import ModelSpec, list_models, submit_embedding_job

# List available models
for spec in list_models():
    print(f"{spec.name}: dim={spec.dim} lang={spec.lang}")

# Submit a batch embedding job via Ray Jobs API
job_id = submit_embedding_job(
    s3_input_path="s3://bucket/data.parquet",
    s3_output_path="s3://bucket/embedded.lance",
    model_name="bge-small-zh",
    text_column="content",
    batch_size=64,
    concurrency=4,
)
```

## Available Models

| Model | Language | Dimension | Speed |
|---|---|---|---|
| `bge-small-zh` | Chinese | 512 | Fast |
| `bge-base-zh` | Chinese | 768 | Moderate |
| `bge-small-en` | English | 384 | Fast |
| `bge-base-en` | English | 768 | Moderate |

List all registered models:

```bash
uv run tributo embed list
```

## Export a Custom Model

Convert a HuggingFace model to ONNX + tokenizer:

```bash
uv run tributo embed export \
  --model BAAI/bge-large-zh-v1.5 \
  --output-dir /opt/models/bge-large-zh
```

## Output Formats

- **Lance** (`.lance`) — columnar format with vector index. Recommended for downstream vector search.
- **Parquet** (`.parquet`) — standard columnar format. Use when you only need the embedding arrays.

## Performance Tuning

| Parameter | Guidance |
|---|---|
| `concurrency` | Number of parallel Ray actors. Start with `2`; increase if workers have spare memory. |
| `batch_size` | Larger = higher throughput, more memory. 32–64 is a good range for BGE models. |
| `num_cpus_per_actor` | Default `1`. Each actor holds ~100 MB for the ONNX model. |

## See Also

- Submit via Ray Jobs API: `uv run tributo embed batch ...`
