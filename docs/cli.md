# CLI Reference

Tributo provides a unified CLI built on [Click](https://click.palletsprojects.com/).

## Global Options

| Option | Description |
|---|---|
| `--help` | Show help for any command or subcommand. |

## `tributo submit`

Submit a Ray Job.

```bash
uv run tributo submit \
  --address http://127.0.0.1:8265 \
  --entrypoint "python my_script.py" \
  [--config job_config.json] \
  [--num-cpus 2] \
  [--num-gpus 0] \
  [--memory 1073741824]
```

| Option | Required | Description |
|---|---|---|
| `--address` | Yes | Ray Dashboard URL. |
| `--entrypoint` | Yes | Shell command to run on the cluster. |
| `--config` | No | Path to JSON job configuration file. |
| `--num-cpus` | No | Number of CPUs to allocate for the entrypoint. |
| `--num-gpus` | No | Number of GPUs to allocate for the entrypoint. |
| `--memory` | No | Memory to allocate for the entrypoint (in bytes). |

## `tributo status`

Check a job's status.

```bash
uv run tributo status --address http://127.0.0.1:8265 <job_id>
```

## `tributo logs`

Retrieve a job's logs.

```bash
uv run tributo logs --address http://127.0.0.1:8265 <job_id>
```

## `tributo stop`

Stop a running job.

```bash
uv run tributo stop --address http://127.0.0.1:8265 <job_id>
```

## `tributo serve`

Manage ONNX inference services.

```bash
# Start serving
uv run tributo serve start --model-path /path/to/model.onnx [--app-name my-app] [--num-replicas 2]

# Check status
uv run tributo serve status [--app-name my-app]

# Stop serving
uv run tributo serve stop [--app-name my-app]

# Streaming LLM inference
uv run tributo serve streaming start --model-path /path/to/model --tokenizer-path /path/to/tokenizer
uv run tributo serve streaming status
uv run tributo serve streaming stop
```

## `tributo embed`

Batch text embedding and model management.

```bash
# Run batch embedding
uv run tributo embed batch \
  --input s3://bucket/data.parquet \
  --output s3://bucket/embedded.lance \
  --model bge-small-zh \
  --text-column content \
  --batch-size 64 \
  --concurrency 4

# List registered models
uv run tributo embed list

# Export model (HF -> ONNX + tokenizer)
uv run tributo embed export --model bge-small-zh --output-dir /opt/models/bge-small-zh

# Start embedding HTTP service
uv run tributo embed serve start --model-path /opt/models/bge-small-zh
uv run tributo embed serve status
uv run tributo embed serve stop
```

## `tributo tune`

Run hyperparameter optimization.

```bash
uv run tributo tune run --config tune_config.json
```

