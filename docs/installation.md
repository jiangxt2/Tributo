# Installation

Tributo requires Python 3.12 or 3.13. Install the smallest dependency set that
matches the workload you intend to run.

## Core package

```bash
python -m pip install tributo
```

The core package includes Ray Jobs, Ray Serve, Ray Tune, Pydantic, ONNX
Runtime, PyArrow, pandas, and S3 filesystem support.

## Optional capabilities

| Workload | Installation |
| --- | --- |
| Ray Data table formats | `python -m pip install "tributo[data]"` |
| Daft ingestion | `python -m pip install "tributo[data,data-daft]"` |
| PostgreSQL ingestion | `python -m pip install "tributo[postgresql]"` |
| Distributed training | `python -m pip install "tributo[training]"` |
| Hugging Face source/exporter and streaming | `python -m pip install "tributo[hf]"` or `python -m pip install "tributo[model-export-hf]"` |
| Torch model export | `python -m pip install "tributo[model-export-torch]"` |
| MLflow registry | `python -m pip install "tributo[registry]"` |
| gRPC serving | `python -m pip install "tributo[grpc]"` |
| Kafka stream source | `python -m pip install "tributo[streaming-inference]"` |

Optional extras install dependencies; they do not imply that every protocol or
reserved problem type has a concrete implementation. Check the
[support matrix](reference/support-matrix.md) before selecting a production
path.

## Source checkout

For repository development:

```bash
git clone https://github.com/jiangxt2/tributo.git
cd tributo
uv sync --extra dev --locked
uv run tributo --help
```

Use `uv.lock` for the project runtime and test environment. Documentation uses
the separate `requirements-doc.lock` so Read the Docs does not install the
full machine learning stack.

## Connect to Ray

Most commands that manage jobs require the Ray Dashboard address:

```bash
uv run tributo status \
  --address http://127.0.0.1:8265 \
  <job-id>
```

The dashboard endpoint must be reachable from the process running the CLI.
Data and training execution occurs in the Ray job runtime rather than in the
local CLI process.

## Configuration format

Tributo accepts JSON configuration:

```json
{
  "entrypoint": "python train.py",
  "num_cpus": 2
}
```

YAML files are not supported.
