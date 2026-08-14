# Tributo

<p align="center">
  <strong>Ray-native machine learning SDK for distributed data, training, model bundles, and inference.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/tributo"><img src="https://img.shields.io/pypi/v/tributo?color=blue" alt="PyPI"></a>
  <a href="https://github.com/jiangxt2/tributo/actions/workflows/pr-test-suite.yml"><img src="https://github.com/jiangxt2/tributo/actions/workflows/pr-test-suite.yml/badge.svg" alt="CI"></a>
  <a href="https://tributo.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/tributo/badge/?version=latest" alt="Documentation"></a>
  <a href="https://codecov.io/gh/jiangxt2/tributo"><img src="https://codecov.io/gh/jiangxt2/tributo/branch/master/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/tributo"><img src="https://img.shields.io/pypi/pyversions/tributo" alt="Python versions"></a>
  <a href="https://github.com/jiangxt2/tributo/blob/master/LICENSE"><img src="https://img.shields.io/github/license/jiangxt2/tributo" alt="License"></a>
</p>

---

## What Tributo provides

Tributo adds machine learning control-plane contracts on top of Ray. Ray owns
distributed execution. Tributo validates requests, selects registered
implementations, records credential-free provenance, and publishes verified
model bundles.

- Submit and manage workloads through Ray Jobs.
- Read bounded data through explicit Ray Data or Daft bindings.
- Run distributed XGBoost, DNN, positive-unlabeled learning, and the formal
  algorithm execution contract.
- Publish validated, multi-format model bundles to local storage or S3.
- Run bundle-backed batch inference and Ray Serve HTTP or gRPC endpoints.
- Build, search, optimize, or compact Lance vector indexes as separate
  workflows.
- Produce batch explainability reports through registered adapters.

Tributo is an SDK, not a multi-tenant platform or Kubernetes control plane.
See the [documentation](https://tributo.readthedocs.io/en/latest/) and
[support matrix](https://tributo.readthedocs.io/en/latest/reference/support-matrix/)
for verified paths and explicit boundaries.

---

## Quick start

```bash
pip install tributo
```

```python
from tributo import TributoClient

client = TributoClient("http://127.0.0.1:8265")
job_id = client.submit(entrypoint="python my_script.py")
print(client.get_status(job_id))
```

[Read the full quickstart](https://tributo.readthedocs.io/en/latest/getting-started/quickstart/).

---

## Architecture

<p align="center">
  <a href="docs/architecture/system-landscape.md">
    <img
      src="docs/images/tributo-system-landscape.svg"
      alt="Tributo system landscape showing the framework boundary, Ray runtime, external systems, and platform non-goals."
      width="736"
    >
  </a>
</p>

For a detailed map of current capabilities, planned work, and API stability
guarantees, see the [System Landscape](docs/architecture/system-landscape.md),
[Architecture Documentation](docs/architecture/), and
[API Stability Inventory](docs/STABILITY.md).

---

## Installation

```bash
git clone https://github.com/jiangxt2/tributo.git
cd tributo

# Core install
uv sync

# With XGBoost training + ONNX export
uv sync --extra training

# With data connectors (Lance / Iceberg)
uv sync --extra data

# With Hugging Face sources/exporters
uv sync --extra model-export-hf

# With Lance vector-index operations
uv sync --extra vector-index

# Development dependencies
uv sync --extra dev

# Dual-engine files/tables and PostgreSQL
uv sync --extra data --extra data-daft --extra postgresql
```

---

## Data sources

Data sources are opt-in extras where applicable. Install the extra for the
dialect or backend you use:

| Source | Physical reader | Extra | Status |
|---|---|---|---|
| Local/S3 Parquet and CSV | Ray Data / Daft public readers | core / `tributo[data-daft]` | Alpha; real dual-engine Conformance |
| Local/S3 Iceberg and Lance | Ray Data / Daft public readers | `tributo[data,data-daft]` | Alpha; real dual-engine Conformance |
| PostgreSQL structured table | Ray Data / Daft SQL readers | `tributo[postgresql,data-daft]` | Alpha; real PostgreSQL Conformance |
| HDFS Parquet/CSV | Ray Data + PyArrow Hadoop filesystem | Ray runtime with HDFS libraries | Adapter present; cluster gate pending |
| ClickHouse | independent `daft-olap-connectors` | external package | Adapter present; package/infrastructure gates pending |
| Doris | independent `ray-doris` / `daft-olap-connectors` | external packages | Adapters present; package/infrastructure gates pending |
| ORC / Hive external tables | no locked public reader path | — | Unsupported |

Adapter presence is not a support claim. ClickHouse, Doris, HDFS, and Hive are
reported as available only after their locked external dependencies and real
infrastructure gates pass. Tributo never installs optional connectors at
runtime.

---

## Modules

### Distributed XGBoost training

XGBoost on Ray Train with S3 data sources and automatic ONNX export.

```bash
uv run python examples/xgboost_s3_training.py
```

### Positive-unlabeled learning

The formal nnPU/uPU implementation uses the shared Ray Train/PyTorch DDP kernel
with an explicit class prior and PU-specific fail-closed data checks. The Beta
`PUTrainerImpl` compatibility entry delegates to the same distributed kernel
and accepts one or more Ray Train workers. See the
[PU learning guide](docs/how-to/pu-learning.md) and
[support matrix](docs/reference/support-matrix.md).

### Custom batch inference

Distributed inference with a user-supplied Ray `BasePredictor`, output to
Parquet or Lance.

```python
from tributo.inference import BasePredictor, InferenceConfig, run_batch_inference


class MyPredictor(BasePredictor):
    def _load_model(self):
        self.model = load_model_from_your_runtime()

    def __call__(self, batch):
        return predict_with_your_model(self.model, batch)


run_batch_inference(config, predictor_cls=MyPredictor)
```

### ONNX inference serving

Ray Serve deployment for ONNX models with HTTP API.

```bash
uv run tributo serve start --model-path /path/to/model.onnx
uv run tributo serve status
uv run tributo serve stop
```

### Batch inference

XGBoost + ONNX distributed batch inference.

```python
from tributo.inference.pipeline import InferenceConfig, run_batch_inference

config = InferenceConfig(
    input_uri="s3://bucket/input.parquet",
    output_uri="s3://bucket/output/",
    model_uri="s3://bucket/model.onnx",
)
run_batch_inference(config)
```

### Streaming LLM inference

SSE-based streaming inference for LLMs on Ray Serve.

```bash
uv run tributo serve streaming start --model-path /path/to/model --tokenizer-path /path/to/tokenizer
uv run tributo serve streaming status
```

### Hyperparameter tuning with Ray Tune

Random search / BayesOpt with FIFO / ASHA / HyperBand schedulers.
Tune trials execute setup and fit only: they report the configured metric and
checkpoint without publishing production Bundles. After selecting parameters,
run the Trainer explicitly to publish the single production Bundle.
Ray owns experiment and checkpoint state below the Tune output root's
`trials/` namespace; Tributo does not remove that recovery state. Treat the
Tune output root as Ray-managed storage, not as a production Bundle destination.

```bash
uv run tributo tune run \
  --trainer xgboost \
  --config train_config.json \
  --space search_space.json \
  --output ./tune_results \
  --num-samples 50 \
  --search-alg bayesopt
```

---

## Project structure

```
src/tributo/
├── __init__.py          # Public API (TributoClient, JobConfig, exceptions)
├── job.py               # TributoClient core
├── config.py            # JobConfig (Pydantic models)
├── exceptions.py        # Exception hierarchy
├── cli.py               # CLI entry point (Click)
├── _common/             # Shared utilities (runtime_env, IO, logging, Serve)
├── data/                # Ingestion, transforms, typed handles, and writing
├── algorithms/          # Formal distributed algorithm execution contracts
├── training/            # Trainer compatibility, XGBoost, DNN, PU, and Tune
├── exporting/           # Model bundle planning, validation, and publication
├── inference/           # Bundle-backed distributed batch inference
├── serving/             # Ray Serve HTTP, gRPC, and streaming transports
├── vector_index/        # Lance vector-index jobs and maintenance
├── explainability/      # Batch explanation planning and adapters
├── streaming/           # Unbounded input protocols such as Kafka
├── registry/            # MLflow tracking and registry integration
├── integrations/        # Optional framework, format, sink, and hook adapters
├── pipeline/            # Alpha in-process DAG compatibility utility
└── util/                # @PublicAPI decorator, stability annotations
```

---

## Development

```bash
# Run tests
uv run pytest

# Format
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/tributo
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
