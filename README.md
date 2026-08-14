# Tributo

<p align="center">
  <strong>Telecom-native ML framework for Ray clusters.</strong><br>
  PU Learning · Distributed Training · Batch Inference & ONNX Serving
</p>

<p align="center">
  <a href="https://pypi.org/project/tributo"><img src="https://img.shields.io/pypi/v/tributo?color=blue" alt="PyPI"></a>
  <a href="https://github.com/jiangxt2/tributo/actions/workflows/pr-test-suite.yml"><img src="https://github.com/jiangxt2/tributo/actions/workflows/pr-test-suite.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/jiangxt2/tributo"><img src="https://codecov.io/gh/jiangxt2/tributo/branch/master/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/tributo"><img src="https://img.shields.io/pypi/pyversions/tributo" alt="Python versions"></a>
  <a href="https://github.com/jiangxt2/tributo/blob/master/LICENSE"><img src="https://img.shields.io/github/license/jiangxt2/tributo" alt="License"></a>
</p>

---

## Why Tributo

Tributo tackles three problems that no existing open-source framework solves end-to-end on Ray:

**① PU Learning — train classifiers when you only have positive labels**

Real-world fraud detection, churn prediction, and identity resolution all share the same data problem: confirmed positives are scarce, but unlabeled samples are plentiful and *not* truly negative. Tributo's PU Learning module runs nnPU/uPU loss training on Ray, requires an explicit positive class prior, and exports ONNX models for production serving. The current DNN and PU trainers are single-worker implementations; XGBoost is the distributed trainer.

```python
from tributo.training.pu_trainer import run_pu_training_from_json

# Single-worker PU training on Ray → ONNX export
result = run_pu_training_from_json("pu_config.json")
print(result["onnx_path"])  # Ready for serving
```

**② Behavioral Sequence Pre-training — model irregular time-series at scale (coming in v2.0)**

Call records, SMS logs, and app usage traces have irregular intervals and multi-granularity patterns that standard Transformers cannot handle. Tributo's planned Temporal Transformer will encode continuous timestamps and support three self-supervised objectives (masked span, interval prediction, next-activity) for pre-training on billion-row event tables. See the roadmap for timeline.

```python
# Coming in v2.0 — stay tuned
# from tributo.pretrain import TemporalTransformer, SequenceConfig
#
# config = SequenceConfig.from_json("pretrain_config.json")
# model = TemporalTransformer(config)
# model.pretrain(data_path="s3://bucket/events/*.parquet")
```

**③ Large-scale inference pipeline — bounded ingestion → distributed inference → Lance storage**

A typed ingestion Gateway selects Ray Data or Daft explicitly and delegates
physical reads to that engine. Inference consumers select Ray Data;
there is no implicit Daft-to-Ray materialization. Lance writes remain a
separate output concern.

Lance reads use the public Ray Data or Daft `read_lance` API and do not collect
the dataset into a driver-side Arrow table.

```python
from tributo.data import (
    IngestionRequest,
    ParquetSourceConfig,
    RayDataHandle,
    get_connector,
    open_ingestion,
)

result = open_ingestion(
    IngestionRequest(
        source=ParquetSourceConfig(path="s3://bucket/data.parquet"),
        engine="ray",
    )
)
try:
    assert isinstance(result.handle, RayDataHandle)
    get_connector("lance").write(
        result.handle.dataset,
        path="s3://bucket/index.lance",
    )
finally:
    result.close()
```

---

## Quick Start

```bash
pip install tributo
```

```python
from tributo import TributoClient

client = TributoClient("http://127.0.0.1:8265")
job_id = client.submit(entrypoint="python my_script.py")
print(client.get_status(job_id))
```

[→ Full Quickstart Guide](docs/quickstart.md)

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

# Development dependencies
uv sync --extra dev

# Dual-engine files/tables and PostgreSQL
uv sync --extra data --extra data-daft --extra postgresql
```

---

## Data Sources

Data sources are opt-in extras where applicable — install the extra for the
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

### Distributed XGBoost Training

XGBoost on Ray Train with S3 data sources and automatic ONNX export.

```bash
uv run python examples/xgboost_s3_training.py
```

### PU Learning (Positive-Unlabeled)

nnPU/uPU training with an explicit class prior and PU-specific metrics. The trainer currently requires one Ray worker. See the [PU Learning guide](docs/how-to/pu-learning.md) and [support matrix](docs/reference/support-matrix.md).

### Custom Batch Inference

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

### ONNX Inference Serving

Ray Serve deployment for ONNX models with HTTP API.

```bash
uv run tributo serve start --model-path /path/to/model.onnx
uv run tributo serve status
uv run tributo serve stop
```

### Batch Inference

XGBoost + ONNX distributed batch inference.

```python
from tributo.inference.pipeline import InferenceConfig, run_batch_inference

config = InferenceConfig(
    s3_input_path="s3://bucket/input.parquet",
    s3_output_path="s3://bucket/output/",
    model_uri="s3://bucket/model.onnx",
)
run_batch_inference(config)
```

### Streaming LLM Inference

SSE-based streaming inference for LLMs on Ray Serve.

```bash
uv run tributo serve streaming start --model-path /path/to/model --tokenizer-path /path/to/tokenizer
uv run tributo serve streaming status
```

### Hyperparameter Tuning (Ray Tune)

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

## Project Structure

```
src/tributo/
├── __init__.py          # Public API (TributoClient, JobConfig, exceptions)
├── job.py               # TributoClient core
├── config.py            # JobConfig (Pydantic models)
├── exceptions.py        # Exception hierarchy
├── cli.py               # CLI entry point (Click)
├── _common/             # Shared utilities (runtime_env, IO, logging, Serve)
├── data/                # Data connectors (Parquet / Lance / Iceberg)
├── training/            # XGBoost, DNN, PU Learning, Tune on Ray Train
├── serving/             # ONNX inference service (HTTP / gRPC / streaming)
├── inference/           # Distributed batch inference
├── registry/            # Model registry (MLflow integration)
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

Apache 2.0 — see [LICENSE](LICENSE).
