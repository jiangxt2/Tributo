# Tributo

Tributo is a Ray-native machine learning SDK for data access, distributed
training, model bundles, and batch or online inference.

## Start here

- [Install Tributo](installation.md) with the extras required by your workload.
- Complete the [Quickstart](quickstart.md) against a Ray cluster.
- Browse the [user guides](user-guide/index.md) for task-oriented workflows.
- Use the [Python API](api.md) and [CLI](cli.md) references as source-aligned
  contracts.

## Current capabilities

| Area | Supported path |
| --- | --- |
| Ray Jobs | Submit, inspect, stream logs, and stop jobs |
| Data | Explicit Ray Data/Daft ingestion for verified file, table, and PostgreSQL inputs; optional Connector adapters fail closed until validated |
| Training | Local/Kubernetes Ray profiles; distributed XGBoost, DNN, PU and constrained algorithm SPI; Ray Tune integration |
| Model lifecycle | Validated multi-format bundles with local or S3 publication |
| Inference | Ray Data batch inference and Ray Serve HTTP/gRPC endpoints |
| Embeddings | Ray Jobs batch processing and online serving |

See the [support matrix](reference/support-matrix.md) for maturity levels and
known boundaries. An enum, protocol, or prototype alone is not treated as an
implemented capability.

## Minimal job submission

```bash
pip install tributo
```

```python
from tributo import TributoClient

client = TributoClient("http://127.0.0.1:8265")
job_id = client.submit(entrypoint="python my_script.py")
print(client.get_status(job_id))
```

Configuration files accepted by Tributo are JSON. YAML configuration is
explicitly rejected.

```{toctree}
:hidden:
:maxdepth: 4

Overview <overview/index>
Getting Started <quickstart>
Installation <installation>
Use Cases <user-guide/index>
Examples <examples/index>
Integrations <integrations/index>
Data <data/index>
Training <training/index>
Model Lifecycle <model-lifecycle/index>
Inference <inference/index>
Embeddings <embeddings/index>
Reference <reference/index>
Ray Jobs and Clusters <ray-jobs/index>
Monitoring and Troubleshooting <operations/index>
Developer Guides <developer/index>
Architecture <architecture/index>
Security <security/index>
```
