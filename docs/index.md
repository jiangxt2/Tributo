# Tributo

Tributo is a Ray-native machine learning SDK. Ray executes distributed work.
Tributo validates requests, selects registered implementations, records
credential-free evidence, and publishes verified model bundles.

## Choose a starting point

- [Install Tributo](getting-started/installation.md) with only the extras your
  workload needs.
- [Run the local quickstart](getting-started/quickstart.md) to train a formal
  distributed algorithm and publish a Bundle.
- [Submit work to a Ray cluster](ray-jobs/index.md) through Ray Jobs.
- Check the [support matrix](reference/support-matrix.md) before selecting a
  production path.

## Explore the components

| Component | What you can do |
| --- | --- |
| Data | Plan bounded reads, run portable transforms, and delegate native writes to Ray Data or Daft |
| Algorithms and training | Run distributed XGBoost, X-Learner causal uplift, DNN, PU, Multinomial Naive Bayes, and Ray Tune workflows |
| Model lifecycle | Export, validate, publish, read, and govern model Bundles |
| Inference and serving | Run Bundle-backed batch inference, explainability jobs, and Ray Serve transports |
| Vector indexing | Build, search, optimize, or compact Lance vector indexes as separate workflows |

Tributo is not a Kubernetes control plane or a multi-tenant machine learning
platform. It does not provision clusters, manage RBAC, assign quotas, or run
approval workflows.

```{note}
Tributo persists configuration as JSON. It rejects YAML configuration. YAML in
a KubeRay manifest belongs to Kubernetes, not to the Tributo configuration
contract.
```

```{toctree}
:hidden:
:maxdepth: 4

Overview <overview/index>
Getting started <getting-started/index>
Examples <examples/index>
Data <data/index>
Algorithms and training <algorithms/index>
Model lifecycle <model-lifecycle/index>
Inference and serving <inference/index>
Vector indexing <vector-index/index>
Reference <reference/index>
Ray clusters <ray-jobs/index>
Operations <operations/index>
Developer guides <developer/index>
Security <security/index>
```
