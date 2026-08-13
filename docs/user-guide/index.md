# Use cases and user guides

These guides describe currently implemented workflows. Each guide states the
runtime service, optional dependency, and maturity constraints that apply.

## Choose a workflow

- [Prepare distributed data](../data/index.md).
- [Train and tune models](../training/index.md).
- [Publish model bundles](../model-lifecycle/index.md).
- [Run batch or online inference](../inference/index.md).
- [Run batch explainability](../how-to/explainability.md).
- [Generate embeddings](../embeddings/index.md).
- [Operate and troubleshoot jobs](../operations/index.md).

## Execution model

The primary execution path is:

```text
canonical source -> Ray Dataset -> trainer -> model bundle -> inference
```

Bounded data providers, Kafka streaming sources, and model output sinks are
separate contracts. A working read connector does not imply that Tributo can
write inference results back to the same system.
