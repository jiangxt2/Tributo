# API Reference

Tributo's public API is annotated with stability guarantees via `@PublicAPI(stability=...)`.

**Stability levels:**

| Level | Meaning |
|---|---|
| `stable` | Backward-compatible. Breaking changes require a major version bump. |
| `beta` | Generally stable; minor breaking changes possible with deprecation notice. |
| `alpha` | Active development; APIs may change without notice. |

## Core (`tributo`)

::: tributo.TributoClient
    options:
      stability: stable

::: tributo.JobConfig
    options:
      stability: stable

## Training (`tributo.training`)

::: tributo.training.build_trainer
    options:
      stability: beta

::: tributo.training.data_loader.load_ray_dataset_from_config
    options:
      stability: beta

## Embeddings (`tributo.embeddings`)

::: tributo.embeddings.ModelSpec
    options:
      stability: beta

::: tributo.embeddings.submit_embedding_job
    options:
      stability: beta

## Serving (`tributo.serving`)

::: tributo.serving.start_serving
    options:
      stability: beta

::: tributo.serving.stop_serving
    options:
      stability: beta

## Inference (`tributo.inference`)

::: tributo.inference.InferenceConfig
    options:
      stability: beta

::: tributo.inference.run_batch_inference
    options:
      stability: beta

## Exceptions

All exceptions inherit from `TributoError`:

| Exception | Description |
|---|---|
| `TributoError` | Base exception for all Tributo errors. |
| `JobSubmissionError` | Failed to submit a job to Ray. |
| `JobExecutionError` | Job failed during execution. |
| `JobConfigurationError` | Invalid job configuration. |
| `JobTimeoutError` | Job exceeded its timeout. |
| `ModelExportError` | ONNX model export failed. |
| `DataSourceError` | Data source read/write error. |
