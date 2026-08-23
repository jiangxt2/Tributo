# Core API

```{important}
This page is generated from top-level `@PublicAPI` annotations. Do not edit it
by hand. Run `python tools/generate_public_api_reference.py` after changing
a public annotation or moving a public object.
```

Stable, Beta, and Alpha objects appear because Ray-style API policy requires
documentation for every public stability tier.

## `tributo._common.dependencies`

```{autoclass} tributo._common.dependencies.DependencySpec
:no-members:
```

```{autoclass} tributo._common.dependencies.DependencyState
:no-members:
```

```{autoclass} tributo._common.dependencies.DependencyStatus
:no-members:
```

```{autoexception} tributo._common.dependencies.DependencyUnavailableError
```

```{autoexception} tributo._common.dependencies.MissingOptionalDependency
```

```{autofunction} tributo._common.dependencies.probe_dependency
```

```{autofunction} tributo._common.dependencies.require_dependency
```


## `tributo._common.storage_profiles`

```{autoclass} tributo._common.storage_profiles.StorageProfile
:no-members:
```

```{autoclass} tributo._common.storage_profiles.StorageProfileResolver
:no-members:
```


## `tributo.config`

```{autoclass} tributo.config.AlgorithmExecutionConfig
:no-members:
```

```{autoclass} tributo.config.AlgorithmInputConfig
:no-members:
```

```{autoclass} tributo.config.AlgorithmWorkerResourcesConfig
:no-members:
```

```{autoclass} tributo.JobConfig
:no-members:
```

```{autoclass} tributo.config.LocalRayRuntimeConfig
:no-members:
```


## `tributo.exceptions`

```{autoexception} tributo.exceptions.AliasConflict
```

```{autoexception} tributo.exceptions.ArtifactCorruptedError
```

```{autoexception} tributo.exceptions.BundleCommitBusyError
```

```{autoexception} tributo.exceptions.BundleExportError
```

```{autoexception} tributo.exceptions.DataQueryError
```

```{autoexception} tributo.DataSourceError
```

```{autoexception} tributo.exceptions.EmptyInputError
```

```{autoexception} tributo.exceptions.EngineNotAvailableError
```

```{autoexception} tributo.exceptions.InputColumnMissingError
```

```{autoexception} tributo.JobConfigurationError
```

```{autoexception} tributo.JobExecutionError
```

```{autoexception} tributo.JobSubmissionError
```

```{autoexception} tributo.JobTimeoutError
```

```{autoexception} tributo.exceptions.KafkaCommitError
```

```{autoexception} tributo.exceptions.KafkaPoisonMessageError
```

```{autoexception} tributo.ModelExportError
```

```{autoexception} tributo.exceptions.ModelFormatUnsupportedError
```

```{autoexception} tributo.exceptions.ModelLoadError
```

```{autoexception} tributo.exceptions.ModelSchemaMismatchError
```

```{autoexception} tributo.exceptions.PluginLoadIssue
```

```{autoexception} tributo.exceptions.PostPublishCallbackError
```

```{autoexception} tributo.exceptions.PredictionError
```

```{autoexception} tributo.exceptions.ResourceBudgetExceededError
```

```{autoexception} tributo.exceptions.ResultMaterializationError
```

```{autoexception} tributo.exceptions.ResultWriteError
```

```{autoexception} tributo.exceptions.SessionFatalError
```

```{autoexception} tributo.exceptions.StreamSourceError
```

```{autoexception} tributo.TributoError
```

```{autoexception} tributo.exceptions.UnsupportedArtifactFormat
```


## `tributo.job`

```{autoclass} tributo.RayJob
:no-members:
```

```{autoclass} tributo.TributoClient
:no-members:
```


## `tributo.ray_jobs`

```{autoclass} tributo.ray_jobs.RayJobSubmission
:no-members:
```

```{autofunction} tributo.ray_jobs.get_ray_job_logs
```

```{autofunction} tributo.ray_jobs.get_ray_job_status
```

```{autofunction} tributo.ray_jobs.stop_ray_job
```

```{autofunction} tributo.ray_jobs.submit_ray_job
```


## `tributo.runtime`

```{autoclass} tributo.RuntimeExecutionMode
:no-members:
```

```{autoclass} tributo.RuntimeLifecycle
:no-members:
```

```{autoclass} tributo.RuntimeSubmissionMode
:no-members:
```

```{autoclass} tributo.RuntimeTarget
:no-members:
```


## `tributo.runtime_providers`

```{autoclass} tributo.RuntimeLease
:no-members:
```
