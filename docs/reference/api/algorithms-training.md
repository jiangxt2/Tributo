# Algorithms and training API

```{important}
This page is generated from top-level `@PublicAPI` annotations. Do not edit it
by hand. Run `python tools/generate_public_api_reference.py` after changing
a public annotation or moving a public object.
```

Stable, Beta, and Alpha objects appear because Ray-style API policy requires
documentation for every public stability tier.

## `tributo.algorithms.api.artifacts`

```{autoclass} tributo.algorithms.api.artifacts.AlgorithmArtifact
:no-members:
```

```{autoclass} tributo.algorithms.api.artifacts.AlgorithmBundleManifest
:no-members:
```

```{autoclass} tributo.algorithms.api.artifacts.AlgorithmDistributionReceipt
:no-members:
```

```{autoclass} tributo.algorithms.api.artifacts.ArtifactDistributionMode
:no-members:
```

```{autoclass} tributo.algorithms.api.artifacts.ArtifactFile
:no-members:
```

```{autoclass} tributo.algorithms.api.artifacts.ImageProfile
:no-members:
```


## `tributo.algorithms.api.context`

```{autoclass} tributo.algorithms.api.context.UserExecutionContext
:no-members:
```


## `tributo.algorithms.api.descriptor`

```{autoclass} tributo.algorithms.api.descriptor.DistributedAlgorithmDescriptor
:no-members:
```


## `tributo.algorithms.api.distribution`

```{autoclass} tributo.algorithms.api.distribution.CollectivePolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.DistributedExactness
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.DistributionSpec
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.DistributionStrategy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.ExecutionProfile
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.FrameworkNativePolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.InputDistribution
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.IterativeOptimizationPolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.JoblibEstimatorPolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.MapReducePolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.MetricReduction
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.ParallelEnsemblePolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.ResultPolicy
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.StateCoordination
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.StateField
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.WorkerRange
:no-members:
```

```{autoclass} tributo.algorithms.api.distribution.WorkerResources
:no-members:
```


## `tributo.algorithms.api.errors`

```{autoexception} tributo.algorithms.api.errors.AlgorithmConfigurationError
```

```{autoexception} tributo.algorithms.api.errors.AlgorithmDependencyError
```

```{autoexception} tributo.algorithms.api.errors.AlgorithmExecutionError
```

```{autoexception} tributo.algorithms.api.errors.AlgorithmInputError
```

```{autoexception} tributo.algorithms.api.errors.AlgorithmResolutionError
```


## `tributo.algorithms.api.execution`

```{autoclass} tributo.algorithms.api.execution.ExecutionReceipt
:no-members:
```

```{autoclass} tributo.algorithms.api.execution.ExecutionRequest
:no-members:
```

```{autoclass} tributo.algorithms.api.execution.StateCoordinationEvidence
:no-members:
```

```{autoclass} tributo.algorithms.api.execution.WorkerExecutionEvidence
:no-members:
```


## `tributo.algorithms.api.models`

```{autoclass} tributo.algorithms.api.models.AlgorithmExecutionResult
:no-members:
```

```{autoclass} tributo.algorithms.api.models.AlgorithmOperation
:no-members:
```

```{autoclass} tributo.algorithms.api.models.AlgorithmRegistration
:no-members:
```

```{autoclass} tributo.algorithms.api.models.AlgorithmRequest
:no-members:
```

```{autoclass} tributo.algorithms.api.models.AlgorithmResolution
:no-members:
```

```{autoclass} tributo.algorithms.api.models.AlgorithmRunResult
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ArtifactDraft
:no-members:
```

```{autoclass} tributo.algorithms.api.models.BackendInputCompatibility
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ContractBinding
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ContractBindingSet
:no-members:
```

```{autoclass} tributo.algorithms.api.models.EnvironmentSpec
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ExecutionMode
:no-members:
```

```{autoclass} tributo.algorithms.api.models.FailureCategory
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ImplementationDescriptor
:no-members:
```

```{autoclass} tributo.algorithms.api.models.InputBinding
:no-members:
```

```{autoclass} tributo.algorithms.api.models.InputBindingSet
:no-members:
```

```{autoclass} tributo.algorithms.api.models.InputCoverageContract
:no-members:
```

```{autoclass} tributo.algorithms.api.models.QualifiedReference
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ResolvedAlgorithmPlan
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ResolvedInputDescriptor
:no-members:
```

```{autoclass} tributo.algorithms.api.models.ResolvedInputDescriptorSet
:no-members:
```

```{autoclass} tributo.algorithms.api.models.RuntimeBinding
:no-members:
```

```{autoclass} tributo.algorithms.api.models.RuntimeTopology
:no-members:
```

```{autoclass} tributo.algorithms.api.models.WorkerExecutionResult
:no-members:
```


## `tributo.algorithms.api.support`

```{autoclass} tributo.algorithms.api.support.AlgorithmSupportEvidence
:no-members:
```

```{autoclass} tributo.algorithms.api.support.AlgorithmSupportEvidenceRegistry
:no-members:
```

```{autoclass} tributo.algorithms.api.support.DistributedSemantics
:no-members:
```

```{autoclass} tributo.algorithms.api.support.SupportTier
:no-members:
```


## `tributo.algorithms.composition`

```{autofunction} tributo.algorithms.composition.build_algorithm_dispatcher
```


## `tributo.algorithms.conformance`

```{autoclass} tributo.algorithms.conformance.AlgorithmPackageConformanceReport
:no-members:
```

```{autofunction} tributo.algorithms.conformance.validate_algorithm_descriptor_conformance
```

```{autofunction} tributo.algorithms.conformance.validate_installed_algorithm_package
```


## `tributo.algorithms.core.builder`

```{autoclass} tributo.algorithms.core.builder.AlgorithmBuilder
:no-members:
```


## `tributo.algorithms.core.runtime`

```{autoclass} tributo.algorithms.core.runtime.LocalRuntimeOptions
:no-members:
```

```{autoclass} tributo.algorithms.core.runtime.RayRuntimeSession
:no-members:
```


## `tributo.algorithms.spi.contracts`

```{autoclass} tributo.algorithms.spi.contracts.AlgorithmContractValidator
:no-members:
```


## `tributo.algorithms.spi.execution`

```{autoclass} tributo.algorithms.spi.execution.AlgorithmExecutionContext
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.CollectiveAlgorithm
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.EnsembleUnitSpec
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.Evaluable
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.ExecutionEnvelope
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.Fittable
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.FrameworkNativeAlgorithm
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.IterativeOptimizationAlgorithm
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.JoblibEstimatorRecipe
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.MapReduceAlgorithm
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.ParallelEnsembleAlgorithm
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.PortableRuntimeAdapter
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.Predictable
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.RuntimeExecutionEnvelope
:no-members:
```

```{autoclass} tributo.algorithms.spi.execution.Transformable
:no-members:
```


## `tributo.algorithms.spi.input`

```{autoclass} tributo.algorithms.spi.input.InputExecutionContext
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.InputResolutionContext
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.InputResolverPort
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.InputRuntimeAdapter
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.MaterializedTabularInputView
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.PreparedInput
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.ResolvedInputLease
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.RuntimeInputBinding
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.TabularBatchInputView
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.WorkerInputAdapter
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.WorkerInputPayload
:no-members:
```

```{autoclass} tributo.algorithms.spi.input.WorkerInputPayloadSet
:no-members:
```


## `tributo.algorithms.spi.torch`

```{autoclass} tributo.algorithms.spi.torch.MetricPlan
:no-members:
```

```{autoclass} tributo.algorithms.spi.torch.OptimizationPlan
:no-members:
```

```{autoclass} tributo.algorithms.spi.torch.TorchTrainingRecipe
:no-members:
```

```{autoclass} tributo.algorithms.spi.torch.TrainingRecipeV2
:no-members:
```

```{autoclass} tributo.algorithms.spi.torch.TrainingStepResult
:no-members:
```


## `tributo.integrations.algorithm_inputs.ingestion`

```{autoclass} tributo.integrations.algorithm_inputs.ingestion.IngestionInputInvocation
:no-members:
```

```{autoclass} tributo.integrations.algorithm_inputs.ingestion.IngestionInputResolver
:no-members:
```

```{autoclass} tributo.integrations.algorithm_inputs.ingestion.IngestionInputRuntimeAdapter
:no-members:
```

```{autoclass} tributo.integrations.algorithm_inputs.ingestion.IngestionRequestRef
:no-members:
```


## `tributo.training.algorithm_spec`

```{autoclass} tributo.training.algorithm_spec.AlgorithmSpec
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.AlgorithmStatus
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.Capability
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.DataContract
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.DataLoadingMode
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.ExecutionKind
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.ProblemFamily
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.ProblemType
:no-members:
```

```{autoclass} tributo.training.algorithm_spec.ResourceHints
:no-members:
```


## `tributo.training.base`

```{autoclass} tributo.training.base.BaseTrainer
:no-members:
```


## `tributo.training.catalog`

```{autoclass} tributo.training.catalog.AlgorithmCatalog
:no-members:
```

```{autofunction} tributo.training.catalog.get_algorithm_catalog
```


## `tributo.training.causal_estimator`

```{autoclass} tributo.training.causal_estimator.BaseCausalEstimator
:no-members:
```

```{autoclass} tributo.training.causal_estimator.CausalEffect
:no-members:
```

```{autoclass} tributo.training.causal_estimator.CausalGraph
:no-members:
```

```{autoclass} tributo.training.causal_estimator.RefutationResult
:no-members:
```


## `tributo.training.checkpoint`

```{autoclass} tributo.training.checkpoint.ResumeCheckpointV1
:no-members:
```

```{autoclass} tributo.training.checkpoint.ResumeConfig
:no-members:
```

```{autofunction} tributo.training.checkpoint.capture_rng_state
```

```{autofunction} tributo.training.checkpoint.checkpoint_config
```

```{autofunction} tributo.training.checkpoint.checkpoint_directory
```

```{autofunction} tributo.training.checkpoint.compute_payload_digest
```

```{autofunction} tributo.training.checkpoint.load_initial_checkpoint
```

```{autofunction} tributo.training.checkpoint.materialize_checkpoint_directory
```

```{autofunction} tributo.training.checkpoint.publish_checkpoint_directory
```

```{autofunction} tributo.training.checkpoint.read_resume_manifest
```

```{autofunction} tributo.training.checkpoint.restore_rng_state
```

```{autofunction} tributo.training.checkpoint.write_resume_manifest
```


## `tributo.training.exporters.artifact_protocol`

```{autoclass} tributo.training.exporters.artifact_protocol.ArtifactExporter
:no-members:
```

```{autofunction} tributo.training.exporters.artifact_protocol.is_known_artifact_kind
```


## `tributo.training.exporters.causal_report`

```{autoclass} tributo.training.exporters.causal_report.CausalReportExporter
:no-members:
```


## `tributo.training.exporters.safetensors`

```{autoclass} tributo.training.exporters.safetensors.SafetensorsExporter
:no-members:
```


## `tributo.training.exporters.torch_onnx_exporter`

```{autofunction} tributo.training.exporters.torch_onnx_exporter.export_pytorch_to_onnx
```


## `tributo.training.exporters.torchscript`

```{autoclass} tributo.training.exporters.torchscript.TorchScriptExporter
:no-members:
```


## `tributo.training.features.column_types`

```{autoclass} tributo.training.features.column_types.DenseFeat
:no-members:
```

```{autoclass} tributo.training.features.column_types.SparseFeat
:no-members:
```


## `tributo.training.flavor`

```{autoclass} tributo.training.flavor.ModelFlavor
:no-members:
```

```{autoclass} tributo.training.flavor.ONNXFlavor
:no-members:
```


## `tributo.training.graph_trainer`

```{autoclass} tributo.training.graph_trainer.BaseGraphTrainer
:no-members:
```


## `tributo.training.job_submitter`

```{autoclass} tributo.training.job_submitter.JobAttempt
:no-members:
```

```{autoclass} tributo.training.job_submitter.TrainingJobResult
:no-members:
```

```{autofunction} tributo.training.job_submitter.submit_training_job
```

```{autofunction} tributo.training.job_submitter.submit_training_job_with_identity
```

```{autofunction} tributo.training.job_submitter.submit_training_job_with_retry
```

```{autofunction} tributo.training.job_submitter.wait_for_job
```


## `tributo.training.local_runner`

```{autofunction} tributo.training.local_runner.run_local_trial
```


## `tributo.training.portable_tune`

```{autoclass} tributo.training.portable_tune.PortableTuneRunner
:no-members:
```


## `tributo.training.registry`

```{autofunction} tributo.training.registry.get_trainer
```

```{autofunction} tributo.training.registry.list_trainers
```

```{autofunction} tributo.training.registry.register
```


## `tributo.training.results`

```{autoclass} tributo.training.results.BundleStatus
:no-members:
```

```{autoclass} tributo.training.results.TrainingHookStatus
:no-members:
```

```{autoclass} tributo.training.results.TrainingResult
:no-members:
```

```{autoclass} tributo.training.results.TrainingStatus
:no-members:
```


## `tributo.training.tune_config`

```{autoclass} tributo.training.tune_config.TuneSearchConfig
:no-members:
```


## `tributo.training.tune_runner`

```{autoclass} tributo.training.tune_runner.TuneRunner
:no-members:
```

```{autofunction} tributo.training.tune_runner.extract_best_params
```


## `tributo.training.tune_space`

```{autoclass} tributo.training.tune_space.SearchParamSpec
:no-members:
```

```{autoclass} tributo.training.tune_space.SearchSpaceSpec
:no-members:
```

```{autofunction} tributo.training.tune_space.parse_search_space
```

```{autofunction} tributo.training.tune_space.resolve_local_overrides
```

```{autofunction} tributo.training.tune_space.to_ray_param_space
```

```{autofunction} tributo.training.tune_space.validate_search_targets
```

```{autofunction} tributo.training.tune_space.warn_search_space_conflicts
```
