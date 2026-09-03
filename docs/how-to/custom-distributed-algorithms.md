# Custom distributed algorithms

Tributo accepts trusted Python packages through a narrow descriptor SPI. A
package can be pre-installed in the selected image Profile, supplied as a
reviewed code-only Wheel through Ray Job `py_modules`, or supplied as a
reviewed offline Bundle containing a complete Wheelhouse. Tributo does not
resolve plugin dependencies online, isolate arbitrary code, hot reload
packages, or maintain multiple plugin versions.

A wheel supplied through `py_modules` must be code-only: its Core Metadata must
not contain `Requires-Dist` entries. Ray installs such wheels with pip into a
job-local target directory, so declared dependencies would invoke dependency
resolution during job startup. Tributo, sklearn, and every other runtime
dependency must already exist in the reviewed Ray image. A package installed
while building that image can use normal locked build-time dependencies.

## Select the runtime distribution mode

The runtime contract is explicit at Job submission time. The platform selects
an immutable `ImageProfile`; the caller supplies an `AlgorithmArtifact` only
when the algorithm is not already in that image:

```python
from tributo.algorithms import AlgorithmArtifact, ImageProfile
from tributo.training import submit_training_job


profile = ImageProfile(
    profile_id="sklearn.cpu.v1",
    image_uri="registry.internal/tributo/sklearn:2026.08",
    image_digest="<64-lower-case-sha256>",
    python_version="3.12",
    sys_platform="linux",
    platform_machine="x86_64",
    wheel_tags=("py3-none-any",),
    algorithm_ids=("example.random_forest",),
    installed_distributions={
        "pip": "24.3.1",
        "scikit-learn": "1.6.1",
        "numpy": "2.2.6",
    },
)
artifact = AlgorithmArtifact(
    source="/trusted-artifacts/my_algorithm-1.0.0-py3-none-any.whl",
    package_name="my-algorithm",
    package_version="1.0.0",
)

job_id = submit_training_job(
    "python train.py",
    algorithm_artifact=artifact,
    image_profile=profile,
)
```

The default `image_py_modules` mode validates the Wheel locally and passes it
to Ray as `runtime_env.py_modules`. Its Core Metadata must contain no
`Requires-Dist` entries. Dependencies must already be present in the selected
image Profile; `declared_dependencies` or a registration's
`EnvironmentSpec.dependencies` are checked against that inventory before the
Ray Job is submitted. A local Wheel may instead be an immutable HTTPS/S3 URI
with its SHA-256, package identity, entry-point names, and Wheel tags declared
explicitly.

The Wheel's `tributo.algorithms` entry point is automatically selected through
the existing plugin discovery path. Tributo sets `TRIBUTO_PLUGINS` for the
entry point recorded by the artifact, so the Driver's normal descriptor
discovery and `TrainingAlgorithmRegistry` bootstrap register it for this Job.
One Job has one active algorithm entry point. This is Job-local registration;
it does not mutate a global registry.

## Use an offline Wheelhouse

Use `offline_wheelhouse` when the algorithm has Python dependencies that are
not in the image and the complete, already-built dependency closure is allowed
to travel with the Job. A standard Bundle is:

```text
algorithm-bundle/
├── manifest.json
├── algorithm.whl                         # optional friendly root copy
├── requirements.lock
└── wheelhouse/
    ├── my_algorithm-1.0.0-py3-none-any.whl
    ├── dependency_a-3.2.0-py3-none-any.whl
    └── dependency_b-1.4.0-py3-none-any.whl
```

`requirements.lock` must keep all installation controls inside the file:

```text
--no-index
--find-links ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse
${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse/my_algorithm-1.0.0-py3-none-any.whl
```

Tributo verifies the manifest file digests, Wheel metadata, Wheel tags,
`Requires-Dist` closure, and the absence of HTTP(S), VCS, index, editable,
constraint, and nested-requirements references. The Wheel may retain standard
`Requires-Dist` metadata in this mode. The image must explicitly allow offline
pip, and all relevant dependency Wheels must be present in the Bundle or
already listed in the image Profile.

The selected image must also contain `pip` in its Ray process environment.
Ray 2.55.1 creates the `runtime_env.pip` environment by cloning the active
Python environment; an image built with UV should bootstrap pip during image
construction (for example, with Python's bundled `ensurepip`) and list its
version in `ImageProfile.installed_distributions`. This is an image-build
operation and does not access an index during Job startup.

```python
offline_artifact = AlgorithmArtifact(
    source="/trusted-artifacts/my-algorithm-bundle",
    mode="offline_wheelhouse",
)

job_id = submit_training_job(
    "python train.py",
    algorithm_artifact=offline_artifact,
    image_profile=profile,
)
```

In this mode the Bundle is Ray's single `working_dir`, because Ray expands
`${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}` relative to that directory. The
training entrypoint therefore must be present in the selected image or in the
Bundle itself. Tributo does not silently merge a second project directory into
the Bundle; if the training code must be uploaded separately, use the
code-only `py_modules` mode or build one reviewed application Bundle containing
both the entrypoint and the offline dependency files.

For an internal HTTPS/S3 Bundle URI, the URI must identify an immutable ZIP
archive; a remote directory URI is not a valid Ray `working_dir` source. The
archive must expand directly to `manifest.json`, `requirements.lock`, and
`wheelhouse/` (without an extra top-level directory). Pass the archive
SHA-256, package identity, entry-point names, and the attested `manifest`
metadata in `AlgorithmArtifact`. Tributo can then perform the same fail-closed
preflight without downloading or resolving anything on the submitting host;
Ray fetches the fixed `working_dir` archive when it creates the Job runtime
environment. This still requires the Ray nodes to be authorized to read the
internal artifact store, but it does not require a pip index or a separate pip
service.

Ray's `pip_install_options` is an argv list, not a shell command. Therefore
the `${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}` expansion belongs in
`requirements.lock`, while Tributo supplies the fixed options
`--disable-pip-version-check` and `--no-cache-dir`. The generated runtime
environment is equivalent to:

```yaml
pip:
  packages:
    - "-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/requirements.lock"
  pip_check: true
  pip_install_options:
    - --disable-pip-version-check
    - --no-cache-dir
```

`pip_check: true` is a whole-environment health check in Ray 2.55.1's normal
`--system-site-packages` runtime environment; it is not the Wheelhouse
closure check. Image Profiles should be published only after a clean baseline
`python -m pip check`. Ray's pip API has no option to ignore selected baseline
lines: if an explicitly approved historical conflict is recorded in
`pip_check_baseline`, Tributo sets `pip_check` to `false` for that Bundle and
records the waiver in the distribution receipt. Such a Profile must be
treated as an exception and remediated before production promotion; the
waiver must never be attributed to the user algorithm.

## Keep `from_sklearn()` compatible

`from_sklearn()` remains a managed-estimator compatibility path. It does not
turn an arbitrary sklearn estimator into a distributed single-model trainer.
Its existing `EnvironmentSpec` is the dependency declaration reused by the
same artifact preflight:

```python
from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import EnvironmentSpec

registration = AlgorithmBuilder.from_sklearn(
    spec=SKLEARN_SPEC,
    implementation_id="example.random_forest",
    implementation_version="1.0.0",
    estimator_factory="example_algorithms.random_forest:create_estimator",
    environment=EnvironmentSpec(
        environment_id="example.random_forest.v1",
        dependencies=("scikit-learn>=1.6,<1.7",),
    ),
    allowed_config_keys=(),
)

job_id = submit_training_job(
    "python train.py",
    algorithm_artifact=artifact,
    image_profile=profile,
    environment=registration.environment,
)
```

This validates the declared sklearn dependency against the image or offline
Wheelhouse, while preserving the existing managed-estimator topology and
support classification. It does not add an implicit `pip install` path.

## Choose an image Profile

Profiles are immutable compatibility records, not an algorithm resolver. A
deployment may publish separate CPU, GPU, PyTorch, sklearn, or algorithm-family
Profiles. Python, Ray, native-library, CUDA/NCCL, and Wheel-tag compatibility
must be established when the image is built and tested. Tributo validates the
selected Profile and its digest at submission; it does not switch clusters or
build an image during training. `wheel_tags` and the target marker fields
(`python_version`, `sys_platform`, and `platform_machine`) are mandatory for
artifact preflight. A non-empty `algorithm_ids` allowlist is enforced against
an offline Bundle's manifest; an empty allowlist means that the Profile does
not restrict the algorithm family.

## Use a low-code PyTorch recipe

For Core-owned training, subclass `TorchRecipe` and implement the typed
`build_modules`, `adapt_batch`, `training_step`, `validation_step`,
`configure_optimizers`, `metric_plan`, and `artifact_plan` hooks. Framework-
owned training instead subclasses `RayTorchAdapter`; the Adapter validates its
environment, binds role datasets, receives a Core-selected checkpoint context,
and cannot create a nested Trainer or declare a second execution plan.

```python
from tributo.algorithms import (
    TorchBatch,
    TorchLossContribution,
    TorchMetricPlan,
    TorchOptimizationPlan,
    TorchRecipe,
    TorchStepResult,
)


class BinaryLinearRecipe(TorchRecipe):
    def build_modules(self, context):
        import torch

        return {"model": torch.nn.Linear(2, 1), "loss": torch.nn.BCEWithLogitsLoss()}

    def adapt_batch(self, batch, context):
        import torch

        features = torch.as_tensor(batch["features"])
        targets = torch.as_tensor(batch["label"])
        return TorchBatch(positional=(features,), targets=targets, local_rows=len(targets))

    def training_step(self, modules, batch, context):
        import torch

        predictions = modules["model"](batch.positional[0])
        numerator = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions, batch.targets.float(), reduction="sum"
        )
        return TorchStepResult(
            outputs={"prediction": predictions},
            loss=TorchLossContribution(numerator, batch.local_rows),
        )

    def validation_step(self, modules, batch, context):
        return self.training_step(modules, batch, context)

    def configure_optimizers(self, modules, context):
        import torch

        return TorchOptimizationPlan(torch.optim.Adam(modules["model"].parameters(), lr=1e-3))

    def metric_plan(self, context):
        return TorchMetricPlan({"train_loss": "sum_count"})

    def artifact_plan(self, context):
        return {"source_kind": "torch_module", "roles": {"inference": "onnx-model"}}
```

Use `AlgorithmBuilder.from_torch()` to lower a Recipe, or
`AlgorithmBuilder.from_torch_adapter()` for a framework Adapter. Losses always
submit explicit numerator/normalizer pairs; model-specific composite reducers
remain owned by the algorithm Wheel while Core owns collectives and scaling.

The complete independent package used by conformance lives under
the installed algorithm Wheel. It contains no Tributo private Runtime import,
Ray worker loop, checkpoint upload, Bundle publisher, or deployment code.
Default Recipes receive role-routed datasets; the Core Runtime owns sharding,
checkpoint handling, evidence and Bundle publication.

## Choose the state coordination strategy

Implement exactly one interface:

- `CollectiveAlgorithm` for iterative models that synchronize gradients or
  parameters through Ray Train collectives;
- `MapReduceAlgorithm` for models with bounded, associative partial state;
- `FrameworkNativeAlgorithm` when an installed framework owns sharding,
  communication, and consolidated checkpoints.

Inheriting an interface is necessary but not sufficient. The package must also
publish an immutable `DistributedAlgorithmDescriptor` containing one atomic
`AlgorithmRegistration`: algorithm identity, implementation references,
environment requirements, result policy, and a matching `DistributionSpec`.
Use `AlgorithmBuilder.from_distributed_algorithm()` to assemble these existing
contracts. It fills only the fields mechanically implied by the strategy:
`ExecutionMode`, Runtime ID, Runtime topology, input distribution, state
coordination, and the standard Ray Data Worker input adapter. Worker ranges,
resources, strategy policy, package identity, environment, configuration keys,
and support evidence remain explicit.

This Builder removes declaration boilerplate. It does not inspect or rewrite an
arbitrary single-machine `fit()` implementation. The algorithm must still
implement the synchronization, associative reduction, or framework-native
semantics of its selected interface.

The formal Builder currently accepts only algorithms whose declared operations
are exactly `("fit",)`. An algorithm that also exposes `evaluate` or `predict`
must construct an `AlgorithmRegistration` directly until those operations have
formal distributed runtime contracts.

## MapReduce example shape

```python
from tributo.algorithms.spi import MapReduceAlgorithm


class DistributedCounter(MapReduceAlgorithm):
    def map_partition(
        self, batches, context
    ): ...  # Return bounded NumPy state matching state_schema().

    def merge_states(
        self, left, right
    ): ...  # Must be associative; retries require side-effect freedom.

    def finalize_model(self, state): ...  # Build one consolidated model.

    def state_schema(self): ...

    def empty_partition(self): ...

    @property
    def retry_safe(self):
        return True
```

The `multinomial_nb` built-in is a Bundle-producing reference implementation.
The independent package under `tests/fixtures/distributed_algorithm_plugin`
directly implements the same public interface without importing a Tributo
builtin. Every Ray map
task reads one exclusive shard, returns only bounded sufficient statistics,
and the runtime constructs a balanced reduction tree. The finalizer always
runs, including for fit-only execution. A sequential `partial_fit` loop or a
Driver-side list of all partial models does not satisfy this contract.

The input runtime adapter must also supply one immutable expected total row
count on every map payload. The built-in ingestion adapter obtains it from the
bounded Ray Dataset before `streaming_split`. The finalizer sums the rows
actually consumed by all map workers and refuses publication unless that sum
equals the Driver count. Thus `input_complete=true` means that every planned
shard was exhausted and the observed total matches the bounded input; it is
runtime coverage evidence, not a sandbox against deliberately malicious
trusted package code.

The v1 Ray batch adapter is single-pass, so `MapReducePolicy.max_retries` must
be `0`. A future retryable adapter must first add a replayable shard identity
and prove that a retried map consumes exactly the same rows; declaring
`retry_safe=True` on the algorithm alone is not sufficient.

## Build the descriptor

The following abbreviated call shows the public Builder path. The strategy
policy remains explicit because its mathematical guarantees cannot be inferred
from the algorithm class.

```python
from tributo.algorithms import AlgorithmBuilder
from tributo.algorithms.api import (
    DistributionStrategy,
    EnvironmentSpec,
    ExecutionProfile,
    MapReducePolicy,
    ResultPolicy,
    StateField,
    WorkerRange,
    WorkerResources,
)


DESCRIPTOR = AlgorithmBuilder.from_distributed_algorithm(
    spec=ALGORITHM_SPEC,
    implementation_id="example.counter.map_reduce",
    implementation_version="1.0.0",
    implementation="example_algorithms.counter:DistributedCounter",
    executable_factory="example_algorithms.counter:create_algorithm",
    distribution="example-algorithms",
    framework=None,
    environment=EnvironmentSpec(
        environment_id="example.counter.v1",
        dependencies=("example-algorithms==1.0.0",),
    ),
    allowed_config_keys=(),
    strategy=DistributionStrategy.RAY_MAP_REDUCE,
    supported_worker_range=WorkerRange(1, 32),
    supported_execution_profiles=(ExecutionProfile.LOCAL,),
    resources_per_worker=WorkerResources(num_cpus=1),
    policy=MapReducePolicy(
        state_schema=(StateField("count", "int64", ()),),
        max_partial_state_bytes=4096,
        reducer_ref="example_algorithms.counter:DistributedCounter.merge_states",
        finalizer_ref="example_algorithms.counter:DistributedCounter.finalize_model",
    ),
    package_name="example-algorithms",
    package_version="1.0.0",
    tributo_version_spec=">=1,<2",
    result_policy=ResultPolicy.FIT_ONLY,
    tested=True,
)
```

An explicitly supplied `BackendInputCompatibility` may broaden compatible input
views, but it must retain the strategy topology and standard Worker adapter.
Conflicts fail while the descriptor is being built, before a Ray Job is
submitted.

Choose `FIT_ONLY` when the completed in-memory model is intentionally not a
published product. The Runtime skips exporter, flavor, artifact, and Bundle
publication, while retaining input, Worker, shard, state-coordination, and
finalization evidence. Algorithm finalization/merge semantics still run in
`FIT_ONLY`; only the publication chain is skipped. Collective and
framework-native Runtime implementations expose portable user metrics after
removing runtime evidence fields, and fail closed if a user metric is not
portable. The current
MapReduce SPI has no metrics channel, so its `FIT_ONLY` result contains no
metrics even though `finalize_model` still runs. Such a result cannot be loaded
by Serving or registered as a model. Choose `BUNDLE_REQUIRED` for any model
that must be persisted or consumed after training; both `exporter` and
`flavor_id` are then mandatory and Bundle publication remains fail-closed.

## Descriptor entry point

Expose the descriptor from the installed package:

```toml
[project.entry-points."tributo.algorithms"]
distributed_counter = "example_algorithms.descriptors:DISTRIBUTED_COUNTER"
```

Discovery validates the entry-point identity, descriptor API version,
publishing package metadata, Tributo compatibility, DistributionSpec/
implementation-mode agreement, Runtime and input topology, result policy, and
inheritance from the declared strategy interface. Invalid descriptors are
diagnosed and excluded; the Registry is updated atomically. A package test can
run the exact same conformance rules before job submission:

```python
from tributo.plugin import validate_distributed_algorithm_descriptor


validate_distributed_algorithm_descriptor(
    DESCRIPTOR,
    entry_point_name="distributed_counter",
)
```

These checks are fail-closed: the strategy, execution mode, Runtime ID,
topology, and Worker input adapter must remain aligned, so a previously
accepted descriptor with conflicting fields can be rejected after an upgrade.

The package must be installed when this check runs because conformance compares
the descriptor with installed package metadata. See
`tests/fixtures/distributed_algorithm_plugin` for a complete independent
package fixture.

For `BUNDLE_REQUIRED`, formal fit exporters receive the Dispatcher-generated
`run_id` as a keyword argument. A Bundle-producing exporter must bind that value
as both the Bundle request and run identity; it must not substitute the
enclosing Ray Job ID.

## Execution evidence

A successful multi-worker run is classified as distributed only when its
receipt proves all of the following:

- at least two unique Ray workers and ranks;
- unique shard identities and complete input coverage;
- one synchronized, framework-native, or associatively reduced model state;
- no Driver materialization of training rows;
- requested resource satisfaction;
- a formal Bundle artifact only when `result_policy=bundle_required`.

The Receipt records `result_policy`. A `fit_only` Receipt may prove true
distributed training with no artifact IDs; it does not claim that a reusable
model was published.

The same algorithm can run with one worker under the local profile. That is a
supported single-machine run, not a distributed-training claim. Tributo's
distributed Gate submits the independently packaged fixture through Ray Jobs
to an isolated Docker cluster and requires two different Ray node IDs. This
validates the algorithm semantics reused by Kubernetes; deployment-profile
compatibility and direct environment evidence remain separate Catalog facts.

## sklearn boundary

Tributo does not transform arbitrary sklearn `fit()` implementations. An
sklearn-backed algorithm is eligible only when an adapter proves bounded
mergeable state, or when a separate distributed backend implements and tests
the model's state semantics. `n_jobs`, Joblib, OpenMP, BLAS threads, and Tune
trial parallelism are not by themselves distributed single-model training.
