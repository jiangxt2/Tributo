# Custom distributed algorithms

Tributo accepts trusted Python packages through a narrow descriptor SPI. A
package can be pre-installed in the runtime image or supplied as a reviewed,
locked wheel through Ray Job `py_modules`. Tributo does not resolve plugin
dependencies online, isolate arbitrary code, hot reload packages, or maintain
multiple plugin versions.

A wheel supplied through `py_modules` must be code-only: its Core Metadata must
not contain `Requires-Dist` entries. Ray installs such wheels with pip into a
job-local target directory, so declared dependencies would invoke dependency
resolution during job startup. Tributo, sklearn, and every other runtime
dependency must already exist in the reviewed Ray image. A package installed
while building that image can use normal locked build-time dependencies.

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
