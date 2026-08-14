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
environment requirements, input compatibility, exporter/flavor, and a matching
`DistributionSpec`. The descriptor also declares the publishing distribution,
its exact installed version, and the compatible Tributo package range.

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

The `multinomial_nb` built-in is the reference implementation. Every Ray map
task reads one exclusive shard, returns only bounded sufficient statistics,
the runtime constructs a balanced reduction tree, and the finalizer publishes
a validated Bundle. A sequential `partial_fit` loop or a Driver-side list of
all partial models does not satisfy this contract.

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

## Descriptor entry point

Expose the descriptor from the installed package:

```toml
[project.entry-points."tributo.algorithms"]
distributed_counter = "example_algorithms.descriptors:DISTRIBUTED_COUNTER"
```

Discovery validates the entry-point identity, descriptor API version,
publishing package metadata, Tributo compatibility, DistributionSpec/
implementation-mode agreement, and inheritance from the declared strategy
interface. Invalid descriptors are diagnosed and excluded; the Registry is
updated atomically. See
`tests/fixtures/distributed_algorithm_plugin` for a complete independent
package fixture.

Formal fit exporters receive the Dispatcher-generated `run_id` as a keyword
argument. A Bundle-producing exporter must bind that value as both the Bundle
request and run identity; it must not substitute the enclosing Ray Job ID.

## Execution evidence

A successful multi-worker run is classified as distributed only when its
receipt proves all of the following:

- at least two unique Ray workers and ranks;
- unique shard identities and complete input coverage;
- one synchronized, framework-native, or associatively reduced model state;
- no Driver materialization of training rows;
- requested resource satisfaction and a formal Bundle result.

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
