# Debug logging

Tributo uses Python logging. Configure the current SDK process explicitly:

```python
from tributo._common.logging import configure_logging

configure_logging(level="DEBUG", log_format="json")
```

The existing DeveloperAPI remains an explicit initialization entry point.
Importing a module does not configure output or start background work.
Caller-owned handlers, output formats, thresholds, filters and propagation remain
under caller control. With no caller output path, Tributo installs one named
package handler; repeated configuration updates that handler.

Tributo's own output uses a bounded field list. Unknown extra objects and raw
exception chains are omitted; exception type names remain available. Applications
that need their own record projection can retain their existing output handler.

## Environment options

| Variable | Values and default |
| --- | --- |
| `TRIBUTO_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR, CRITICAL; INFO |
| `TRIBUTO_LOG_FORMAT` | json or text; JSON in Jobs, text in CLI |
| `TRIBUTO_LOGGER_LEVELS` | JSON object of namespace levels; empty |
| `TRIBUTO_LOG_BATCHES` | 0 or 1; 0 |
| `TRIBUTO_LOG_STEPS` | 0 or 1; 0 |

Explicit SDK parameters take precedence in the current process. Explicit Job
environment values take precedence over these known submitting-process
variables. Other host environment variables are not copied. A Job does not
infer its policy from the submitting process's handlers.

For example:

```bash
export TRIBUTO_LOGGER_LEVELS='{"tributo.inference":"DEBUG","ray_doris":"DEBUG"}'
export TRIBUTO_LOG_BATCHES=1
```

The mapping is limited to 32 entries and configuration to 4096 bytes. Root, Ray and
Torch logging policy remain outside this mapping. Connector detail switches
remain independent. Malformed values fail validation without echoing input.

## Processes and ownership

Owned Job drivers and newly owned runtimes install the worker hook. Original
user hooks run afterward and retain final control over logging. Ray Jobs use
an importable function path; local Ray can compose a serializable callable.
Serve options are serialized as JSON, so its user setup hook must also use an
importable function path. It runs after Tributo initialization; replica
bootstrap preserves the hook's final logging settings.

Custom entrypoints are not rewritten. Initialize them explicitly before the
first business operation and pass the logging runtime environment when
creating their Ray connection:

```python
import ray
from tributo._common.logging import bootstrap_logging
from tributo._common.logging_policy import driver_logging_runtime_env

bootstrap_logging()
ray.init(address="auto", **driver_logging_runtime_env())
```

These helpers are internal startup adapters. Configure an existing external
Ray session at its creation; driver reconfiguration does not alter workers
that have already started. A local subprocess driver has its own Ray Job and
must initialize its own hook. Owned inference actors and Train wrappers also
receive an explicit serializable policy. Serve replica options preserve
existing resource and runtime settings.

An owned local subprocess receives the exact Ray address created by its parent.
Its default policy preserves explicit SDK settings. Explicit child environment
options override only the corresponding policy values in that child, without
changing the parent configuration.

## Event meaning and volume

Text and JSON include the same safe event, phase, status and available identity
fields. Missing identities are omitted. Planning a lazy dataset emits planned,
not a claim that data was read. Export blocked/partial states and write receipt
commit state retain their original meanings. A successful immutable Bundle
publication does not imply asynchronous Hook delivery succeeded.

Batch and training progress require both DEBUG visibility and their independent
detail switch. Each physical execution emits the first eligible point, then
only multiples of 100 separated by at least one second, capped at 100 progress
records. Failure and cleanup events do not use that progress budget.

SSE request logs follow actual iteration. Returning a StreamingResponse or
yielding a completion marker does not prove the client received the stream.
Cancellation, validation rejection and generation errors have distinct terminal
states; audit metrics for existing HTTP/gRPC paths are not counted twice.

Logging does not count datasets, read extra tensor values, synchronize CUDA or
add collectives. DEBUG does not enable native Ray or PyTorch diagnostics. Ray
standard output/error forwarding and duplicate suppression retain existing behavior;
forwarded output can contain Ray prefixes. Raw native output is not promised
to conform to the application JSON schema.
