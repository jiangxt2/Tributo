# Debug logging

Tributo uses standard Python logging. Configure an SDK process explicitly:

```python
from tributo._common.logging import configure_logging

configure_logging(level="DEBUG", log_format="json")
```

Imports do not configure output. Caller-owned handlers, formats, thresholds,
filters and propagation remain under caller control. Without a caller output
path, Tributo installs one package handler; repeated configuration updates it.
JSON and text share bounded fields and omit arbitrary extra objects and raw
exception chains. Exception type names remain available.

## Configuration

| Variable | Values and default |
| --- | --- |
| `TRIBUTO_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR, CRITICAL; INFO |
| `TRIBUTO_LOG_FORMAT` | json or text; JSON in Jobs, text in CLI |
| `TRIBUTO_LOGGER_LEVELS` | JSON object of component levels; empty |

For example, set `TRIBUTO_LOGGER_LEVELS` to
`{"tributo.inference":"DEBUG","ray_doris":"DEBUG"}`.
The mapping accepts at most 32 entries and configuration at most 4096 bytes.
Root, Ray and Torch diagnostics remain caller-managed. Only explicitly selected
connector namespaces are configured. Existing connector detail switches remain
independent and can be supplied explicitly through `runtime_env.env_vars`.
This change introduces no per-batch or per-step output or sampling controls.

Explicit SDK arguments take precedence in the current process. Explicit Job
environment options take precedence over the three known submitting-process
options. Other host variables are not copied. Handlers are never transported.

## Processes and ownership

Owned Job drivers and newly owned runtimes install the worker setup hook.
Existing user hooks run afterward and retain final control. Ray Jobs and Serve
require importable string hooks; local Ray can compose a serializable callable.
Serve resource and runtime options are preserved.

Custom entrypoints initialize explicitly with `bootstrap_logging()` and pass
`driver_logging_runtime_env()` when creating their Ray connection. Both helpers
are internal startup adapters. Configure caller-owned Ray sessions at creation;
changing a driver does not alter workers that have already started.

Owned inference actors and Train wrappers receive a serializable policy before
model or algorithm initialization. A local subprocess receives its parent's
actual Ray address, preserving SDK settings and applying explicit child options
only in that child.

## Diagnostic boundaries

The small set of added messages identifies an ingestion plan as planned, a
loaded model as ready, and a training worker as started, succeeded, failed or
`cancelled`. Available run, attempt, Job, task, worker and rank identities are
included; missing values are omitted. Elapsed time uses the monotonic clock.
Lazy planning never claims rows were read. Existing write receipts, export
states, request audits and SSE behavior retain their original contracts.

Logging adds no Dataset actions, database queries, tensor reads or collectives.
DEBUG does not enable native Ray or Torch diagnostics. Ray `stdout` forwarding
and duplicate suppression remain unchanged; forwarded and native output can
retain Ray prefixes and need not match the application JSON schema.
