# CLI reference

The command reference below is generated from `tributo.cli:main`. Adding,
removing, or changing a Click command updates this page without maintaining a
second command table.

```{click} tributo.cli:main
:prog: tributo
:nested: full
```

## Configuration

Commands that accept configuration files use JSON. YAML files are rejected.
The `--source` option used by data commands accepts canonical source JSON.

## Exit behavior

Click reports malformed options and missing arguments with exit code 2.
Tributo runtime failures use a nonzero exit status and write their error
message to standard error.

```bash
uv run tributo --help
uv run tributo submit --help
uv run tributo serve grpc start --help
```

Commands that contact Ray require a reachable dashboard or serving endpoint.
Generating `--help` and this reference does not contact those services.

## Component guides

- Use [Algorithms and training](algorithms/index.md) for `algo` and `tune`.
- Use [Model lifecycle](model-lifecycle/index.md) for `export`, `export-gc`, and
  `registry`.
- Use [Inference and serving](inference/index.md) for `explain` and `serve`.
- Use [Vector indexing](vector-index/index.md) for `vector`.
- Use [Ray clusters](ray-jobs/index.md) for `submit`, `status`, `logs`, `stop`,
  and `inspect`.
