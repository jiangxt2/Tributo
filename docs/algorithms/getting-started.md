# Run a formal algorithm

Tributo Core does not bundle production algorithms. Install a compatible,
independently versioned Wheel from the official
[tributo-algorithms repository](https://github.com/jiangxt2/tributo-algorithms)
or another trusted provider before executing a formal request. Registry
discovery never installs packages at runtime.

Use the Core-only [local quickstart](../getting-started/quickstart.md) to verify
the source checkout and bounded-data path before adding an algorithm package.
Every formal request declares:

- an algorithm and operation;
- an explicit owned-local or attached-cluster execution profile;
- worker count and optional reviewed resource overrides;
- a bounded ingestion request and tabular roles;
- algorithm-specific configuration, including an explicit Bundle destination.

Validate the JSON shape through the same CLI that executes it:

```bash
uv run --locked --no-sync tributo algo run --config execution.json
```

Use `uv run --locked --no-sync tributo algo list --json` to inspect registered
algorithms. Replace `ALGORITHM_NAME` with an ID from that list to inspect its
configuration schema:

```bash
uv run --locked --no-sync tributo algo config-schema ALGORITHM_NAME
```

An empty list is expected when no algorithm Wheel is installed.
