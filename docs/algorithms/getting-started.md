# Run a formal algorithm

The [local quickstart](../getting-started/quickstart.md) provides a complete
Multinomial Naive Bayes example. Every formal request declares:

- an algorithm and operation;
- an explicit local or Kubernetes execution profile;
- worker count and optional reviewed resource overrides;
- a bounded ingestion request and tabular roles;
- algorithm-specific configuration, including an explicit Bundle destination.

Validate the JSON shape through the same CLI that executes it:

```bash
tributo algo run --config execution.json
```

Use `tributo algo list --json` to inspect registered algorithms and
`tributo algo config-schema` to inspect an algorithm's configuration schema.
Registry discovery does not install packages at runtime.
