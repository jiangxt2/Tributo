# Monitoring and troubleshooting

Use Ray job status and logs as the primary execution boundary. Preserve job,
run, attempt, bundle, request, and trace identifiers when diagnosing a
distributed workflow.

```{toctree}
:maxdepth: 1

../user-guide/troubleshooting
```

Operational errors must not expose storage credentials, database passwords,
tokens, or connection URIs containing user information.
