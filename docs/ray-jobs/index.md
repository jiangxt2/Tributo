# Ray Jobs and clusters

Tributo submits and manages workloads through the Ray Jobs API. The client and
CLI can submit jobs, inspect status, stream logs, and stop running jobs.

Tributo expects an existing reachable Ray cluster and dashboard endpoint. It
does not provision or administer the underlying cluster.

Use the [Python API](../api.md) for `TributoClient` and `JobConfig`, or the
[CLI reference](../cli.md) for job-management commands.
