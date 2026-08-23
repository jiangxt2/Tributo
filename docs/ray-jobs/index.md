# Ray Jobs and clusters

Tributo uses Ray's native runtime and submission APIs. The client and CLI can
run an owned local workload, submit jobs through the Ray Jobs API, inspect
status, stream logs, and stop running jobs.

Use `--master local --wait` for an owned local Ray runtime. Use an HTTP(S) Ray
dashboard/Jobs endpoint for an attached cluster. A `managed://` target may use
an explicitly configured Ray Cluster Launcher provider; owned managed jobs
wait for completion before releasing the provider-owned cluster. Kubernetes
and KubeRay remain external deployment providers.

Ray Jobs is the primary path for training, inference, Tune, Explainability and
detached workloads. Ray Client (`ray://`) is an interactive compatibility path,
not a replacement for long-running Ray Train or Ray Tune submissions.

Use the [Python API](../api.md) for `TributoClient` and `JobConfig`, or the
[CLI reference](../cli.md) for job-management commands.
