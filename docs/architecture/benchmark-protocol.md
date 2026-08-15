# Benchmark protocol

This document proposes a reproducible protocol for a future performance gate.
Tributo ships a deterministic data generator and a data-provider benchmark
runner under `tests/benchmark/`, but it does not yet ship a repository-owned
benchmark dataset, stored baseline, or blocking benchmark CI job. Do not cite
this page as evidence that a performance comparison ran.

## Record the environment

| Variable | Value | Why |
|----------|-------|-----|
| Python | Exact major, minor, and patch | The package supports Python 3.12 and 3.13 |
| Ray | 2.55.1 | Pinned in `pyproject.toml` |
| OS | Exact OS, release, and architecture | Required for comparison |

| Hardware | Record CPU model, core count, RAM | Required for cross-run comparison |
| Tributo commit | Full SHA | Baseline and candidate commits must be recorded |
| Dependencies | Locked via `uv.lock` | No floating dependencies |

## Record the dataset

| Attribute | Requirement |
|-----------|------------|
| Format | Record the physical format and compression |
| Size | Record row count and encoded bytes |
| Schema | Record a stable schema fingerprint |
| Generation | Record generator revision, arguments, and deterministic seed |
| Location | Record local or remote storage and cache state without credentials |


`tests/benchmark/generate_data.py` creates deterministic inputs for the
provider benchmark in `tests/benchmark/benchmark_data_provider.py`. The
v1.0.0 source tree does not commit a canonical `tests/benchmark/data/`
dataset, so every comparison must record the generator revision and arguments
used to create its inputs.

## Apply measurement rules

### Warm-up

- Every benchmark run should include at least one warm-up iteration whose
  results are discarded.
- Warm-up ensures JIT compilation, filesystem cache, and Ray actor pools are
  in steady state.

### Repetitions

- Each benchmark should run at least three measured iterations.
- Report: mean, standard deviation, min, max for each metric.
- If standard deviation exceeds 10% of the mean, run at least five iterations.

### Metrics

| Path | Primary Metric | Secondary Metric |
|------|---------------|-----------------|
| Training | Throughput (samples/sec) | Peak worker RSS |
| Export | Wall-clock time (sec) | Export artifact size (bytes) |
| Inference/Serving | p95 latency (ms) | Throughput (requests/sec) |
| Data loading | Wall-clock time (sec) | Peak driver/worker RSS |

### RSS Measurement

- Worker RSS: `ray.util.memory.worker_memory` or `/proc/<pid>/status` VmRSS.
- Driver RSS: `psutil.Process().memory_info().rss`.
- Report peak value across the run, not time-averaged.

## Evaluate proposed stop-loss thresholds

These thresholds are review guidance until a benchmark gate is implemented and
approved. They do not automatically block a change.

| Scenario | Threshold | Metric |
|----------|-----------|--------|
| Training (no semantic change) | > 10% drop | Throughput |
| Training (no semantic change) | > 20% increase | Peak worker RSS |
| Inference/Serving | > 20% increase | p95 latency |
| Any path | Regression | Required artifact or data results |

### Calculate a threshold

- Baseline = mean of at least three runs on the baseline commit.
- Candidate = mean of at least three runs on the candidate commit.
- Change% = `(candidate - baseline) / baseline * 100`.
- A single run exceeding threshold triggers a **re-test** (not a pause).
- Two consecutive comparisons over the threshold require review; no automated
  pause exists.

## Review a regression

1. Document the regression in `decision-log.md`.
2. Identify root cause (profile, not speculate).
3. Propose fix in a separate PR (do not mix with the migration PR).
4. Re-benchmark after fix on both baseline and candidate commits.
5. Thresholds must pass **twice consecutively** before unpausing.

## Benchmark Script Template

```python
"""Benchmark: <description>

Baseline commit: <sha>
Candidate commit: <sha>
Environment: macOS 15.x arm64 / Linux x86_64
Python: 3.12.x
Ray: 2.55.1
"""

import time
import psutil
import ray
from tributo import JobConfig

DATASET_PATH = "<recorded-dataset-path>"


def measure():
    # Warm-up (discarded)
    ...

    results = []
    for i in range(3):
        start = time.perf_counter()
        rss_start = psutil.Process().memory_info().rss
        # ... run benchmarked operation ...
        elapsed = time.perf_counter() - start
        rss_peak = psutil.Process().memory_info().rss - rss_start
        results.append((elapsed, rss_peak))

    return results


if __name__ == "__main__":
    ray.init()
    results = measure()
    # Print in machine-readable format
    for i, (t, rss) in enumerate(results):
        print(f"RUN,{i},{t:.3f},{rss}")
    ray.shutdown()
```

## CI status and test-tier integration

No benchmark workflow or stored baseline implements this protocol. A future
gate must wire the existing deterministic generator and benchmark runner (or
commit an immutable dataset), add an immutable baseline record,
machine-readable results, variance handling, and an explicit workflow before
this page can describe blocking CI behavior.

Small deterministic contracts for benchmark result parsing may run in the
budgeted scheduled tier. A benchmark that uses the reference dataset, starts a
Ray cluster, requires S3/MinIO, or can exceed the scheduled suite budget is an
external validation. It is never selected by a GitHub Actions event.

An external benchmark records the exact baseline and candidate commits,
environment, dataset identity, and result log in the validation ledger. It
compares against the approved stored baseline and applies the thresholds above.
A threshold-triggered repeat follows the long-running-test rerun rules; it is
not an automatic nightly retry.

<!-- END -->
