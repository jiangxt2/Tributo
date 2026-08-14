# Benchmark Protocol

This document defines the reproducible benchmark protocol for measuring
performance before and after architecture changes. All condition track Go/No-Go
decisions and migration stop-loss thresholds (§6.9 of the architecture roadmap)
reference this protocol.

## Fixed Environment

| Variable | Value | Why |
|----------|-------|-----|
| Python | 3.12.x (latest patch) | Matches `requires-python` lower bound |
| Ray | 2.55.1 | Pinned in `pyproject.toml` |
| OS | macOS 15.x (arm64) for dev; Linux x86_64 for external reference runs | Record which was used |
| Hardware | Record CPU model, core count, RAM | Required for cross-run comparison |
| Tributo commit | Full SHA | Baseline and candidate commits must be recorded |
| Dependencies | Locked via `uv.lock` | No floating dependencies |

## Dataset

| Attribute | Requirement |
|-----------|------------|
| Format | Parquet (benchmark dataset stored in `tests/benchmark/data/`) |
| Size | ≥ 1 GB for training benchmarks; ≥ 100 MB for export benchmarks |
| Schema | Fixed schema documented in benchmark script header |
| Generation | Script in `tests/benchmark/generate_data.py` — deterministic seed |
| Location | Local filesystem for dev; S3/MinIO for external reference runs |

## Measurement Rules

### Warm-up

- Every benchmark run MUST include at least **1 warm-up iteration** whose
  results are discarded.
- Warm-up ensures JIT compilation, filesystem cache, and Ray actor pools are
  in steady state.

### Repetitions

- Each benchmark MUST run **≥ 3 measured iterations**.
- Report: mean, standard deviation, min, max for each metric.
- If standard deviation > 10% of mean, run ≥ 5 iterations.

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

## Stop-Loss Thresholds

These thresholds are defined in §6.9 of the architecture roadmap. A migration
is paused if thresholds are exceeded on **two consecutive** benchmark runs.

| Scenario | Threshold | Metric |
|----------|-----------|--------|
| Training (no semantic change) | > 10% drop | Throughput |
| Training (no semantic change) | > 20% increase | Peak worker RSS |
| Inference/Serving | > 20% increase | p95 latency |
| Any path | Regression | Required artifact or data results |

### Threshold Calculation

- Baseline = mean of ≥ 3 runs on the **baseline commit**.
- Candidate = mean of ≥ 3 runs on the **candidate commit**.
- Change% = `(candidate - baseline) / baseline * 100`.
- A single run exceeding threshold triggers a **re-test** (not a pause).
- Two consecutive runs exceeding threshold triggers **automatic pause**.

## Recovery After Pause

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

DATASET_PATH = "tests/benchmark/data/..."


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

## Test-Tier Integration

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
