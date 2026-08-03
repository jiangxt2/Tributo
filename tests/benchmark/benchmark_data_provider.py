"""D1+D2 benchmark instance: provider path vs legacy baseline (data loading).

Follows benchmark-protocol.md §Measurement Rules: at least 1 warm-up
iteration (discarded) and ≥ 3 measured iterations, reporting mean/std/min/max
wall-clock and peak driver RSS.

Baseline (legacy): the pre-D1+D2 canonical loader dispatch
(``TRIBUTO_DATA_BACKEND=legacy``).  Candidate (provider): the default
ProviderRegistry path.  Both read the same fixed Parquet dataset generated
by ``generate_data.py``.

Each path runs in its own subprocess so the ``ru_maxrss`` peak is
process-scoped (getrusage reports the process-lifetime high-water mark —
running both paths in one process would leak the legacy peak into the
provider reading).

Usage::

    uv run python tests/benchmark/benchmark_data_provider.py \\
        --data tests/benchmark/data/train.parquet --iterations 3
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _peak_driver_rss_mb() -> float:
    """Peak driver RSS (stdlib — no extra dependency for the benchmark).

    macOS reports bytes; Linux reports kilobytes.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _measure_single(
    backend: str, path: str, iterations: int, warmup: int
) -> dict[str, object]:
    """Measure one backend in THIS process (invoked via subprocess).

    ``DATA_BACKEND`` is assigned directly on the module (runtime lookup in
    ``load_ray_dataset_from_source``) — deliberately NOT via env + reload,
    which would re-read the environment and silently override the override.
    """
    from tributo.training import data_loader as dl_module

    dl_module.DATA_BACKEND = backend

    loader = dl_module.load_ray_dataset_from_source
    # warm-up (discarded). to_pandas() forces full materialization —
    # ds.count() would only read the parquet footer row-count metadata.
    for _ in range(warmup):
        loader({"type": "parquet", "path": path}).to_pandas()

    times: list[float] = []
    for _ in range(iterations):
        # Timing covers the whole call — including provider resolution and
        # routing — matching the documented "overall call wall-clock".
        start = time.perf_counter()
        ds = loader({"type": "parquet", "path": path})
        ds.to_pandas()
        times.append(time.perf_counter() - start)

    return {"times": times, "rss_mb": _peak_driver_rss_mb()}


def _run_child(
    backend: str, path: str, iterations: int, warmup: int
) -> dict[str, object]:
    """Run one backend in a fresh subprocess (clean RSS, clean Ray cluster)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--path",
            backend,
            "--data",
            path,
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark child for {backend!r} failed:\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _fmt(name: str, result: dict[str, object]) -> str:
    times = result["times"]
    assert isinstance(times, list)
    values = [float(t) for t in times]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    rss = float(result["rss_mb"])
    return (
        f"| {name} | {mean:.3f} | {stdev:.3f} | {min(values):.3f} | "
        f"{max(values):.3f} | {rss:.0f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=str, default="tests/benchmark/data/train.parquet"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--path",
        choices=["legacy", "provider"],
        default=None,
        help="Measure a single backend (subprocess mode).",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found: {args.data} — run generate_data.py first")

    if args.path is not None:
        result = _measure_single(args.path, args.data, args.iterations, args.warmup)
        print(json.dumps(result))
        return

    legacy = _run_child("legacy", args.data, args.iterations, args.warmup)
    provider = _run_child("provider", args.data, args.iterations, args.warmup)

    print("\n| Path | mean (s) | stdev (s) | min (s) | max (s) | peak RSS (MB) |")
    print("|------|----------|-----------|---------|---------|---------------|")
    print(_fmt("legacy (baseline)", legacy))
    print(_fmt("provider (D1+D2)", provider))


if __name__ == "__main__":
    main()
