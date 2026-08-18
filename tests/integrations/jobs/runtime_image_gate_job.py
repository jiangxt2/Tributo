"""Ray Jobs API payload for the full runtime image gate."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys

import ray

from tools.runtime_image_contract import (
    REQUIRED_DISTRIBUTION_VERSIONS,
    REQUIRED_DISTRIBUTIONS,
    REQUIRED_IMPORTS,
)


@ray.remote
def worker_probe() -> dict[str, object]:
    """Verify that a Ray worker sees the same installed runtime closure."""
    for name in REQUIRED_DISTRIBUTIONS:
        importlib.metadata.version(name)
    for name, version in REQUIRED_DISTRIBUTION_VERSIONS.items():
        assert importlib.metadata.version(name) == version
    for name in REQUIRED_IMPORTS:
        importlib.import_module(name)
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "machine": platform.machine(),
        "ray": importlib.metadata.version("ray"),
        "tributo": importlib.metadata.version("tributo"),
        "daft-clickhouse": importlib.metadata.version("daft-clickhouse"),
        "daft-doris": importlib.metadata.version("daft-doris"),
        "ray-doris": importlib.metadata.version("ray-doris"),
    }


def main() -> None:
    assert sys.version_info[:2] == (3, 12)
    expected_machine = {
        "linux/amd64": "x86_64",
        "linux/arm64": "aarch64",
    }[os.environ["TRIBUTO_RUNTIME_PLATFORM"]]
    assert platform.machine() == expected_machine
    assert importlib.metadata.version("ray") == "2.55.1"
    assert importlib.metadata.version("tributo") == "1.0.0"
    for name in REQUIRED_DISTRIBUTIONS:
        importlib.metadata.version(name)
    for name, version in REQUIRED_DISTRIBUTION_VERSIONS.items():
        assert importlib.metadata.version(name) == version
    for name in REQUIRED_IMPORTS:
        importlib.import_module(name)

    ray.init(address="auto", namespace="tributo-runtime-image-gate")
    try:
        worker_result = ray.get(worker_probe.remote())
        assert worker_result == {
            "python": "3.12",
            "machine": expected_machine,
            "ray": "2.55.1",
            "tributo": "1.0.0",
            "daft-clickhouse": "1.0",
            "daft-doris": "1.0",
            "ray-doris": "1.0",
        }
        dataset = ray.data.from_items([{"value": 1}, {"value": 2}])
        assert dataset.count() == 2
        assert ray.cluster_resources().get("CPU", 0) >= 1
        print(json.dumps({"status": "passed", "worker": worker_result}, sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
