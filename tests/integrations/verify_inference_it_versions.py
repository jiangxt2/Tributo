"""Fail-fast runtime version and service checks for inference IT."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import urllib.request


def _expected(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required version contract {name}")
    return value


def main() -> None:
    expected_python = _expected("TRIBUTO_EXPECTED_PYTHON_PREFIX")
    actual_python = sys.version.split()[0]
    if not actual_python.startswith(expected_python):
        raise RuntimeError(
            f"Python version mismatch: expected {expected_python}*, got {actual_python}"
        )

    packages = {
        "ray": _expected("TRIBUTO_EXPECTED_RAY_VERSION"),
        "mlflow": _expected("TRIBUTO_EXPECTED_MLFLOW_VERSION"),
        "onnx": _expected("TRIBUTO_EXPECTED_ONNX_VERSION"),
        "onnxruntime": _expected("TRIBUTO_EXPECTED_ONNXRUNTIME_VERSION"),
        "pyarrow": _expected("TRIBUTO_EXPECTED_PYARROW_VERSION"),
        "xgboost": _expected("TRIBUTO_EXPECTED_XGBOOST_VERSION"),
    }
    actual = {name: importlib.metadata.version(name) for name in packages}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in packages.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"locked package version mismatch: {mismatches}")

    uv_output = subprocess.run(
        ["uv", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    expected_uv_version = _expected("TRIBUTO_EXPECTED_UV_VERSION")
    uv_version_fields = uv_output.split()
    if uv_version_fields[:2] != ["uv", expected_uv_version]:
        raise RuntimeError(
            f"uv version mismatch: expected {expected_uv_version!r}, got {uv_output!r}"
        )

    for url in (
        "http://mlflow:5000/health",
        "http://minio:9000/minio/health/live",
    ):
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"service health check failed with HTTP {response.status}"
                )

    print(
        json.dumps(
            {"python": actual_python, "uv": uv_output, **actual},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
