"""Shared import and distribution contract for the Tributo runtime image."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_image_inputs(root: Path = ROOT) -> dict[str, str]:
    """Read shared pinned image inputs without importing the SDK."""
    payload = json.loads((root / "tools/tributo-runtime-full.json").read_text())
    inputs = {}
    for key in ("base_image", "base_image_mirror", "uv_version"):
        value = payload[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid runtime image input {key!r}")
        inputs[key] = value
    match = re.fullmatch(
        r"rayproject/ray:(?P<ray>[^-@]+)-py(?P<python>3\d{1,2})@sha256:[0-9a-f]{64}",
        inputs["base_image"],
    )
    if match is None:
        raise ValueError("Runtime base image must be a pinned Ray Python image")
    baseline = json.loads(
        (root / "src/tributo/_common/runtime_versions.json").read_text()
    )
    if match["ray"] != baseline["ray"]:
        raise ValueError("Runtime base image differs from the audited SDK Ray version")
    inputs["ray_version"] = match["ray"]
    inputs["python_version"] = f"{match['python'][0]}.{match['python'][1:]}"
    return inputs


def locked_versions(names: Sequence[str], root: Path = ROOT) -> dict[str, str]:
    """Resolve explicitly selected image checks from the committed lockfile."""
    packages = tomllib.loads((root / "uv.lock").read_text())["package"]
    result = {}
    for name in names:
        versions = {item["version"] for item in packages if item["name"] == name}
        if len(versions) != 1:
            raise ValueError(
                f"Expected one locked version for {name!r}, found {versions}"
            )
        result[name] = versions.pop()
    return result


_INPUTS = load_image_inputs()
BASE_IMAGE = _INPUTS["base_image"]
BASE_IMAGE_MIRROR = _INPUTS["base_image_mirror"]
UV_VERSION = _INPUTS["uv_version"]
RAY_VERSION = _INPUTS["ray_version"]
PYTHON_VERSION = _INPUTS["python_version"]
LOCAL_BASE_IMAGE = f"tributo-ray-base:{RAY_VERSION}-py{PYTHON_VERSION.replace('.', '')}"

REQUIRED_IMPORTS = (
    "tributo.data",
    "tributo.algorithms",
    "tributo.training",
    "tributo.exporting",
    "tributo.inference",
    "tributo.serving",
    "tributo.vector_index",
    "tributo.explainability",
    "tributo.registry",
    "daft",
    "ray_hive",
    "ray_clickhouse",
    "lance_ray",
    "lance",
    "pyiceberg",
    "torch",
    "onnx",
    "onnxruntime",
    "transformers",
    "shap",
    "mlflow",
    "grpc",
    "torch_geometric",
    "daft_doris",
    "ray_doris",
)

REQUIRED_DISTRIBUTIONS = (
    "ray",
    "ray-hive",
    "ray-clickhouse",
    "tributo",
    "daft",
    "lance-ray",
    "pylance",
    "pyiceberg",
    "torch",
    "onnx",
    "onnxruntime",
    "transformers",
    "shap",
    "mlflow",
    "grpcio",
    "torch-geometric",
    "dowhy",
    "econml",
    "bayesian-optimization",
    "daft-doris",
    "ray-doris",
    "clickhouse-connect",
    "PyMySQL",
    "thrift",
    "adbc-driver-manager",
    "adbc-driver-flightsql",
)

REQUIRED_DISTRIBUTION_VERSIONS = locked_versions(
    ("ray-hive", "ray-clickhouse", "daft-doris", "ray-doris", "thrift")
)
