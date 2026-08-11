"""Static contract tying Docker IT images to uv.lock package versions."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from tests.support.it_versions import load_it_component_versions

ROOT = Path(__file__).parents[2]


def _locked_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return {package["name"]: package["version"] for package in lock["package"]}


def test_component_versions_match_lock_and_are_digest_pinned() -> None:
    versions = load_it_component_versions()
    locked = _locked_versions()
    package_keys = {
        "boto3": "BOTO3_VERSION",
        "botocore": "BOTOCORE_VERSION",
        "ray": "RAY_VERSION",
        "mlflow": "MLFLOW_VERSION",
        "xgboost": "XGBOOST_VERSION",
        "onnx": "ONNX_VERSION",
        "onnxruntime": "ONNXRUNTIME_VERSION",
        "onnxmltools": "ONNXMLTOOLS_VERSION",
        "torch": "TORCH_VERSION",
        "transformers": "TRANSFORMERS_VERSION",
        "pyarrow": "PYARROW_VERSION",
        "pandas": "PANDAS_VERSION",
    }
    for package, key in package_keys.items():
        assert versions[key] == locked[package]

    assert re.fullmatch(r"3\.(12|13)", versions["PYTHON_VERSION"])
    python_tag = versions["PYTHON_VERSION"].replace(".", "")
    assert f":{versions['RAY_VERSION']}-py{python_tag}@sha256:" in versions["RAY_IMAGE"]
    assert f":{versions['UV_VERSION']}@sha256:" in versions["UV_IMAGE"]
    assert f":{versions['MINIO_RELEASE']}@sha256:" in versions["MINIO_IMAGE"]
    assert f":{versions['POSTGRES_VERSION']}@sha256:" in versions["POSTGRES_IMAGE"]
    for key in (
        "RAY_IMAGE",
        "UV_IMAGE",
        "TOOL_IMAGE",
        "MINIO_IMAGE",
        "POSTGRES_IMAGE",
    ):
        assert re.search(r"@sha256:[0-9a-f]{64}$", versions[key])


def test_dockerfiles_and_compose_consume_the_version_contract() -> None:
    dockerfile = (ROOT / "tests/integrations/Dockerfile.data-ingestion").read_text()
    compose = (
        ROOT / "tests/integrations/docker-compose.data-ingestion.yml"
    ).read_text()
    profile = json.loads(
        (ROOT / "tests/integrations/runtime-profiles.json").read_text()
    )["profiles"]["data-ingestion"]
    versions = load_it_component_versions()

    assert profile["base_image"] == versions["RAY_IMAGE"]
    assert profile["uv_image"] == versions["UV_IMAGE"]
    assert profile["tool_image"] == versions["TOOL_IMAGE"]
    assert profile["minio_image"] == versions["MINIO_IMAGE"]
    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "ARG UV_IMAGE" in dockerfile
    assert "FROM ${UV_IMAGE} AS uv" in dockerfile
    assert "COPY --from=uv" in dockerfile
    assert "image: ${TRIBUTO_IT_RUNTIME_IMAGE:" in compose
    assert "image: ${TRIBUTO_IT_MINIO_IMAGE:" in compose
    assert "build:" not in compose
    assert "MLFLOW_VERSION" in compose
    assert "mlflow-init:" in compose
    assert "mlflow-data:/mlflow" in compose
    assert "--artifacts-destination" in compose
    assert "--default-artifact-root" not in compose
