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
    assert re.fullmatch(r"\d+\.\d+\.\d+", versions["UV_VERSION"])
    assert f":{versions['HIVE_VERSION']}@sha256:" in versions["HIVE_IMAGE"]
    assert f":{versions['MINIO_RELEASE']}@sha256:" in versions["MINIO_IMAGE"]
    assert f":{versions['POSTGRES_VERSION']}@sha256:" in versions["POSTGRES_IMAGE"]
    for key in (
        "RAY_IMAGE",
        "HIVE_IMAGE",
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
    assert profile["dependency_mode"] == "host-uv-export"
    assert profile["uv_version"] == versions["UV_VERSION"]
    assert profile["hive_image"] == versions["HIVE_IMAGE"]
    assert profile["minio_image"] == versions["MINIO_IMAGE"]
    assert profile["postgres_image"] == versions["POSTGRES_IMAGE"]
    assert "postgresql" in profile["extras"]
    assert "uv_image" not in profile
    assert "tool_image" not in profile
    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "ARG UV_IMAGE" not in dockerfile
    assert "FROM ${UV_IMAGE}" not in dockerfile
    assert "COPY --from=uv" not in dockerfile
    assert "COPY --from=locked-requirements" in dockerfile
    assert "python -m pip install --require-hashes" in dockerfile
    assert "image: ${TRIBUTO_IT_RUNTIME_IMAGE:" in compose
    assert "image: ${TRIBUTO_IT_MINIO_IMAGE:" in compose
    assert compose.count("image: ${TRIBUTO_IT_RUNTIME_IMAGE:") >= 3
    assert "TRIBUTO_IT_TOOL_IMAGE" not in compose
    assert "python:3.12.13-alpine3.22" not in compose
    assert "hiveserver2:" in compose
    assert "hive-init:" in compose
    assert "setup_hive_fixture.py" in compose
    assert "/opt/hive/bin/beeline" not in compose
    assert "JAVA_TOOL_OPTIONS" not in compose
    assert "build:" not in compose
    assert "MLFLOW_VERSION" in compose
    assert "mlflow-init:" in compose
    assert "mlflow-data:/mlflow" in compose
    assert "--artifacts-destination" in compose
    assert "--default-artifact-root" not in compose


def test_full_runtime_profile_matches_the_image_builder_contract() -> None:
    profiles = json.loads(
        (ROOT / "tests/integrations/runtime-profiles.json").read_text()
    )["profiles"]
    profile = profiles["runtime-full"]
    dockerfile = (ROOT / profile["dockerfile"]).read_text()
    config = json.loads((ROOT / "tools/tributo-runtime-full.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    locked = _locked_versions()

    assert profile["base_image"] == config["base_image"]
    assert profile["base_image_mirror"] == config["base_image_mirror"]
    assert "uv_image" not in profile
    assert "uv_image_mirror" not in profile
    assert "uv_image" not in config
    assert "uv_image_mirror" not in config
    assert profile["dependency_mode"] == config["dependency_mode"] == "host-uv-export"
    assert profile["uv_version"] == config["uv_version"]
    assert (
        profile["base_image"].rsplit("@", 1)[1]
        == profile["base_image_mirror"].rsplit("@", 1)[1]
    )
    assert profile["extras"] == config["runtime_extras"]
    assert profile["python_version"] == "3.12"
    assert profile["version_contract"]["ray"] == "2.55.1"
    assert project["project"]["optional-dependencies"]["hive-ray"] == ["ray-hive==1.0"]
    for package, contract_key in {
        "daft-clickhouse": "daft_clickhouse",
        "daft-doris": "daft_doris",
        "ray-doris": "ray_doris",
        "ray-hive": "ray_hive",
        "thrift": "thrift",
    }.items():
        assert profile["version_contract"][contract_key] == locked[package]
    assert "TRIBUTO_BASE_IMAGE" in dockerfile
    assert "python -m venv /opt/tributo/.venv" in dockerfile
    assert "ARG UV_IMAGE" not in dockerfile
    assert "FROM ${UV_IMAGE}" not in dockerfile
    assert "uv sync" not in dockerfile
    assert "COPY --from=locked-requirements" in dockerfile
    assert "COPY --from=project-wheelhouse" in dockerfile
    assert "COPY --from=external-wheelhouse" in dockerfile
    assert "python -m pip install --require-hashes" in dockerfile

    runner = (ROOT / "scripts/run_runtime_image_it.sh").read_text()
    assert 'case "$runtime_platform" in' in runner
    assert 'export TRIBUTO_RUNTIME_PLATFORM="$runtime_platform"' in runner
    assert "-u HTTP_PROXY" in runner
    assert "-u http_proxy" in runner
    assert "--wait-timeout 180" in runner
