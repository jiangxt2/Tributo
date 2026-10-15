"""Independent dependency and profile checks for the shared runtime inputs."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from tests.support.runtime_versions import DAFT_VERSION, RAY_VERSION, locked_version
from tools import build_algorithm_bundle, runtime_image_contract, tributo_it
from tributo._common import runtime_versions
from tributo.algorithms.api.artifacts import AlgorithmBundleManifest, ImageProfile
from tributo.data.writing import native_bindings
from tributo.kuberay_submission import KubeRayDeploymentConfig

ROOT = Path(__file__).parents[1]


def test_sdk_defaults_match_dependency_declarations_and_lock() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    ray = next(
        Requirement(item)
        for item in project["dependencies"]
        if Requirement(item).name == "ray"
    )
    assert str(ray.specifier) == f"=={runtime_versions.SUPPORTED_RAY_VERSION}"
    assert runtime_versions.SUPPORTED_RAY_VERSION == RAY_VERSION
    assert runtime_versions.SUPPORTED_DAFT_VERSION == DAFT_VERSION
    assert DAFT_VERSION in SpecifierSet(runtime_versions.DAFT_SQL_VERSION_SPEC)
    for model in (ImageProfile, AlgorithmBundleManifest, KubeRayDeploymentConfig):
        assert model.model_fields["ray_version"].default == RAY_VERSION
    assert native_bindings.RAY_ENGINE_VERSION == RAY_VERSION
    assert native_bindings.DAFT_ENGINE_VERSION == DAFT_VERSION
    assert build_algorithm_bundle.RAY_VERSION == RAY_VERSION


def test_ray_jobs_uploads_the_independent_locked_test_baseline(tmp_path: Path) -> None:
    from ray._private.runtime_env.packaging import get_uri_for_directory

    (tmp_path / ".rayignore").write_bytes((ROOT / ".rayignore").read_bytes())
    lock = tmp_path / "uv.lock"
    lock.write_text("first locked baseline")
    first = get_uri_for_directory(str(tmp_path), include_gitignore=True)
    lock.write_text("different locked baseline")
    assert get_uri_for_directory(str(tmp_path), include_gitignore=True) != first


@pytest.fixture
def shared_profile_root(tmp_path: Path) -> Path:
    for name in (
        "tools/tributo-runtime-full.json",
        "src/tributo/_common/runtime_versions.json",
        "uv.lock",
        "pyproject.toml",
        "tests/integrations/runtime-profiles.json",
        "tests/integrations/Dockerfile.data-ingestion",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_shared_profile_resolves_pinned_inputs_and_selected_lock_versions(
    shared_profile_root: Path,
) -> None:
    profile = tributo_it.load_profile("data-ingestion", root=shared_profile_root)
    assert profile.base_image == runtime_image_contract.BASE_IMAGE
    assert profile.version_contract["ray"] == RAY_VERSION
    assert profile.version_contract["ray_hive"] == locked_version("ray-hive")
    assert "locked_version_contract" not in profile.definition
    assert profile.definition["python_version"] == "3.12"


@pytest.mark.parametrize("field,value", [("shared_image_inputs", "yes")])
def test_shared_profile_rejects_invalid_input_selection(
    shared_profile_root: Path, field: str, value: object
) -> None:
    path = shared_profile_root / "tests/integrations/runtime-profiles.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(tributo_it.TributoITError, match="must be a boolean"):
        tributo_it.load_profile("data-ingestion", root=shared_profile_root)


def test_shared_profile_rejects_conflicting_explicit_and_locked_checks(
    shared_profile_root: Path,
) -> None:
    path = shared_profile_root / "tests/integrations/runtime-profiles.json"
    payload = json.loads(path.read_text())
    payload["profiles"]["data-ingestion"]["version_contract"]["ray"] = "0.0.0"
    path.write_text(json.dumps(payload))
    with pytest.raises(tributo_it.TributoITError, match="duplicate version contract"):
        tributo_it.load_profile("data-ingestion", root=shared_profile_root)


def test_image_input_rejects_a_ray_version_outside_the_sdk_baseline(
    shared_profile_root: Path,
) -> None:
    path = shared_profile_root / "tools/tributo-runtime-full.json"
    payload = json.loads(path.read_text())
    payload["base_image"] = payload["base_image"].replace(f":{RAY_VERSION}-", ":0.0.0-")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="audited SDK Ray version"):
        runtime_image_contract.load_image_inputs(shared_profile_root)


@pytest.mark.parametrize("versions", [[], ["1.0", "2.0"]])
def test_locked_image_check_rejects_missing_or_ambiguous_versions(
    tmp_path: Path, versions: list[str]
) -> None:
    content = "".join(
        f'[[package]]\nname = "ray"\nversion = "{version}"\n' for version in versions
    )
    (tmp_path / "uv.lock").write_text("package = []\n" if not versions else content)
    with pytest.raises(ValueError, match="Expected one locked version"):
        runtime_image_contract.locked_versions(["ray"], tmp_path)


@pytest.mark.parametrize("distribution", ["ray", "pyarrow", "pyiceberg"])
def test_generated_image_check_rejects_installed_version_drift(
    monkeypatch: pytest.MonkeyPatch, distribution: str
) -> None:
    profile = tributo_it.load_profile("data-ingestion-hdfs")
    definition = {
        **profile.definition,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "version_contract": {
            name: locked_version(name) for name in ("ray", "pyarrow", "pyiceberg")
        },
    }
    identity = tributo_it.RuntimeIdentity(
        profile=replace(profile, definition=definition),
        platform="linux/arm64",
        runtime_key="test",
        lock_sha256="a" * 64,
        contract_sha256="b" * 64,
        local_tag="test:1",
    )
    monkeypatch.setattr(importlib.metadata, "version", locked_version)
    code = tributo_it._version_check_code(identity)
    exec(code, {})
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.0.0" if name == distribution else locked_version(name),
    )
    with pytest.raises(AssertionError):
        exec(code, {})
