"""Unit coverage for algorithm artifact preflight and runtime wiring."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from tributo._common.algorithm_distribution import (
    algorithm_runtime_env_patch,
    prepare_algorithm_distribution,
)
from tributo._common.runtime_env import build_runtime_env
from tributo.algorithms import (
    AlgorithmArtifact,
    ArtifactDistributionMode,
    ImageProfile,
)
from tributo.exceptions import JobConfigurationError


def _python_spec() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f">={major}.{minor},<{major}.{minor + 1}"


def _profile(
    installed: dict[str, str] | None = None,
    *,
    baseline: tuple[str, ...] = (),
) -> ImageProfile:
    return ImageProfile(
        profile_id="cpu.test",
        image_uri="tributo:test",
        image_digest="a" * 64,
        python_spec=_python_spec(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        wheel_tags=("py3-none-any",),
        installed_distributions=(
            installed if installed is not None else {"pip": "24.3.1"}
        ),
        pip_check_baseline=baseline,
    )


def _write_wheel(
    directory: Path,
    package: str,
    version: str = "1.0.0",
    *,
    requires: tuple[str, ...] = (),
    plugin_name: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{package.replace('-', '_')}-{version}-py3-none-any.whl"
    path = directory / filename
    dist_info = f"{package.replace('-', '_')}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.3",
        f"Name: {package}",
        f"Version: {version}",
        *[f"Requires-Dist: {requirement}" for requirement in requires],
        "",
    ]
    wheel = "Wheel-Version: 1.0\nGenerator: tributo-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    entry_points = ""
    if plugin_name is not None:
        entry_points = (
            "[tributo.algorithms]\n"
            f"{plugin_name} = {package.replace('-', '_')}:DESCRIPTOR\n"
        )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{package.replace('-', '_')}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        archive.writestr(f"{dist_info}/WHEEL", wheel)
        if entry_points:
            archive.writestr(f"{dist_info}/entry_points.txt", entry_points)
    return path


def _bundle(
    root: Path,
    *,
    algorithm_requires: tuple[str, ...] = (),
    requirements: str | None = None,
    include_dependency: bool = True,
) -> Path:
    wheelhouse = root / "wheelhouse"
    algorithm = _write_wheel(
        wheelhouse,
        "demo-algorithm",
        requires=algorithm_requires,
        plugin_name="demo_algorithm",
    )
    dependency = None
    if include_dependency and algorithm_requires:
        dependency = _write_wheel(wheelhouse, "dep-a", requires=("dep-b==1.0.0",))
        _write_wheel(wheelhouse, "dep-b")
    requirements_path = root / "requirements.lock"
    requirements_path.parent.mkdir(parents=True, exist_ok=True)
    requirements_path.write_text(
        requirements
        or "\n".join(
            (
                "--no-index",
                "--find-links ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse",
                f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/wheelhouse/{algorithm.name}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    files = []
    for path in (
        algorithm,
        dependency,
        *(wheelhouse.glob("dep_b-*.whl")),
        requirements_path,
    ):
        if path is None:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    manifest = {
        "algorithm_id": "demo.algorithm",
        "package_name": "demo-algorithm",
        "package_version": "1.0.0",
        "wheel": f"wheelhouse/{algorithm.name}",
        "requirements": "requirements.lock",
        "wheelhouse": "wheelhouse",
        "files": files,
        "plugin_names": ["demo_algorithm"],
        "wheel_entry_points": ["demo_algorithm"],
        "python_spec": _python_spec(),
        "profile_ids": ["cpu.test"],
        "network": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _remote_manifest() -> dict[str, object]:
    algorithm_name = "demo_algorithm-1.0.0-py3-none-any.whl"
    dependency_name = "dep_a-1.0.0-py3-none-any.whl"
    files = [
        {"path": "algorithm.whl", "size": 101, "sha256": "a" * 64},
        {"path": "requirements.lock", "size": 202, "sha256": "b" * 64},
        {
            "path": f"wheelhouse/{algorithm_name}",
            "size": 101,
            "sha256": "a" * 64,
        },
    ]
    wheel_requires: dict[str, list[str]] = {
        "demo-algorithm": ["dep-a==1.0.0"],
    }
    requirements_entries = [
        "--no-index",
        "--find-links ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/wheelhouse",
        f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/wheelhouse/{algorithm_name}",
    ]
    files.append(
        {
            "path": f"wheelhouse/{dependency_name}",
            "size": 303,
            "sha256": "c" * 64,
        }
    )
    wheel_requires["dep-a"] = []
    return {
        "algorithm_id": "demo.algorithm",
        "package_name": "demo-algorithm",
        "package_version": "1.0.0",
        "wheel": "algorithm.whl",
        "requirements": "requirements.lock",
        "wheelhouse": "wheelhouse",
        "files": files,
        "plugin_names": ["demo_algorithm"],
        "python_spec": _python_spec(),
        "profile_ids": ["cpu.test"],
        "network": False,
        "requirements_entries": requirements_entries,
        "wheel_requires": wheel_requires,
        "wheel_entry_points": ["demo_algorithm"],
    }


def test_code_only_wheel_is_validated_and_wired_to_py_modules(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-algorithm", plugin_name="demo_algorithm")
    artifact = AlgorithmArtifact(
        source=str(wheel),
        package_name="demo-algorithm",
        plugin_names=("demo_algorithm",),
    )

    prepared = prepare_algorithm_distribution(
        artifact,
        _profile(),
        declared_dependencies=("demo-algorithm==1.0.0",),
    )
    patch = algorithm_runtime_env_patch(prepared)

    assert (
        prepared.receipt.artifact_sha256
        == hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    assert patch["py_modules"] == [str(wheel)]
    assert patch["env_vars"]["TRIBUTO_PLUGINS"] == "demo_algorithm"
    assert "TRIBUTO_ALGORITHM_PREFLIGHT_RECEIPT" in patch["env_vars"]
    assert prepared.receipt.tributo_version
    assert prepared.receipt.ray_version == "2.55.1"
    assert prepared.receipt.python_version == _profile().python_version


def test_py_modules_rejects_requires_dist(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-algorithm", requires=("numpy>=2",))
    artifact = AlgorithmArtifact(source=str(wheel))

    with pytest.raises(JobConfigurationError, match="no Requires-Dist"):
        prepare_algorithm_distribution(artifact, _profile())


def test_remote_code_wheel_requires_immutable_metadata() -> None:
    artifact = AlgorithmArtifact(
        source="https://artifacts.example.invalid/demo-algorithm-1.0.0.whl",
        sha256="b" * 64,
        package_name="demo-algorithm",
        package_version="1.0.0",
        plugin_names=("demo_algorithm",),
        wheel_tags=("py3-none-any",),
    )

    prepared = prepare_algorithm_distribution(artifact, _profile())

    assert prepared.receipt.artifact_uri.startswith("https://")
    assert prepared.receipt.artifact_sha256 == "b" * 64


def test_remote_code_wheel_requires_immutable_tags() -> None:
    with pytest.raises(JobConfigurationError, match="Wheel tags"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(
                source="https://artifacts.example.invalid/demo-algorithm-1.0.0.whl",
                sha256="b" * 64,
                package_name="demo-algorithm",
                package_version="1.0.0",
                plugin_names=("demo_algorithm",),
            ),
            _profile(),
        )


def test_remote_offline_bundle_uses_attested_manifest_without_network() -> None:
    artifact = AlgorithmArtifact(
        source="s3://internal-artifacts/demo-bundle-1.0.0.zip",
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
        sha256="d" * 64,
        package_name="demo-algorithm",
        package_version="1.0.0",
        plugin_names=("demo_algorithm",),
        wheel_tags=("py3-none-any",),
        manifest=_remote_manifest(),
    )

    prepared = prepare_algorithm_distribution(artifact, _profile())
    patch = algorithm_runtime_env_patch(prepared)

    assert prepared.receipt.artifact_sha256 == "d" * 64
    assert patch["working_dir"] == artifact.source
    assert patch["pip"]["packages"] == [
        "-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/requirements.lock"
    ]
    assert patch["env_vars"]["TRIBUTO_PLUGINS"] == "demo_algorithm"


def test_remote_offline_bundle_requires_complete_manifest_closure() -> None:
    with pytest.raises(ValueError, match="attested manifest"):
        AlgorithmArtifact(
            source="https://artifacts.example.invalid/demo-bundle.zip",
            mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
            sha256="d" * 64,
            package_name="demo-algorithm",
            package_version="1.0.0",
        )

    artifact = AlgorithmArtifact(
        source="https://artifacts.example.invalid/demo-bundle.zip",
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
        sha256="d" * 64,
        package_name="demo-algorithm",
        package_version="1.0.0",
        wheel_tags=("py3-none-any",),
        manifest={
            **_remote_manifest(),
            "wheel_requires": {
                "demo-algorithm": ["missing-package==1.0.0"],
                "dep-a": [],
            },
        },
    )
    with pytest.raises(JobConfigurationError, match="absent from the image Profile"):
        prepare_algorithm_distribution(
            artifact,
            _profile(),
        )


def test_remote_offline_bundle_requires_zip_archive() -> None:
    with pytest.raises(ValueError, match="ZIP Bundle archives"):
        AlgorithmArtifact(
            source="s3://internal-artifacts/demo-bundle-1.0.0.tar",
            mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
            sha256="d" * 64,
            package_name="demo-algorithm",
            package_version="1.0.0",
            manifest=_remote_manifest(),
        )


def test_offline_wheelhouse_validates_transitive_local_closure(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle", algorithm_requires=("dep-a==1.0.0",))
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    prepared = prepare_algorithm_distribution(artifact, _profile())
    patch = algorithm_runtime_env_patch(prepared)

    assert prepared.receipt.plugin_names == ("demo_algorithm",)
    assert patch["working_dir"] == str(root)
    assert patch["pip"]["packages"] == [
        "-r ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/requirements.lock"
    ]
    assert patch["pip"]["pip_check"] is True
    assert patch["pip"]["pip_install_options"] == [
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    assert "pip_check_full_environment" in prepared.receipt.checks


def test_offline_wheelhouse_requires_profile_permission(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )
    profile = _profile().model_copy(update={"allow_offline_pip": False})

    with pytest.raises(JobConfigurationError, match="disallows offline pip"):
        prepare_algorithm_distribution(artifact, profile)


def test_offline_wheelhouse_requires_image_pip(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )
    profile = _profile().model_copy(update={"installed_distributions": {}})

    with pytest.raises(JobConfigurationError, match="must declare pip"):
        prepare_algorithm_distribution(artifact, profile)


def test_build_runtime_env_orders_core_extension_and_code_wheel(
    tmp_path: Path,
) -> None:
    wheel = _write_wheel(tmp_path, "demo-algorithm")
    artifact = AlgorithmArtifact(source=str(wheel))
    extension_module = tmp_path / "execution-driver.whl"

    runtime_env = build_runtime_env(
        algorithm_artifact=artifact,
        image_profile=_profile(),
        extra_py_modules=[extension_module],
    )

    assert len(runtime_env["py_modules"]) == 3
    assert Path(runtime_env["py_modules"][0]).name == "tributo"
    assert runtime_env["py_modules"][-2:] == [str(extension_module), str(wheel)]
    working_dir = Path(runtime_env["working_dir"])
    assert (working_dir / "pyproject.toml").is_file()


def test_build_runtime_env_uses_bundle_as_working_dir_for_offline_mode(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "bundle")
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    runtime_env = build_runtime_env(
        algorithm_artifact=artifact,
        image_profile=_profile(),
    )

    assert runtime_env["working_dir"] == str(root)
    assert len(runtime_env["py_modules"]) == 1
    assert runtime_env["pip"]["pip_check"] is True


def test_offline_wheelhouse_rejects_missing_transitive_dependency(
    tmp_path: Path,
) -> None:
    root = _bundle(
        tmp_path / "bundle",
        algorithm_requires=("dep-a==1.0.0",),
        include_dependency=False,
    )
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    with pytest.raises(JobConfigurationError, match="absent from the image Profile"):
        prepare_algorithm_distribution(artifact, _profile())


@pytest.mark.parametrize(
    "requirements",
    [
        "--find-links https://public.example.invalid/wheelhouse\n",
        "--no-index\nhttps://public.example.invalid/demo.whl\n",
        "--no-index\n-r nested.txt\n",
        "--no-index\n--index-url https://public.example.invalid/simple\n",
    ],
)
def test_offline_requirements_reject_network_or_nested_references(
    tmp_path: Path,
    requirements: str,
) -> None:
    root = _bundle(tmp_path / "bundle", requirements=requirements)
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    with pytest.raises(JobConfigurationError, match="preflight failed"):
        prepare_algorithm_distribution(artifact, _profile())


def test_offline_manifest_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    (root / "requirements.lock").write_text("tampered\n", encoding="utf-8")
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    with pytest.raises(JobConfigurationError, match="digest or size mismatch"):
        prepare_algorithm_distribution(artifact, _profile())


def test_bundle_compatibility_checks_tributo_and_profile_algorithm_id(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "bundle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tributo_version_spec"] = ">=99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(JobConfigurationError, match="Tributo version"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(root), mode="offline_wheelhouse"),
            _profile(),
        )

    root = _bundle(tmp_path / "algorithm-id-bundle")
    profile = _profile().model_copy(update={"algorithm_ids": ("other.algorithm",)})
    with pytest.raises(JobConfigurationError, match="not allowed by image Profile"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(root), mode="offline_wheelhouse"),
            profile,
        )


def test_profile_wheel_tags_are_required(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-algorithm")
    with pytest.raises(
        JobConfigurationError, match="must declare supported Wheel tags"
    ):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(wheel)),
            _profile().model_copy(update={"wheel_tags": ()}),
        )


def test_dependency_markers_use_image_profile_platform(tmp_path: Path) -> None:
    root = _bundle(
        tmp_path / "bundle",
        algorithm_requires=(
            "missing-linux-dependency==1.0.0; sys_platform == 'linux'",
        ),
        include_dependency=False,
    )
    profile = _profile().model_copy(update={"sys_platform": "linux"})
    with pytest.raises(JobConfigurationError, match="missing-linux-dependency"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(root), mode="offline_wheelhouse"),
            profile,
        )


def test_local_root_and_wheelhouse_duplicate_digests_are_checked(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "bundle")
    algorithm = next((root / "wheelhouse").glob("demo_algorithm-*.whl"))
    alias = root / "algorithm.whl"
    shutil.copy2(algorithm, alias)
    alias.write_bytes(alias.read_bytes() + b"tampered")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append(
        {
            "path": "algorithm.whl",
            "size": alias.stat().st_size,
            "sha256": hashlib.sha256(alias.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(JobConfigurationError, match="root algorithm.whl"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(root), mode="offline_wheelhouse"),
            _profile(),
        )


def test_local_manifest_plugin_names_must_exist_in_wheel(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plugin_names"] = ["not_in_wheel"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(JobConfigurationError, match="absent from Wheel entry points"):
        prepare_algorithm_distribution(
            AlgorithmArtifact(source=str(root), mode="offline_wheelhouse"),
            _profile(),
        )


def test_profile_pip_check_baseline_is_recorded_as_a_warning(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")
    artifact = AlgorithmArtifact(
        source=str(root),
        mode=ArtifactDistributionMode.OFFLINE_WHEELHOUSE,
    )

    prepared = prepare_algorithm_distribution(
        artifact,
        _profile(
            baseline=("legacy-package has requirement old>=1, but you have old 0",)
        ),
    )

    assert prepared.receipt.warnings == (
        "approved image pip-check baseline: legacy-package has requirement old>=1, but you have old 0",
    )
    assert prepared.pip_check is False
    assert "pip_check_skipped_approved_baseline" in prepared.receipt.checks
    assert algorithm_runtime_env_patch(prepared)["pip"]["pip_check"] is False


def test_job_owned_identity_cannot_be_overridden(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-algorithm")
    artifact = AlgorithmArtifact(source=str(wheel))
    prepared = prepare_algorithm_distribution(artifact, _profile())

    with pytest.raises(JobConfigurationError, match="identity conflicts"):
        algorithm_runtime_env_patch(
            prepared,
            existing_env_vars={"TRIBUTO_IMAGE_DIGEST": "not-the-profile"},
        )
