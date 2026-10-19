"""Tests for safe publication of a validated Tributo runtime image."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.publish_tributo_image import (
    BuildEvidence,
    PublishError,
    _remote_tag_is_missing,
    image_tag,
    publish_image,
)


def _evidence(
    *,
    platform: str = "linux/arm64",
    local_digest: str = "a" * 64,
    tributo_version: str = "1.2.0",
) -> BuildEvidence:
    architecture = "aarch64" if platform == "linux/arm64" else "x86_64"
    return BuildEvidence(
        image="tributo-runtime-full:local",
        platform=platform,
        image_digest=local_digest,
        manifest_sha256="b" * 64,
        tributo_version=tributo_version,
        ray_version="2.59.0",
        python_version="3.12",
        image_profile={
            "profile_id": "tributo.runtime.full",
            "image_uri": f"tributo-runtime-full:local@sha256:{local_digest}",
            "image_digest": local_digest,
            "ray_version": "2.59.0",
            "python_spec": ">=3.12,<3.14",
            "python_version": "3.12",
            "sys_platform": "linux",
            "platform_machine": architecture,
            "wheel_tags": [],
            "installed_distributions": {},
            "allow_offline_pip": True,
            "pip_check_baseline": [],
            "algorithm_ids": [],
        },
    )


def _write_build_evidence(output_dir: Path, evidence: BuildEvidence) -> None:
    output_dir.mkdir(parents=True)
    manifest = {
        "image": evidence.image,
        "platform": evidence.platform,
        "image_digest": evidence.image_digest,
        "manifest_sha256": evidence.manifest_sha256,
        "tributo_version": evidence.tributo_version,
        "ray_version": evidence.ray_version,
        "python_version": evidence.python_version,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / "image-profile.json").write_text(
        json.dumps(evidence.image_profile), encoding="utf-8"
    )


def test_image_tag_uses_package_runtime_and_architecture_versions() -> None:
    assert image_tag(_evidence()) == ("1.2.0-py312-ray2.59.0-cpu-arm64")


def test_image_tag_tracks_a_later_formal_release_version() -> None:
    assert image_tag(_evidence(tributo_version="1.3.0")) == (
        "1.3.0-py312-ray2.59.0-cpu-arm64"
    )


@pytest.mark.parametrize(
    ("output", "reference", "expected"),
    [
        ("ERROR: docker.io/ns/image:tag: not found", "ns/image:tag", True),
        ("unauthorized: ns/image:tag not found", "ns/image:tag", False),
        ("ERROR: docker.io/ns/other:tag: not found", "ns/image:tag", False),
        ("connection refused", "ns/image:tag", False),
    ],
)
def test_remote_missing_check_fails_closed(
    output: str, reference: str, expected: bool
) -> None:
    assert _remote_tag_is_missing(output, reference) is expected


def test_publisher_records_registry_digest_and_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    output_dir = tmp_path / "image"
    _write_build_evidence(output_dir, evidence)
    remote_digest = "sha256:" + "c" * 64
    target = "tosche911/tributo-runtime:1.2.0-py312-ray2.59.0-cpu-arm64"
    local_inspect: dict[str, Any] = {
        "Id": f"sha256:{evidence.image_digest}",
        "Os": "linux",
        "Architecture": "arm64",
        "Config": {"Labels": {"org.tributo.manifest-sha256": evidence.manifest_sha256}},
    }
    calls: list[list[str]] = []

    def fake_run_docker(
        args: list[str] | tuple[str, ...], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = list(args)
        calls.append(command)
        if command[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                ["docker", *command], 0, json.dumps(local_inspect), ""
            )
        if command[:3] == ["buildx", "imagetools", "inspect"]:
            if "--raw" in command:
                raw = {
                    "schemaVersion": 2,
                    "config": {"digest": f"sha256:{evidence.image_digest}"},
                    "layers": [],
                }
                return subprocess.CompletedProcess(
                    ["docker", *command], 0, json.dumps(raw), ""
                )
            if command[-1] == target and len(calls) == 2:
                return subprocess.CompletedProcess(
                    ["docker", *command], 1, "", f"ERROR: {target}: not found"
                )
            return subprocess.CompletedProcess(
                ["docker", *command], 0, f"Digest: {remote_digest}\n", ""
            )
        return subprocess.CompletedProcess(["docker", *command], 0, "", "")

    monkeypatch.setattr("tools.publish_tributo_image._run_docker", fake_run_docker)

    result = publish_image(
        output_dir, "tosche911/tributo-runtime", expected_version="1.2.0"
    )

    assert result["image_uri"] == f"docker.io/{target}@{remote_digest}"
    assert result["source_image_id"] == f"sha256:{evidence.image_digest}"
    assert result["already_published"] is False
    profile = json.loads(
        (output_dir / "published-image-profile.json").read_text(encoding="utf-8")
    )
    assert profile["image_uri"] == result["image_uri"]
    assert profile["image_digest"] == "c" * 64
    assert any(command[:2] == ["tag", evidence.image] for command in calls)
    assert ["push", target] in calls


def test_publisher_refuses_existing_tag_with_a_different_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    output_dir = tmp_path / "image"
    _write_build_evidence(output_dir, evidence)
    calls: list[list[str]] = []

    def fake_run_docker(
        args: list[str] | tuple[str, ...], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = list(args)
        calls.append(command)
        if command[:2] == ["image", "inspect"]:
            local_inspect = {
                "Id": f"sha256:{evidence.image_digest}",
                "Os": "linux",
                "Architecture": "arm64",
                "Config": {
                    "Labels": {"org.tributo.manifest-sha256": evidence.manifest_sha256}
                },
            }
            return subprocess.CompletedProcess(
                ["docker", *command], 0, json.dumps(local_inspect), ""
            )
        if command[:3] == ["buildx", "imagetools", "inspect"]:
            if "--raw" in command:
                raw = {
                    "schemaVersion": 2,
                    "config": {"digest": "sha256:" + "d" * 64},
                    "layers": [],
                }
                return subprocess.CompletedProcess(
                    ["docker", *command], 0, json.dumps(raw), ""
                )
            return subprocess.CompletedProcess(
                ["docker", *command], 0, "Digest: sha256:" + "c" * 64 + "\n", ""
            )
        return subprocess.CompletedProcess(["docker", *command], 0, "", "")

    monkeypatch.setattr("tools.publish_tributo_image._run_docker", fake_run_docker)

    with pytest.raises(PublishError, match="points to a different image"):
        publish_image(output_dir, "tosche911/tributo-runtime", expected_version="1.2.0")

    assert not any(command[:1] == ["push"] for command in calls)


def test_publisher_recovers_an_identical_existing_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    output_dir = tmp_path / "image"
    _write_build_evidence(output_dir, evidence)
    remote_digest = "sha256:" + "c" * 64
    calls: list[list[str]] = []

    def fake_run_docker(
        args: list[str] | tuple[str, ...], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = list(args)
        calls.append(command)
        if command[:2] == ["image", "inspect"]:
            local_inspect = {
                "Id": f"sha256:{evidence.image_digest}",
                "Os": "linux",
                "Architecture": "arm64",
                "Config": {
                    "Labels": {"org.tributo.manifest-sha256": evidence.manifest_sha256}
                },
            }
            return subprocess.CompletedProcess(
                ["docker", *command], 0, json.dumps(local_inspect), ""
            )
        if command[:3] == ["buildx", "imagetools", "inspect"]:
            if "--raw" in command:
                raw = {
                    "schemaVersion": 2,
                    "config": {"digest": f"sha256:{evidence.image_digest}"},
                    "layers": [],
                }
                return subprocess.CompletedProcess(
                    ["docker", *command], 0, json.dumps(raw), ""
                )
            return subprocess.CompletedProcess(
                ["docker", *command], 0, f"Digest: {remote_digest}\n", ""
            )
        return subprocess.CompletedProcess(["docker", *command], 0, "", "")

    monkeypatch.setattr("tools.publish_tributo_image._run_docker", fake_run_docker)

    result = publish_image(
        output_dir, "tosche911/tributo-runtime", expected_version="1.2.0"
    )

    assert result["already_published"] is True
    assert not any(command[:1] in (["tag"], ["push"]) for command in calls)


def test_publisher_refuses_a_different_package_version_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    output_dir = tmp_path / "image"
    _write_build_evidence(output_dir, evidence)
    calls: list[list[str]] = []

    def fake_run_docker(
        args: list[str] | tuple[str, ...], *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(["docker", *args], 0, "", "")

    monkeypatch.setattr("tools.publish_tributo_image._run_docker", fake_run_docker)

    with pytest.raises(PublishError, match="does not match the requested release"):
        publish_image(output_dir, "tosche911/tributo-runtime", expected_version="1.3.0")

    assert calls == []
