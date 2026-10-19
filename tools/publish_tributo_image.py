"""Publish a locally validated Tributo runtime image to Docker Hub."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.build_tributo_image import (
    ImageBuildError,
    ImageProfile,
    build_image,
    load_config,
    project_version,
)

ROOT = Path(__file__).resolve().parents[1]


SUPPORTED_PLATFORMS = {"linux/amd64": "amd64", "linux/arm64": "arm64"}
PROFILE_MACHINES = {"linux/amd64": "x86_64", "linux/arm64": "aarch64"}
REPOSITORY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*/[a-z0-9]+(?:[._-][a-z0-9]+)*$")
TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REGISTRY_DIGEST = re.compile(r"(?m)^Digest:\s*(sha256:[0-9a-f]{64})\s*$")


class PublishError(RuntimeError):
    """Raised when a runtime image cannot be safely published."""


@dataclass(frozen=True)
class BuildEvidence:
    """Validated local build metadata needed to publish one image."""

    image: str
    platform: str
    image_digest: str
    manifest_sha256: str
    tributo_version: str
    ray_version: str
    python_version: str
    image_profile: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"cannot read build evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublishError(f"build evidence must be a JSON object: {path}")
    return payload


def _build_evidence(output_dir: Path) -> BuildEvidence:
    manifest = _read_json(output_dir / "manifest.json")
    profile = _read_json(output_dir / "image-profile.json")
    image = manifest.get("image")
    platform = manifest.get("platform")
    image_digest = manifest.get("image_digest")
    manifest_sha256 = manifest.get("manifest_sha256")
    tributo_version = manifest.get("tributo_version")
    ray_version = manifest.get("ray_version")
    python_version = manifest.get("python_version")
    if not isinstance(image, str) or not image:
        raise PublishError("manifest.json has no local image reference")
    if platform not in SUPPORTED_PLATFORMS:
        raise PublishError(f"unsupported runtime image platform: {platform!r}")
    for field_name, value in (
        ("image_digest", image_digest),
        ("manifest_sha256", manifest_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise PublishError(f"manifest.json has an invalid {field_name}")
    if not all(
        isinstance(value, str) and value
        for value in (tributo_version, ray_version, python_version)
    ):
        raise PublishError("manifest.json is missing a runtime version")
    if profile.get("image_digest") != image_digest:
        raise PublishError("manifest and image profile digests differ")
    profile_reference = profile.get("image_uri")
    if (
        not isinstance(profile_reference, str)
        or profile_reference.split("@", 1)[0] != image
    ):
        raise PublishError("image profile does not identify the built local image")
    try:
        validated_profile = ImageProfile.model_validate(profile)
    except Exception as exc:
        raise PublishError(f"invalid image-profile.json: {exc}") from exc
    profile_payload = validated_profile.model_dump(mode="json")
    if profile_payload.get("platform_machine") != PROFILE_MACHINES[platform]:
        raise PublishError("image profile platform differs from the build manifest")
    return BuildEvidence(
        image=image,
        platform=platform,
        image_digest=image_digest,
        manifest_sha256=manifest_sha256,
        tributo_version=tributo_version,
        ray_version=ray_version,
        python_version=python_version,
        image_profile=profile_payload,
    )


def image_tag(evidence: BuildEvidence) -> str:
    """Build an architecture-specific, versioned runtime image tag."""
    python_tag = evidence.python_version.replace(".", "")
    architecture = SUPPORTED_PLATFORMS[evidence.platform]
    value = (
        f"{evidence.tributo_version}-py{python_tag}-ray{evidence.ray_version}"
        f"-cpu-{architecture}"
    )
    if TAG.fullmatch(value) is None:
        raise PublishError(f"runtime versions produce an invalid Docker tag: {value!r}")
    return value


def _docker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(name, None)
    return environment


def _run_docker(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = ["docker", *args]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=_docker_environment(),
        text=True,
    )
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise PublishError(
            f"Docker command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{details}" if details else "")
        )
    return result


def _repository_reference(repository: str, tag: str) -> str:
    if REPOSITORY.fullmatch(repository) is None:
        raise PublishError(
            "repository must be a Docker Hub namespace/repository, "
            f"for example 'tosche911/tributo-runtime': {repository!r}"
        )
    if TAG.fullmatch(tag) is None:
        raise PublishError(f"invalid Docker image tag: {tag!r}")
    return f"{repository}:{tag}"


def _remote_tag_is_missing(output: str, reference: str) -> bool:
    lowered = output.lower()
    lowered_reference = reference.lower()
    fatal_markers = (
        "failed to authorize",
        "unauthorized",
        "forbidden",
        "permission denied",
        "connection refused",
        "network is unreachable",
        "dial tcp",
        "tls handshake timeout",
    )
    if any(marker in lowered for marker in fatal_markers):
        return False
    missing_markers = (
        "not found",
        "manifest unknown",
        "manifest not found",
        "name unknown",
    )
    return lowered_reference in lowered and any(
        marker in lowered for marker in missing_markers
    )


def _local_image_id(evidence: BuildEvidence) -> str:
    result = _run_docker(
        [
            "image",
            "inspect",
            "--format",
            "{{json .}}",
            evidence.image,
        ]
    )
    try:
        inspected = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PublishError("docker image inspect returned invalid JSON") from exc
    if not isinstance(inspected, dict):
        raise PublishError("docker image inspect returned incomplete build evidence")
    image_id = inspected.get("Id")
    platform = f"{inspected.get('Os', '')}/{inspected.get('Architecture', '')}"
    config = inspected.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    sealed_manifest_sha256 = (
        labels.get("org.tributo.manifest-sha256") if isinstance(labels, dict) else None
    )
    if image_id != f"sha256:{evidence.image_digest}":
        raise PublishError("local image ID differs from image-profile.json")
    if platform != evidence.platform:
        raise PublishError(
            f"local image platform differs from build manifest: {platform!r}"
        )
    if sealed_manifest_sha256 != evidence.manifest_sha256:
        raise PublishError("local image manifest label differs from build manifest")
    return image_id


def _inspect_remote_manifest(target: str) -> tuple[str, str] | None:
    inspect = _run_docker(["buildx", "imagetools", "inspect", target], check=False)
    if inspect.returncode != 0:
        output = "\n".join((inspect.stdout, inspect.stderr)).strip()
        if _remote_tag_is_missing(output, target):
            return None
        raise PublishError(
            f"could not safely inspect Docker Hub tag: {target}"
            + (f"\n{output}" if output else "")
        )
    match = REGISTRY_DIGEST.search(inspect.stdout)
    if match is None:
        raise PublishError(f"registry did not report a digest for {target}")
    registry_digest = match.group(1)
    raw_result = _run_docker(
        ["buildx", "imagetools", "inspect", "--raw", f"{target}@{registry_digest}"]
    )
    try:
        raw_manifest = json.loads(raw_result.stdout)
    except json.JSONDecodeError as exc:
        raise PublishError("registry returned an invalid image manifest") from exc
    if not isinstance(raw_manifest, dict) or "manifests" in raw_manifest:
        raise PublishError("architecture-specific tag resolved to an image index")
    config = raw_manifest.get("config")
    config_digest = config.get("digest") if isinstance(config, dict) else None
    if (
        not isinstance(config_digest, str)
        or IMAGE_DIGEST.fullmatch(config_digest) is None
    ):
        raise PublishError("registry manifest has an invalid image config digest")
    return registry_digest, config_digest


def publish_image(
    output_dir: Path, repository: str, *, expected_version: str
) -> dict[str, Any]:
    """Push one validated platform image and write its registry-backed profile."""
    output_dir = output_dir.resolve()
    evidence = _build_evidence(output_dir)
    if evidence.tributo_version != expected_version:
        raise PublishError(
            "built image Tributo version does not match the requested release: "
            f"expected={expected_version!r}, actual={evidence.tributo_version!r}"
        )
    tag = image_tag(evidence)
    target = _repository_reference(repository, tag)
    profile_path = output_dir / "published-image-profile.json"
    registry_path = output_dir / "registry-image.json"
    if profile_path.exists() or registry_path.exists():
        raise PublishError(
            "refusing to overwrite existing registry publication evidence"
        )
    local_image_id = _local_image_id(evidence)

    remote_manifest = _inspect_remote_manifest(target)
    already_published = remote_manifest is not None
    if remote_manifest is not None and remote_manifest[1] != local_image_id:
        raise PublishError(
            f"Docker Hub tag {target} points to a different image; refusing to overwrite"
        )
    if remote_manifest is None:
        _run_docker(["tag", evidence.image, target])
        _run_docker(["push", target])
        remote_manifest = _inspect_remote_manifest(target)
        if remote_manifest is None:
            raise PublishError(f"published Docker Hub tag is not visible: {target}")
        if remote_manifest[1] != local_image_id:
            raise PublishError(
                f"Docker Hub tag {target} does not match the validated local image"
            )
    registry_digest, _ = remote_manifest
    registry_uri = f"docker.io/{target}@{registry_digest}"
    published_profile = dict(evidence.image_profile)
    published_profile["image_uri"] = registry_uri
    published_profile["image_digest"] = registry_digest.removeprefix("sha256:")
    try:
        published_profile = ImageProfile.model_validate(published_profile).model_dump(
            mode="json"
        )
    except Exception as exc:
        raise PublishError(f"published image profile is invalid: {exc}") from exc

    registry_record = {
        "schema_version": 1,
        "image_uri": registry_uri,
        "tag": target,
        "registry_digest": registry_digest,
        "source_image": evidence.image,
        "source_image_id": local_image_id,
        "manifest_sha256": evidence.manifest_sha256,
        "platform": evidence.platform,
        "already_published": already_published,
    }
    profile_path.write_text(
        json.dumps(published_profile, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(registry_record, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return registry_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "tools" / "tributo-runtime-full.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--platform", default="linux/arm64")
    parser.add_argument("--expected-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        actual_version = project_version(ROOT)
        if actual_version != args.expected_version:
            raise PublishError(
                "source checkout Tributo version does not match the requested "
                f"release: expected={args.expected_version!r}, actual={actual_version!r}"
            )
        config = load_config(
            args.config.resolve(), root=ROOT, platform_override=args.platform
        )
        build_image(
            config,
            root=ROOT,
            output_dir=args.output_dir.resolve(),
        )
        published = publish_image(
            args.output_dir,
            args.repository,
            expected_version=args.expected_version,
        )
    except (ImageBuildError, PublishError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(published, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
