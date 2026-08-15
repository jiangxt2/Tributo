#!/usr/bin/env python3
"""Prepare and run Tributo Docker integration-test infrastructure safely.

The module deliberately keeps Docker image preparation outside Compose. Runtime
images are content-addressed, source is copied once into a run-scoped volume,
and cleanup only targets the unique Compose project created for this run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform as host_platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "tests" / "integrations" / "docker-compose.data-ingestion.yml"
DEFAULT_LOCK_TIMEOUT_SECONDS = 3600.0
MANAGED_LABEL = "io.tributo.it.managed-runtime"
PROFILE_LABEL = "io.tributo.it.profile"
RUNTIME_KEY_LABEL = "io.tributo.it.runtime-key"
LOCK_SHA_LABEL = "io.tributo.it.lock-sha256"
BASE_IMAGE_LABEL = "io.tributo.it.base-image"
PLATFORM_LABEL = "io.tributo.it.platform"
CONTRACT_SHA_LABEL = "io.tributo.it.contract-sha256"
CREATED_BY_LABEL = "io.tributo.it.created-by"
SNAPSHOT_MANIFEST = ".tributo-source-manifest.json"
SNAPSHOT_DIGEST = ".tributo-source-sha256"
SNAPSHOT_READY = ".tributo-source-ready"
SOURCE_ENTRIES = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "ci",
    "scripts",
    "src",
    "tests",
)
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "logs",
}
EXCLUDED_FILE_NAMES = {".coverage", ".pypirc", "id_rsa", "id_ed25519"}
PROJECT_PREFIXES = frozenset({"tributo-ingestion", "tributo-lance-vector"})
DIGEST_REFERENCE_PATTERN = re.compile(r"^.+:[^/@]+@sha256:[0-9a-f]{64}$")


class TributoITError(RuntimeError):
    """Raised when an integration-test lifecycle contract is violated."""


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved runtime inputs for one integration-test profile."""

    name: str
    definition: dict[str, Any]
    root: Path

    @property
    def dockerfile(self) -> Path:
        return self.root / str(self.definition["dockerfile"])

    @property
    def repository(self) -> str:
        return str(self.definition["runtime_repository"])

    @property
    def base_image(self) -> str:
        return str(self.definition["base_image"])

    @property
    def uv_image(self) -> str:
        return str(self.definition["uv_image"])

    @property
    def tool_image(self) -> str:
        return str(self.definition["tool_image"])

    @property
    def minio_image(self) -> str:
        return str(self.definition["minio_image"])

    @property
    def version_contract(self) -> dict[str, str]:
        raw = self.definition["version_contract"]
        return {str(key): str(value) for key, value in raw.items()}


@dataclass(frozen=True)
class RuntimeIdentity:
    """Immutable identity and validation metadata for one runtime image."""

    profile: RuntimeProfile
    platform: str
    runtime_key: str
    lock_sha256: str
    contract_sha256: str
    local_tag: str


@dataclass(frozen=True)
class PreparedRuntime:
    """Result of a runtime prepare operation."""

    identity: RuntimeIdentity
    image_id: str
    source: str


@dataclass(frozen=True)
class ContainerSnapshot:
    """Diagnostic state for a container that predates one IT run."""

    state: str
    name: str
    compose_project: str


def _run(
    args: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(str(part) for part in args)}", flush=True)
    result = subprocess.run(
        [str(part) for part in args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise TributoITError(
            f"command failed with exit code {result.returncode}: "
            f"{shlex.join(str(part) for part in args)}"
            + (f"\n{details}" if details else "")
        )
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_parts(parts: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in parts:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_profile(
    name: str,
    *,
    root: Path = ROOT,
    profile_file: Path | None = None,
) -> RuntimeProfile:
    """Load and minimally validate a named runtime profile."""
    path = profile_file or root / "tests" / "integrations" / "runtime-profiles.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise TributoITError(f"unsupported runtime profile schema in {path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise TributoITError(f"runtime profile {name!r} is not defined in {path}")
    definition = profiles[name]
    required = {
        "base_image",
        "dockerfile",
        "extras",
        "minio_image",
        "python_version",
        "runtime_repository",
        "tool_image",
        "uv_image",
        "version_contract",
    }
    if not isinstance(definition, dict):
        raise TributoITError(f"runtime profile {name!r} must be an object")
    missing = sorted(required - definition.keys())
    if missing:
        raise TributoITError(f"runtime profile {name!r} is missing fields: {missing}")
    image_fields = [
        str(definition["base_image"]),
        str(definition["minio_image"]),
        str(definition["tool_image"]),
        str(definition["uv_image"]),
    ]
    invalid_images = [
        reference
        for reference in image_fields
        if not DIGEST_REFERENCE_PATTERN.fullmatch(reference)
    ]
    if invalid_images:
        raise TributoITError(
            "runtime profile image references must use readable tag@sha256: "
            f"{invalid_images}"
        )
    dockerfile = root / str(definition["dockerfile"])
    if not dockerfile.is_file():
        raise TributoITError(f"runtime Dockerfile is missing: {dockerfile}")
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    missing_extras = [
        str(extra)
        for extra in definition["extras"]
        if f"--extra {extra}" not in dockerfile_text
    ]
    if missing_extras:
        raise TributoITError(
            f"runtime Dockerfile does not install profile extras: {missing_extras}"
        )
    return RuntimeProfile(name=name, definition=definition, root=root)


def normalize_platform(value: str) -> str:
    """Normalize Docker platform architecture aliases."""
    try:
        operating_system, architecture = value.strip().split("/", 1)
    except ValueError as exc:
        raise TributoITError(f"invalid Docker platform {value!r}") from exc
    aliases = {"aarch64": "arm64", "x86_64": "amd64"}
    architecture = aliases.get(architecture, architecture)
    if operating_system != "linux" or architecture not in {"amd64", "arm64"}:
        raise TributoITError(f"unsupported Docker platform {value!r}")
    return f"{operating_system}/{architecture}"


def docker_platform() -> str:
    """Return the target platform of the active Docker daemon."""
    configured = os.environ.get("TRIBUTO_IT_PLATFORM")
    if configured:
        return normalize_platform(configured)
    result = _run(["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"])
    value = result.stdout.strip()
    if not value or value == "/":
        value = f"linux/{host_platform.machine()}"
    return normalize_platform(value)


def _effective_dockerignore(profile: RuntimeProfile) -> tuple[str, bytes]:
    """Return the Docker-selected ignore file identity and content."""
    dockerfile_specific = profile.dockerfile.with_name(
        f"{profile.dockerfile.name}.dockerignore"
    )
    root_ignore = profile.root / ".dockerignore"
    for candidate in (dockerfile_specific, root_ignore):
        if candidate.is_file():
            relative_path = candidate.relative_to(profile.root).as_posix()
            return relative_path, candidate.read_bytes()
    return "<none>", b""


def runtime_identity(profile: RuntimeProfile, platform: str) -> RuntimeIdentity:
    """Compute the content identity for a runtime without source-tree inputs."""
    platform = normalize_platform(platform)
    lock_content = (profile.root / "uv.lock").read_bytes()
    contract_content = _canonical_json(profile.version_contract)
    ignore_path, ignore_content = _effective_dockerignore(profile)
    runtime_definition = {
        key: profile.definition[key]
        for key in (
            "base_image",
            "extras",
            "python_version",
            "uv_image",
            "version_contract",
        )
    }
    parts = [
        ("schema", b"tributo-it-runtime-v2"),
        ("profile", _canonical_json(runtime_definition)),
        ("dockerfile", profile.dockerfile.read_bytes()),
        (f"dockerignore:{ignore_path}", ignore_content),
        ("pyproject.toml", (profile.root / "pyproject.toml").read_bytes()),
        ("uv.lock", lock_content),
        ("platform", platform.encode("utf-8")),
    ]
    runtime_key = _hash_parts(parts)[:24]
    local_tag = f"{profile.repository}:{profile.name}-{runtime_key}"
    return RuntimeIdentity(
        profile=profile,
        platform=platform,
        runtime_key=runtime_key,
        lock_sha256=hashlib.sha256(lock_content).hexdigest(),
        contract_sha256=hashlib.sha256(contract_content).hexdigest(),
        local_tag=local_tag,
    )


def _image_inspect(reference: str) -> dict[str, Any] | None:
    result = _run(["docker", "image", "inspect", reference], check=False)
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise TributoITError(f"unexpected docker image inspect result for {reference}")
    inspected = payload[0]
    if not isinstance(inspected, dict):
        raise TributoITError(f"invalid docker image inspect result for {reference}")
    return inspected


def _expected_labels(identity: RuntimeIdentity) -> dict[str, str]:
    return {
        MANAGED_LABEL: "true",
        PROFILE_LABEL: identity.profile.name,
        RUNTIME_KEY_LABEL: identity.runtime_key,
        LOCK_SHA_LABEL: identity.lock_sha256,
        BASE_IMAGE_LABEL: identity.profile.base_image,
        PLATFORM_LABEL: identity.platform,
        CONTRACT_SHA_LABEL: identity.contract_sha256,
        CREATED_BY_LABEL: "tools/tributo_it.py",
    }


def _version_check_code(identity: RuntimeIdentity) -> str:
    contract = identity.profile.version_contract
    major, minor = str(identity.profile.definition["python_version"]).split(".", 1)
    checks = [
        "import importlib.metadata as metadata,sys",
        f"assert sys.version_info[:2] == ({int(major)}, {int(minor)})",
        f"assert metadata.version('ray') == {contract['ray']!r}",
        f"assert metadata.version('daft').startswith({contract['daft_prefix']!r})",
    ]
    for key, distribution in (
        ("pylance", "pylance"),
        ("lance_ray", "lance-ray"),
        ("pyarrow", "pyarrow"),
    ):
        if key in contract:
            checks.append(
                f"assert metadata.version({distribution!r}) == {contract[key]!r}"
            )
    return "; ".join(checks)


def validate_runtime_image(
    identity: RuntimeIdentity,
    reference: str | None = None,
) -> str:
    """Fail closed unless an image exactly matches the expected runtime."""
    target = reference or identity.local_tag
    inspected = _image_inspect(target)
    if inspected is None:
        raise TributoITError(f"runtime image is missing: {target}")
    labels = inspected.get("Config", {}).get("Labels") or {}
    mismatches = {
        key: {"expected": value, "actual": labels.get(key)}
        for key, value in _expected_labels(identity).items()
        if labels.get(key) != value
    }
    actual_platform = normalize_platform(
        f"{inspected.get('Os', '')}/{inspected.get('Architecture', '')}"
    )
    if actual_platform != identity.platform:
        mismatches[PLATFORM_LABEL] = {
            "expected": identity.platform,
            "actual": actual_platform,
        }
    configured_user = str(inspected.get("Config", {}).get("User") or "")
    if configured_user in {"", "0", "0:0", "root"}:
        mismatches["Config.User"] = {
            "expected": "non-root",
            "actual": configured_user or "<empty>",
        }
    if mismatches:
        raise TributoITError(
            f"runtime image validation failed for {target}: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--entrypoint",
            "python",
            target,
            "-c",
            _version_check_code(identity),
        ]
    )
    image_id = str(inspected.get("Id") or "")
    if not image_id.startswith("sha256:"):
        raise TributoITError(f"runtime image has invalid ID: {image_id!r}")
    return image_id


def _docker_daemon_identity() -> str:
    result = _run(
        ["docker", "info", "--format", "{{.ID}}|{{.Name}}|{{.DockerRootDir}}"]
    )
    identity = result.stdout.strip()
    if not identity:
        raise TributoITError("Docker daemon did not report an identity")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _lock_directory() -> Path:
    lock_dir = Path(tempfile.gettempdir()) / f"tributo-it-locks-{os.getuid()}"
    lock_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_dir.chmod(0o700)
    current_stat = lock_dir.stat()
    if (
        current_stat.st_uid != os.getuid()
        or stat.S_IMODE(current_stat.st_mode) != 0o700
    ):
        raise TributoITError(f"unsafe runtime lock directory: {lock_dir}")
    return lock_dir


def _lock_timeout() -> float:
    raw = os.environ.get(
        "TRIBUTO_IT_RUNTIME_LOCK_TIMEOUT_SECONDS",
        str(DEFAULT_LOCK_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise TributoITError(
            "TRIBUTO_IT_RUNTIME_LOCK_TIMEOUT_SECONDS must be numeric"
        ) from exc
    if timeout <= 0:
        raise TributoITError(
            "TRIBUTO_IT_RUNTIME_LOCK_TIMEOUT_SECONDS must be greater than zero"
        )
    return timeout


@contextmanager
def runtime_lock(identity: RuntimeIdentity) -> Iterator[Path]:
    """Hold a profile/key/platform/daemon-scoped advisory file lock."""
    daemon = _docker_daemon_identity()
    lock_key = _hash_parts(
        [
            ("profile", identity.profile.name.encode("utf-8")),
            ("runtime-key", identity.runtime_key.encode("utf-8")),
            ("platform", identity.platform.encode("utf-8")),
            ("daemon", daemon.encode("utf-8")),
        ]
    )
    lock_path = _lock_directory() / f"{lock_key}.lock"
    timeout = _lock_timeout()
    deadline = time.monotonic() + timeout
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                now = time.monotonic()
                if now >= deadline:
                    lock_file.seek(0)
                    holder = lock_file.read().strip() or "<unavailable>"
                    present = _image_inspect(identity.local_tag) is not None
                    raise TributoITError(
                        "timed out waiting for runtime lock "
                        f"{lock_key} after {timeout:.1f}s; holder={holder}; "
                        f"target_present={present}"
                    ) from exc
                time.sleep(min(0.2, deadline - now))
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "profile": identity.profile.name,
                    "runtime_key": identity.runtime_key,
                    "started_monotonic": time.monotonic(),
                },
                sort_keys=True,
            )
        )
        lock_file.flush()
        os.fsync(lock_file.fileno())
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _registry_reference(registry: str, identity: RuntimeIdentity) -> str:
    return f"{registry.rstrip('/')}:{identity.profile.name}-{identity.runtime_key}"


def _registry_miss(output: str, reference: str) -> bool:
    """Return whether an authenticated registry lookup found no image.

    Buildx reports a missing GHCR tag as ``<reference>: not found`` instead of
    one of the OCI ``manifest unknown`` variants.  Match that response only
    when it names the exact requested reference so authentication, permission,
    and transport failures remain fatal.
    """
    lowered = output.lower()
    lowered_reference = reference.lower()
    fatal_markers = (
        "failed to authorize",
        "unauthorized",
        "forbidden",
        "permission_denied",
        "permission denied",
        "network is unreachable",
        "connection refused",
        "dial tcp",
    )
    if any(marker in lowered for marker in fatal_markers):
        return False
    oci_miss_markers = (
        "manifest unknown",
        "manifest not found",
        "name unknown",
        "404 not found",
    )
    for line in lowered.splitlines():
        candidate = re.sub(r"^error:\s*", "", line.strip())
        if not candidate.startswith(f"{lowered_reference}:"):
            continue
        detail = candidate[len(lowered_reference) + 1 :].strip()
        if any(marker in detail for marker in oci_miss_markers):
            return True
    return (
        re.search(
            rf"(?m)^(?:error:\s*)?{re.escape(reference)}:\s*not found\s*$",
            output,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _registry_wait_seconds() -> float:
    raw = os.environ.get("TRIBUTO_IT_RUNTIME_REGISTRY_WAIT_SECONDS", "0")
    try:
        wait_seconds = float(raw)
    except ValueError as exc:
        raise TributoITError(
            "TRIBUTO_IT_RUNTIME_REGISTRY_WAIT_SECONDS must be numeric"
        ) from exc
    if wait_seconds < 0:
        raise TributoITError(
            "TRIBUTO_IT_RUNTIME_REGISTRY_WAIT_SECONDS must not be negative"
        )
    return wait_seconds


def _pull_registry_runtime(
    reference: str, *, wait_seconds: float
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    deadline = time.monotonic() + wait_seconds
    while True:
        inspect = _run(
            ["docker", "buildx", "imagetools", "inspect", reference],
            check=False,
        )
        result = inspect
        pinned_reference: str | None = None
        if inspect.returncode == 0:
            match = re.search(
                r"^Digest:\s*(sha256:[0-9a-f]{64})\s*$",
                inspect.stdout,
                flags=re.MULTILINE,
            )
            if match is None:
                raise TributoITError(
                    f"registry did not report an immutable digest for {reference}"
                )
            resolved_digest = match.group(1)
            pinned_reference = f"{reference}@{resolved_digest}"
            existing = _image_inspect(reference)
            if existing is not None:
                repo_digests = [
                    str(value) for value in existing.get("RepoDigests") or []
                ]
                if not any(
                    value.endswith(f"@{resolved_digest}") for value in repo_digests
                ):
                    raise TributoITError(
                        "local registry tag differs from its resolved remote digest; "
                        f"refusing to overwrite {reference}"
                    )
                return (
                    subprocess.CompletedProcess(
                        ["docker", "pull", pinned_reference], 0, "already present", ""
                    ),
                    pinned_reference,
                )
            result = _run(["docker", "pull", pinned_reference], check=False)
            if result.returncode == 0:
                return result, pinned_reference
        if time.monotonic() >= deadline:
            return result, pinned_reference
        remaining = deadline - time.monotonic()
        print(
            f"Registry runtime is not ready; retrying for up to {remaining:.0f}s: "
            f"{reference}",
            flush=True,
        )
        time.sleep(min(10.0, max(0.0, remaining)))


def _build_runtime(identity: RuntimeIdentity) -> str:
    labels = _expected_labels(identity)
    descriptor, metadata_name = tempfile.mkstemp(
        prefix="tributo-buildx-", suffix=".json"
    )
    os.close(descriptor)
    metadata_file = Path(metadata_name)
    try:
        command = [
            "docker",
            "buildx",
            "build",
            "--load",
            "--platform",
            identity.platform,
            "--file",
            str(identity.profile.dockerfile),
            "--tag",
            identity.local_tag,
            "--metadata-file",
            str(metadata_file),
            "--build-arg",
            f"BASE_IMAGE={identity.profile.base_image}",
            "--build-arg",
            f"UV_IMAGE={identity.profile.uv_image}",
        ]
        for key, value in sorted(labels.items()):
            command.extend(("--label", f"{key}={value}"))
        for cache_from in filter(
            None, os.environ.get("TRIBUTO_IT_BUILDX_CACHE_FROM", "").splitlines()
        ):
            command.extend(("--cache-from", cache_from))
        for cache_to in filter(
            None, os.environ.get("TRIBUTO_IT_BUILDX_CACHE_TO", "").splitlines()
        ):
            command.extend(("--cache-to", cache_to))
        command.append(str(identity.profile.root))
        _run(command, capture_output=False)
        if metadata_file.stat().st_size == 0:
            raise TributoITError("Buildx did not produce its required metadata file")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        print(
            "Buildx metadata: "
            + json.dumps(
                {
                    key: metadata[key]
                    for key in sorted(metadata)
                    if key in {"containerimage.digest", "containerimage.config.digest"}
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return validate_runtime_image(identity)
    finally:
        metadata_file.unlink(missing_ok=True)


def prepare_runtime(
    profile: RuntimeProfile,
    *,
    platform: str,
    registry: str | None = None,
    allow_build: bool = True,
) -> PreparedRuntime:
    """Reuse, pull, or build one immutable runtime under its file lock."""
    identity = runtime_identity(profile, platform)
    existing = _image_inspect(identity.local_tag)
    if existing is not None:
        return PreparedRuntime(
            identity=identity,
            image_id=validate_runtime_image(identity),
            source="local",
        )

    with runtime_lock(identity):
        existing = _image_inspect(identity.local_tag)
        if existing is not None:
            return PreparedRuntime(
                identity=identity,
                image_id=validate_runtime_image(identity),
                source="concurrent-local",
            )

        if registry:
            remote = _registry_reference(registry, identity)
            pull, pinned_remote = _pull_registry_runtime(
                remote, wait_seconds=_registry_wait_seconds()
            )
            if pull.returncode == 0:
                if pinned_remote is None:
                    raise TributoITError(
                        f"registry runtime was pulled without a digest: {remote}"
                    )
                validate_runtime_image(identity, pinned_remote)
                _run(["docker", "tag", pinned_remote, identity.local_tag])
                return PreparedRuntime(
                    identity=identity,
                    image_id=validate_runtime_image(identity),
                    source="registry",
                )
        if not allow_build:
            pull_output = (
                "\n".join((pull.stdout, pull.stderr)).strip() if registry else ""
            )
            details = f": {pull_output}" if pull_output else ""
            raise TributoITError(
                f"runtime {identity.local_tag} is unavailable and local build is "
                f"disabled{details}"
            )
        return PreparedRuntime(
            identity=identity,
            image_id=_build_runtime(identity),
            source="build",
        )


def ensure_digest_image(reference: str) -> str:
    """Ensure an immutable third-party image is present without implicit pulls."""
    if "@sha256:" not in reference or ":" not in reference.split("@", 1)[0]:
        raise TributoITError(
            f"third-party image must use readable tag@sha256 form: {reference}"
        )
    expected_digest = reference.split("@", 1)[1]
    readable_tag = reference.split("@", 1)[0]
    tagged = _image_inspect(readable_tag)
    if tagged is not None:
        tagged_repo_digests = [str(value) for value in tagged.get("RepoDigests") or []]
        if not any(
            value.endswith(f"@{expected_digest}") for value in tagged_repo_digests
        ):
            raise TributoITError(
                "readable third-party tag differs from the pinned digest; "
                f"refusing to overwrite {readable_tag}"
            )

    inspected = _image_inspect(reference) or tagged
    if inspected is None:
        _run(["docker", "pull", reference], capture_output=False)
        inspected = _image_inspect(reference)
    if inspected is None:
        raise TributoITError(
            f"third-party image is unavailable after pull: {reference}"
        )
    repo_digests = [str(value) for value in inspected.get("RepoDigests") or []]
    if not any(value.endswith(f"@{expected_digest}") for value in repo_digests):
        raise TributoITError(
            f"third-party image digest mismatch for {reference}: {repo_digests}"
        )
    image_id = str(inspected["Id"])
    if tagged is not None and tagged.get("Id") != image_id:
        raise TributoITError(
            f"readable third-party tag points to another image: {readable_tag}"
        )
    if tagged is None:
        _run(["docker", "tag", image_id, readable_tag])
        tagged = _image_inspect(readable_tag)
    if tagged is None or tagged.get("Id") != image_id:
        raise TributoITError(f"could not bind readable third-party tag: {readable_tag}")
    return image_id


def _excluded(relative: Path, *, is_directory: bool) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
        return True
    name = relative.name
    if is_directory:
        return False
    return (
        name in EXCLUDED_FILE_NAMES
        or name.startswith(".env")
        or name.endswith((".pyc", ".pyo", ".log", ".pem", ".key"))
    )


def _validate_symlink(source_root: Path, source_path: Path) -> str:
    target = os.readlink(source_path)
    if os.path.isabs(target):
        raise TributoITError(
            f"absolute symlink is not allowed in source snapshot: {source_path}"
        )
    resolved = (source_path.parent / target).resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise TributoITError(
            f"symlink escapes source snapshot root: {source_path} -> {target}"
        ) from exc
    return target


def _copy_snapshot_entry(source_root: Path, staging: Path, relative: Path) -> None:
    source = source_root / relative
    destination = staging / relative
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(_validate_symlink(source_root, source))
        return
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
        return
    if not source.is_dir():
        raise TributoITError(f"unsupported source entry type: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copystat(source, destination, follow_symlinks=False)
    with os.scandir(source) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            child_relative = relative / entry.name
            if _excluded(
                child_relative, is_directory=entry.is_dir(follow_symlinks=False)
            ):
                continue
            _copy_snapshot_entry(source_root, staging, child_relative)


def _snapshot_manifest(staging: Path) -> tuple[list[dict[str, object]], str]:
    records: list[dict[str, object]] = []
    for current, directories, files in os.walk(staging, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            relative = path.relative_to(staging).as_posix()
            path_stat = path.lstat()
            mode = stat.S_IMODE(path_stat.st_mode)
            if path.is_symlink():
                records.append(
                    {
                        "mode": mode,
                        "path": relative,
                        "target": os.readlink(path),
                        "type": "symlink",
                    }
                )
            elif path.is_dir():
                records.append({"mode": mode, "path": relative, "type": "directory"})
            elif path.is_file():
                records.append(
                    {
                        "mode": mode,
                        "path": relative,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size": path_stat.st_size,
                        "type": "file",
                    }
                )
            else:
                raise TributoITError(f"unsupported snapshot entry type: {path}")
    manifest_bytes = _canonical_json(records) + b"\n"
    return records, hashlib.sha256(manifest_bytes).hexdigest()


def _metadata_value(value: object, *, field: str) -> str:
    if value is None:
        raise TributoITError(f"missing project metadata field {field!r}")
    rendered = str(value)
    if not rendered or "\n" in rendered or "\r" in rendered:
        raise TributoITError(f"invalid project metadata field {field!r}")
    return rendered


def _create_project_dist_info(staging: Path) -> Path:
    """Project installed-metadata projection for source-only execution."""
    pyproject = tomllib.loads((staging / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise TributoITError("pyproject.toml has no [project] metadata")
    name = _metadata_value(project.get("name"), field="project.name")
    version = _metadata_value(project.get("version"), field="project.version")
    normalized_name = re.sub(r"[-_.]+", "_", name).lower()
    if not re.fullmatch(r"[a-z0-9_]+", normalized_name):
        raise TributoITError(f"unsafe normalized project name: {normalized_name!r}")
    if not re.fullmatch(r"[a-zA-Z0-9_.+!-]+", version):
        raise TributoITError(f"unsafe project version: {version!r}")

    dist_info = staging / f"{normalized_name}-{version}.dist-info"
    dist_info.mkdir(mode=0o755)
    metadata_lines = [
        "Metadata-Version: 2.3",
        f"Name: {name}",
        f"Version: {version}",
    ]
    requires_python = project.get("requires-python")
    if requires_python is not None:
        metadata_lines.append(
            "Requires-Python: "
            + _metadata_value(requires_python, field="project.requires-python")
        )
    _atomic_write(
        dist_info / "METADATA",
        ("\n".join(metadata_lines) + "\n").encode("utf-8"),
    )
    _atomic_write(
        dist_info / "WHEEL",
        (
            "Wheel-Version: 1.0\n"
            "Generator: tributo-source-snapshot\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("ascii"),
    )

    entry_point_sections: dict[str, dict[str, object]] = {}
    scripts = project.get("scripts")
    if isinstance(scripts, dict) and scripts:
        entry_point_sections["console_scripts"] = scripts
    project_entry_points = project.get("entry-points")
    if isinstance(project_entry_points, dict):
        for group, entries in project_entry_points.items():
            if isinstance(entries, dict) and entries:
                entry_point_sections[str(group)] = entries
    entry_point_lines: list[str] = []
    for group, entries in sorted(entry_point_sections.items()):
        safe_group = _metadata_value(group, field="project.entry-points group")
        if any(character in safe_group for character in "[]"):
            raise TributoITError(f"unsafe entry-point group: {safe_group!r}")
        entry_point_lines.append(f"[{safe_group}]")
        for entry_name, target in sorted(entries.items()):
            safe_name = _metadata_value(
                entry_name, field=f"project.entry-points.{safe_group} name"
            )
            safe_target = _metadata_value(
                target, field=f"project.entry-points.{safe_group}.{safe_name}"
            )
            if "=" in safe_name:
                raise TributoITError(f"unsafe entry-point name: {safe_name!r}")
            entry_point_lines.append(f"{safe_name} = {safe_target}")
        entry_point_lines.append("")
    if entry_point_lines:
        _atomic_write(
            dist_info / "entry_points.txt",
            "\n".join(entry_point_lines).encode("utf-8"),
        )
    return dist_info


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def create_source_snapshot(
    source_root: Path,
    destination: Path,
    *,
    owner_uid: int,
    owner_gid: int,
) -> str:
    """Copy the controlled source allowlist and commit a content manifest."""
    source_root = source_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = list(destination.iterdir())
    if unexpected:
        raise TributoITError(
            f"source snapshot destination must be empty: {destination}"
        )
    staging = destination / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o755)
    try:
        for name in SOURCE_ENTRIES:
            relative = Path(name)
            source = source_root / relative
            if not source.exists() and not source.is_symlink():
                if name in {".python-version", "README.md"}:
                    continue
                raise TributoITError(
                    f"required source snapshot entry is missing: {source}"
                )
            _copy_snapshot_entry(source_root, staging, relative)
        _create_project_dist_info(staging)
        records, digest = _snapshot_manifest(staging)
        manifest_bytes = _canonical_json(records) + b"\n"
        _atomic_write(staging / SNAPSHOT_MANIFEST, manifest_bytes)
        _atomic_write(staging / SNAPSHOT_DIGEST, f"{digest}\n".encode("ascii"))
        for path in sorted(
            staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
        ):
            os.chown(path, owner_uid, owner_gid, follow_symlinks=False)
        os.chown(staging, owner_uid, owner_gid)
        metadata_names = {SNAPSHOT_MANIFEST, SNAPSHOT_DIGEST}
        content_entries = sorted(
            (entry for entry in staging.iterdir() if entry.name not in metadata_names),
            key=lambda entry: entry.name,
        )
        metadata_entries = [
            staging / name for name in (SNAPSHOT_MANIFEST, SNAPSHOT_DIGEST)
        ]
        for entry in [*content_entries, *metadata_entries]:
            entry.replace(destination / entry.name)
        _atomic_write(destination / SNAPSHOT_READY, f"{digest}\n".encode("ascii"))
        os.chown(destination / SNAPSHOT_READY, owner_uid, owner_gid)
        staging.rmdir()
        return digest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _compose_environment(
    project: str,
    runtime: PreparedRuntime,
    profile: RuntimeProfile,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "COMPOSE_PROJECT_NAME": project,
            "TRIBUTO_IT_RUNTIME_IMAGE": runtime.identity.local_tag,
            "TRIBUTO_IT_SOURCE_ROOT": str(ROOT),
            "TRIBUTO_IT_TOOL_IMAGE": profile.tool_image,
            "TRIBUTO_IT_MINIO_IMAGE": profile.minio_image,
        }
    )
    return env


def _compose_args(*args: str) -> list[str]:
    return ["docker", "compose", "--file", str(COMPOSE_FILE), *args]


def resolved_compose_config(env: dict[str, str]) -> dict[str, Any]:
    result = _run(_compose_args("config", "--format", "json"), env=env)
    config = json.loads(result.stdout)
    if not isinstance(config, dict):
        raise TributoITError("Docker Compose did not return an object config")
    return config


def _volume_mount(service: dict[str, Any], target: str) -> dict[str, Any] | None:
    for volume in service.get("volumes") or []:
        if isinstance(volume, dict) and volume.get("target") == target:
            return volume
    return None


def validate_compose_contract(
    config: dict[str, Any],
    runtime: PreparedRuntime,
    profile: RuntimeProfile,
) -> None:
    """Validate the resolved config rather than trusting YAML inheritance."""
    services = config.get("services")
    if not isinstance(services, dict):
        raise TributoITError("resolved Compose config has no services")
    for name, service in services.items():
        if "build" in service:
            raise TributoITError(f"Compose service {name} must not define build")
        if service.get("container_name"):
            raise TributoITError(f"Compose service {name} must not fix container_name")
        if service.get("pull_policy") != "never":
            raise TributoITError(f"Compose service {name} must set pull_policy: never")

    for name in ("ray-head", "ray-worker"):
        service = services[name]
        if service.get("image") != runtime.identity.local_tag:
            raise TributoITError(f"{name} does not use the prepared runtime")
        source_mount = _volume_mount(service, "/workspace/tributo-src")
        if source_mount is None or not source_mount.get("read_only"):
            raise TributoITError(f"{name} must mount the source snapshot read-only")
        work_mount = _volume_mount(service, "/workspace/tributo-work")
        if work_mount is None or work_mount.get("read_only"):
            raise TributoITError(f"{name} must mount a writable work volume")
        dependency = (service.get("depends_on") or {}).get("source-init") or {}
        if dependency.get("condition") != "service_completed_successfully":
            raise TributoITError(
                f"{name} must depend on successful source-init completion"
            )

    source_init = services["source-init"]
    workspace_init = services["workspace-init"]
    if source_init.get("image") != profile.tool_image:
        raise TributoITError("source-init must use the pinned tool image")
    if workspace_init.get("image") != profile.tool_image:
        raise TributoITError("workspace-init must use the pinned tool image")
    if source_init.get("restart") != "no":
        raise TributoITError('source-init must set restart: "no"')
    source_input = _volume_mount(source_init, "/host-source")
    if source_input is None or not source_input.get("read_only"):
        raise TributoITError("source-init checkout input must be read-only")
    if services["minio"].get("image") != profile.minio_image:
        raise TributoITError("MinIO must use its pinned readable tag@digest")
    runtime_users = {
        name
        for name, service in services.items()
        if service.get("image") == runtime.identity.local_tag
    }
    if runtime_users != {"ray-head", "ray-worker"}:
        raise TributoITError(
            f"unexpected services use the Tributo runtime: {sorted(runtime_users)}"
        )


def _image_has_repo_digests(image_id: str) -> bool:
    """Return whether an untagged-looking image is a legitimate pulled digest."""
    result = _run(
        [
            "docker",
            "image",
            "inspect",
            image_id,
            "--format",
            "{{json .RepoDigests}}",
        ]
    )
    digests = json.loads(result.stdout.strip() or "[]")
    return bool(digests)


def _diagnostic_image_ids() -> set[str]:
    """List dangling or untagged build artifacts, excluding pulled digests."""
    dangling = _run(
        [
            "docker",
            "image",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            "dangling=true",
        ]
    )
    none_rows = _run(
        [
            "docker",
            "image",
            "ls",
            "--all",
            "--no-trunc",
            "--format",
            "{{.Repository}}\t{{.Tag}}\t{{.ID}}",
        ]
    )
    image_ids = {line.strip() for line in dangling.stdout.splitlines() if line.strip()}
    for line in none_rows.stdout.splitlines():
        fields = line.split("\t")
        if (
            len(fields) == 3
            and "<none>" in fields[:2]
            and not _image_has_repo_digests(fields[2])
        ):
            image_ids.add(fields[2])
    return image_ids


def _capture_image_diagnostic_baseline() -> set[str] | None:
    """Capture best-effort image state without making it an ownership check."""
    try:
        return _diagnostic_image_ids()
    except Exception as exc:
        print(
            "Docker image diagnostic baseline unavailable; owned-project checks "
            f"remain authoritative: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _report_new_image_artifacts(project: str, before: set[str] | None) -> None:
    """Report new dangling/untagged images without attributing or deleting them."""
    if before is None:
        return
    try:
        added = sorted(_diagnostic_image_ids() - before)
    except Exception as exc:
        print(
            "Docker image diagnostic snapshot failed after scoped cleanup; "
            f"owned-project checks remain authoritative: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return
    if added:
        print(
            "New dangling or untagged Docker image artifacts detected and ignored "
            f"by the ownership-scoped audit for {project}: {added}",
            file=sys.stderr,
            flush=True,
        )


def _container_states() -> dict[str, ContainerSnapshot]:
    result = _run(
        [
            "docker",
            "ps",
            "--all",
            "--no-trunc",
            "--format",
            '{{.ID}}\t{{.State}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}',
        ]
    )
    return {
        fields[0]: ContainerSnapshot(
            state=fields[1],
            name=fields[2],
            compose_project=fields[3],
        )
        for line in result.stdout.splitlines()
        if len(fields := line.split("\t", 3)) == 4
    }


def _capture_container_diagnostic_baseline() -> dict[str, ContainerSnapshot] | None:
    """Capture best-effort global state for attribution-only diagnostics."""
    try:
        return _container_states()
    except Exception as exc:
        print(
            "Docker diagnostic baseline unavailable; owned-project checks remain "
            f"authoritative: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _report_external_container_activity(
    project: str,
    before: dict[str, ContainerSnapshot] | None,
) -> None:
    """Report concurrent Docker activity without assigning it to this IT run."""
    if before is None:
        return
    try:
        after = _container_states()
    except Exception as exc:
        print(
            "Docker diagnostic snapshot failed after scoped cleanup; "
            f"owned-project checks remain authoritative: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return
    changed = {
        container_id: {
            "name": snapshot.name,
            "compose_project": snapshot.compose_project or None,
            "before": snapshot.state,
            "after": (
                after[container_id].state if container_id in after else "<missing>"
            ),
        }
        for container_id, snapshot in before.items()
        if container_id not in after or after[container_id].state != snapshot.state
    }
    changed.update(
        {
            container_id: {
                "name": snapshot.name,
                "compose_project": snapshot.compose_project or None,
                "before": "<missing>",
                "after": snapshot.state,
            }
            for container_id, snapshot in after.items()
            if container_id not in before
        }
    )
    if changed:
        print(
            "Concurrent external Docker activity detected and ignored by the "
            f"ownership-scoped audit for {project}: "
            f"{json.dumps(changed, sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )


def _project_resource_ids(project: str) -> dict[str, list[str]]:
    label = f"com.docker.compose.project={project}"
    commands = {
        "containers": [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label={label}",
        ],
        "networks": [
            "docker",
            "network",
            "ls",
            "--quiet",
            "--filter",
            f"label={label}",
        ],
        "volumes": [
            "docker",
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"label={label}",
        ],
    }
    return {
        kind: [line for line in _run(command).stdout.splitlines() if line]
        for kind, command in commands.items()
    }


def _assert_project_absent(project: str) -> None:
    resources = _project_resource_ids(project)
    leftovers = {kind: values for kind, values in resources.items() if values}
    if leftovers:
        raise TributoITError(
            f"Compose project {project} still owns resources: "
            f"{json.dumps(leftovers, sort_keys=True)}"
        )


def _service_ids(env: dict[str, str]) -> dict[str, str]:
    identities: dict[str, str] = {}
    for service in ("ray-head", "ray-worker", "minio"):
        result = _run(_compose_args("ps", "--quiet", service), env=env)
        values = [line for line in result.stdout.splitlines() if line]
        if len(values) != 1:
            raise TributoITError(
                f"expected exactly one running {service} container, found {values}"
            )
        identities[service] = values[0]
    return identities


def _wait_for_services(env: dict[str, str], timeout_seconds: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        ray_result = _run(
            _compose_args(
                "exec",
                "-T",
                "ray-head",
                "ray",
                "status",
                "--address=127.0.0.1:6379",
            ),
            env=env,
            check=False,
        )
        minio_result = _run(
            _compose_args(
                "exec",
                "-T",
                "ray-head",
                "python",
                "-c",
                "import urllib.request; "
                "urllib.request.urlopen('http://minio:9000/minio/health/live', "
                "timeout=2).read()",
            ),
            env=env,
            check=False,
        )
        if ray_result.returncode == 0 and minio_result.returncode == 0:
            return
        last_output = "\n".join(
            part.strip()
            for part in (ray_result.stderr, minio_result.stderr)
            if part.strip()
        )
        time.sleep(2)
    raise TributoITError(
        f"Ray and MinIO were not ready within {timeout_seconds:.0f}s: {last_output}"
    )


def _snapshot_digest_in_service(env: dict[str, str], service: str) -> str:
    result = _run(
        _compose_args(
            "exec",
            "-T",
            service,
            "cat",
            f"/workspace/tributo-src/{SNAPSHOT_READY}",
        ),
        env=env,
    )
    digest = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise TributoITError(
            f"invalid source snapshot digest from {service}: {digest!r}"
        )
    return digest


def _project_version_in_service(env: dict[str, str], service: str) -> str:
    result = _run(
        _compose_args(
            "exec",
            "-T",
            service,
            "python",
            "-c",
            "import importlib.metadata as m; print(m.version('tributo'))",
        ),
        env=env,
    )
    return result.stdout.strip()


def _run_streamed(command: Sequence[str], env: dict[str, str], log_path: Path) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise TributoITError("integration-test process stdout pipe is unavailable")
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise TributoITError(
            f"integration test failed with exit code {return_code}; log={log_path}"
        )


def _collect_logs(env: dict[str, str], path: Path) -> None:
    result = _run(
        _compose_args("logs", "--no-color", "--timestamps"),
        env=env,
        check=False,
    )
    path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise TributoITError(f"failed to collect Compose logs: {path}")


def _default_project(prefix: str = "tributo-ingestion") -> str:
    if prefix not in PROJECT_PREFIXES:
        raise TributoITError(f"unsupported integration-test project prefix: {prefix}")
    return f"{prefix}-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _validate_project(project: str, prefix: str = "tributo-ingestion") -> None:
    if prefix not in PROJECT_PREFIXES:
        raise TributoITError(f"unsupported integration-test project prefix: {prefix}")
    pattern = re.compile(rf"^{re.escape(prefix)}-[a-z0-9][a-z0-9_-]*$")
    if not pattern.fullmatch(project):
        raise TributoITError(
            f"COMPOSE_PROJECT_NAME must be a unique {prefix}-* identifier"
        )


def _run_docker_ray_suite(
    profile: RuntimeProfile,
    *,
    project_prefix: str,
    test_module: str,
    display_name: str,
) -> None:
    """Run one module on the shared isolated Ray/MinIO lifecycle."""
    project = os.environ.get("COMPOSE_PROJECT_NAME") or _default_project(project_prefix)
    _validate_project(project, project_prefix)
    platform = docker_platform()
    registry = os.environ.get("TRIBUTO_IT_RUNTIME_REGISTRY")
    allow_build = os.environ.get("TRIBUTO_IT_ALLOW_LOCAL_BUILD", "1") == "1"
    log_dir = Path(os.environ.get("TRIBUTO_IT_LOG_DIR", "/tmp/tributo-it-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log = log_dir / f"{project}-test.log"
    service_log = log_dir / f"{project}-services.log"
    if any(_project_resource_ids(project).values()):
        raise TributoITError(
            f"refusing to take over existing Compose project resources: {project}"
        )
    images_before = _capture_image_diagnostic_baseline()
    containers_before = _capture_container_diagnostic_baseline()

    env: dict[str, str] | None = None
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    previous_handlers: dict[signal.Signals, Any] = {}

    def _interrupt(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, _interrupt)
    try:
        prepared = prepare_runtime(
            profile,
            platform=platform,
            registry=registry,
            allow_build=allow_build,
        )
        reused = prepare_runtime(
            profile,
            platform=platform,
            registry=registry,
            allow_build=False,
        )
        if reused.image_id != prepared.image_id or reused.source != "local":
            raise TributoITError(
                "hot runtime prepare did not reuse the same local image"
            )
        print(
            f"Runtime reuse verified: {prepared.identity.local_tag} "
            f"({prepared.image_id}, initial source={prepared.source})",
            flush=True,
        )
        ensure_digest_image(profile.tool_image)
        ensure_digest_image(profile.minio_image)
        env = _compose_environment(project, prepared, profile)
        config = resolved_compose_config(env)
        validate_compose_contract(config, prepared, profile)
        _run(
            _compose_args(
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
            ),
            env=env,
            capture_output=False,
        )
        _wait_for_services(env)
        before_ids = _service_ids(env)
        head_digest = _snapshot_digest_in_service(env, "ray-head")
        worker_digest = _snapshot_digest_in_service(env, "ray-worker")
        if head_digest != worker_digest:
            raise TributoITError(
                "source snapshot differs across Ray nodes: "
                f"{head_digest} != {worker_digest}"
            )
        print(
            f"Source snapshot verified on Ray head and worker: {head_digest}",
            flush=True,
        )
        project_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())[
            "project"
        ]
        expected_project_version = _metadata_value(
            project_metadata["version"], field="project.version"
        )
        project_versions = {
            service: _project_version_in_service(env, service)
            for service in ("ray-head", "ray-worker")
        }
        if set(project_versions.values()) != {expected_project_version}:
            raise TributoITError(
                "source snapshot project metadata differs across Ray services: "
                f"expected={expected_project_version}, actual={project_versions}"
            )
        print(
            "Source snapshot project metadata verified: "
            f"tributo=={expected_project_version}",
            flush=True,
        )
        _run_streamed(
            _compose_args(
                "exec",
                "-T",
                "--env",
                "TRIBUTO_DOCKER_RAY_TEST=1",
                "ray-head",
                "python",
                "-m",
                test_module,
            ),
            env,
            test_log,
        )
        after_ids = _service_ids(env)
        if after_ids != before_ids:
            raise TributoITError(
                "long-running containers were recreated during the suite: "
                f"before={before_ids}, after={after_ids}"
            )
    except BaseException as exc:
        primary_error = exc
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if env is not None:
            try:
                _collect_logs(env, service_log)
            except BaseException as exc:
                cleanup_errors.append(str(exc))
            try:
                _run(
                    _compose_args(
                        "down",
                        "--volumes",
                        "--remove-orphans",
                        "--timeout",
                        "30",
                    ),
                    env=env,
                    capture_output=False,
                )
            except BaseException as exc:
                cleanup_errors.append(str(exc))
        try:
            _assert_project_absent(project)
            print(f"Scoped Compose cleanup verified: {project}", flush=True)
        except BaseException as exc:
            cleanup_errors.append(str(exc))
        _report_external_container_activity(project, containers_before)
        _report_new_image_artifacts(project, images_before)

    if primary_error is not None or cleanup_errors:
        messages = []
        if primary_error is not None:
            messages.append(f"test lifecycle failed: {primary_error}")
        messages.extend(f"cleanup contract failed: {error}" for error in cleanup_errors)
        raise TributoITError("\n".join(messages)) from primary_error
    print(
        f"{display_name} Docker IT passed; logs: {test_log}, {service_log}",
        flush=True,
    )


def run_data_ingestion(profile: RuntimeProfile) -> None:
    """Run the complete Data Ingestion Docker suite with scoped cleanup."""
    _run_docker_ray_suite(
        profile,
        project_prefix="tributo-ingestion",
        test_module="tests.integrations.test_data_ingestion_dual_engine",
        display_name="Data Ingestion",
    )


def run_lance_vector_index(profile: RuntimeProfile) -> None:
    """Run distributed Lance vector indexing and search on Ray and MinIO."""
    _run_docker_ray_suite(
        profile,
        project_prefix="tributo-lance-vector",
        test_module="tests.integrations.test_lance_vector_index",
        display_name="Lance Vector Index",
    )


def publish_runtime(
    profile: RuntimeProfile,
    *,
    platform: str,
    registry: str,
) -> None:
    """Publish a missing immutable runtime after local validation."""
    identity = runtime_identity(profile, platform)
    remote = _registry_reference(registry, identity)
    pull, pinned_remote = _pull_registry_runtime(remote, wait_seconds=0)
    if pull.returncode == 0:
        if pinned_remote is None:
            raise TributoITError(
                f"registry runtime was resolved without a digest: {remote}"
            )
        validate_runtime_image(identity, pinned_remote)
        print(f"Registry runtime already exists and is valid: {pinned_remote}")
        return
    pull_output = "\n".join((pull.stdout, pull.stderr)).strip()
    if not _registry_miss(pull_output, remote):
        raise TributoITError(
            f"could not safely inspect registry runtime {remote}: {pull_output}"
        )
    prepared = prepare_runtime(
        profile,
        platform=platform,
        allow_build=True,
    )
    _run(["docker", "tag", prepared.identity.local_tag, remote])
    _run(["docker", "push", remote], capture_output=False)
    published, published_reference = _pull_registry_runtime(remote, wait_seconds=0)
    if published.returncode != 0 or published_reference is None:
        raise TributoITError(f"published runtime could not be resolved: {remote}")
    validate_runtime_image(prepared.identity, published_reference)
    print(f"Published immutable runtime: {published_reference}", flush=True)


def runtime_gc_dry_run(profile_name: str | None, platform: str) -> None:
    """Report obsolete managed runtimes without deleting any image."""
    current_keys: set[tuple[str, str]] = set()
    worktree_lines = _run(
        ["git", "worktree", "list", "--porcelain"]
    ).stdout.splitlines()
    roots = [
        Path(line.removeprefix("worktree "))
        for line in worktree_lines
        if line.startswith("worktree ")
    ]
    for root in roots:
        profile_path = root / "tests" / "integrations" / "runtime-profiles.json"
        if not profile_path.is_file():
            continue
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        names = sorted((payload.get("profiles") or {}).keys())
        for name in names:
            if profile_name is not None and name != profile_name:
                continue
            profile = load_profile(name, root=root, profile_file=profile_path)
            current_keys.add((name, runtime_identity(profile, platform).runtime_key))

    container_ids = _run(["docker", "ps", "--all", "--quiet"]).stdout.splitlines()
    referenced_images: set[str] = set()
    for container_id in container_ids:
        container_inspect = _run(["docker", "inspect", container_id])
        payload = json.loads(container_inspect.stdout)
        if payload:
            referenced_images.add(str(payload[0].get("Image") or ""))
    image_ids = sorted(
        set(
            _run(
                [
                    "docker",
                    "image",
                    "ls",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    f"label={MANAGED_LABEL}=true",
                ]
            ).stdout.splitlines()
        )
    )
    report: list[dict[str, object]] = []
    for image_id in image_ids:
        inspected = _image_inspect(image_id)
        if inspected is None:
            continue
        labels = inspected.get("Config", {}).get("Labels") or {}
        image_profile = str(labels.get(PROFILE_LABEL) or "")
        image_key = str(labels.get(RUNTIME_KEY_LABEL) or "")
        if profile_name is not None and image_profile != profile_name:
            continue
        reasons: list[str] = []
        required_labels = {
            BASE_IMAGE_LABEL,
            CONTRACT_SHA_LABEL,
            CREATED_BY_LABEL,
            LOCK_SHA_LABEL,
            PLATFORM_LABEL,
            PROFILE_LABEL,
            RUNTIME_KEY_LABEL,
        }
        if (
            any(not labels.get(label) for label in required_labels)
            or labels.get(CREATED_BY_LABEL) != "tools/tributo_it.py"
        ):
            reasons.append("invalid-managed-runtime-labels")
        if (image_profile, image_key) in current_keys:
            reasons.append("current-worktree-key")
        if image_id in referenced_images:
            reasons.append("referenced-by-container")
        report.append(
            {
                "candidate": not reasons,
                "image_id": image_id,
                "profile": image_profile,
                "reasons": reasons or ["obsolete-managed-runtime"],
                "runtime_key": image_key,
                "tags": inspected.get("RepoTags") or [],
            }
        )
    print(json.dumps({"dry_run": True, "images": report}, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    key_parser = subparsers.add_parser("runtime-key")
    key_parser.add_argument("--profile", default="data-ingestion")
    key_parser.add_argument("--platform")

    prepare_parser = subparsers.add_parser("prepare-runtime")
    prepare_parser.add_argument("--profile", default="data-ingestion")
    prepare_parser.add_argument("--platform")
    prepare_parser.add_argument("--registry")
    prepare_parser.add_argument("--no-local-build", action="store_true")

    publish_parser = subparsers.add_parser("publish-runtime")
    publish_parser.add_argument("--profile", default="data-ingestion")
    publish_parser.add_argument("--platform", required=True)
    publish_parser.add_argument("--registry", required=True)

    snapshot_parser = subparsers.add_parser("create-source-snapshot")
    snapshot_parser.add_argument("--source", type=Path, required=True)
    snapshot_parser.add_argument("--destination", type=Path, required=True)
    snapshot_parser.add_argument("--owner-uid", type=int, default=1000)
    snapshot_parser.add_argument("--owner-gid", type=int, default=100)

    run_parser = subparsers.add_parser("run-data-ingestion")
    run_parser.add_argument("--profile", default="data-ingestion")

    vector_parser = subparsers.add_parser("run-lance-vector-index")
    vector_parser.add_argument("--profile", default="data-ingestion")

    gc_parser = subparsers.add_parser("runtime-gc-dry-run")
    gc_parser.add_argument("--profile")
    gc_parser.add_argument("--platform")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create-source-snapshot":
            digest = create_source_snapshot(
                args.source,
                args.destination,
                owner_uid=args.owner_uid,
                owner_gid=args.owner_gid,
            )
            print(digest)
            return 0

        if args.command == "run-data-ingestion":
            run_data_ingestion(load_profile(args.profile))
            return 0

        if args.command == "run-lance-vector-index":
            run_lance_vector_index(load_profile(args.profile))
            return 0

        platform = (
            normalize_platform(args.platform) if args.platform else docker_platform()
        )
        if args.command == "runtime-gc-dry-run":
            runtime_gc_dry_run(args.profile, platform)
            return 0

        profile = load_profile(args.profile)
        if args.command == "runtime-key":
            identity = runtime_identity(profile, platform)
            print(
                json.dumps(
                    {
                        "local_tag": identity.local_tag,
                        "platform": identity.platform,
                        "profile": profile.name,
                        "runtime_key": identity.runtime_key,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "prepare-runtime":
            prepared = prepare_runtime(
                profile,
                platform=platform,
                registry=args.registry,
                allow_build=not args.no_local_build,
            )
            print(
                json.dumps(
                    {
                        "image_id": prepared.image_id,
                        "local_tag": prepared.identity.local_tag,
                        "source": prepared.source,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "publish-runtime":
            publish_runtime(profile, platform=platform, registry=args.registry)
            return 0
        raise TributoITError(f"unsupported command: {args.command}")
    except TributoITError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
