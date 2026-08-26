"""Build and attest the pinned Tributo full runtime image.

The command is intentionally JSON-configured and fail-closed.  It builds the
same image twice: the first build discovers the dependency closure, and the
second build seals that closure into an OCI label.  The emitted manifest and
``ImageProfile`` are the hand-off consumed by Ray runtime selection and image
validation; this tool does not submit a Ray job or publish an image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as host_platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.host_uv_export import (  # noqa: E402
    HostDependencyExport,
    HostUVExportError,
    export_locked_requirements,
)
from tools.runtime_image_contract import (  # noqa: E402
    REQUIRED_DISTRIBUTION_VERSIONS,
    REQUIRED_DISTRIBUTIONS,
    REQUIRED_IMPORTS,
)

try:
    from tributo.algorithms.api.artifacts import ImageProfile
except ModuleNotFoundError:  # pragma: no cover - direct source checkout fallback
    _SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
    if str(_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SOURCE_ROOT))
    from tributo.algorithms.api.artifacts import ImageProfile


DOCKERFILE = ROOT / "docker" / "tributo-runtime" / "Dockerfile"
BASE_IMAGE = (
    "rayproject/ray:2.55.1-py312@"
    "sha256:911245f2478ad2e9f67ac13978dc2a75bcae0498b9f188b10bba703324b78379"
)
BASE_IMAGE_MIRROR = (
    "docker.m.daocloud.io/rayproject/ray:2.55.1-py312@"
    "sha256:911245f2478ad2e9f67ac13978dc2a75bcae0498b9f188b10bba703324b78379"
)
LOCAL_BASE_IMAGE = "tributo-ray-base:2.55.1-py312"
UV_VERSION = "0.11.23"
PLATFORM_AUTO = "auto"
SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")
RAY_VERSION = "2.55.1"
PYTHON_VERSION = "3.12"
PYTHON_SPEC = ">=3.12,<3.14"
PROFILE_ID = "tributo.runtime.full"
IMAGE_DIGEST = re.compile(r"^[0-9a-f]{64}$")
IMAGE_REFERENCE = re.compile(r"^.+:[^/@]+@sha256:[0-9a-f]{64}$")
PULL_DIGEST = re.compile(r"(?m)^Digest:\s+(sha256:[0-9a-f]{64})\s*$")
RUNTIME_IMAGE = re.compile(r"^[^\s@]+:[^\s@]+$")

RUNTIME_EXTRAS = (
    "data",
    "data-daft",
    "hive-ray",
    "vector-index",
    "postgresql",
    "clickhouse",
    "mysql",
    "doris-flight",
    "s3",
    "model-export",
    "model-export-torch",
    "hf",
    "model-export-hf",
    "training",
    "tune",
    "explainability",
    "identity",
    "streaming",
    "grpc",
    "registry",
    "graph",
    "causal",
    "streaming-inference",
)

ALPHA_CAPABILITIES = (
    "explainability",
    "vector_index",
    "kafka_streaming",
    "pipeline",
    "graph",
    "causal",
)
ARM64_PIP_CHECK_BASELINE = (
    "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform",
)


class ImageBuildError(RuntimeError):
    """Raised when an image cannot satisfy the reproducible build contract."""


@dataclass(frozen=True)
class RuntimeImageConfig:
    """Validated inputs for one full runtime image build."""

    image: str
    base_image: str
    base_image_mirror: str
    uv_version: str
    platform: str
    runtime_extras: tuple[str, ...]
    external_wheelhouse: Path | None


def canonical_json(value: object) -> bytes:
    """Serialize attestation data without whitespace or nondeterminism."""
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lower-case SHA-256 digest for ``value``."""
    return hashlib.sha256(value).hexdigest()


def normalize_platform(value: object) -> str:
    """Normalize and validate a supported Linux container platform."""
    if not isinstance(value, str):
        raise ImageBuildError("platform must be a string")
    try:
        operating_system, architecture = value.strip().split("/", 1)
    except ValueError as exc:
        raise ImageBuildError(f"invalid Docker platform {value!r}") from exc
    architecture = {"aarch64": "arm64", "x86_64": "amd64"}.get(
        architecture, architecture
    )
    normalized = f"{operating_system}/{architecture}"
    if normalized not in SUPPORTED_PLATFORMS:
        raise ImageBuildError(f"unsupported Docker platform {value!r}")
    return normalized


def detect_host_platform() -> str:
    """Return the native Linux target corresponding to the host architecture."""
    machine = host_platform.machine().lower()
    architecture = {"aarch64": "arm64", "arm64": "arm64"}.get(
        machine,
        {"x86_64": "amd64", "amd64": "amd64"}.get(machine),
    )
    if architecture is None:
        raise ImageBuildError(
            "cannot select a native runtime image for host architecture "
            f"{machine!r}; pass --platform linux/amd64 or --platform linux/arm64"
        )
    return f"linux/{architecture}"


def resolve_platform(value: object, *, override: str | None = None) -> str:
    """Resolve config ``auto`` or an explicit platform to a concrete target."""
    candidate = override if override is not None else value
    if candidate == PLATFORM_AUTO:
        return detect_host_platform()
    return normalize_platform(candidate)


def platform_machine(value: str) -> str:
    """Return the Linux machine marker for a concrete Docker platform."""
    normalized = normalize_platform(value)
    return "x86_64" if normalized == "linux/amd64" else "aarch64"


def platform_wheel_tags(value: str) -> tuple[str, ...]:
    """Return the platform-specific CPython 3.12 tags accepted by the image."""
    normalized = normalize_platform(value)
    architecture = "x86_64" if normalized == "linux/amd64" else "aarch64"
    return (
        f"cp312-cp312-manylinux_2_17_{architecture}",
        f"cp312-cp312-manylinux2014_{architecture}",
        "py3-none-any",
    )


def pip_check_baseline(value: str) -> tuple[str, ...]:
    """Return only the documented vendor metadata exception for one target."""
    return (
        ARM64_PIP_CHECK_BASELINE if normalize_platform(value) == "linux/arm64" else ()
    )


def local_base_image(value: str) -> str:
    """Return an architecture-scoped local Ray base-image tag."""
    architecture = normalize_platform(value).rsplit("/", 1)[1]
    return f"{LOCAL_BASE_IMAGE}-{architecture}"


def _validate_digest_reference(value: object, field: str) -> str:
    rendered = str(value)
    if IMAGE_REFERENCE.fullmatch(rendered) is None:
        raise ImageBuildError(
            f"{field} must use tag@sha256:<64 lower-case hex> form: {rendered!r}"
        )
    return rendered


def _validate_config_payload(
    payload: object,
    *,
    root: Path,
    platform_override: str | None = None,
) -> RuntimeImageConfig:
    if not isinstance(payload, dict):
        raise ImageBuildError("image configuration must be a JSON object")
    expected_keys = {
        "schema_version",
        "image",
        "base_image",
        "base_image_mirror",
        "dependency_mode",
        "uv_version",
        "platform",
        "runtime_extras",
        "external_wheelhouse",
    }
    unknown = sorted(set(payload) - expected_keys)
    missing = sorted(expected_keys - set(payload))
    if unknown or missing:
        raise ImageBuildError(
            f"image configuration keys differ; missing={missing}, unknown={unknown}"
        )
    if payload["schema_version"] != 1:
        raise ImageBuildError("unsupported image configuration schema")

    image = str(payload["image"])
    if RUNTIME_IMAGE.fullmatch(image) is None:
        raise ImageBuildError("image must be a readable tag such as repository:tag")
    base_image = _validate_digest_reference(payload["base_image"], "base_image")
    if base_image != BASE_IMAGE:
        raise ImageBuildError(f"base_image must be the pinned Ray image {BASE_IMAGE!r}")
    base_image_mirror = _validate_digest_reference(
        payload["base_image_mirror"], "base_image_mirror"
    )
    if base_image_mirror != BASE_IMAGE_MIRROR:
        raise ImageBuildError(
            "base_image_mirror must be the pinned DaoCloud Ray mirror "
            f"{BASE_IMAGE_MIRROR!r}"
        )
    if payload["dependency_mode"] != "host-uv-export":
        raise ImageBuildError("dependency_mode must be 'host-uv-export'")
    uv_version = str(payload["uv_version"])
    if uv_version != UV_VERSION:
        raise ImageBuildError(f"uv_version must be the host uv baseline {UV_VERSION!r}")
    configured_platform = payload["platform"]
    if configured_platform != PLATFORM_AUTO:
        normalize_platform(configured_platform)
    platform = resolve_platform(configured_platform, override=platform_override)

    raw_extras = payload["runtime_extras"]
    if not isinstance(raw_extras, list) or any(
        not isinstance(item, str) for item in raw_extras
    ):
        raise ImageBuildError("runtime_extras must be a JSON string list")
    extras = tuple(raw_extras)
    if len(set(extras)) != len(extras):
        raise ImageBuildError("runtime_extras must not contain duplicates")
    if extras != RUNTIME_EXTRAS:
        raise ImageBuildError(
            "runtime_extras must exactly match the full CPU runtime closure: "
            f"expected={list(RUNTIME_EXTRAS)!r}, actual={list(extras)!r}"
        )

    external = payload["external_wheelhouse"]
    external_path: Path | None
    if external is None:
        external_path = None
    else:
        if not isinstance(external, str) or not external:
            raise ImageBuildError("external_wheelhouse must be null or a path")
        candidate = Path(external)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ImageBuildError("external_wheelhouse must be a relative path")
        external_path = (root / candidate).resolve()
        try:
            external_path.relative_to(root.resolve())
        except ValueError as exc:
            raise ImageBuildError(
                "external_wheelhouse must stay under repository root"
            ) from exc
        if not external_path.is_dir():
            raise ImageBuildError(
                f"external_wheelhouse is not a directory: {external_path}"
            )

    return RuntimeImageConfig(
        image=image,
        base_image=base_image,
        base_image_mirror=base_image_mirror,
        uv_version=uv_version,
        platform=platform,
        runtime_extras=RUNTIME_EXTRAS,
        external_wheelhouse=external_path,
    )


def load_config(
    path: Path,
    *,
    root: Path = ROOT,
    platform_override: str | None = None,
) -> RuntimeImageConfig:
    """Load and validate a JSON runtime image configuration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImageBuildError(f"cannot read image configuration: {path}") from exc
    return _validate_config_payload(
        payload, root=root, platform_override=platform_override
    )


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def wheel_records(path: Path | None) -> list[dict[str, Any]]:
    """Return safe, digest-only records for an optional external wheelhouse."""
    if path is None:
        return []
    if not path.is_dir():
        raise ImageBuildError(f"external wheelhouse is not a directory: {path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for wheel in sorted(path.iterdir(), key=lambda item: item.name):
        if wheel.suffix != ".whl":
            continue
        if wheel.is_symlink() or not wheel.is_file():
            raise ImageBuildError(f"external wheel must be a regular file: {wheel}")
        try:
            name, version, _build, _tags = parse_wheel_filename(wheel.name)
        except InvalidWheelFilename as exc:
            raise ImageBuildError(
                f"invalid external wheel filename: {wheel.name!r}"
            ) from exc
        normalized_name = canonicalize_name(str(name))
        if normalized_name in seen:
            raise ImageBuildError(
                f"duplicate external wheel distribution: {normalized_name!r}"
            )
        seen.add(normalized_name)
        size, digest = _hash_file(wheel)
        records.append(
            {
                "filename": wheel.name,
                "name": normalized_name,
                "version": str(version),
                "size": size,
                "sha256": digest,
            }
        )
    return records


@contextmanager
def prepared_wheelhouse(path: Path | None) -> Iterator[Path]:
    """Copy external wheels into a safe named build context."""
    with tempfile.TemporaryDirectory(prefix="tributo-image-wheelhouse-") as temporary:
        destination = Path(temporary)
        if path is not None:
            for source in sorted(path.iterdir(), key=lambda item: item.name):
                if source.suffix != ".whl":
                    continue
                if source.is_symlink() or not source.is_file():
                    raise ImageBuildError(
                        f"external wheel must be a regular file: {source}"
                    )
                shutil.copy2(source, destination / source.name)
        yield destination


def build_command(
    config: RuntimeImageConfig,
    *,
    root: Path,
    wheelhouse_context: Path,
    requirements_context: Path,
    project_wheelhouse_context: Path,
    manifest_sha256: str,
    metadata_file: Path | None = None,
) -> list[str]:
    """Construct the shell-free Buildx command for one image build."""
    if not IMAGE_DIGEST.fullmatch(manifest_sha256) and manifest_sha256 != "unsealed":
        raise ImageBuildError("manifest_sha256 must be a digest or 'unsealed'")
    command = [
        "docker",
        "buildx",
        "build",
        "--load",
        "--platform",
        config.platform,
        "--file",
        str(root / "docker" / "tributo-runtime" / "Dockerfile"),
        "--tag",
        config.image,
        "--build-arg",
        f"BASE_IMAGE={local_base_image(config.platform)}",
        "--build-arg",
        f"TRIBUTO_BASE_IMAGE={config.base_image}",
        "--build-arg",
        f"TRIBUTO_PLATFORM={config.platform}",
        "--build-arg",
        f"TRIBUTO_MANIFEST_SHA256={manifest_sha256}",
        "--build-arg",
        f"TRIBUTO_RUNTIME_EXTRAS={','.join(config.runtime_extras)}",
        "--build-arg",
        f"TRIBUTO_VERSION={project_version(root)}",
        "--build-context",
        f"external-wheelhouse={wheelhouse_context}",
        "--build-context",
        f"locked-requirements={requirements_context}",
        "--build-context",
        f"project-wheelhouse={project_wheelhouse_context}",
    ]
    if metadata_file is not None:
        command.extend(("--metadata-file", str(metadata_file)))
    command.append(str(root))
    return command


def project_version(root: Path = ROOT) -> str:
    """Read the package version without importing the project."""
    try:
        payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str(payload["project"]["version"])
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ImageBuildError(
            "cannot read project.version from pyproject.toml"
        ) from exc


class _CommandLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def command(self, args: Sequence[str]) -> None:
        self.lines.append("$ " + " ".join(str(item) for item in args))

    def result(self, result: subprocess.CompletedProcess[str]) -> None:
        if result.stdout:
            self.lines.append(result.stdout.rstrip())
        if result.stderr:
            self.lines.append(result.stderr.rstrip())


def _no_pandafan_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment that cannot route Docker traffic through PandaFan."""
    environment = dict(os.environ if base is None else base)
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


def _run(
    args: Sequence[str],
    *,
    cwd: Path = ROOT,
    capture_output: bool = True,
    log: _CommandLog | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(item) for item in args]
    if log is not None:
        log.command(rendered)
    print("+ " + " ".join(rendered), flush=True)
    command_env = env
    if command_env is None and rendered and rendered[0] == "docker":
        command_env = _no_pandafan_environment()
    result = subprocess.run(
        rendered,
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        check=False,
        env=dict(command_env) if command_env is not None else None,
    )
    if log is not None and capture_output:
        log.result(result)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise ImageBuildError(
            f"command failed with exit code {result.returncode}: {' '.join(rendered)}"
            + (f"\n{details}" if details else "")
        )
    return result


def _host_dependency_export(
    config: RuntimeImageConfig,
    *,
    root: Path,
    log: _CommandLog,
) -> HostDependencyExport:
    """Export the full runtime dependency closure with local uv."""
    try:
        return export_locked_requirements(
            root=root,
            extras=config.runtime_extras,
            baseline_uv_version=config.uv_version,
            run=lambda args, cwd: _run(args, cwd=cwd, log=log),
            warning_prefix="full runtime baseline uv is",
        )
    except HostUVExportError as exc:
        raise ImageBuildError(str(exc)) from exc


@contextmanager
def prepared_requirements(export: HostDependencyExport) -> Iterator[Path]:
    """Materialize host-exported requirements as a named build context."""
    with tempfile.TemporaryDirectory(
        prefix="tributo-runtime-requirements-"
    ) as temporary:
        context = Path(temporary)
        (context / "requirements.txt").write_bytes(export.content)
        yield context


@contextmanager
def prepared_project_wheelhouse(*, root: Path, log: _CommandLog) -> Iterator[Path]:
    """Build the project wheel with host uv for offline image installation."""
    uv = shutil.which("uv")
    if uv is None:
        raise ImageBuildError("host uv is required to build the Tributo project wheel")
    with tempfile.TemporaryDirectory(
        prefix="tributo-runtime-project-wheel-"
    ) as temporary:
        context = Path(temporary)
        _run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(context),
                "--no-create-gitignore",
            ],
            cwd=root,
            log=log,
        )
        wheels = sorted(context.glob("tributo-*.whl"))
        if len(wheels) != 1:
            raise ImageBuildError(
                f"expected exactly one Tributo project wheel, found {len(wheels)}"
            )
        yield context


def _prepare_pinned_image(
    *,
    mirror: str,
    canonical: str,
    local: str,
    platform: str,
    log: _CommandLog | None = None,
) -> None:
    """Pull a mirror image, verify its digest/platform, and create a local tag."""
    mirror_tag = mirror.rsplit("@", 1)[0]
    pull_result = _run(
        ["docker", "pull", "--platform", platform, mirror_tag],
        log=log,
    )
    expected_digest = canonical.rsplit("@", 1)[-1]
    pull_output = "\n".join(
        part for part in (pull_result.stdout, pull_result.stderr) if part
    )
    pull_matches = PULL_DIGEST.findall(pull_output)
    if pull_matches != [expected_digest]:
        raise ImageBuildError(
            f"mirror image digest mismatch for {mirror}: "
            f"expected={expected_digest}, pull_output={pull_output!r}"
        )
    inspected = _inspect_image(mirror_tag, log=log)
    actual_platform = f"{inspected.get('Os', '')}/{inspected.get('Architecture', '')}"
    if actual_platform != platform:
        raise ImageBuildError(
            f"mirror image platform mismatch for {mirror}: "
            f"expected={platform}, actual={actual_platform}"
        )
    _run(["docker", "tag", mirror_tag, local], log=log)


def _prepare_pinned_images(
    config: RuntimeImageConfig,
    *,
    log: _CommandLog | None = None,
) -> None:
    """Prepare both digest-pinned image inputs without using the host proxy."""
    _prepare_pinned_image(
        mirror=config.base_image_mirror,
        canonical=config.base_image,
        local=local_base_image(config.platform),
        platform=config.platform,
        log=log,
    )


def _inspect_image(image: str, *, log: _CommandLog | None = None) -> dict[str, Any]:
    result = _run(["docker", "image", "inspect", image], log=log)
    try:
        payload = json.loads(result.stdout)
        inspected = payload[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ImageBuildError(
            f"unexpected docker image inspect result for {image}"
        ) from exc
    if not isinstance(inspected, dict):
        raise ImageBuildError(f"invalid docker image inspect result for {image}")
    return inspected


def _container_python(
    image: str,
    code: str,
    *,
    platform: str,
    log: _CommandLog | None = None,
) -> str:
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            platform,
            "--pull",
            "never",
            "--entrypoint",
            "python",
            image,
            "-c",
            code,
        ],
        log=log,
    )
    return result.stdout.strip()


def _validate_image(
    config: RuntimeImageConfig,
    *,
    root: Path = ROOT,
    manifest_sha256: str,
    log: _CommandLog | None = None,
) -> dict[str, Any]:
    inspected = _inspect_image(config.image, log=log)
    labels = inspected.get("Config", {}).get("Labels") or {}
    expected_labels = {
        "org.tributo.base-image": config.base_image,
        "org.tributo.ray-version": RAY_VERSION,
        "org.tributo.python-version": PYTHON_VERSION,
        "org.tributo.platform": config.platform,
        "org.tributo.runtime-extras": ",".join(config.runtime_extras),
        "org.tributo.manifest-sha256": manifest_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": labels.get(key)}
        for key, value in expected_labels.items()
        if labels.get(key) != value
    }
    actual_platform = f"{inspected.get('Os', '')}/{inspected.get('Architecture', '')}"
    if actual_platform != config.platform:
        mismatches["platform"] = {
            "expected": config.platform,
            "actual": actual_platform,
        }
    user = str(inspected.get("Config", {}).get("User") or "")
    if user in {"", "0", "0:0", "root"}:
        mismatches["Config.User"] = {
            "expected": "non-root",
            "actual": user or "<empty>",
        }
    if mismatches:
        raise ImageBuildError(f"runtime image labels/platform invalid: {mismatches}")

    pip_check_code = (
        "import subprocess, sys; "
        "result = subprocess.run([sys.executable, '-m', 'pip', 'check'], "
        "capture_output=True, text=True); "
        "output = (result.stdout + result.stderr).strip(); "
        f"baseline = {pip_check_baseline(config.platform)!r}; "
        "assert result.returncode == 0 or tuple(output.splitlines()) == baseline, "
        "(result.returncode, output)"
    )
    _container_python(config.image, pip_check_code, platform=config.platform, log=log)
    seal_code = (
        "from pathlib import Path; "
        f"assert Path('/opt/tributo-image/manifest-seal').read_text().strip() "
        f"== {manifest_sha256!r}"
    )
    _container_python(config.image, seal_code, platform=config.platform, log=log)
    import_code = (
        "import importlib, importlib.metadata as metadata, sys; "
        f"assert sys.version_info[:2] == (3, 12); "
        f"assert metadata.version('ray') == {RAY_VERSION!r}; "
        f"assert metadata.version('tributo') == {project_version(root)!r}; "
        + "; ".join(f"metadata.version({name!r})" for name in REQUIRED_DISTRIBUTIONS)
        + "; "
        + "; ".join(
            f"assert metadata.version({name!r}) == {version!r}"
            for name, version in REQUIRED_DISTRIBUTION_VERSIONS.items()
        )
        + "; "
        + "; ".join(f"importlib.import_module({name!r})" for name in REQUIRED_IMPORTS)
    )
    _container_python(config.image, import_code, platform=config.platform, log=log)
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            config.platform,
            "--pull",
            "never",
            "--entrypoint",
            "tributo",
            config.image,
            "--help",
        ],
        log=log,
    )
    image_id = str(inspected.get("Id") or "")
    if (
        not image_id.startswith("sha256:")
        or IMAGE_DIGEST.fullmatch(image_id.removeprefix("sha256:")) is None
    ):
        raise ImageBuildError(f"runtime image has invalid local ID: {image_id!r}")
    return inspected


def _installed_distributions(
    image: str, *, platform: str, log: _CommandLog | None = None
) -> dict[str, str]:
    output = _container_python(
        image,
        "import json, pathlib; print(pathlib.Path('/opt/tributo-image/installed-distributions.json').read_text())",
        platform=platform,
        log=log,
    )
    try:
        raw = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ImageBuildError(
            "image installed-distributions.json is not valid JSON"
        ) from exc
    if not isinstance(raw, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in raw.items()
    ):
        raise ImageBuildError(
            "image installed-distributions.json must map names to versions"
        )
    return dict(
        sorted((canonicalize_name(name), version) for name, version in raw.items())
    )


def _build_once(
    config: RuntimeImageConfig,
    *,
    root: Path,
    wheelhouse_context: Path,
    requirements_context: Path,
    project_wheelhouse_context: Path,
    manifest_sha256: str,
    log: _CommandLog,
) -> None:
    with tempfile.NamedTemporaryFile(
        prefix="tributo-image-buildx-", suffix=".json"
    ) as metadata:
        command = build_command(
            config,
            root=root,
            wheelhouse_context=wheelhouse_context,
            requirements_context=requirements_context,
            project_wheelhouse_context=project_wheelhouse_context,
            manifest_sha256=manifest_sha256,
            metadata_file=Path(metadata.name),
        )
        _run(command, cwd=root, capture_output=False, log=log)
        if Path(metadata.name).stat().st_size == 0:
            raise ImageBuildError("Buildx did not produce its required metadata file")


def _manifest_core(
    config: RuntimeImageConfig,
    *,
    root: Path,
    dependency_export: HostDependencyExport,
    distributions: dict[str, str],
    external_wheels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "image": config.image,
        "base_image": config.base_image,
        "dependency_mode": "host-uv-export",
        "host_uv_version": dependency_export.uv_version,
        "requirements_sha256": dependency_export.sha256,
        "image_sources": {
            "base_image": {
                "canonical": config.base_image,
                "mirror": config.base_image_mirror,
                "local": local_base_image(config.platform),
            },
        },
        "platform": config.platform,
        "python_version": PYTHON_VERSION,
        "python_spec": PYTHON_SPEC,
        "ray_version": RAY_VERSION,
        "tributo_version": project_version(root),
        "runtime_extras": list(config.runtime_extras),
        "installed_distributions": distributions,
        "external_wheels": external_wheels,
        "alpha_capabilities": list(ALPHA_CAPABILITIES),
        "required_imports": list(REQUIRED_IMPORTS),
        "required_distributions": list(REQUIRED_DISTRIBUTIONS),
        "required_distribution_versions": dict(REQUIRED_DISTRIBUTION_VERSIONS),
        "pip_check_baseline": list(pip_check_baseline(config.platform)),
    }


def _profile_manifest(
    config: RuntimeImageConfig,
    *,
    root: Path,
    dependency_export: HostDependencyExport,
    distributions: dict[str, str],
    external_wheels: list[dict[str, Any]],
    manifest_sha256: str,
    inspected: dict[str, Any],
) -> tuple[dict[str, Any], ImageProfile]:
    image_id = str(inspected["Id"])
    digest = image_id.removeprefix("sha256:")
    image_uri = f"{config.image}@sha256:{digest}"
    manifest = {
        **_manifest_core(
            config,
            root=root,
            dependency_export=dependency_export,
            distributions=distributions,
            external_wheels=external_wheels,
        ),
        "manifest_sha256": manifest_sha256,
        "image_uri": image_uri,
        "image_digest": digest,
        "profile_id": PROFILE_ID,
    }
    profile = ImageProfile(
        profile_id=PROFILE_ID,
        image_uri=image_uri,
        image_digest=digest,
        ray_version=RAY_VERSION,
        python_spec=PYTHON_SPEC,
        python_version=PYTHON_VERSION,
        sys_platform="linux",
        platform_machine=platform_machine(config.platform),
        wheel_tags=platform_wheel_tags(config.platform),
        installed_distributions=distributions,
        allow_offline_pip=True,
        pip_check_baseline=pip_check_baseline(config.platform),
    )
    return manifest, profile


def write_outputs(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    profile: ImageProfile,
    distributions: dict[str, str],
    log: _CommandLog,
) -> None:
    """Write all attestations, refusing to overwrite an existing result."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ImageBuildError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest.json": manifest,
        "image-profile.json": profile.model_dump(mode="json"),
        "installed-distributions.json": distributions,
        "capabilities.json": {
            "schema_version": 1,
            "profile_id": PROFILE_ID,
            "status": "alpha",
            "capabilities": [
                {"id": capability, "status": "alpha"}
                for capability in ALPHA_CAPABILITIES
            ],
        },
    }
    for filename, payload in files.items():
        (output_dir / filename).write_bytes(canonical_json(payload) + b"\n")
    (output_dir / "build.log").write_text("\n".join(log.lines) + "\n", encoding="utf-8")


def build_image(
    config: RuntimeImageConfig,
    *,
    root: Path = ROOT,
    output_dir: Path,
    output_archive: Path | None = None,
) -> tuple[dict[str, Any], ImageProfile]:
    """Build, validate, seal, and emit one full runtime image."""
    dockerfile = root / "docker" / "tributo-runtime" / "Dockerfile"
    if not dockerfile.is_file():
        raise ImageBuildError(f"runtime Dockerfile is missing: {dockerfile}")
    external_wheels = wheel_records(config.external_wheelhouse)
    log = _CommandLog()
    dependency_export = _host_dependency_export(config, root=root, log=log)
    _prepare_pinned_images(config, log=log)
    with (
        prepared_wheelhouse(config.external_wheelhouse) as wheelhouse_context,
        prepared_requirements(dependency_export) as requirements_context,
        prepared_project_wheelhouse(root=root, log=log) as project_wheelhouse_context,
    ):
        _build_once(
            config,
            root=root,
            wheelhouse_context=wheelhouse_context,
            requirements_context=requirements_context,
            project_wheelhouse_context=project_wheelhouse_context,
            manifest_sha256="unsealed",
            log=log,
        )
        _validate_image(config, root=root, manifest_sha256="unsealed", log=log)
        distributions = _installed_distributions(
            config.image, platform=config.platform, log=log
        )
        core = _manifest_core(
            config,
            root=root,
            dependency_export=dependency_export,
            distributions=distributions,
            external_wheels=external_wheels,
        )
        manifest_sha256 = sha256_bytes(canonical_json(core))
        _build_once(
            config,
            root=root,
            wheelhouse_context=wheelhouse_context,
            requirements_context=requirements_context,
            project_wheelhouse_context=project_wheelhouse_context,
            manifest_sha256=manifest_sha256,
            log=log,
        )
        inspected = _validate_image(
            config, root=root, manifest_sha256=manifest_sha256, log=log
        )
        final_distributions = _installed_distributions(
            config.image, platform=config.platform, log=log
        )
        if final_distributions != distributions:
            raise ImageBuildError(
                "sealed image dependency closure differs from discovery build"
            )
        manifest, profile = _profile_manifest(
            config,
            root=root,
            dependency_export=dependency_export,
            distributions=final_distributions,
            external_wheels=external_wheels,
            manifest_sha256=manifest_sha256,
            inspected=inspected,
        )
        write_outputs(
            output_dir,
            manifest=manifest,
            profile=profile,
            distributions=final_distributions,
            log=log,
        )
        if output_archive is not None:
            output_archive.parent.mkdir(parents=True, exist_ok=True)
            _run(
                ["docker", "save", "--output", str(output_archive), config.image],
                log=log,
            )
    return manifest, profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "tools" / "tributo-runtime-full.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--platform",
        dest="platform_override",
        help="target platform; defaults to the native host architecture",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    config = load_config(
        args.config.resolve(),
        root=root,
        platform_override=args.platform_override,
    )
    manifest, profile = build_image(
        config,
        root=root,
        output_dir=args.output_dir.resolve(),
        output_archive=args.output_archive.resolve() if args.output_archive else None,
    )
    print(
        json.dumps(
            {
                "image_uri": profile.image_uri,
                "image_digest": profile.image_digest,
                "manifest_sha256": manifest["manifest_sha256"],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
