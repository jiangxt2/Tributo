"""Unit tests for the reproducible full-runtime image builder."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.build_tributo_image import (
    BASE_IMAGE,
    BASE_IMAGE_MIRROR,
    DOCKERFILE,
    PLATFORM_AUTO,
    RUNTIME_EXTRAS,
    UV_IMAGE,
    UV_IMAGE_MIRROR,
    ImageBuildError,
    _manifest_core,
    _prepare_pinned_image,
    build_command,
    build_image,
    canonical_json,
    detect_host_platform,
    load_config,
    local_base_image,
    local_uv_image,
    pip_check_baseline,
    prepared_wheelhouse,
    sha256_bytes,
    wheel_records,
)
from tools.runtime_image_contract import (
    REQUIRED_DISTRIBUTION_VERSIONS,
    REQUIRED_DISTRIBUTIONS,
    REQUIRED_IMPORTS,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "tools" / "tributo-runtime-full.json"


def test_full_runtime_config_is_pinned_and_complete() -> None:
    config = load_config(CONFIG, root=ROOT)

    assert config.base_image == BASE_IMAGE
    assert config.uv_image == UV_IMAGE
    assert config.base_image_mirror == BASE_IMAGE_MIRROR
    assert config.uv_image_mirror == UV_IMAGE_MIRROR
    assert config.platform == detect_host_platform()
    assert config.runtime_extras == RUNTIME_EXTRAS
    assert config.external_wheelhouse is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_image", "rayproject/ray:2.55.1-py312"),
        ("uv_image", "ghcr.io/astral-sh/uv:0.11.23"),
        ("platform", "linux/ppc64le"),
    ],
)
def test_config_rejects_unpinned_or_unsupported_values(
    field: str, value: str, tmp_path: Path
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    candidate = tmp_path / "tributo-runtime-invalid.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImageBuildError):
        load_config(candidate, root=ROOT)


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("arm64", "linux/arm64"),
        ("aarch64", "linux/arm64"),
        ("x86_64", "linux/amd64"),
    ],
)
def test_auto_platform_follows_host_architecture(
    machine: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.build_tributo_image.host_platform.machine", lambda: machine
    )

    assert load_config(CONFIG, root=ROOT).platform == expected


def test_explicit_platform_overrides_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tools.build_tributo_image.host_platform.machine", lambda: "arm64"
    )

    config = load_config(CONFIG, root=ROOT, platform_override="linux/amd64")

    assert config.platform == "linux/amd64"


def test_auto_is_the_default_config_value() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["platform"] == PLATFORM_AUTO


def test_config_rejects_missing_and_unknown_keys(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload.pop("runtime_extras")
    payload["unexpected"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ImageBuildError, match="keys differ"):
        load_config(path, root=ROOT)


def test_wheel_records_are_digest_only_and_deterministic(tmp_path: Path) -> None:
    wheel = tmp_path / "example_pkg-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel-content")

    records = wheel_records(tmp_path)

    assert records == [
        {
            "filename": wheel.name,
            "name": "example-pkg",
            "version": "1.2.3",
            "size": len(b"wheel-content"),
            "sha256": sha256_bytes(b"wheel-content"),
        }
    ]
    assert "path" not in records[0]


def test_wheelhouse_context_copies_only_wheels(tmp_path: Path) -> None:
    wheel = tmp_path / "example_pkg-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel-content")
    (tmp_path / "README.txt").write_text("ignored", encoding="utf-8")

    with prepared_wheelhouse(tmp_path) as context:
        assert (context / wheel.name).read_bytes() == b"wheel-content"
        assert not (context / "README.txt").exists()


def test_build_command_is_shell_free_and_uses_named_context() -> None:
    config = load_config(CONFIG, root=ROOT)
    command = build_command(
        config,
        root=ROOT,
        wheelhouse_context=Path("/tmp/empty-wheelhouse"),
        manifest_sha256="unsealed",
    )

    assert command[:4] == ["docker", "buildx", "build", "--load"]
    assert "--platform" in command
    assert "--file" in command
    assert str(DOCKERFILE) in command
    assert "--push" not in command
    assert "external-wheelhouse=/tmp/empty-wheelhouse" in command
    assert f"BASE_IMAGE={local_base_image(config.platform)}" in command
    assert f"UV_IMAGE={local_uv_image(config.platform)}" in command
    assert f"TRIBUTO_BASE_IMAGE={BASE_IMAGE}" in command
    assert f"TRIBUTO_PLATFORM={config.platform}" in command
    assert f"TRIBUTO_RUNTIME_EXTRAS={','.join(RUNTIME_EXTRAS)}" in command


def test_runtime_image_contract_is_shared_with_the_gate_job() -> None:
    gate_job = (
        ROOT / "tests" / "integrations" / "jobs" / "runtime_image_gate_job.py"
    ).read_text(encoding="utf-8")

    from tools import build_tributo_image

    assert build_tributo_image.REQUIRED_IMPORTS is REQUIRED_IMPORTS
    assert build_tributo_image.REQUIRED_DISTRIBUTIONS is REQUIRED_DISTRIBUTIONS
    assert (
        build_tributo_image.REQUIRED_DISTRIBUTION_VERSIONS
        is REQUIRED_DISTRIBUTION_VERSIONS
    )
    assert "from tools.runtime_image_contract import" in gate_job


def test_runtime_image_dockerfile_seals_and_generates_its_inventory() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    expected_manifest_line = (
        "RUN printf '%s\\n' \""
        + "$"
        + '{TRIBUTO_MANIFEST_SHA256}" > /opt/tributo-image/manifest-seal'
    )
    assert expected_manifest_line in dockerfile
    assert "COPY --chown=ray:users tools/generate_distributions.py" in dockerfile
    assert "RUN python /opt/tributo-image/generate_distributions.py" in dockerfile


def test_pinned_image_pull_is_platform_scoped_and_digest_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "a" * 64
    mirror = f"docker.m.daocloud.io/example/runtime:1@sha256:{digest}"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(
            returncode=0,
            stdout=f"Digest: sha256:{digest}\n",
            stderr="",
        )

    monkeypatch.setattr("tools.build_tributo_image._run", fake_run)
    monkeypatch.setattr(
        "tools.build_tributo_image._inspect_image",
        lambda image, **kwargs: {"Os": "linux", "Architecture": "arm64"},
    )

    _prepare_pinned_image(
        mirror=mirror,
        canonical=f"example/runtime:1@sha256:{digest}",
        local="tributo-example:arm64",
        platform="linux/arm64",
    )

    assert calls == [
        [
            "docker",
            "pull",
            "--platform",
            "linux/arm64",
            "docker.m.daocloud.io/example/runtime:1",
        ],
        [
            "docker",
            "tag",
            "docker.m.daocloud.io/example/runtime:1",
            "tributo-example:arm64",
        ],
    ]


def test_docker_environment_removes_all_proxy_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10080")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:10080")

    from tools.build_tributo_image import _no_pandafan_environment

    environment = _no_pandafan_environment()

    assert all(
        name not in environment
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
    )


def test_manifest_hash_is_canonical() -> None:
    left = {"b": 2, "a": [1, True]}
    right = {"a": [1, True], "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_bytes(canonical_json(left)) == sha256_bytes(canonical_json(right))


def test_build_image_seals_dependency_closure_without_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG, root=ROOT)
    built_manifests: list[str] = []

    def fake_build_once(*args: object, manifest_sha256: str, **kwargs: object) -> None:
        built_manifests.append(manifest_sha256)

    def fake_validate(
        *args: object, manifest_sha256: str, **kwargs: object
    ) -> dict[str, str]:
        return {"Id": "sha256:" + "a" * 64}

    distributions = {
        "bayesian-optimization": "1.4.3",
        "pip": "25.0",
        "ray": "2.55.1",
        "tributo": "1.0.0",
    }
    monkeypatch.setattr("tools.build_tributo_image._build_once", fake_build_once)
    monkeypatch.setattr(
        "tools.build_tributo_image._prepare_pinned_images", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("tools.build_tributo_image._validate_image", fake_validate)
    monkeypatch.setattr(
        "tools.build_tributo_image._installed_distributions",
        lambda *args, **kwargs: distributions,
    )

    manifest, profile = build_image(config, root=ROOT, output_dir=tmp_path / "result")

    assert built_manifests[0] == "unsealed"
    assert len(built_manifests[1]) == 64
    assert manifest["manifest_sha256"] == built_manifests[1]
    assert profile.image_digest == "a" * 64
    assert (tmp_path / "result" / "image-profile.json").is_file()


def test_manifest_core_contains_alpha_and_runtime_closure() -> None:
    config = load_config(CONFIG, root=ROOT)
    core = _manifest_core(
        config,
        root=ROOT,
        distributions={"ray": "2.55.1"},
        external_wheels=[],
    )

    assert core["platform"] == config.platform
    assert core["runtime_extras"] == list(RUNTIME_EXTRAS)
    assert core["alpha_capabilities"] == [
        "explainability",
        "vector_index",
        "kafka_streaming",
        "pipeline",
        "graph",
        "causal",
    ]
    assert "torch_geometric" in core["required_imports"]
    assert "ray_hive" in core["required_imports"]
    assert "bayesian-optimization" in core["required_distributions"]
    assert core["required_distribution_versions"]["ray-hive"] == "1.0"
    assert core["required_distribution_versions"]["thrift"] == "0.24.0"
    assert core["pip_check_baseline"] == list(pip_check_baseline(config.platform))
    assert core["image_sources"]["base_image"]["mirror"] == BASE_IMAGE_MIRROR
    assert core["image_sources"]["uv_image"]["mirror"] == UV_IMAGE_MIRROR
