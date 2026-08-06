"""Shared test fixtures and utilities."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.support.object_storage import S3InfrastructureUnavailable, S3Service

# Skip test files that import optional dependencies not installed in the
# current environment.  This prevents collection-phase ImportErrors when
# running `pytest -m "not integration"` without extras like mlflow, httpx,
# pyiceberg etc.
_TESTS_DIR = Path(__file__).parent

_OPTIONAL_IMPORTS = {
    "integrations": ["mlflow"],
    "integrations/test_e2e_streaming.py": ["httpx"],
    "serving/test_streaming_http.py": ["httpx"],
    "serving/test_streaming_integration.py": ["httpx"],
    "serving/test_streaming_deployment.py": ["transformers"],
    "data/test_iceberg_connector.py": ["pyiceberg"],
    "data/test_lance_connector.py": ["lance"],
    "registry/test_model_registry.py": ["mlflow"],
    "registry/test_integration.py": ["mlflow"],
    "training/test_pu_trainer.py": ["torch"],
    "training/test_identity_e2e.py": ["torch"],
}

collect_ignore: list[str] = []

for path, modules in _OPTIONAL_IMPORTS.items():
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            target = _TESTS_DIR / path
            if target.is_dir():
                collect_ignore.extend(str(p) for p in target.rglob("*.py"))
            else:
                collect_ignore.append(str(target))
            break


@pytest.fixture
def mock_ray_address():
    """Provide a mock Ray cluster address."""
    return "http://127.0.0.1:8265"


@pytest.fixture
def sample_entrypoint():
    """Provide a sample job entrypoint."""
    return "python -c 'print(\"Hello from Ray\")'"


@contextmanager
def _bounded_ray_local_runtime() -> Iterator[None]:
    """Own one bounded local Ray runtime without selecting a Daft runner."""
    import ray
    from ray._private import ray_constants

    already_initialized = ray.is_initialized()
    previous_uv_runtime_env = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    try:
        if not already_initialized:
            try:
                ray.init(
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    num_cpus=1,
                )
            except (PermissionError, OSError) as exc:
                pytest.skip(
                    "Local Ray runtime is unavailable on this host: "
                    f"{exc}. Run Ray contract tests on a host that permits "
                    "Ray process discovery and local workers."
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if (
                    "operation not permitted" in message
                    or "timed out waiting" in message
                ):
                    pytest.skip(
                        "Local Ray runtime is unavailable on this host: "
                        f"{exc}. Run Ray contract tests on a host that permits "
                        "Ray process discovery and local workers."
                    )
                raise
        yield
    finally:
        if not already_initialized and ray.is_initialized():
            ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous_uv_runtime_env


@pytest.fixture(scope="module")
def ray_local_runtime() -> Iterator[None]:
    """Run Ray data contract tests in one bounded local runtime per module."""
    with _bounded_ray_local_runtime():
        yield


@pytest.fixture(scope="module")
def native_daft_ray_local_runtime() -> Iterator[None]:
    """Lock Daft to its native runner before starting a local Ray runtime."""
    import daft

    runner = daft.get_or_create_runner()
    if runner.name != "native":
        pytest.fail(
            "Native Daft conformance requires a fresh process whose Daft runner "
            "has not already been locked to Ray"
        )
    with _bounded_ray_local_runtime():
        yield


@pytest.fixture(scope="session")
def s3_service(request: pytest.FixtureRequest) -> Iterator[S3Service | None]:
    """Own the S3 service required by the selected S3 test tier."""
    markers = {
        marker
        for item in request.session.items
        for marker in ("s3_contract", "minio_compat")
        if item.get_closest_marker(marker) is not None
    }
    if not markers:
        yield None
        return

    try:
        service = (
            S3Service.start_minio()
            if "minio_compat" in markers
            else S3Service.start_contract()
        )
    except S3InfrastructureUnavailable as exc:
        pytest.skip(str(exc))
    try:
        yield service
    finally:
        service.close()


@pytest.fixture
def s3_environment(
    s3_service: S3Service | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject S3 settings only for a test that explicitly opts into them."""
    if s3_service is None:  # pragma: no cover - guarded by marker selection
        raise pytest.UsageError("S3 test service was not initialized")
    monkeypatch.setenv("S3_ENDPOINT", s3_service.endpoint)
    monkeypatch.setenv(
        "AWS_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    )
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY",
        os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    )
    monkeypatch.setenv(
        "AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
