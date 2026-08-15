"""Shared test fixtures and utilities."""

from __future__ import annotations

import ast
import importlib
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.ci_test_plan import load_manifest, markers_for_test_path
from tests.support.object_storage import S3InfrastructureUnavailable, S3Service

# Skip test files that import optional dependencies not installed in the
# current environment.  This prevents collection-phase ImportErrors when
# running `pytest -m "not integration"` without extras like mlflow, httpx,
# pyiceberg etc.
_TESTS_DIR = Path(__file__).parent

_OPTIONAL_IMPORTS = {
    "integrations/test_e2e_streaming.py": ["httpx"],
    "serving/test_streaming_http.py": ["httpx"],
    "serving/test_streaming_integration.py": ["httpx"],
    "serving/test_streaming_deployment.py": ["transformers"],
    "registry/test_model_registry.py": ["mlflow"],
    "registry/test_integration.py": ["mlflow"],
    "training/test_pu_trainer.py": ["torch"],
    "training/test_identity_e2e.py": ["torch"],
}

collect_ignore: list[str] = []


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the manifest-owned execution tier before marker deselection."""
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(root)
    for item in items:
        path = Path(str(item.path)).resolve()
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            continue
        for marker in markers_for_test_path(manifest, relative_path):
            item.add_marker(marker)


def _marker_selects_integration(expression: str) -> bool:
    """Return whether an integration-only test satisfies a marker expression."""

    def evaluate(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name):
            return node.id == "integration"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return all(evaluate(value) for value in node.values)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return any(evaluate(value) for value in node.values)
        raise ValueError("unsupported marker expression")

    try:
        parsed = ast.parse(expression, mode="eval")
        return evaluate(parsed)
    except (SyntaxError, ValueError):
        return False


def _mlflow_integration_requested(arguments: Sequence[str] | None = None) -> bool:
    targets = (
        (_TESTS_DIR / "integrations/test_e2e_mlflow.py").resolve(),
        (_TESTS_DIR / "registry/test_integration.py").resolve(),
    )
    selected_arguments = list(arguments if arguments is not None else sys.argv[1:])
    marker_expression: str | None = None
    for index, argument in enumerate(selected_arguments):
        if argument.startswith("-m="):
            marker_expression = argument[3:]
        elif argument == "-m" and index + 1 < len(selected_arguments):
            marker_expression = selected_arguments[index + 1]

    if marker_expression is not None:
        return _marker_selects_integration(marker_expression)

    for argument in selected_arguments:
        if argument.startswith("-"):
            continue
        candidate = argument.split("::", 1)[0]
        if not candidate:
            continue
        selected_path = Path(candidate).resolve()
        if selected_path in targets:
            return True
    return False


try:
    importlib.import_module("mlflow")
except ImportError:
    if not _mlflow_integration_requested():
        collect_ignore.extend(
            [
                str(_TESTS_DIR / "integrations/test_e2e_mlflow.py"),
                str(_TESTS_DIR / "registry/test_integration.py"),
            ]
        )

for path, modules in _OPTIONAL_IMPORTS.items():
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError:
            if (
                path == "registry/test_integration.py"
                and _mlflow_integration_requested()
            ):
                break
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
        if "minio_compat" in markers:
            pytest.fail(
                f"Required MinIO compatibility infrastructure is unavailable: {exc}"
            )
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
