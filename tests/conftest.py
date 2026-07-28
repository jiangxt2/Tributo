"""Shared test fixtures and utilities."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Skip test files that import optional dependencies not installed in the
# current environment.  This prevents collection-phase ImportErrors when
# running `pytest -m "not integration"` without extras like mlflow, httpx,
# pyiceberg etc.
_TESTS_DIR = Path(__file__).parent

_OPTIONAL_IMPORTS = {
    "integrations": ["mlflow"],
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
