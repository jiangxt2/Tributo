"""Shared test fixtures and utilities."""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_ray_address():
    """Provide a mock Ray cluster address."""
    return "http://127.0.0.1:8265"


@pytest.fixture
def sample_entrypoint():
    """Provide a sample job entrypoint."""
    return "python -c 'print(\"Hello from Ray\")'"
