"""Small algorithm package used by the offline Ray Jobs Gate."""

from __future__ import annotations

import importlib.metadata
import os
import socket

from tributo_test_offline_dependency import MARKER


def verify_runtime() -> dict[str, str]:
    """Return evidence that the algorithm and its private dependency are loaded."""
    return {
        "algorithm_marker": "offline-algorithm-imported",
        "dependency_marker": MARKER,
        "dependency_version": importlib.metadata.version(
            "tributo-test-offline-dependency"
        ),
        "node": socket.gethostname(),
        "distribution_mode": os.environ.get("TRIBUTO_ALGORITHM_DISTRIBUTION_MODE", ""),
    }
