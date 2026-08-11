"""Owned real MLflow service for registry integration tests."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class MLflowService:
    """Compose-owned MLflow tracking and model-registry endpoint."""

    tracking_uri: str

    @classmethod
    def start(cls) -> "MLflowService":
        """Attach only to the lifecycle-owned inference Compose service."""
        tracking_uri = os.environ.get("TRIBUTO_MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError(
                "Inference integration tests require the isolated Compose runner; "
                "use scripts/run_inference_it.sh"
            )
        service = cls(tracking_uri=tracking_uri.rstrip("/"))
        service._wait_until_healthy()
        return service

    def close(self) -> None:
        """Leave lifecycle ownership to the scoped Compose runner."""

    def _wait_until_healthy(self) -> None:
        health_url = f"{self.tracking_uri}/health"
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(1)
        raise RuntimeError(
            f"MLflow did not become healthy at {health_url}: {last_error}"
        )


__all__ = ["MLflowService"]
