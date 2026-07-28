"""XGBoost model export utilities: ONNX export wrapper and metric persistence."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from tributo._common.storage import upload_file, write_json
from tributo.training.onnx_exporter import export_from_checkpoint

logger = logging.getLogger(__name__)


# ── ONNX export wrapper ──


def export_onnx(
    checkpoint: Any,
    onnx_path: str,
    n_features: int,
    *,
    onnx_opset: int = 12,
    s3_cfg: dict | None = None,
) -> tuple[str, str, int]:
    """Export ONNX model to local or S3, returns (path, sha256_hex, size_bytes).

    Args:
        checkpoint: Ray Train Checkpoint object.
        onnx_path: Target path (local path or s3:// URI).
        n_features: Number of feature columns.
        onnx_opset: ONNX opset version.
        s3_cfg: S3 connection config dictionary (required for S3 paths).

    Returns:
        (Export path, SHA256 hex string, file size in bytes).
    """
    if onnx_path.startswith("s3://"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_onnx = str(Path(tmp_dir) / "model.onnx")
            export_from_checkpoint(
                checkpoint=checkpoint,
                output_path=local_onnx,
                n_features=n_features,
                target_opset=onnx_opset,
                validate=True,
            )
            file_hash, file_size = _hash_file(local_onnx)
            upload_file(local_onnx, onnx_path, s3_cfg=s3_cfg)
    else:
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        export_from_checkpoint(
            checkpoint=checkpoint,
            output_path=onnx_path,
            n_features=n_features,
            target_opset=onnx_opset,
            validate=True,
        )
        file_hash, file_size = _hash_file(onnx_path)
    logger.info(
        "ONNX exported to %s (sha256=%s, size=%d)", onnx_path, file_hash, file_size
    )
    return onnx_path, file_hash, file_size


def _hash_file(path: str) -> tuple[str, int]:
    """Compute SHA256 hex digest and file size."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def save_metrics(
    summary: dict[str, Any],
    metrics_path: str,
    s3_cfg: dict | None = None,
) -> None:
    """Save training metrics to local or S3.

    Args:
        summary: Metric summary dictionary.
        metrics_path: Target path (local path or s3:// URI).
        s3_cfg: S3 connection config dictionary (required for S3 paths).
    """
    write_json(metrics_path, summary, s3_cfg=s3_cfg)
    logger.info("Metrics saved to %s", metrics_path)
