"""Trainer checkpoint and resume contracts.

The checkpoint used by Ray Train has two deliberately separate concerns:

* ``ResumeCheckpointV1`` contains mutable training state needed to continue a
  run (optimizer, RNG and progress state).
* E2's ``ExportCheckpointV1`` remains the model-export contract.  Resume-only
  files are stored beside, but never inside, that export metadata.

The helpers in this module keep the envelope framework-neutral and leave
framework-specific binary serialization to the trainer implementations.
"""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, cast

from pydantic import ConfigDict, Field, field_validator

from tributo._common.config import StrictConfigModel
from tributo.util.annotations import PublicAPI

RESUME_MANIFEST_FILENAME = "resume.json"
_DIGEST_CHUNK_BYTES = 1024 * 1024


@PublicAPI(stability="beta")
class ResumeConfig(StrictConfigModel):
    """Configuration for single-worker trainer checkpoint recovery.

    ``checkpoint_path`` is an explicit initial checkpoint directory.  When it
    is omitted, Ray Train supplies the latest persisted checkpoint through
    ``ray.train.get_checkpoint()`` after a worker failure.  Multi-worker
    checkpoint coordination remains the T4-D follow-up.
    """

    enabled: bool = False
    checkpoint_path: str | None = None
    resume_id: str | None = Field(default=None, min_length=1)
    checkpoint_interval: int = Field(default=1, ge=1)
    num_to_keep: int = Field(default=2, ge=1)

    @property
    def effective_enabled(self) -> bool:
        """Return whether resume state should be written and restored."""
        return (
            self.enabled
            or self.checkpoint_path is not None
            or self.resume_id is not None
        )


@PublicAPI(stability="beta")
class ResumeCheckpointV1(StrictConfigModel):
    """Versioned envelope for a trainer resume checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    resume_id: str = Field(..., min_length=1)
    trainer_type: str = Field(..., min_length=1)
    completed_step: int = Field(..., ge=0)
    framework: str = Field(..., min_length=1)
    framework_version: str = Field(..., min_length=1)
    payload_digest: str = Field(..., min_length=64, max_length=64)
    payload_files: tuple[str, ...] = Field(..., min_length=1)
    payload_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_digest")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        if any(char not in "0123456789abcdef" for char in value):
            raise ValueError("payload_digest must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("payload_files")
    @classmethod
    def _validate_payload_files(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("payload_files must not contain duplicates")
        for filename in value:
            path = Path(filename)
            if (
                not filename
                or "\\" in filename
                or "\x00" in filename
                or path.is_absolute()
                or ".." in path.parts
            ):
                raise ValueError(
                    f"payload file must be relative and safe: {filename!r}"
                )
        return value


def _canonical_json(value: Any) -> bytes:
    """Serialize JSON using the deterministic form used for digests."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_payload_digest(
    checkpoint_dir: str | Path, payload_files: tuple[str, ...]
) -> str:
    """Compute a digest over payload file names and bytes."""
    root = Path(checkpoint_dir)
    root_resolved = root.resolve()
    file_digests: dict[str, str] = {}
    for filename in sorted(payload_files):
        path = root / filename
        if path.is_symlink() or not path.resolve().is_relative_to(root_resolved):
            raise ValueError(
                f"Resume checkpoint payload escapes its directory: {filename}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint payload is missing: {filename}")
        file_digests[filename] = _sha256_file(path)
    return hashlib.sha256(_canonical_json(file_digests)).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a payload incrementally to avoid loading large models at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_DIGEST_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_resume_manifest(
    checkpoint_dir: str | Path,
    *,
    resume_id: str | None = None,
    trainer_type: str,
    completed_step: int,
    framework: str,
    framework_version: str,
    payload_files: tuple[str, ...],
    payload_metadata: dict[str, Any] | None = None,
) -> ResumeCheckpointV1:
    """Write and return a verified ``ResumeCheckpointV1`` envelope."""
    root = Path(checkpoint_dir)
    digest = compute_payload_digest(root, payload_files)
    envelope = ResumeCheckpointV1(
        resume_id=resume_id or f"resume-{digest[:32]}",
        trainer_type=trainer_type,
        completed_step=completed_step,
        framework=framework,
        framework_version=framework_version,
        payload_digest=digest,
        payload_files=payload_files,
        payload_metadata=payload_metadata or {},
    )
    (root / RESUME_MANIFEST_FILENAME).write_text(
        envelope.model_dump_json(indent=2), encoding="utf-8"
    )
    return envelope


def read_resume_manifest(
    checkpoint_dir: str | Path,
    *,
    expected_trainer_type: str | None = None,
    expected_resume_id: str | None = None,
) -> ResumeCheckpointV1:
    """Read and verify a resume envelope and all declared payload files."""
    root = Path(checkpoint_dir)
    manifest_path = root / RESUME_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Resume checkpoint is missing {RESUME_MANIFEST_FILENAME}")
    envelope = ResumeCheckpointV1.model_validate_json(manifest_path.read_text())
    if expected_trainer_type and envelope.trainer_type != expected_trainer_type:
        raise ValueError(
            f"Resume checkpoint trainer_type {envelope.trainer_type!r} does not match "
            f"{expected_trainer_type!r}"
        )
    if expected_resume_id and envelope.resume_id != expected_resume_id:
        raise ValueError(
            f"Resume checkpoint ID {envelope.resume_id!r} does not match "
            f"{expected_resume_id!r}"
        )
    actual_digest = compute_payload_digest(root, envelope.payload_files)
    if actual_digest != envelope.payload_digest:
        raise ValueError(
            "Resume checkpoint payload digest mismatch: "
            f"expected {envelope.payload_digest}, got {actual_digest}"
        )
    return envelope


@contextmanager
def checkpoint_directory(checkpoint: Any) -> Iterator[Path]:
    """Yield a local directory for a Ray ``Checkpoint`` or local path."""
    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        if not path.is_dir():
            raise NotADirectoryError(f"Checkpoint path is not a directory: {path}")
        yield path
        return

    if not hasattr(checkpoint, "as_directory"):
        raise TypeError(f"Expected a Ray Checkpoint or path, got {type(checkpoint)!r}")
    with checkpoint.as_directory() as raw_path:
        yield Path(raw_path)


def load_initial_checkpoint(path: str | None) -> Any | None:
    """Create a Ray checkpoint for an explicit local resume directory."""
    if path is None:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(
            f"Checkpoint path is not a directory: {checkpoint_path}"
        )
    from ray.train import Checkpoint

    return Checkpoint.from_directory(str(checkpoint_path))


def capture_rng_state() -> dict[str, Any]:
    """Capture available Python, NumPy, and Torch RNG state in JSON-safe form.

    Torch state is optional so this helper remains usable in NumPy-only
    environments, such as XGBoost workers.
    """
    import numpy as np

    numpy_state = cast(tuple[str, Any, int, int, float], np.random.get_state())
    state: dict[str, Any] = {
        "python": _to_jsonable(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    try:
        import torch
    except ImportError:
        return state

    state["torch"] = torch.get_rng_state().tolist()
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            value.tolist() for value in torch.cuda.get_rng_state_all()
        ]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG state entries available in *state*.

    Torch is imported only when the state contains a Torch entry.  A state
    captured without Torch therefore restores successfully in NumPy-only
    environments.
    """
    import numpy as np

    random.setstate(_from_jsonable(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if "torch" not in state:
        return

    import torch

    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(
            [torch.tensor(value, dtype=torch.uint8) for value in state["torch_cuda"]]
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_from_jsonable(item) for item in value)
    return value


def checkpoint_config(resume: ResumeConfig) -> Any:
    """Build Ray's retention configuration without importing Ray eagerly."""
    from ray.train import CheckpointConfig

    return CheckpointConfig(num_to_keep=resume.num_to_keep)


__all__ = [
    "RESUME_MANIFEST_FILENAME",
    "ResumeCheckpointV1",
    "ResumeConfig",
    "capture_rng_state",
    "checkpoint_config",
    "checkpoint_directory",
    "compute_payload_digest",
    "load_initial_checkpoint",
    "read_resume_manifest",
    "restore_rng_state",
    "write_resume_manifest",
]
