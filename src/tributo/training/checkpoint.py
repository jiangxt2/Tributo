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
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Literal, cast
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator

from tributo._common.config import StrictConfigModel
from tributo.data.persistence import (
    CheckpointStore,
    ObjectStore,
    default_checkpoint_store,
    default_object_store,
)
from tributo.util.annotations import PublicAPI

RESUME_MANIFEST_FILENAME = "resume.json"
_DIGEST_CHUNK_BYTES = 1024 * 1024


@PublicAPI(stability="beta")
class ResumeConfig(StrictConfigModel):
    """Configuration for trainer checkpoint recovery.

    ``checkpoint_path`` is an explicit initial checkpoint directory.  When it
    is omitted, Ray Train supplies the latest persisted checkpoint through
    ``ray.train.get_checkpoint()`` after a worker failure.  Multi-worker
    Distributed trainers additionally validate their world size and formal
    distribution contract before restoring mutable training state.
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


@PublicAPI(stability="beta")
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


@PublicAPI(stability="beta")
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


@PublicAPI(stability="beta")
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
@PublicAPI(stability="beta")
def checkpoint_directory(
    checkpoint: Any,
    *,
    store: CheckpointStore | None = None,
) -> Generator[Path, None, None]:
    """Yield a local directory through the selected CheckpointStore."""
    with (store or default_checkpoint_store()).open_directory(checkpoint) as path:
        yield path


def _is_s3_uri(value: str | Path) -> bool:
    return urlsplit(str(value)).scheme.lower() == "s3"


@contextmanager
@PublicAPI(stability="beta")
def materialize_checkpoint_directory(
    checkpoint: Any,
    *,
    object_store: ObjectStore | None = None,
    checkpoint_store: CheckpointStore | None = None,
) -> Generator[Path, None, None]:
    """Expose a local or S3 checkpoint directory with bounded cleanup.

    Distributed runtimes use this helper at every restore boundary. S3
    checkpoints are materialized into a private temporary directory, so a
    Worker never assumes that another Worker-local path is shared.
    """
    if not isinstance(checkpoint, (str, Path)) or not _is_s3_uri(checkpoint):
        with checkpoint_directory(checkpoint, store=checkpoint_store) as path:
            yield path
        return

    store = object_store or default_object_store()
    temporary = Path(tempfile.mkdtemp(prefix="tributo-checkpoint-"))
    try:
        for item in store.list_files(str(checkpoint)):
            relative = Path(item.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"checkpoint object path escapes its prefix: {item.relative_path!r}"
                )
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(store.read_bytes(item.uri))
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@PublicAPI(stability="beta")
def publish_checkpoint_directory(
    checkpoint_dir: str | Path,
    target: str | Path,
    *,
    object_store: ObjectStore | None = None,
) -> str:
    """Publish a complete checkpoint directory through the Core transport.

    Payload files are uploaded before the manifest. Readers therefore never
    observe a checkpoint as complete until its manifest is present. Local
    paths are copied into the requested directory; S3 paths use the shared
    object-store binding and return the original URI.
    """
    source = Path(checkpoint_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"checkpoint source is not a directory: {source}")
    files = tuple(
        sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    )
    if not files:
        raise ValueError("checkpoint directory must contain at least one file")
    target_value = str(target)
    if not _is_s3_uri(target_value):
        destination = Path(target_value)
        destination.mkdir(parents=True, exist_ok=True)
        for path in files:
            relative = path.relative_to(source)
            destination_path = destination / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination_path)
        return target_value

    store = object_store or default_object_store()
    parsed = urlsplit(target_value)
    suffix = parsed.path.lstrip("/").rstrip("/")
    prefix = f"s3://{parsed.netloc}/{suffix + '/' if suffix else ''}"
    manifest_files = [
        path
        for path in files
        if path.name in {RESUME_MANIFEST_FILENAME, "manifest.json"}
    ]
    payload_files = [path for path in files if path not in manifest_files]
    for path in (*payload_files, *manifest_files):
        relative = path.relative_to(source).as_posix()
        store.write_bytes(
            prefix + relative,
            path.read_bytes(),
            content_type="application/json" if path.suffix == ".json" else None,
        )
    return target_value.rstrip("/")


@PublicAPI(stability="beta")
def load_initial_checkpoint(
    path: str | None,
    *,
    store: CheckpointStore | None = None,
) -> Any | None:
    """Create an initial runtime checkpoint through the selected store."""
    return (store or default_checkpoint_store()).load_initial(path)


@PublicAPI(stability="beta")
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


@PublicAPI(stability="beta")
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


@PublicAPI(stability="beta")
def checkpoint_config(resume: ResumeConfig) -> Any:
    """Build Ray's retention configuration without importing Ray eagerly."""
    from ray.train import CheckpointConfig

    return CheckpointConfig(num_to_keep=resume.num_to_keep)


__all__ = [
    "ResumeCheckpointV1",
    "ResumeConfig",
    "capture_rng_state",
    "checkpoint_config",
    "checkpoint_directory",
    "compute_payload_digest",
    "load_initial_checkpoint",
    "materialize_checkpoint_directory",
    "publish_checkpoint_directory",
    "read_resume_manifest",
    "restore_rng_state",
    "write_resume_manifest",
]
