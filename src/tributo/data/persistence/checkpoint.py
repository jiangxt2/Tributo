"""Physical checkpoint access for training resume state.

The data module owns the storage boundary while the training module owns the
checkpoint envelope, payload digest, and restore semantics.  This adapter is
deliberately directory-based: Ray Checkpoint payloads are not assumed to be
Parquet tables.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, ClassVar, Iterator, Protocol, runtime_checkable

from tributo.util.annotations import DeveloperAPI


@runtime_checkable
@DeveloperAPI
class CheckpointStore(Protocol):
    """Physical checkpoint boundary used by training resume helpers."""

    store_id: ClassVar[str]

    def open_directory(self, checkpoint: Any) -> AbstractContextManager[Path]:
        """Expose a checkpoint as a local directory for payload inspection."""
        ...

    def load_initial(self, path: str | None) -> Any | None:
        """Create the runtime checkpoint object for an explicit local path."""
        ...


@DeveloperAPI
class RayCheckpointStore:
    """Default adapter around the public Ray Train checkpoint API."""

    store_id: ClassVar[str] = "ray-checkpoint-v1"

    @contextmanager
    def open_directory(self, checkpoint: Any) -> Iterator[Path]:
        """Yield a local directory for a Ray Checkpoint or local path."""
        if isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            if not path.is_dir():
                raise NotADirectoryError(f"Checkpoint path is not a directory: {path}")
            yield path
            return

        if not hasattr(checkpoint, "as_directory"):
            raise TypeError(
                f"Expected a Ray Checkpoint or path, got {type(checkpoint)!r}"
            )
        with checkpoint.as_directory() as raw_path:
            yield Path(raw_path)

    def load_initial(self, path: str | None) -> Any | None:
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


_DEFAULT_CHECKPOINT_STORE: CheckpointStore = RayCheckpointStore()


@DeveloperAPI
def default_checkpoint_store() -> CheckpointStore:
    """Return the process-local Ray checkpoint storage adapter."""
    return _DEFAULT_CHECKPOINT_STORE


__all__ = [
    "CheckpointStore",
    "RayCheckpointStore",
    "default_checkpoint_store",
]
