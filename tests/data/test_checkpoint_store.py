"""Tests for the unified physical CheckpointStore boundary."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from tributo.data.persistence import (
    RayCheckpointStore,
    default_checkpoint_store,
)
from tributo.training.checkpoint import (
    checkpoint_directory,
    materialize_checkpoint_directory,
    publish_checkpoint_directory,
)


def test_default_checkpoint_store_is_the_ray_adapter() -> None:
    store = default_checkpoint_store()

    assert isinstance(store, RayCheckpointStore)
    assert store.store_id == "ray-checkpoint-v1"


def test_checkpoint_store_exposes_local_directory(tmp_path: Path) -> None:
    store = RayCheckpointStore()

    with store.open_directory(tmp_path) as path:
        assert path == tmp_path


def test_training_helper_delegates_to_injected_checkpoint_store(
    tmp_path: Path,
) -> None:
    class FakeStore:
        @contextmanager
        def open_directory(self, checkpoint: object) -> Iterator[Path]:
            assert checkpoint == "checkpoint-token"
            yield tmp_path

        def load_initial(self, path: str | None) -> object | None:
            return path

    with checkpoint_directory("checkpoint-token", store=FakeStore()) as path:
        assert path == tmp_path


def test_checkpoint_store_rejects_missing_local_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError, match="not a directory"):
        with RayCheckpointStore().open_directory(tmp_path / "missing"):
            pass


def test_core_checkpoint_transport_publishes_and_materializes_local_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.bin").write_bytes(b"state")
    (source / "manifest.json").write_text('{"state": "state.bin"}')
    target = tmp_path / "published"

    assert publish_checkpoint_directory(source, target) == str(target)
    with materialize_checkpoint_directory(target) as materialized:
        assert (materialized / "state.bin").read_bytes() == b"state"
        assert (materialized / "manifest.json").is_file()
