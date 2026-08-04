"""Tests for the versioned trainer resume checkpoint contract."""

from __future__ import annotations

import builtins
import random
from pathlib import Path

import pytest

from tributo.training.checkpoint import (
    ResumeCheckpointV1,
    ResumeConfig,
    read_resume_manifest,
    write_resume_manifest,
)


class TestResumeCheckpointV1:
    def test_manifest_digest_covers_declared_payload(self, tmp_path: Path) -> None:
        (tmp_path / "model.bin").write_bytes(b"model")
        (tmp_path / "state.json").write_text('{"step": 3}')

        envelope = write_resume_manifest(
            tmp_path,
            resume_id="run-42",
            trainer_type="dnn",
            completed_step=3,
            framework="pytorch",
            framework_version="2.5.0",
            payload_files=("model.bin", "state.json"),
        )

        assert envelope.resume_id == "run-42"
        assert (
            read_resume_manifest(
                tmp_path,
                expected_trainer_type="dnn",
                expected_resume_id="run-42",
            ).payload_digest
            == envelope.payload_digest
        )

        (tmp_path / "state.json").write_text('{"step": 4}')
        with pytest.raises(ValueError, match="digest mismatch"):
            read_resume_manifest(tmp_path)

    def test_payload_paths_are_safe(self) -> None:
        with pytest.raises(ValueError, match="relative and safe"):
            ResumeCheckpointV1(
                resume_id="run-42",
                trainer_type="dnn",
                completed_step=1,
                framework="pytorch",
                framework_version="2.5.0",
                payload_digest="a" * 64,
                payload_files=("../optimizer.pt",),
            )

        with pytest.raises(ValueError, match="relative and safe"):
            ResumeCheckpointV1(
                resume_id="run-42",
                trainer_type="dnn",
                completed_step=1,
                framework="pytorch",
                framework_version="2.5.0",
                payload_digest="a" * 64,
                payload_files=("nested\\optimizer.pt",),
            )

    def test_resume_config_is_opt_in(self) -> None:
        assert not ResumeConfig().effective_enabled
        assert ResumeConfig(enabled=True).effective_enabled
        assert ResumeConfig(checkpoint_path="/tmp/checkpoint").effective_enabled

    def test_rng_state_round_trip_with_torch(self) -> None:
        torch = pytest.importorskip("torch")
        import numpy as np

        from tributo.training.checkpoint import capture_rng_state, restore_rng_state

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        assert actual == pytest.approx(expected)

    def test_rng_state_is_usable_without_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from tributo.training.checkpoint import capture_rng_state, restore_rng_state

        original_import = builtins.__import__
        torch_import_attempted = False

        def reject_torch(name: str, *args: object, **kwargs: object) -> object:
            nonlocal torch_import_attempted
            if name == "torch":
                torch_import_attempted = True
                raise ImportError("Torch is unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", reject_torch)
        random.seed(7)
        np.random.seed(7)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()))

        random.seed(99)
        np.random.seed(99)
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()))

        assert torch_import_attempted
        assert "torch" not in state
        assert actual == pytest.approx(expected)
