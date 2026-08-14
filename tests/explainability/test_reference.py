"""Reference provider tests for explainability background data."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tributo.explainability.contracts import ExplainabilityLimits, ReferenceBinding
from tributo.explainability.reference import FileReferenceProvider


def test_file_reference_provider_rejects_pickle_and_applies_row_limit(tmp_path) -> None:
    path = tmp_path / "reference.npy"
    np.save(path, np.arange(12, dtype=np.float32).reshape(6, 2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    resolved = FileReferenceProvider().resolve(
        ReferenceBinding(uri=str(path), digest=digest, rows=6),
        ExplainabilityLimits(max_background_rows=3),
    )

    assert resolved.rows == 3
    assert resolved.digest == digest
    assert resolved.data.shape == (3, 2)


def test_file_reference_provider_rejects_mismatched_declared_digest(tmp_path) -> None:
    path = tmp_path / "reference.npy"
    np.save(path, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="digest"):
        FileReferenceProvider().resolve(
            ReferenceBinding(uri=str(path), digest="a" * 64),
            ExplainabilityLimits(),
        )
