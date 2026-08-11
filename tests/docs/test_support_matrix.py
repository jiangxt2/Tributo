"""Tests for the marker-scoped algorithm support matrix generator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import tools.generate_algorithm_support_matrix as support_matrix_generator
from tools.generate_algorithm_support_matrix import (
    BEGIN_MARKER,
    END_MARKER,
    SupportMatrixGenerationError,
    check_support_matrix,
    render_generated_region,
    replace_generated_region,
    write_support_matrix,
)
from tributo.training.support_snapshot import AlgorithmSupportRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(name: str = "example") -> AlgorithmSupportRecord:
    return AlgorithmSupportRecord(
        name=name,
        problem_types=("binary_classification",),
        data_modality=("tabular",),
        tags=(),
        execution_kind="train",
        supported_tasks=("train",),
        capabilities=("tunable", "exportable"),
        data_loading="canonical_driver",
        gpu_required=False,
        status="ready",
        extras_group="training",
    )


def _document(region: str) -> str:
    return f"# Before\n\nManual before.\n\n{region}\n\nManual after.\n"


def test_render_is_idempotent_and_preserves_manual_content() -> None:
    snapshot = (_record(),)
    original = _document(f"{BEGIN_MARKER}\nstale\n{END_MARKER}")

    rendered_once = replace_generated_region(original, snapshot)
    rendered_twice = replace_generated_region(rendered_once, snapshot)

    assert rendered_once == rendered_twice
    assert rendered_once.startswith("# Before\n\nManual before.")
    assert rendered_once.endswith("\n\nManual after.\n")
    assert render_generated_region(snapshot) in rendered_once
    assert "| Algorithm | Lifecycle | Stability | Availability |" in rendered_once
    assert "<code>beta</code>" in rendered_once


@pytest.mark.parametrize(
    "document",
    (
        "no markers",
        f"{BEGIN_MARKER}\n{BEGIN_MARKER}\n{END_MARKER}",
        f"{BEGIN_MARKER}\n{END_MARKER}\n{END_MARKER}",
        f"{END_MARKER}\n{BEGIN_MARKER}",
    ),
)
def test_marker_errors_fail_closed(document: str) -> None:
    with pytest.raises(SupportMatrixGenerationError):
        replace_generated_region(document, (_record(),))


def test_check_detects_manual_drift_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "support-matrix.md"
    stale = _document(f"{BEGIN_MARKER}\nmanually edited\n{END_MARKER}")
    path.write_text(stale, encoding="utf-8")

    errors = check_support_matrix(path, (_record(),))

    assert errors == [
        f"{path}: generated algorithm support region is stale; "
        "run tools/generate_algorithm_support_matrix.py"
    ]
    assert path.read_text(encoding="utf-8") == stale


def test_check_rejects_missing_markers_before_loading_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "support-matrix.md"
    path.write_text("manual content only\n", encoding="utf-8")
    monkeypatch.setattr(
        support_matrix_generator,
        "load_support_snapshot",
        lambda: pytest.fail("catalog must not load before marker validation"),
    )

    errors = check_support_matrix(path)

    assert len(errors) == 1
    assert "found begin=0, end=0" in errors[0]


def test_write_updates_only_generated_region(tmp_path: Path) -> None:
    path = tmp_path / "support-matrix.md"
    path.write_text(
        _document(f"{BEGIN_MARKER}\nstale\n{END_MARKER}"),
        encoding="utf-8",
    )

    assert write_support_matrix(path, (_record("safe|name"),)) is True
    updated = path.read_text(encoding="utf-8")
    assert "Manual before." in updated
    assert "Manual after." in updated
    assert "<code>safe&#124;name</code>" in updated
    assert check_support_matrix(path, (_record("safe|name"),)) == []
    assert write_support_matrix(path, (_record("safe|name"),)) is False


def test_static_docs_gate_does_not_import_optional_training_frameworks() -> None:
    script = """
import sys
from pathlib import Path
from tools.check_docs import run_checks

errors = run_checks(Path("docs"), static_only=True)
assert errors == [], errors
loaded_roots = {name.partition(".")[0] for name in sys.modules}
assert "torch" not in loaded_roots
assert "xgboost" not in loaded_roots
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
