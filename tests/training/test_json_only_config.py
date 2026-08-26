"""Tests that YAML config files are rejected and JSON is accepted."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tributo.training.tune_space import parse_search_space


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_tune_space_rejects_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            "search_space:\n  lr:\n    type: loguniform\n    lower: 0.001\n    upper: 0.1\n"
        )
        path = f.name
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        parse_search_space(path)


def test_tune_space_accepts_json(tmp_path: Path):
    path = tmp_path / "space.json"
    _write(
        path,
        {
            "search_space": {
                "lr": {"type": "loguniform", "lower": 0.001, "upper": 0.1},
            }
        },
    )
    space = parse_search_space(str(path))
    paths = {p.path for p in space.parameters}
    assert "lr" in paths
