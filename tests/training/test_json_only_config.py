"""Tests that YAML config files are rejected and JSON is accepted."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from tributo.training.dnn_trainer import run_dnn_training_from_json
from tributo.training.pu_trainer import run_pu_training_from_json
from tributo.training.tune_space import parse_search_space
from tributo.training.xgboost_trainer import run_training_from_json


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_xgboost_rejects_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("data:\n  type: csv\n")
        path = f.name
    with pytest.raises(ValueError, match="YAML"):
        run_training_from_json(path)


def test_xgboost_accepts_json(tmp_path: Path):
    path = tmp_path / "cfg.json"
    _write(path, {"data": {"type": "csv", "path": "x.csv", "label_col": "y"}})
    # The provider path should preserve the old early missing-file error and
    # must not start Ray just to discover that the local input is absent.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(FileNotFoundError, match=r"x\.csv"):
            run_training_from_json(str(path))


def test_pu_rejects_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("data:\n  type: csv\n")
        path = f.name
    with pytest.raises(ValueError, match="YAML"):
        run_pu_training_from_json(path)


def test_dnn_rejects_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("data:\n  type: csv\n")
        path = f.name
    with pytest.raises(ValueError, match="YAML"):
        run_dnn_training_from_json(path)


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
