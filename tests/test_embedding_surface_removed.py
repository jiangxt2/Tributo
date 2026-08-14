"""Regression tests for the complete removal of Tributo's old Embed surface."""

from __future__ import annotations

import importlib
import importlib.metadata

from click.testing import CliRunner

from tributo import plugin
from tributo.cli import main


def test_embedding_package_is_absent_without_a_compatibility_shim() -> None:
    try:
        importlib.import_module("tributo.embeddings")
    except ModuleNotFoundError as error:
        assert error.name == "tributo.embeddings"
    else:
        raise AssertionError("tributo.embeddings must be removed completely")


def test_cli_and_model_plugin_surface_no_longer_expose_embed() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "embed" not in result.output
    assert not hasattr(plugin, "discover_model_plugins")
    assert not importlib.metadata.entry_points().select(group="tributo.models")
