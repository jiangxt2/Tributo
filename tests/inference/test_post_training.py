"""Tests for the Training-to-Inference entry adapter boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from tributo.data import IngestionRequest
from tributo.data.source_config import ParquetSourceConfig
from tributo.exporting.models import BundleRef
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ParquetResultSinkRequest,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.inference.post_training import (
    PostTrainingInferenceAction,
    run_post_training_inference,
    submit_post_training_inference,
)


def _bundle_ref() -> BundleRef:
    return BundleRef(
        canonical_uri="/models/bundle",
        bundle_id="bundle-1",
        manifest_sha256="a" * 64,
    )


def _action(mode: str = "inline") -> PostTrainingInferenceAction:
    return PostTrainingInferenceAction(
        input=IngestionRequest(
            source=ParquetSourceConfig(path="/data/input"), engine="ray"
        ),
        input_binding=InputBindingSpec(
            tensors=(TensorInputBinding(tensor_name="x", columns=("feature",)),)
        ),
        output_binding=OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="score", column="score", semantic="score"
                ),
            )
        ),
        result_sink=ParquetResultSinkRequest(uri="/data/output"),
        mode=mode,
    )


def test_bind_uses_published_bundle_and_parent_identity() -> None:
    request = _action().bind(_bundle_ref(), parent_run_id="training-run")

    assert request.model.kind == "bundle"
    assert request.model.uri == "/models/bundle"
    assert request.model.expected_manifest_sha256 == "a" * 64
    assert request.parent_run_id == "training-run"
    assert request.run_id is None
    assert request.input.engine == "tributo.ray_data"


def test_inline_uses_normal_inference_api() -> None:
    result = object()
    with patch(
        "tributo.inference.post_training.run_inference", return_value=result
    ) as run:
        actual = run_post_training_inference(
            _action("inline"), _bundle_ref(), parent_run_id="training-run"
        )

    assert actual is result
    request = run.call_args.args[0]
    assert request.parent_run_id == "training-run"


def test_detached_uses_normal_ray_jobs_adapter(tmp_path: Path) -> None:
    env_vars = {"TRIBUTO_STORAGE_PROFILE_MODEL": "profile-json"}
    with patch(
        "tributo.inference.job_runner.submit_inference_request",
        return_value="job-1",
    ) as submit:
        job_id = submit_post_training_inference(
            _action("detached"),
            _bundle_ref(),
            parent_run_id="training-run",
            dashboard_url="http://ray-head:8265",
            env_vars=env_vars,
            project_root=tmp_path,
        )

    assert job_id == "job-1"
    request = submit.call_args.args[0]
    assert request.parent_run_id == "training-run"
    assert submit.call_args.kwargs == {
        "dashboard_url": "http://ray-head:8265",
        "env_vars": env_vars,
        "project_root": tmp_path,
    }


def test_mode_specific_entry_points_fail_fast() -> None:
    with pytest.raises(ValueError, match="mode='inline'"):
        run_post_training_inference(
            _action("detached"), _bundle_ref(), parent_run_id="training-run"
        )
    with pytest.raises(ValueError, match="mode='detached'"):
        submit_post_training_inference(
            _action("inline"),
            _bundle_ref(),
            parent_run_id="training-run",
            dashboard_url="http://ray-head:8265",
        )


def test_adapter_source_has_no_training_import() -> None:
    path = Path(__file__).parents[2] / "src/tributo/inference/post_training.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.startswith("tributo.training") for name in imports)


def test_adapter_does_not_import_private_contract_symbols() -> None:
    path = Path(__file__).parents[2] / "src/tributo/inference/post_training.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_contract_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "tributo.inference.contracts"
        for alias in node.names
    }

    assert not any(name.startswith("_") for name in imported_contract_symbols)
