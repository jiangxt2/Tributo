"""Shared fixtures for E3 bundle-serving tests.

``build_test_bundle`` writes a minimal, valid local bundle (manifest +
artifact files) whose integrity checks pass.  By default the model file
is a dummy byte blob — enough for runtime-logic tests that inject a fake
flavor; ``make_dummy_onnx`` produces a real ONNX model for the end-to-end
ONNX Runtime path (skipped when skl2onnx/sklearn are unavailable).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

from tributo.exporting.manifest import (
    ExportManifest,
    ManifestExecution,
    ManifestSignature,
    ManifestSourceInfo,
    SignatureField,
)
from tributo.exporting.models import ArtifactFile, LogicalArtifact, ProducerInfo

DUMMY_MODEL_BYTES = b"not-a-real-onnx-model"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_dummy_onnx(tmp_path: Path) -> str:
    """Generate a minimal trainable ONNX classifier file (via skl2onnx).

    Skipped when skl2onnx or sklearn are not installed.
    """
    import numpy as np
    import pytest

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("skl2onnx or sklearn not installed")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    # sklearn 1.6 passes an 'iprint' solver option that scipy >= 1.14
    # warns about; pytest runs with filterwarnings=error, so silence the
    # unrelated solver warning around the fit.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Unknown solver options: iprint", category=Warning
        )
        clf = LogisticRegression().fit(X, y)

    initial_types = [("float_input", FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_types,
        options={id(clf): {"zipmap": False}},
    )  # plain float probability matrix, like the XGBoost ONNX path

    path = str(tmp_path / "dummy.onnx")
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def build_test_bundle(
    tmp_path: Path,
    *,
    with_signature: bool = True,
    flavor_id: str = "onnx-runtime-v1",
    model_bytes: bytes | None = None,
    onnx_path: str | None = None,
    roles: dict[str, str] | None = None,
    extra_files: dict[str, tuple[str, bytes]] | None = None,
    input_field_name: str = "float_input",
    input_field_shape: tuple[int | str, ...] = (),
    output_field_shapes: dict[str, tuple[int | str, ...]] | None = None,
    architecture_id: str | None = None,
    source_kind: str = "xgboost_result",
    extra_artifacts: dict[str, dict[str, tuple[str, bytes]]] | None = None,
) -> Path:
    """Write a minimal valid local bundle and return its root directory.

    Args:
        tmp_path: pytest tmp dir (or any writable dir).
        with_signature: Whether the manifest carries typed signatures.
        flavor_id: Flavor id recorded on the artifact.
        model_bytes: Raw model file bytes (defaults to a dummy blob).
        onnx_path: Copy an existing ONNX file as the entrypoint instead.
        roles: role → artifact name mapping (defaults to inference→model).
        extra_files: extra ``relative_path → (role, bytes)`` files.
        input_field_name: Typed signature input field name (must match the
            real model's input name for end-to-end tests).  Output fields
            default to the skl2onnx fixture's actual outputs
            (label:int64, probabilities:float32).
    """
    bundle_dir = tmp_path / "bundle"
    artifact_dir = bundle_dir / "artifacts" / "model"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if onnx_path is not None:
        shutil.copy(onnx_path, artifact_dir / "model.onnx")
    else:
        (artifact_dir / "model.onnx").write_bytes(model_bytes or DUMMY_MODEL_BYTES)

    file_entries: dict[str, tuple[str, bytes]] = {
        "model.onnx": ("model", (artifact_dir / "model.onnx").read_bytes())
    }
    if extra_files:
        for relative_path, (role, data) in extra_files.items():
            target = artifact_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            file_entries[relative_path] = (role, data)

    def _build_artifact(
        name: str, file_entries: dict[str, tuple[str, bytes]]
    ) -> LogicalArtifact:
        files = tuple(
            ArtifactFile(
                relative_path=path,
                sha256=_sha256(data),
                size_bytes=len(data),
                role=role,
            )
            for path, (role, data) in sorted(file_entries.items())
        )
        return LogicalArtifact(
            name=name,
            format="onnx" if name == "model" else "aux",
            flavor_id=flavor_id,
            files=files,
            entrypoint=sorted(file_entries)[0],
            tree_digest=LogicalArtifact.compute_tree_digest(files),
            producer=ProducerInfo(exporter_id="test-exporter"),
        )

    artifacts = [_build_artifact("model", file_entries)]
    for artifact_name, aux_files in (extra_artifacts or {}).items():
        artifact_dir = bundle_dir / "artifacts" / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, tuple[str, bytes]] = {}
        for relative_path, (role, data) in aux_files.items():
            target = artifact_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written[relative_path] = (role, data)
        artifacts.append(_build_artifact(artifact_name, written))

    manifest = ExportManifest(
        schema_version=1,
        bundle_id="bundle-e3-test",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status="succeeded",
        canonical_uri=str(bundle_dir),
        tributo_version="0.0.0",
        source_info=ManifestSourceInfo(
            source_kind=source_kind, architecture_id=architecture_id
        ),
        input_signature=ManifestSignature(
            input_fields=(
                SignatureField(
                    name=input_field_name, dtype="float32", shape=input_field_shape
                ),
            )
            if with_signature
            else ()
        ),
        # Matches the skl2onnx fixture's real outputs (zipmap=False):
        # label is int64, probabilities is float32.
        output_signature=ManifestSignature(
            output_fields=(
                SignatureField(
                    name="label",
                    dtype="int64",
                    shape=(output_field_shapes or {}).get("label", ()),
                ),
                SignatureField(
                    name="probabilities",
                    dtype="float32",
                    shape=(output_field_shapes or {}).get("probabilities", ()),
                ),
            )
            if with_signature
            else ()
        ),
        artifacts=tuple(artifacts),
        roles=roles or {"inference": "model"},
        execution=ManifestExecution(execution_id="exec-e3-test"),
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle_dir
