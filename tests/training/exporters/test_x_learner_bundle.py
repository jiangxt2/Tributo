"""End-to-end fixed X-Learner checkpoint, Bundle, and inference tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tributo.exceptions import ModelLoadError
from tributo.exporting.bundle_reader import BundleReader
from tributo.exporting.models import BundleOutputConfig, BundleRef, ExportTarget
from tributo.exporting.service import BundleExportService
from tributo.inference.bundle_predictor import BundleBatchPredictor
from tributo.inference.contracts import (
    InputBindingSpec,
    OutputBindingSpec,
    ResolvedModelSelection,
    TensorInputBinding,
    TensorOutputBinding,
)
from tributo.integrations.flavors.x_learner import XLearnerFlavor
from tributo.integrations.sources.ray_x_learner import RayXLearnerSourceProvider
from tributo.training.x_learner import X_LEARNER_STAGES, XLearnerTrainingResult


def _checkpoint(root: Path, stage: str, objective: str) -> Path:
    xgboost = pytest.importorskip("xgboost")
    path = root / stage
    path.mkdir(parents=True)
    features = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = (
        np.asarray([0.0, 0.2, 0.8, 1.0], dtype=np.float32)
        if objective == "reg:squarederror"
        else np.asarray([0, 0, 1, 1], dtype=np.float32)
    )
    booster = xgboost.train(
        {"objective": objective, "max_depth": 1, "eta": 0.3},
        xgboost.DMatrix(features, label=labels, feature_names=["x1", "x2"]),
        num_boost_round=2,
    )
    (path / "model.ubj").write_bytes(bytes(booster.save_raw(raw_format="ubj")))
    return path


def test_x_learner_bundle_round_trip_and_batch_prediction(tmp_path: Path) -> None:
    objectives = {
        "mu0": "binary:logistic",
        "mu1": "binary:logistic",
        "tau0": "reg:squarederror",
        "tau1": "reg:squarederror",
        "propensity": "binary:logistic",
    }
    result = XLearnerTrainingResult(
        checkpoints={
            stage: _checkpoint(tmp_path / "checkpoints", stage, objectives[stage])
            for stage in X_LEARNER_STAGES
        },
        metrics={"ate": 0.1, "qini": 0.2},
        feature_names=("x1", "x2"),
        response_threshold=0.5,
        propensity_clip=(0.01, 0.99),
        stage_evidence={},
    )
    bundle = tmp_path / "bundle"
    with RayXLearnerSourceProvider().open_source(result) as source:
        published = BundleExportService().export_bundle(
            source,
            BundleOutputConfig(
                bundle_uri=str(bundle),
                targets=[
                    ExportTarget(
                        name="x-learner-model",
                        format="x-learner",
                        exporter_id="x-learner-v1",
                    ),
                    ExportTarget(
                        name="causal-report",
                        format="json",
                        exporter_id="causal-report-v1",
                    ),
                ],
                roles={
                    "inference": "x-learner-model",
                    "causal_report": "causal-report",
                },
            ),
            tributo_version="1.0.0",
        )
    assert published.status == "succeeded"
    predictor = BundleBatchPredictor(
        ResolvedModelSelection(
            bundle_ref=BundleRef(
                canonical_uri=published.canonical_uri,
                bundle_id=published.bundle_id,
                manifest_sha256=published.manifest_sha256,
            ),
            role="inference",
            flavor_id="x-learner-v1",
            source_provenance="tributo-bundle",
        ),
        InputBindingSpec(
            tensors=(
                TensorInputBinding(
                    tensor_name="float_input",
                    columns=("x1", "x2"),
                    dtype="float32",
                ),
            )
        ),
        OutputBindingSpec(
            tensors=(
                TensorOutputBinding(
                    tensor_name="cate",
                    column="cate",
                    semantic="score",
                ),
                TensorOutputBinding(
                    tensor_name="quadrant",
                    column="quadrant",
                    semantic="label",
                ),
            )
        ),
    )
    try:
        prediction = predictor(
            {
                "x1": np.asarray([0.0, 1.0], dtype=np.float32),
                "x2": np.asarray([1.0, 0.0], dtype=np.float32),
            }
        )
    finally:
        predictor.close()
    assert set(prediction) == {"cate", "quadrant"}
    assert prediction["cate"].shape == (2,)
    assert prediction["quadrant"].shape == (2,)

    with BundleReader().open_artifact(
        published.canonical_uri,
        role="causal_report",
    ) as report_artifact:
        report = json.loads(report_artifact.entrypoint_path.read_text(encoding="utf-8"))
        assert report["study"]["ate"] == 0.1

    with BundleReader().open_artifact(
        published.canonical_uri,
        role="inference",
    ) as model_artifact:
        metadata = json.loads(
            model_artifact.entrypoint_path.read_text(encoding="utf-8")
        )
        duplicate = json.loads(json.dumps(metadata))
        duplicate["components"]["mu1"] = duplicate["components"]["mu0"]
        model_artifact.entrypoint_path.write_text(
            json.dumps(duplicate),
            encoding="utf-8",
        )
        with pytest.raises(ModelLoadError, match="artifact is invalid"):
            XLearnerFlavor().load(
                model_artifact,
                role="inference",
                architecture_id="x_learner",
            )

        swapped = json.loads(json.dumps(metadata))
        swapped["components"]["mu0"], swapped["components"]["tau0"] = (
            swapped["components"]["tau0"],
            swapped["components"]["mu0"],
        )
        model_artifact.entrypoint_path.write_text(
            json.dumps(swapped),
            encoding="utf-8",
        )
        with pytest.raises(ModelLoadError, match="artifact is invalid"):
            XLearnerFlavor().load(
                model_artifact,
                role="inference",
                architecture_id="x_learner",
            )

        metadata["api_version"] = 2
        model_artifact.entrypoint_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        with pytest.raises(ModelLoadError, match="artifact is invalid"):
            XLearnerFlavor().load(
                model_artifact,
                role="inference",
                architecture_id="x_learner",
            )


def test_x_learner_source_rejects_component_objective_drift(tmp_path: Path) -> None:
    objectives = {
        "mu0": "reg:squarederror",
        "mu1": "binary:logistic",
        "tau0": "reg:squarederror",
        "tau1": "reg:squarederror",
        "propensity": "binary:logistic",
    }
    result = XLearnerTrainingResult(
        checkpoints={
            stage: _checkpoint(tmp_path / "checkpoints", stage, objectives[stage])
            for stage in X_LEARNER_STAGES
        },
        metrics={},
        feature_names=("x1", "x2"),
        response_threshold=0.5,
        propensity_clip=(0.01, 0.99),
        stage_evidence={},
    )

    with pytest.raises(ValueError, match="incompatible XGBoost objective"):
        with RayXLearnerSourceProvider().open_source(result):
            pass
