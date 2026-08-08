"""Ray Job for XGBoost → ONNX + UBJ + JSON Bundle → runtime."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


def _delete_bucket(client: Any, bucket: str) -> None:
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        objects = page.get("Contents", [])
        if objects:
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
            )
    client.delete_bucket(Bucket=bucket)


def main() -> int:
    """Train, export, read, and execute one immutable S3 bundle."""
    from tributo._common.storage import get_boto3_client
    from tributo.exporting.bundle_reader import BundleReader
    from tributo.exporting.models import BundleOutputConfig, ExportTarget
    from tributo.exporting.runtime import BundleModelLoader
    from tributo.exporting.service import BundleExportService
    from tributo.integrations.sources.ray_xgboost import RayXGBoostSourceProvider

    rng = np.random.default_rng(42)
    features = rng.random((32, 4), dtype=np.float32)
    labels = (features[:, 0] + features[:, 1] > 1.0).astype(np.int32)
    matrix = xgb.DMatrix(
        features,
        label=labels,
        feature_names=[f"f{index}" for index in range(features.shape[1])],
    )
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "seed": 42},
        matrix,
        num_boost_round=3,
    )

    root = Path(tempfile.mkdtemp(prefix="tributo-model-export-job-"))
    bucket = f"tributo-model-export-job-{uuid.uuid4().hex[:12]}"
    client = get_boto3_client(path_style=True)
    client.create_bucket(Bucket=bucket)
    try:
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        booster.save_model(str(checkpoint / "model.json"))

        config = BundleOutputConfig(
            bundle_uri=f"s3://{bucket}/models",
            storage_profile="test",
            request_id=f"ray-job-{uuid.uuid4().hex}",
            targets=(
                ExportTarget(
                    name="onnx-model",
                    format="onnx",
                    options={"opset": 12},
                ),
                ExportTarget(name="native-model", format="ubj"),
                ExportTarget(name="json-model", format="xgboost-json"),
            ),
            roles={"inference": "onnx-model"},
        )
        provider = RayXGBoostSourceProvider()
        service = BundleExportService()
        with provider.open_source(str(checkpoint)) as source:
            result = service.export_bundle(source=source, config=config)

        event = service.last_operation_event
        assert event is not None
        assert event.bundle_id == result.bundle_id
        assert event.manifest_sha256 == result.manifest_sha256

        with provider.open_source(str(checkpoint)) as source:
            retried = service.export_bundle(source=source, config=config)
        retried_event = service.last_operation_event
        assert retried.manifest_sha256 == result.manifest_sha256
        assert retried_event is not None
        assert retried_event.event_id == event.event_id

        reader = BundleReader(cache_dir=root / "cache")
        manifest = reader.read_manifest(result.canonical_uri, storage_profile="test")
        assert manifest.roles == {"inference": "onnx-model"}
        formats = {artifact.name: artifact.format for artifact in manifest.artifacts}
        assert formats == {
            "json-model": "xgboost-json",
            "native-model": "ubj",
            "onnx-model": "onnx",
        }
        assert all(artifact.artifact_kind == "model" for artifact in manifest.artifacts)

        evaluation_matrix = xgb.DMatrix(
            features[:2],
            feature_names=[f"f{index}" for index in range(features.shape[1])],
        )
        expected = booster.predict(evaluation_matrix)
        for artifact_name in ("native-model", "json-model"):
            with reader.open_artifact(
                result.canonical_uri,
                artifact_name=artifact_name,
                storage_profile="test",
            ) as native_artifact:
                loaded_booster = xgb.Booster()
                loaded_booster.load_model(str(native_artifact.entrypoint_path))
                actual = loaded_booster.predict(evaluation_matrix)
                np.testing.assert_allclose(actual, expected)

        with BundleModelLoader(bundle_reader=reader).open(
            result.canonical_uri,
            role="inference",
            storage_profile="test",
        ) as runtime:
            input_name = runtime.model.input_names[0]
            predictions = runtime.predict({input_name: features[:2]})
            assert predictions
            assert all(value.shape[0] == 2 for value in predictions.values())

        print(
            "RESULT: "
            + json.dumps(
                {
                    "status": result.status,
                    "bundle_id": result.bundle_id,
                    "manifest_sha256": result.manifest_sha256,
                    "event_id": event.event_id,
                    "artifact_kinds": sorted(
                        {artifact.artifact_kind for artifact in manifest.artifacts}
                    ),
                    "formats": formats,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        _delete_bucket(client, bucket)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
