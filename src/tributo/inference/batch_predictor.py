"""XGBoost ONNX distributed batch inference Actor.

Uses ray.data.map_batches + ActorPoolStrategy to keep the model resident in memory,
avoiding reloading the ONNX model for each batch. Data is stream-processed end-to-end without passing through the Driver.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tributo.exporting.runtime import BundleModelRuntime

from tributo.exceptions import DataSourceError, JobConfigurationError
from tributo.inference.base import BasePredictor
from tributo.util.annotations import PublicAPI

logger = logging.getLogger(__name__)


@PublicAPI(stability="beta")
class XGBoostONNXPredictor(BasePredictor):
    """Stateful Ray Actor: loads the ONNX model and runs inference on batches.

    Lifecycle:
    1. Load the ONNX model into memory on __init__ (once only);
    2. __call__ runs inference on each batch, returning the batch with a prediction column;
    3. The session is released automatically when the Actor is reclaimed.

    The model may be loaded from a published bundle (``bundle_uri`` +
    explicit ``role``) or from a raw ``model_uri`` path (legacy compat).

    predictor_config dict keys:
        return_probs (bool): True returns probability matrix, False returns class labels, default True.
        prediction_column (str): Output column name, default "prediction".
        s3_config (dict): S3 auth config, keys: access_key_id, secret_access_key, endpoint, region.
        feature_names (list[str]): Feature column names, takes precedence over ONNX metadata.
    """

    def __init__(
        self,
        model_uri: str | None = None,
        predictor_config: dict[str, Any] | None = None,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ) -> None:
        self.return_probs = (predictor_config or {}).get("return_probs", True)
        self.prediction_column = (predictor_config or {}).get(
            "prediction_column", "prediction"
        )
        self._s3_config = (predictor_config or {}).get("s3_config") or {}
        self.feature_names: list[str] = []
        self._bundle_uri = bundle_uri
        self._role = role
        self._unsafe = unsafe
        self._storage_profile = storage_profile
        self._runtime: "BundleModelRuntime | None" = None
        self._output_names: tuple[str, ...] = ()

        if (model_uri is None) == (bundle_uri is None):
            raise ValueError(
                "exactly one of 'model_uri' (legacy) or 'bundle_uri' must be provided"
            )

        if bundle_uri is not None:
            self.model_uri = bundle_uri
            self.predictor_config = predictor_config or {}
            self._load_model()
        else:
            assert model_uri is not None  # guarded by the exclusivity check above
            super().__init__(model_uri, predictor_config)

    @classmethod
    def get_feature_names(
        cls,
        model_uri: str | None = None,
        predictor_config: dict[str, Any] | None = None,
        *,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ) -> list[str]:
        """Read feature column names from a bundle or ONNX metadata.

        The bundle path reads ONNX ``feature_names`` metadata first
        (the *table* feature names used to select Dataset columns) and
        falls back to the manifest's typed input signature (a
        *tensor*-level contract); the legacy path reads ONNX metadata.
        """
        if bundle_uri is not None:
            from tributo.exporting.runtime import BundleModelLoader

            loader = BundleModelLoader()
            runtime = loader.open(
                bundle_uri,
                role=role,
                unsafe=unsafe,
                storage_profile=storage_profile,
            )
            try:
                return _feature_names_from_bundle(runtime)
            finally:
                runtime.close()

        assert model_uri is not None
        s3_config = (predictor_config or {}).get("s3_config") or {}

        if model_uri.startswith("s3://"):
            import tempfile
            import uuid

            if importlib.util.find_spec("boto3") is None:
                raise JobConfigurationError("boto3 is required for S3 model download.")

            cache_dir = Path(tempfile.gettempdir()) / "tributo_onnx" / str(uuid.uuid4())
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_path = cache_dir / "model.onnx"

            path = model_uri.removeprefix("s3://")
            bucket, key = path.split("/", 1)

            client = cls._build_s3_client(s3_config)
            try:
                client.download_file(bucket, key, str(local_path))
            except Exception as e:
                raise DataSourceError(
                    f"Failed to download ONNX model from {model_uri}: {e}"
                ) from e
        else:
            if model_uri.startswith("file://"):
                model_uri = model_uri[7:]
            local_path = Path(model_uri)
            if not local_path.exists():
                raise DataSourceError(f"ONNX model not found: {model_uri}")

        return cls._load_feature_names(local_path)

    def _load_model(self) -> None:
        """Load the model: bundle path (shared runtime) or legacy raw path."""
        if self._bundle_uri is not None:
            self._load_model_from_bundle()
            return
        local_path = self._resolve_model(self.model_uri)
        self._init_session(local_path)

        explicit = self.predictor_config.get("feature_names")
        if explicit:
            self.feature_names = explicit
        else:
            self.feature_names = self._load_feature_names(local_path)
            if self.feature_names:
                logger.info(
                    "Loaded feature_names from ONNX metadata: %s", self.feature_names
                )

    def _load_model_from_bundle(self) -> None:
        """Open the bundle through the shared BundleModelLoader."""
        from tributo.exporting.runtime import BundleModelLoader

        assert self._bundle_uri is not None
        loader = BundleModelLoader()
        self._runtime = loader.open(
            self._bundle_uri,
            role=self._role,
            unsafe=self._unsafe,
            storage_profile=self._storage_profile,
        )
        self.input_name = self._runtime.model.input_names[0]
        self._output_names = self._runtime.model.output_names

        explicit = self.predictor_config.get("feature_names")
        if explicit:
            self.feature_names = explicit
        else:
            self.feature_names = _feature_names_from_bundle(self._runtime)
            if self.feature_names:
                logger.info("Loaded feature_names from bundle: %s", self.feature_names)

    @staticmethod
    def _load_feature_names(model_path: str | Path) -> list[str]:
        """Read feature_names from ONNX model metadata."""
        try:
            import json

            import onnx

            model = onnx.load(model_path)
            for prop in model.metadata_props:
                if prop.key == "feature_names":
                    return json.loads(prop.value)  # type: ignore[no-any-return]
        except Exception:
            logger.debug("Failed to load feature_names from ONNX metadata")
        return []

    def _resolve_model(self, model_uri: str) -> str:
        """Resolve model path: supports local and S3 paths."""
        if model_uri.startswith("s3://"):
            return self._download_from_s3(model_uri)

        if model_uri.startswith("file://"):
            model_uri = model_uri[7:]

        path = Path(model_uri)
        if not path.exists():
            raise DataSourceError(f"ONNX model not found: {model_uri}")
        return str(path)

    @staticmethod
    def _build_s3_client(s3_config: dict[str, Any]) -> Any:
        """Build a boto3 S3 client from config dict + environment variables."""
        import os

        import boto3

        endpoint = (
            s3_config.get("endpoint")
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("S3_ENDPOINT")
        )
        access_key = s3_config.get("access_key_id") or os.environ.get(
            "AWS_ACCESS_KEY_ID"
        )
        secret_key = s3_config.get("secret_access_key") or os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        )
        region = s3_config.get("region") or os.environ.get("AWS_REGION", "us-east-1")

        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def _download_from_s3(self, s3_uri: str) -> str:
        """Download the ONNX model from S3 to a local temp directory."""
        import tempfile
        import uuid

        if importlib.util.find_spec("boto3") is None:
            raise JobConfigurationError("boto3 is required for S3 model download.")

        cache_dir = Path(tempfile.gettempdir()) / "tributo_onnx" / str(uuid.uuid4())
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / "model.onnx"

        path = s3_uri.removeprefix("s3://")
        bucket, key = path.split("/", 1)

        client = self._build_s3_client(self._s3_config)
        logger.info("Downloading ONNX model from %s", s3_uri)
        try:
            client.download_file(bucket, key, str(local_path))
        except Exception as e:
            raise DataSourceError(
                f"Failed to download ONNX model from {s3_uri}: {e}"
            ) from e
        logger.info("Model downloaded to %s", local_path)
        return str(local_path)

    def _init_session(self, model_path: str) -> None:
        """Initialize an ONNX Runtime InferenceSession."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime is required. Install with: uv sync") from e

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self.session = ort.InferenceSession(model_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            "ONNX session ready: input=%s shape=%s outputs=%s",
            self.input_name,
            input_shape,
            output_names,
        )

    def close(self) -> None:
        """Release bundle resources (idempotent).

        Call when the Ray Actor's lifetime ends (e.g. from a driver-side
        shutdown helper); prediction keeps working after close because
        the model is in memory.  No-op on the legacy ``model_uri`` path.
        """
        if self._runtime is not None:
            self._runtime.close()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on a batch.

        Args:
            batch: Columnar batch from Ray Data (numpy format).
                   Columns are selected in feature_names order; the batch may contain extra columns.

        Returns:
            Original batch + prediction_column.
        """
        if self.feature_names:
            missing = [name for name in self.feature_names if name not in batch]
            if missing:
                raise KeyError(
                    f"Batch missing required feature columns: {missing}. "
                    f"Available: {list(batch.keys())}"
                )
            features = np.column_stack(
                [batch[name] for name in self.feature_names]
            ).astype(np.float32)
        else:
            # Fallback: compatibility with old models (no feature_names metadata)
            feature_keys = [k for k in batch if k != self.prediction_column]
            features = np.column_stack([batch[k] for k in feature_keys]).astype(
                np.float32
            )

        if self._runtime is not None:
            result = self._runtime.predict({self.input_name: features})
            outputs = [np.asarray(result[n]) for n in self._output_names]
        else:
            outputs = self.session.run(None, {self.input_name: features})

        if self.return_probs and len(outputs) >= 2:
            # XGBoost classification model: outputs[1] is the probability matrix
            predictions = np.asarray(outputs[1], dtype=np.float32)
        else:
            predictions = np.asarray(outputs[0], dtype=np.float32)

        batch[self.prediction_column] = predictions
        return batch


def _feature_names_from_bundle(runtime: "BundleModelRuntime") -> list[str]:
    """Feature column names for a bundle: ONNX metadata first, manifest
    signature as fallback.

    The manifest signature is a *tensor*-level contract (names of the
    model graph inputs), while batch inference selects Dataset columns
    by *table* feature names.  XGBoost ONNX bundles record the original
    feature names in ONNX metadata (``feature_names``), which is exactly
    what the legacy raw-path reads — so the bundle path reads the same
    metadata first and only falls back to the signature.
    """
    metadata_names = XGBoostONNXPredictor._load_feature_names(
        runtime.resolved_artifact.entrypoint_path
    )
    if metadata_names:
        return metadata_names
    fields = runtime.manifest.input_signature.input_fields
    return [f.name for f in fields]
