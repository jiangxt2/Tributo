"""Identity mining online inference service.

Loads preprocessing parameters and ONNX model, provides online inference capabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tributo.serving.model_deployment import ONNXModel
from tributo.training.features.column_types import (
    DenseFeat,
    SparseFeat,
    features_from_dicts,
)
from tributo.training.features.transformer import FeatureTransformer

logger = logging.getLogger(__name__)


class IdentityPredictor:
    """Identity mining online predictor.

    Loads ONNX model and preprocessing parameters, accepts raw feature dictionaries,
    performs preprocessing then calls ONNXModel for inference, returns probabilities.

    Attributes:
        model: ONNX inference model.
        transformer: Feature preprocessor.
        features: Feature column configuration list.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        feature_config_path: str | Path | None = None,
        *,
        bundle_uri: str | None = None,
        role: str = "inference",
        unsafe: bool = False,
        storage_profile: str | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: ONNX model file path or directory containing the complete model package.
            preprocessor_path: Preprocessing parameter file path (optional, only needed when model_path is a file).
            feature_config_path: Feature configuration file path (optional, only needed when model_path is a file).
            bundle_uri: Published bundle URI (stable entry point). When set,
                the preprocessor and feature config are located by artifact
                role inside the bundle instead of the paths above.
            role: Artifact role to serve; defaults to ``inference``.
            unsafe: Permit loading bundles without typed signatures or
                flavors that are not safe.
            storage_profile: Storage profile name for S3 bundles.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if (model_path is None) == (bundle_uri is None):
            raise ValueError(
                "exactly one of 'model_path' (legacy) or 'bundle_uri' must be provided"
            )

        self._runtime: Any = None
        if bundle_uri is not None:
            self._open_bundle(
                bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
            )
            return

        assert model_path is not None  # guarded by the exclusivity check above
        model_path = Path(model_path)

        # If it's a directory, auto-locate files in the model package
        if model_path.is_dir():
            onnx_file = model_path / "model.onnx"
            if not onnx_file.exists():
                raise FileNotFoundError(f"ONNX model not found in {model_path}")
            model_path = onnx_file

            if preprocessor_path is None:
                preprocessor_path = model_path.parent / "preprocessor.json"
            if feature_config_path is None:
                feature_config_path = model_path.parent / "feature_config.json"

        # Load ONNX model
        self.model = ONNXModel(str(model_path))
        logger.info("Loaded ONNX model from %s", model_path)

        # Load preprocessor
        if preprocessor_path is not None:
            preprocessor_path = Path(preprocessor_path)
            if preprocessor_path.exists():
                self.transformer = FeatureTransformer.load(preprocessor_path)
                logger.info("Loaded preprocessor from %s", preprocessor_path)
            else:
                self.transformer = None
                logger.warning("Preprocessor file not found: %s", preprocessor_path)
        else:
            self.transformer = None

        # Load feature config
        if feature_config_path is not None:
            feature_config_path = Path(feature_config_path)
            if feature_config_path.exists():
                config = json.loads(feature_config_path.read_text())
                self.features = self._parse_features(config)
                logger.info("Loaded feature config from %s", feature_config_path)
            else:
                self.features = []
                logger.warning("Feature config file not found: %s", feature_config_path)
        else:
            self.features = []

    def _open_bundle(
        self,
        bundle_uri: str,
        *,
        role: str,
        unsafe: bool,
        storage_profile: str | None,
    ) -> None:
        """Load the model package through the shared BundleModelLoader.

        Auxiliary files are located by artifact role inside the verified
        bundle: ``preprocessor.json`` (role ``preprocessor``) and
        ``model_config.json`` (role ``config``, carrying the ``features``
        list). First-party DNN/PU bundles require both files and matching
        feature semantics; generic ONNX bundles retain the legacy optional
        auxiliary-file behaviour.
        """
        from tributo.exporting.runtime import BundleModelLoader

        loader = BundleModelLoader()
        self._runtime = loader.open(
            bundle_uri, role=role, unsafe=unsafe, storage_profile=storage_profile
        )
        self.model = _BundleModelAdapter(self._runtime)
        logger.info("Loaded bundle model from %s (role=%r)", bundle_uri, role)

        self.transformer = None
        self.features: list[SparseFeat | DenseFeat] = []
        self._bundle_preprocessor_loaded = False
        self._bundle_feature_config_loaded = False

        # Auxiliary-file parsing must not leak the model runtime: any
        # failure below (bad JSON, invalid preprocessor) closes it
        # deterministically before re-raising.  __del__ is only a
        # best-effort fallback, never a substitute for this.
        try:
            # 1) Auxiliary files inside the model artifact (file role).
            self._load_aux_files(
                self._runtime.resolved_artifact, self._runtime.artifact
            )

            # 2) Auxiliary files published as separate artifacts — locate
            # them by manifest role, never by guessing inside the model
            # artifact.
            for aux_role in ("preprocessor", "feature_config", "config"):
                if aux_role not in self._runtime.manifest.roles:
                    continue
                if (
                    self._runtime.manifest.roles[aux_role]
                    == self._runtime.artifact.name
                ):
                    continue  # already handled via the model artifact
                self._load_aux_artifact(bundle_uri, aux_role, storage_profile)
            self._validate_bundle_preprocessing_contract()
        except BaseException:
            self._runtime.close()
            raise

    def _load_aux_artifact(
        self,
        bundle_uri: str,
        aux_role: str,
        storage_profile: str | None,
    ) -> None:
        """Open an auxiliary artifact by manifest role and read its files."""
        from tributo.exporting.bundle_reader import BundleReader

        reader = BundleReader()
        with reader.open_artifact(
            bundle_uri,
            role=aux_role,
            storage_profile=storage_profile,
            # Same snapshot as the model runtime — never re-read.
            manifest=self._runtime.manifest,
            manifest_bytes=self._runtime.manifest_bytes,
        ) as aux:
            self._load_aux_files(aux, aux.descriptor)
            logger.info("Loaded auxiliary artifact role=%r", aux_role)

    def _load_aux_files(self, resolved: Any, artifact: Any) -> None:
        """Read preprocessor/config files from an artifact (by file role)."""
        for file_entry in artifact.files:
            if file_entry.role == "preprocessor":
                self.transformer = FeatureTransformer.load(
                    resolved.path_for(file_entry.relative_path)
                )
                self._bundle_preprocessor_loaded = True
                logger.info("Loaded preprocessor from %s", file_entry.relative_path)
            elif file_entry.role == "config":
                config = json.loads(
                    resolved.path_for(file_entry.relative_path).read_text()
                )
                features_cfg = (
                    config.get("features") if isinstance(config, dict) else config
                )
                if isinstance(features_cfg, list) and features_cfg:
                    self.features = self._parse_features(features_cfg)
                    self._bundle_feature_config_loaded = True
                    logger.info(
                        "Loaded feature config from %s", file_entry.relative_path
                    )

    def _validate_bundle_preprocessing_contract(self) -> None:
        """Fail closed when a first-party DNN/PU Bundle is incomplete."""
        source_kind = self._runtime.manifest.source_info.source_kind
        if source_kind not in {"dnn_result", "pu_result"}:
            return
        if not self._bundle_preprocessor_loaded or self.transformer is None:
            raise ValueError(
                f"{source_kind} Bundle requires a file with role 'preprocessor'"
            )
        if not self._bundle_feature_config_loaded or not self.features:
            raise ValueError(
                f"{source_kind} Bundle requires feature configuration in a file "
                "with role 'config'"
            )
        if self.transformer.features != self.features:
            raise ValueError(
                f"{source_kind} Bundle preprocessing features do not match its "
                "model feature configuration"
            )

        signature_fields = self._runtime.manifest.input_signature.input_fields
        signature_names = tuple(field.name for field in signature_fields)
        feature_names = tuple(feature.name for feature in self.features)
        if signature_names != feature_names:
            raise ValueError(
                f"{source_kind} Bundle preprocessing features {feature_names!r} do "
                f"not match Manifest input signature {signature_names!r}"
            )
        for feature, field in zip(self.features, signature_fields, strict=True):
            expected_dtype = "int64" if isinstance(feature, SparseFeat) else "float32"
            expected_shape: tuple[int | str, ...] = ("batch",)
            if isinstance(feature, DenseFeat) and feature.dimension > 1:
                expected_shape += (feature.dimension,)
            if field.dtype != expected_dtype or field.shape != expected_shape:
                raise ValueError(
                    f"{source_kind} Bundle feature {feature.name!r} expects "
                    f"dtype={expected_dtype!r}, shape={expected_shape!r}, but its "
                    f"Manifest signature declares dtype={field.dtype!r}, "
                    f"shape={field.shape!r}"
                )

    def close(self) -> None:
        """Release bundle resources (idempotent).

        Call when the predictor's lifetime ends; prediction keeps working
        after close (in-memory model contract).  No-op on the legacy
        ``model_path`` path.
        """
        if self._runtime is not None:
            self._runtime.close()

    def _parse_features(
        self, config: list[dict[str, Any]]
    ) -> list[SparseFeat | DenseFeat]:
        """Parse feature configuration."""
        return features_from_dicts(config)

    def predict(
        self,
        features: dict[str, Any],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Predict a single sample.

        Args:
            features: Raw feature dictionary, key is feature name, value is feature value.
            threshold: Classification threshold.

        Returns:
            Dictionary containing prediction results:
            - probability: Prediction probability
            - prediction: Predicted class (0 or 1)
            - features: Preprocessed features (optional)
        """
        # Preprocessing
        if self.transformer is not None:
            # Convert single sample to array format
            input_data = {k: np.array([v]) for k, v in features.items()}
            processed = self.transformer.transform(input_data)
        else:
            # No preprocessor, use raw features directly
            processed = {
                k: np.array([v], dtype=np.float32 if isinstance(v, float) else np.int64)
                for k, v in features.items()
            }

        # Inference - using ONNXModel public method
        ort_inputs = {}
        for name, value in processed.items():
            if isinstance(value, np.ndarray):
                if value.dtype in (np.int64, np.int32):
                    ort_inputs[name] = value.astype(np.int64)
                else:
                    ort_inputs[name] = value.astype(np.float32)
            else:
                ort_inputs[name] = np.array([value], dtype=np.float32)

        probabilities = self._predict_probabilities(ort_inputs)
        if probabilities.size != 1:
            raise ValueError(
                "IdentityPredictor.predict expected one output probability, "
                f"got shape {probabilities.shape}"
            )
        prob = float(probabilities.reshape(-1)[0])

        return {
            "probability": prob,
            "prediction": 1 if prob >= threshold else 0,
        }

    def predict_batch(
        self,
        features_list: list[dict[str, Any]],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Batch prediction.

        Args:
            features_list: List of raw feature dictionaries.
            threshold: Classification threshold.

        Returns:
            List of prediction results.
        """
        if not features_list:
            return []

        # Preprocessing
        if self.transformer is not None:
            # Convert multiple samples to array format
            keys = features_list[0].keys()
            input_data = {k: np.array([f[k] for f in features_list]) for k in keys}
            processed = self.transformer.transform(input_data)
        else:
            keys = features_list[0].keys()
            processed = {}
            for k in keys:
                values = [f[k] for f in features_list]
                sample_val = values[0]
                if isinstance(sample_val, float):
                    processed[k] = np.array(values, dtype=np.float32)
                else:
                    processed[k] = np.array(values, dtype=np.int64)

        # Inference - using ONNXModel public method
        ort_inputs = {}
        for name, value in processed.items():
            if isinstance(value, np.ndarray):
                if value.dtype in (np.int64, np.int32):
                    ort_inputs[name] = value.astype(np.int64)
                else:
                    ort_inputs[name] = value.astype(np.float32)
            else:
                ort_inputs[name] = np.array(value, dtype=np.float32)

        probs = self._predict_probabilities(ort_inputs).reshape(-1).tolist()

        # Build results
        return [
            {
                "probability": float(p),
                "prediction": 1 if float(p) >= threshold else 0,
            }
            for p in probs
        ]

    def _predict_probabilities(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Select probability outputs by name, or apply sigmoid to logits."""
        output_names = self._output_names()
        probability_index = next(
            (
                index
                for index, name in enumerate(output_names)
                if "probab" in name.lower()
            ),
            None,
        )
        if probability_index is not None:
            raw = np.asarray(
                self.model.predict_numpy(inputs, output_index=probability_index)
            )
            if raw.ndim >= 2 and raw.shape[-1] == 2:
                return np.asarray(raw[..., 1], dtype=np.float64)
            if raw.ndim >= 1 and raw.shape[-1] == 1:
                return np.asarray(raw[..., 0], dtype=np.float64)
            return np.asarray(raw, dtype=np.float64)

        logits_index = next(
            (
                index
                for index, name in enumerate(output_names)
                if "logit" in name.lower() or name.lower() == "output"
            ),
            None,
        )
        if logits_index is None:
            raise ValueError(
                "IdentityPredictor requires a probability or logits output; "
                f"available outputs are {list(output_names)}"
            )
        logits = np.asarray(self.model.predict_numpy(inputs, output_index=logits_index))
        if logits.ndim >= 2 and logits.shape[-1] != 1:
            raise ValueError(
                "IdentityPredictor only supports binary logits with one value per row; "
                f"got shape {logits.shape}"
            )
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-values))

    def _output_names(self) -> tuple[str, ...]:
        names = getattr(self.model, "output_names", None)
        if names is not None:
            return tuple(names)
        session = getattr(self.model, "_session", None)
        if session is None:
            raise ValueError("Loaded ONNX model does not expose output metadata")
        return tuple(output.name for output in session.get_outputs())


class _BundleModelAdapter:
    """Adapts a ``BundleModel`` to ONNXModel's ``predict_numpy`` interface.

    Lets the existing ``predict`` / ``predict_batch`` flows stay untouched
    while the underlying model comes from a bundle runtime.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def output_names(self) -> tuple[str, ...]:
        """Expose runtime output names for semantic result selection."""
        return tuple(self._runtime.model.output_names)

    def predict_numpy(
        self,
        inputs: dict[str, np.ndarray],
        output_index: int = 0,
    ) -> np.ndarray:
        """Run inference and return the selected output array."""
        result = self._runtime.predict(inputs)
        name = self._runtime.model.output_names[output_index]
        return np.asarray(result[name])

    def health(self) -> dict[str, Any]:
        """Health check view."""
        return {
            "status": "healthy",
            "input_names": list(self._runtime.model.input_names),
        }
