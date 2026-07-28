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
        model_path: str | Path,
        preprocessor_path: str | Path | None = None,
        feature_config_path: str | Path | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model_path: ONNX model file path or directory containing the complete model package.
            preprocessor_path: Preprocessing parameter file path (optional, only needed when model_path is a file).
            feature_config_path: Feature configuration file path (optional, only needed when model_path is a file).

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
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

        # Execute inference (using logits output)
        logits = self.model.predict_numpy(ort_inputs, output_index=0)
        logit_value = float(logits[0]) if logits.ndim > 0 else float(logits)

        # Apply sigmoid to get probability
        prob = 1.0 / (1.0 + np.exp(-logit_value))

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

        # Execute inference (using logits output)
        logits = self.model.predict_numpy(ort_inputs, output_index=0)

        # Apply sigmoid to get probability
        if isinstance(logits, np.ndarray):
            if logits.ndim == 0:
                probs = [1.0 / (1.0 + np.exp(-float(logits)))]
            else:
                probs = (1.0 / (1.0 + np.exp(-logits))).tolist()
        else:
            # Fallback: non-ndarray type (e.g., list, float)
            logit_val = (
                float(logits) if not isinstance(logits, list) else float(logits[0])
            )
            probs = [1.0 / (1.0 + np.exp(-logit_val))]

        # Build results
        return [
            {
                "probability": float(p),
                "prediction": 1 if float(p) >= threshold else 0,
            }
            for p in probs
        ]
