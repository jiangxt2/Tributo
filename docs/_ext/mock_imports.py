"""Third-party modules mocked by Sphinx documentation builds.

This is the single mock inventory used by Sphinx, sphinx-click, static
validation, and tests. First-party ``tributo`` modules must never be added.
"""

from __future__ import annotations

DOC_MOCK_IMPORTS: tuple[str, ...] = (
    "accelerate",
    "boto3",
    "clickhouse_connect",
    "confluent_kafka",
    "daft",
    "dowhy",
    "econml",
    "grpc",
    "lance",
    "mlflow",
    "numpy",
    "onnx",
    "onnxmltools",
    "onnxruntime",
    "onnxscript",
    "pandas",
    "pyarrow",
    "pyiceberg",
    "pymysql",
    "ray",
    "s3fs",
    "safetensors",
    "skl2onnx",
    "sklearn",
    "starlette",
    "torch",
    "torch_geometric",
    "transformers",
    "xgboost",
)
