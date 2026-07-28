"""Text embedding module for batch and online inference.

Public API:
    - registry: ModelSpec, get_spec, list_models, register
    - job_runner: submit_embedding_job
    - serve_runner: start_embedding_serving, stop_embedding_serving
    - schema: EmbedRequest, EmbedResponse
"""

from __future__ import annotations

from tributo.embeddings.job_runner import submit_embedding_job
from tributo.embeddings.registry import (
    ModelSpec,
    get_spec,
    list_models,
    register,
)
from tributo.embeddings.schema import EmbedRequest, EmbedResponse
from tributo.embeddings.serve_runner import (
    get_embedding_serving_status,
    start_embedding_serving,
    stop_embedding_serving,
)

__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "ModelSpec",
    "get_embedding_serving_status",
    "get_spec",
    "list_models",
    "register",
    "start_embedding_serving",
    "stop_embedding_serving",
    "submit_embedding_job",
]
