"""Request/response schemas for the embedding HTTP service."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
class EmbedRequest(BaseModel):
    """HTTP request body for batch text embedding.

    Attributes:
        texts: List of raw text strings to encode.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of raw text strings to encode",
    )


@PublicAPI(stability="beta")
class EmbedResponse(BaseModel):
    """HTTP response body for batch text embedding.

    Attributes:
        embeddings: List of dense vectors, one per input text.
        model: Name of the model used for encoding.
        dim: Dimensionality of each vector.
    """

    embeddings: list[list[float]]
    model: str
    dim: int
