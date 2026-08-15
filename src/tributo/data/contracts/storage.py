"""Neutral storage configuration contracts shared by data consumers."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.util.annotations import PublicAPI

__all__ = ["S3Config"]


@PublicAPI(stability="beta")
class S3Config(StrictConfigModel):
    """S3 connection configuration, shared across data modules.

    Environment variable fallbacks:
    ``S3_ENDPOINT`` / ``AWS_ENDPOINT_URL`` → endpoint,
    ``AWS_ACCESS_KEY_ID`` → access_key_id,
    ``AWS_SECRET_ACCESS_KEY`` → secret_access_key,
    ``AWS_REGION`` → region.
    """

    access_key_id: Optional[str] = Field(
        default=None, description="AWS Access Key ID", repr=False
    )
    secret_access_key: Optional[str] = Field(
        default=None, description="AWS Secret Access Key", repr=False
    )
    endpoint: Optional[str] = Field(
        default=None, description="S3 endpoint URL", repr=False
    )
    region: Optional[str] = Field(default=None, description="AWS region")
