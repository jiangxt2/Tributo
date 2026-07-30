"""Storage profile resolution.

Maps a profile name to S3-compatible connection parameters without
embedding credentials in model configs or manifests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tributo.util.annotations import PublicAPI


@PublicAPI(stability="beta")
@dataclass(frozen=True)
class StorageProfile:
    """Resolved S3-compatible storage parameters."""

    endpoint: str | None = None
    region: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    use_ssl: bool = True
    path_style: bool = False
    profile_name: str | None = None

    def to_boto3_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for ``boto3.client("s3", **kwargs)``."""
        kwargs: dict[str, Any] = {}
        if self.endpoint:
            kwargs["endpoint_url"] = self.endpoint
        if self.region:
            kwargs["region_name"] = self.region
        if self.access_key_id:
            kwargs["aws_access_key_id"] = self.access_key_id
        if self.secret_access_key:
            kwargs["aws_secret_access_key"] = self.secret_access_key
        return kwargs


@PublicAPI(stability="beta")
class StorageProfileResolver:
    """Resolves a profile *name* to connection parameters.

    Built-in resolution order:

    1. Environment variable ``TRIBUTO_STORAGE_PROFILE_<NAME>`` — JSON string.
    2. ``boto3`` standard credential chain (env vars, ``~/.aws/credentials``,
       instance metadata).

    Plugins may register additional resolvers via entry-point
    ``tributo.storage_resolvers``.
    """

    def __init__(self) -> None:
        self._resolvers: list[Any] = []

    def resolve(self, profile: str | None) -> StorageProfile:
        """Resolve *profile* name to a ``StorageProfile``.

        When *profile* is ``None`` returns default credentials from the
        environment / boto3 chain.
        """
        if profile is None:
            return StorageProfile()

        # Check TRIBUTO_STORAGE_PROFILE_<NAME> env var.
        env_key = f"TRIBUTO_STORAGE_PROFILE_{profile.upper()}"
        raw = os.environ.get(env_key)
        if raw is not None:
            import json

            return StorageProfile(**json.loads(raw))

        # Fallback: use the name as the boto3 profile name.
        return StorageProfile(profile_name=profile)
