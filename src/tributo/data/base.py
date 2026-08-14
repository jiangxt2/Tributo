"""DataConnector abstract base class and shared configuration types.

DataConnector is a configuration-driven data access facade — a higher-level
abstraction than Ray Datasource (the low-level protocol interface).  It
parses format-specific configs and returns ``ray.data.Dataset`` or writes
data consistently across storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

from tributo._common.config import StrictConfigModel
from tributo.data.contracts.modes import WriteMode
from tributo.util.annotations import PublicAPI

if TYPE_CHECKING:
    import ray.data


__all__ = ["DataConnector", "S3Config", "WriteMode"]


@PublicAPI(stability="beta")
class S3Config(StrictConfigModel):
    """S3 connection configuration, shared across all data modules.

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


@PublicAPI(stability="beta")
class DataConnector(ABC):
    """Abstract base class for data connectors.

    Subclasses must implement ``read()`` and ``write()``.  The API uses
    ``**kwargs`` with loose signatures — each subclass defines its own
    Pydantic config model for parameter validation.

    Example::

        connector = get_connector("parquet")
        ds = connector.read(path="s3://bucket/data.parquet", s3=S3Config(...))
        connector.write(ds, path="s3://bucket/output", mode=WriteMode.OVERWRITE)
    """

    @abstractmethod
    def read(self, **kwargs: Any) -> ray.data.Dataset:
        """Read data and return a Ray Dataset.

        Args:
            **kwargs: Format-specific config fields, parsed by the
                subclass Pydantic model.

        Returns:
            A lazy ``ray.data.Dataset``.
        """

    @abstractmethod
    def write(self, dataset: ray.data.Dataset, **kwargs: Any) -> None:
        """Write a Ray Dataset to the target storage.

        Args:
            dataset: The dataset to write.
            **kwargs: Format-specific config fields, parsed by the
                subclass Pydantic model.
        """

    def exists(self, **kwargs: Any) -> bool:
        """Check whether the target path exists.

        Args:
            **kwargs: Format-specific config fields.

        Returns:
            ``True`` if the path exists; ``False`` by default.
        """
        return False
