"""Engine-native runtime credential boundaries for data adapters.

This module is the approved neutral bridge between credential-free control
plane contracts and engine-native runtime objects.  It may import shared
``S3Config``/``StorageProfile`` types, but writing code must not use this
bridge to reach legacy connector execution or storage-format data-plane code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_native_runtime_object(value: Any) -> bool:
    """Return whether *value* is an approved in-process runtime object."""
    from tributo._common.storage_profiles import StorageProfile
    from tributo.data.base import S3Config

    return isinstance(value, (S3Config, StorageProfile))


def coerce_s3_runtime(value: Any) -> Any:
    """Return an approved S3 runtime object for native binding execution."""
    from tributo._common.storage_profiles import StorageProfile
    from tributo.data.base import S3Config

    if isinstance(value, (S3Config, StorageProfile)):
        return value
    if isinstance(value, Mapping):
        try:
            return S3Config.model_validate(value)
        except ValueError:
            raise ValueError("runtime s3 configuration is invalid") from None
    raise ValueError("runtime s3 configuration is invalid")


def credential_free_runtime_value(value: Any) -> Any:
    """Return a serializable runtime view with secret values removed."""
    from tributo._common.storage_profiles import StorageProfile
    from tributo.data.base import S3Config

    if isinstance(value, S3Config):
        return {
            key: item
            for key, item in value.model_dump(mode="python").items()
            if key not in {"access_key_id", "secret_access_key"} and item is not None
        }
    if isinstance(value, StorageProfile):
        return {
            key: item
            for key, item in {
                "endpoint": value.endpoint,
                "region": value.region,
                "use_ssl": value.use_ssl,
                "path_style": value.path_style,
                "profile_name": value.profile_name,
            }.items()
            if item is not None
        }
    if isinstance(value, Mapping):
        return {
            str(key): credential_free_runtime_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [credential_free_runtime_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(
        "runtime options must contain references or approved native runtime objects"
    )
