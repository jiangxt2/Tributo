"""Common utilities shared across Tributo.

This module contains shared utilities at the developer API level.
"""

from __future__ import annotations

import os

from tributo._common.logging import configure_logging
from tributo._common.runtime_env import (
    build_runtime_env,
    find_project_root,
)
from tributo._common.storage import (
    download_from_s3,
    get_boto3_client,
    get_boto3_client_from_config,
    parse_s3_url,
    resolve_to_local,
    upload_file,
    write_bytes,
    write_json,
)

DEFAULT_DASHBOARD_URL = os.environ.get("DEFAULT_DASHBOARD_URL", "http://127.0.0.1:8265")

__all__ = [
    "build_runtime_env",
    "configure_logging",
    "download_from_s3",
    "find_project_root",
    "get_boto3_client",
    "get_boto3_client_from_config",
    "parse_s3_url",
    "resolve_to_local",
    "upload_file",
    "write_bytes",
    "write_json",
]
