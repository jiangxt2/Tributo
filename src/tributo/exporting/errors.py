"""Error-message helpers — safe, sanitised failure text.

Plan requirement: ``FailureInfo.message`` must not leak credentials,
signed query strings, or machine-local absolute paths.
"""

from __future__ import annotations

import re

# s3://user:pass@host/... → s3://***@host/...
_URI_CREDENTIALS_RE = re.compile(r"(s3://)[^/@\s]+@", re.IGNORECASE)
# AWS signed query parameters (?X-Amz-Signature=...) — redact the value.
_SIGNED_QUERY_RE = re.compile(r"[?&]X-Amz-[A-Za-z0-9-]+=[^&\s]*")
# Machine-local absolute paths (macOS / Linux / tmp).
_LOCAL_PATH_RE = re.compile(r"(?:/Users/|/home/|/tmp/)[^\s'\"]*")


def sanitize_error_message(message: str) -> str:
    """Strip credentials, signed query params, and local paths from *message*."""
    cleaned = _URI_CREDENTIALS_RE.sub(r"\1***@", message)
    cleaned = _SIGNED_QUERY_RE.sub("?<redacted>", cleaned)
    cleaned = _LOCAL_PATH_RE.sub("<local-path>", cleaned)
    return cleaned
