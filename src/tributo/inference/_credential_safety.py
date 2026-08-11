"""Internal credential safety helpers for inference boundaries and diagnostics.

Exception messages are untrusted.  Diagnostic redaction intentionally removes
machine-local absolute paths as well as known credential forms.  Losing the
exact path is preferable to exposing usernames, temporary-directory layouts,
or host filesystem details in persisted job logs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlsplit

_CREDENTIAL_KEYS = frozenset(
    {
        "password",
        "secret",
        "secretaccesskey",
        "accesskeyid",
        "token",
        "sessiontoken",
        "credential",
        "clientsecret",
        "apikey",
        "apitoken",
        "authorization",
        "cookie",
        "oauthtoken",
        "privatekey",
        "refreshtoken",
        "setcookie",
        "authtoken",
    }
)

_SAFE_CREDENTIAL_LIKE_KEYS = frozenset(
    {
        "inputsignature",
        "outputsignature",
    }
)

_URI_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")
_QUERY_PARAMETER_RE = re.compile(r"([?&])([^=&\s]+)=([^&\s]*)")
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<quote>[\"']?)(?P<key>\b("
    r"[a-z0-9_-]*(?:password|secret|token|credential|signature)|"
    r"[a-z0-9_-]*api[_-]?key|[a-z0-9_-]*access[_-]?key[_-]?id|"
    r"[a-z0-9_-]*secret[_-]?access[_-]?key|"
    r"authorization|cookie|private[_-]?key|set[_-]?cookie|"
    r"x-amz-[a-z0-9-]+"
    r")\b)(?P=quote)(?P<separator>\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|(?:bearer|basic)\s+[^\s,;&]+|[^\s,;&]+)"
)
_AUTH_SCHEME_RE = re.compile(
    r"(?i)\b(?P<scheme>bearer|basic)\s+(?P<credential>[A-Za-z0-9._~+/=-]+)"
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_LOCAL_PATH_RE = re.compile(r"(?:/Users/|/home/|/tmp/|/private/tmp/)[^\s'\"]*")


def credential_paths(value: Any, path: str = "input") -> set[str]:
    """Find explicit credential fields without reading or returning their values."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if item not in (None, "") and _is_credential_key(str(key)):
                found.add(child)
            found.update(credential_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found.update(credential_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            found.add(path)
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_credential_key(key):
                found.add(path)
    return found


def safe_exception_summary(exc: BaseException) -> str:
    """Retain useful failure context while removing credentials and host paths."""
    message = " ".join(str(exc).splitlines()).strip()
    if not message:
        return type(exc).__name__
    cleaned = _URI_USERINFO_RE.sub(r"\g<scheme><redacted>@", message)
    cleaned = _QUERY_PARAMETER_RE.sub(_redact_query_parameter, cleaned)
    cleaned = _SENSITIVE_ASSIGNMENT_RE.sub(_redact_assignment, cleaned)
    cleaned = _AUTH_SCHEME_RE.sub(r"\g<scheme> <redacted>", cleaned)
    cleaned = _JWT_RE.sub("<redacted>", cleaned)
    cleaned = _AWS_ACCESS_KEY_RE.sub("<redacted>", cleaned)
    cleaned = _LOCAL_PATH_RE.sub("<local-path>", cleaned)
    return cleaned[:1000]


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _is_credential_key(value: str) -> bool:
    normalized = _normalized_key(value)
    if normalized in _SAFE_CREDENTIAL_LIKE_KEYS:
        return False
    return (
        normalized in _CREDENTIAL_KEYS
        or normalized.startswith(("xamz", "awsaccesskey", "awssecretaccesskey"))
        or normalized.endswith(
            (
                "password",
                "secret",
                "token",
                "credential",
                "signature",
                "apikey",
                "accesskeyid",
                "secretaccesskey",
            )
        )
    )


def _redact_query_parameter(match: re.Match[str]) -> str:
    separator, key, value = match.groups()
    if _is_credential_key(key):
        value = "<redacted>"
    return f"{separator}{key}={value}"


def _redact_assignment(match: re.Match[str]) -> str:
    quote = match.group("quote")
    key = match.group("key")
    if not _is_credential_key(key):
        return match.group(0)
    separator = match.group("separator")
    return f"{quote}{key}{quote}{separator}<redacted>"
