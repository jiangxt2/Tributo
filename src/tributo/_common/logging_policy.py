"""Bounded, serializable application logging configuration (no import effects)."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POLICY_ENV = "_TRIBUTO_LOG_POLICY"
USER_HOOK_ENV = "_TRIBUTO_USER_SETUP_MODULE"
JOB_HOOK_ENV = "_TRIBUTO_LOG_JOB_HOOK"
SETUP_HOOK = "tributo._common.worker_logging.setup"
LOG_ENV_KEYS = (
    "TRIBUTO_LOG_LEVEL",
    "TRIBUTO_LOG_FORMAT",
    "TRIBUTO_LOGGER_LEVELS",
)
_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_MAX_BYTES = 4096


def _error() -> ValueError:
    return ValueError("Invalid application logging configuration")


def _level(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in _LEVELS:
        raise _error()
    return value.upper()


def _modules(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or len(value) > 32:
        raise _error()
    result = []
    for name, level in value.items():
        if (
            not isinstance(name, str)
            or len(name) > 128
            or _NAME.fullmatch(name) is None
            or name.split(".")[0] in {"root", "ray", "torch"}
        ):
            raise _error()
        result.append((name, _level(level)))
    return tuple(sorted(result))


def _json(value: str) -> object:
    if len(value.encode("utf-8")) > _MAX_BYTES:
        raise _error()
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        raise _error() from None


@dataclass(frozen=True)
class LoggingPolicy:
    """Validated process policy; contains no handlers, streams, or identities."""

    level: str = "INFO"
    log_format: str = "json"
    logger_levels: tuple[tuple[str, str], ...] = ()

    def serialize(self) -> str:
        result = json.dumps(
            {
                "version": 1,
                "level": self.level,
                "format": self.log_format,
                "logger_levels": dict(self.logger_levels),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(result.encode("utf-8")) > _MAX_BYTES:
            raise _error()
        return result

    @classmethod
    def deserialize(cls, value: str) -> LoggingPolicy:
        raw = _json(value)
        if not isinstance(raw, dict) or set(raw) != {
            "version",
            "level",
            "format",
            "logger_levels",
        }:
            raise _error()
        if (
            type(raw["version"]) is not int
            or raw["version"] != 1
            or raw["format"] not in ("json", "text")
        ):
            raise _error()
        return cls(
            _level(raw["level"]),
            raw["format"],
            _modules(raw["logger_levels"]),
        )


def resolve_policy(
    environ: Mapping[str, str] | None = None, *, default_format: str = "json"
) -> LoggingPolicy:
    """Parse a frozen snapshot first, otherwise only the known environment keys."""
    env = os.environ if environ is None else environ
    if POLICY_ENV in env:
        return LoggingPolicy.deserialize(env[POLICY_ENV])
    if sum(len(env.get(key, "").encode("utf-8")) for key in LOG_ENV_KEYS) > _MAX_BYTES:
        raise _error()
    log_format = env.get("TRIBUTO_LOG_FORMAT", default_format)
    if log_format not in ("json", "text"):
        raise _error()
    return LoggingPolicy(
        _level(env.get("TRIBUTO_LOG_LEVEL", "INFO")),
        log_format,
        _modules(_json(env.get("TRIBUTO_LOGGER_LEVELS", "{}"))),
    )


def override_logging_policy(
    base: LoggingPolicy, env_vars: Mapping[str, str] | None
) -> LoggingPolicy:
    """Apply explicit child options without resetting the parent's SDK policy."""
    raw = {
        "TRIBUTO_LOG_LEVEL": base.level,
        "TRIBUTO_LOG_FORMAT": base.log_format,
        "TRIBUTO_LOGGER_LEVELS": json.dumps(dict(base.logger_levels)),
    }
    supplied = env_vars or {}
    raw.update({key: supplied[key] for key in LOG_ENV_KEYS if key in supplied})
    selected = resolve_policy(raw)
    if POLICY_ENV in supplied and supplied[POLICY_ENV] != selected.serialize():
        raise _error()
    return selected


def forward_logging_env(env_vars: Mapping[str, str] | None) -> dict[str, str]:
    """Copy explicit values first; never scan or copy the rest of the host env."""
    result = dict(env_vars or {})
    for key in LOG_ENV_KEYS:
        if key not in result and key in os.environ:
            result[key] = os.environ[key]
    # Validate only the application logging options before scheduling.
    raw = {key: result[key] for key in LOG_ENV_KEYS if key in result}
    policy = resolve_policy(raw)
    if POLICY_ENV in result and result[POLICY_ENV] != policy.serialize():
        raise _error()
    if USER_HOOK_ENV in result or JOB_HOOK_ENV in result:
        raise _error()
    return result


def logging_runtime_env(
    runtime_env: Mapping[str, Any] | None = None,
    *,
    policy: LoggingPolicy | None = None,
    jobs: bool = False,
) -> dict[str, Any]:
    """Add a process setup hook without mutating the caller's runtime env."""
    result = dict(runtime_env or {})
    env = dict(result.get("env_vars") or {})
    selected = policy or resolve_policy(
        env
        if POLICY_ENV in env
        else {**{k: os.environ[k] for k in LOG_ENV_KEYS if k in os.environ}, **env}
    )
    serialized = selected.serialize()
    if POLICY_ENV in env and env[POLICY_ENV] != serialized:
        raise _error()
    env[POLICY_ENV] = serialized
    previous = result.get("worker_process_setup_hook")
    if previous == SETUP_HOOK:
        result["env_vars"] = env
        return result
    if callable(previous):
        if jobs:
            raise ValueError("Ray Jobs logging setup requires a string hook")
        from tributo._common.worker_logging import compose_setup

        result["worker_process_setup_hook"] = compose_setup(selected, previous)
    else:
        if previous is not None:
            if (
                not isinstance(previous, str)
                or len(previous) > 256
                or _NAME.fullmatch(previous) is None
            ):
                raise _error()
            env[USER_HOOK_ENV] = previous
        result["worker_process_setup_hook"] = SETUP_HOOK
    if jobs:
        env[JOB_HOOK_ENV] = "1"
    result["env_vars"] = env
    return result


def driver_logging_runtime_env(
    *, policy: LoggingPolicy | None = None
) -> dict[str, Any]:
    """Jobs already inherit their submission hook; local child Jobs do not."""
    if os.environ.get(JOB_HOOK_ENV) in {"1", "external"}:
        return {}
    if policy is None:
        from tributo._common.logging import active_logging_policy

        policy = active_logging_policy()
    return {"runtime_env": logging_runtime_env(policy=policy)}


def submission_logging_env(
    entrypoint: str, runtime_env: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Configure only framework-owned Jobs; leave custom commands executable as-is."""
    result = dict(runtime_env or {})
    env = forward_logging_env(result.get("env_vars"))
    if env:
        result["env_vars"] = env
    try:
        command = shlex.split(entrypoint)
    except ValueError:
        raise ValueError("Invalid Ray Job entrypoint") from None
    owned = (
        len(command) >= 3
        and command[1] == "-m"
        and command[2]
        in {
            "tributo.inference.batch_job",
            "tributo.explainability.batch_job",
            "tributo.vector_index.job",
        }
    )
    if owned:
        return logging_runtime_env(result, policy=resolve_policy(env), jobs=True)
    if result.get("worker_process_setup_hook") is not None:
        # A custom Job's hook remains caller-owned. A later runtime manager
        # must inherit it rather than conflict with the injected Job config.
        result["env_vars"] = {**env, JOB_HOOK_ENV: "external"}
        return result
    if isinstance(runtime_env, dict) and env == runtime_env.get("env_vars", {}):
        return runtime_env
    return result if result or runtime_env is not None else None
