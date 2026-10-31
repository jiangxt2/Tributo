"""Execution-domain ownership and frozen producer policy. No optional imports here."""

from __future__ import annotations

import atexit
import importlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from tributo._common.reporting.config import (
    KNOWN_ENV,
    MAX_CONFIG_BYTES,
    POLICY_ENV,
    WORKER_HOOK,
    ReportingConfig,
    ReportingError,
    SenderConfig,
    load_config,
    resolve_reference,
    validate_sdk_environment,
)
from tributo._common.reporting.lifecycle import bounded_call

if TYPE_CHECKING:
    from tributo._common.reporting.collector import ManagedCollector
    from tributo._common.reporting.sender import Sender

_LOCK = threading.RLock()
_ACTIVE: ReportingSession | None = None
_REFERENCES = 0
_PRODUCER_ENTRY: ContextVar[bool] = ContextVar("tributo_producer_entry", default=False)
_P = ParamSpec("_P")
_R = TypeVar("_R")


class ReportingSession:
    """One explicit owner or a process-local producer; business results stay separate."""

    def __init__(
        self,
        config: ReportingConfig | None = None,
        *,
        producer: SenderConfig | None = None,
        service_lifetime: bool = False,
        startup_environment: Mapping[str, str] | None = None,
        producer_context: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.sender_config = producer or (config.sender if config else SenderConfig())
        self.role = "producer" if producer is not None else "owner"
        self.enabled = producer is not None or bool(config and config.enabled)
        self.service_lifetime = service_lifetime
        self.state = "starting" if self.enabled else "disabled"
        context_environment = {**os.environ, **(startup_environment or {})}
        self._context = {
            field: value
            for field, key in (
                ("run_id", "TRIBUTO_RUN_ID"),
                ("attempt_id", "TRIBUTO_ATTEMPT_ID"),
                ("submission_id", "TRIBUTO_SUBMISSION_ID"),
            )
            if self.enabled
            and producer is None
            and isinstance(value := context_environment.get(key), str)
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
        }
        self._context.update(producer_context or {})
        self.operation_id = self._context.get("operation_id", uuid.uuid4().hex)
        self._pid = os.getpid()
        self._environment: dict[str, str] = {}
        references = [self.sender_config.headers_ref]
        if config is not None:
            for profile in (config.read_profile, config.write_profile):
                if profile is not None:
                    references.extend([profile.username_ref, profile.password_ref])
        if self.enabled:
            for reference in references:
                if reference:
                    key = reference.removeprefix("env://")
                    value = (
                        startup_environment[key]
                        if startup_environment is not None
                        and key in startup_environment
                        else os.environ.get(key)
                    )
                    if isinstance(value, str):
                        self._environment[key] = value
        self._sender: Sender | None = None
        self._collector: ManagedCollector | None = None
        self._diagnostics: dict[str, int] = {}
        self._diagnostic_lock = threading.Lock()
        self._services: set[str] = set()
        self._closed = False

    def diagnose(self, code: str) -> None:
        with self._diagnostic_lock:
            count = self._diagnostics.get(code, 0) + 1
            self._diagnostics[code] = count
        if code not in {
            "reporting.checked",
            "producer.checked",
            "reporting.closed",
            "sender.warning",
        }:
            if self.state not in {"draining", "closed", "failed"}:
                self.state = "degraded"
        if count == 1:
            logger = logging.getLogger("tributo.reporting.local")
            logger.propagate = False
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                logger.addHandler(logging.StreamHandler())
            logger.log(
                logging.INFO
                if code in {"reporting.checked", "producer.checked", "reporting.closed"}
                else logging.WARNING,
                "Tributo reporting: %s role=%s",
                code,
                self.role,
            )

    def start(self, *, producer_best_effort: bool = False) -> None:
        if not self.enabled:
            return
        validate_sdk_environment()
        try:
            from tributo._common.reporting.sender import Sender
        except ImportError:
            raise ReportingError(
                "reporting.dependencies.missing_log_reporting_extra"
            ) from None
        deadline = time.monotonic() + (
            self.config.startup_timeout
            if self.config
            else self.sender_config.send_timeout
        )
        try:
            if self.role == "owner":
                if self.config is None or self.config.read_profile is None:
                    raise ReportingError("reporting.config.read_profile")
                resolve_reference(
                    self.config.read_profile.username_ref, environment=self._environment
                )
                resolve_reference(
                    self.config.read_profile.password_ref,
                    empty=True,
                    environment=self._environment,
                )
                # Validate SQL dependency before creating any owned subprocess.
                importlib.import_module("pymysql")

                if self.config.mode == "managed":
                    from tributo._common.reporting.collector import ManagedCollector

                    self._collector = ManagedCollector(
                        self.config, self.diagnose, environment=self._environment
                    )
                    self._collector.start(deadline)
            self._sender = Sender(
                self.sender_config,
                self.diagnose,
                environment=self._environment,
                attributes={"operation_id": self.operation_id, **self._context},
            )
            failures = self._sender.failures
            probe = "tributo.reporting.probe:" + uuid.uuid4().hex
            self._sender.emit(
                probe,
                {"event": "tributo.reporting.probe", "operation_id": self.operation_id},
            )
            self._sender.check_delivery(failures, deadline)
            if self.role == "owner":
                if not bounded_call(
                    lambda: self._readback(probe, deadline), deadline - time.monotonic()
                ):
                    raise ReportingError("reporting.probe.not_queryable")
            self._sender.attach()
            self.state = "ready"
            self.diagnose(
                "reporting.checked" if self.role == "owner" else "producer.checked"
            )
        except BaseException as error:
            if (
                producer_best_effort
                and self.role == "producer"
                and self._sender is not None
                and isinstance(error, Exception)
            ):
                # A dynamically created worker must continue attempting bounded
                # delivery after its initial ACK fails. Static SDK/config errors
                # occur before a Sender exists and remain visibly rejected.
                self._sender.attach()
                self.state = "degraded"
                self.diagnose("producer.probe_failed")
                return
            self.state = "failed"
            self.close()
            raise

    def _readback(self, probe: str, deadline: float) -> None:
        import pymysql

        profile = self.config.read_profile if self.config else None
        if profile is None:
            raise ReportingError("reporting.config.read_profile")
        username = resolve_reference(
            profile.username_ref, environment=self._environment
        )
        password = resolve_reference(
            profile.password_ref, empty=True, environment=self._environment
        )
        while time.monotonic() < deadline:
            remaining = max(1, min(5, int(deadline - time.monotonic())))
            try:
                with pymysql.connect(
                    host=profile.host,
                    port=profile.port,
                    user=username,
                    password=password,
                    database=profile.database,
                    connect_timeout=remaining,
                    read_timeout=remaining,
                    write_timeout=remaining,
                    ssl={"ca": profile.ca_file} if profile.ca_file else None,
                ) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(
                            f"SELECT 1 FROM `{profile.database}`.`{profile.table}` WHERE body = %s LIMIT 1",
                            (probe,),
                        )
                        if cursor.fetchone() is not None:
                            return
            except Exception:
                # Never surface SQL/client errors, endpoints or resolved credentials.
                pass
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise ReportingError("reporting.probe.not_queryable")

    def producer_policy(self) -> str | None:
        if not self.enabled or self._closed:
            return None
        from tributo._common.reporting.sender import CONTEXT, safe_attributes

        scoped = CONTEXT.get()
        context = safe_attributes(
            {"operation_id": self.operation_id, **self._context}
            if scoped is None
            else scoped
        )
        value = json.dumps(
            {
                "version": 1,
                "role": "producer",
                "sender": self.sender_config.model_dump(mode="json"),
                "context": {
                    key: item
                    for key, item in context.items()
                    if key in _POLICY_CONTEXT and isinstance(item, str)
                },
            },
            separators=(",", ":"),
        )
        if len(value.encode()) > MAX_CONFIG_BYTES:
            raise ReportingError("reporting.policy.too_large")
        return value

    def runtime_env(
        self, runtime_env: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.enabled and self._closed:
            raise ReportingError("reporting.session.closed")
        return merge_runtime_env(runtime_env)

    def snapshot(self) -> dict[str, Any]:
        sender = self._sender
        result: dict[str, Any] = {
            "state": self.state,
            "role": self.role,
            "enabled": self.enabled,
            "diagnostics": dict(self._diagnostics),
        }
        if sender:
            result.update(
                offered=sender.offered,
                otlp_accepted=sender.accepted,
                otlp_rejected=sender.rejected,
                failed_batches=sender.failures,
                overflow_events=sender.overflow_events,
            )
        # Queue, Collector acceptance and Doris committed counts are not inferable
        # from SDK flush success and are deliberately not manufactured here.
        return result

    def close(self) -> None:
        if self._closed or os.getpid() != self._pid:
            return
        self._closed = True
        if not self.enabled:
            return
        failed = self.state == "failed"
        self.state = "draining"
        deadline = time.monotonic() + self.sender_config.shutdown_timeout
        if self._services:
            try:
                from ray import serve

                for name in tuple(self._services):
                    try:
                        if not bounded_call(
                            partial(serve.delete, name), deadline - time.monotonic()
                        ):
                            self.diagnose("service.cleanup_incomplete")
                    except Exception:
                        self.diagnose("service.cleanup_failed")
                self._services.clear()
            except Exception:
                self.diagnose("service.cleanup_failed")
        try:
            try:
                if self._sender and not self._sender.close(
                    max(0, deadline - time.monotonic())
                ):
                    self.diagnose("sender.drain_incomplete")
            except Exception:
                self.diagnose("sender.cleanup_failed")
        finally:
            try:
                if self._collector and not self._collector.close(
                    max(0, deadline - time.monotonic())
                ):
                    self.diagnose("collector.drain_incomplete")
            except Exception:
                self.diagnose("collector.cleanup_failed")
            self.state = "failed" if failed else "closed"
            self.diagnose("reporting.closed")


def active_session() -> ReportingSession | None:
    return _ACTIVE if _ACTIVE is not None and _ACTIVE._pid == os.getpid() else None


def ray_options(*, driver: bool = False) -> dict[str, Any]:
    """Compose driver logging and optional reporting; preserve caller-owned tasks."""
    from tributo._common.logging_policy import driver_logging_runtime_env

    options = driver_logging_runtime_env() if driver else {}
    if active_session():
        options["runtime_env"] = merge_runtime_env(options.get("runtime_env"))
        if driver and (injected := os.environ.get("RAY_JOB_CONFIG_JSON_ENV_VAR")):
            # Ray 2.59 rejects even identical keys shared by Job and driver.
            # Keep the submission hook and inherited credentials; add only new
            # producer state, without enabling Ray's global override switch.
            inherited = json.loads(injected).get("runtime_env", {})
            runtime = options["runtime_env"]
            for key, value in inherited.items():
                if key != "env_vars" and runtime.get(key) == value:
                    runtime.pop(key, None)
            runtime["env_vars"] = {
                key: value
                for key, value in runtime.get("env_vars", {}).items()
                if key not in inherited.get("env_vars", {})
                or inherited["env_vars"][key] != value
            }
    return options


def submission_runtime_env(
    runtime_env: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    result = merge_runtime_env(runtime_env, for_job=True)
    if isinstance(runtime_env, dict) and result == runtime_env:
        return runtime_env
    return result if result or runtime_env is not None else None


def current_policy() -> str | None:
    session = active_session()
    return session.producer_policy() if session else os.environ.get(POLICY_ENV)


_POLICY_CONTEXT = frozenset(
    {"operation_id", "execution_id", "run_id", "attempt_id", "submission_id"}
)


def _decode_policy(value: str) -> tuple[SenderConfig, dict[str, str]]:
    try:
        if not isinstance(value, str) or len(value.encode()) > MAX_CONFIG_BYTES:
            raise ValueError
        payload = json.loads(value)
        if (
            not isinstance(payload, dict)
            or set(payload)
            not in (
                {"version", "role", "sender"},
                {"version", "role", "sender", "context"},
            )
            or type(payload["version"]) is not int
            or payload["version"] != 1
            or payload["role"] != "producer"
        ):
            raise ValueError
        context = payload.get("context", {})
        if not isinstance(context, dict) or any(
            key not in _POLICY_CONTEXT
            or not isinstance(item, str)
            or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", item) is None
            for key, item in context.items()
        ):
            raise ValueError
        return SenderConfig.model_validate(payload["sender"]), context
    except (ValueError, TypeError):
        raise ReportingError("reporting.policy.invalid") from None


@contextmanager
def session_scope(
    config: ReportingConfig | None = None,
    *,
    service_lifetime: bool = False,
    startup_environment: Mapping[str, str] | None = None,
) -> Iterator[ReportingSession]:
    global _ACTIVE, _REFERENCES
    if config is not None:
        try:
            config = ReportingConfig.model_validate(
                config.model_dump(mode="python", warnings=False)
            )
        except (AttributeError, ValueError, TypeError):
            raise ReportingError("reporting.config.invalid") from None
    existing = active_session()
    policy = os.environ.get(POLICY_ENV) if config is None and existing is None else None
    decoded = _decode_policy(policy) if policy else None
    producer = (
        existing.sender_config
        if config is None and existing and existing.role == "producer"
        else (decoded[0] if decoded else None)
    )
    resolved = (
        config
        if config is not None
        else (
            existing.config
            if existing
            else (
                None
                if producer
                else ReportingConfig()
                if _PRODUCER_ENTRY.get()
                else load_config(startup_environment)
            )
        )
    )
    candidate = ReportingSession(
        resolved,
        producer=producer,
        service_lifetime=service_lifetime,
        startup_environment=startup_environment,
        producer_context=existing._context
        if existing
        else (decoded[1] if decoded else None),
    )
    if not candidate.enabled:
        if config is not None and existing is not None:
            raise ReportingError("reporting.session.conflict")
        yield candidate
        return
    entry_pid = os.getpid()
    with _LOCK:
        existing = active_session()
        if existing is not None:
            if existing.sender_config != candidate.sender_config or (
                config is not None and existing.config != config
            ):
                raise ReportingError("reporting.session.conflict")
            current = existing
        else:
            candidate.start()
            _ACTIVE = candidate
            current = candidate
        _REFERENCES += 1
    from tributo._common.reporting.sender import CONTEXT, context_scope

    try:
        # An existing scope is authoritative, including intentionally absent IDs.
        context = CONTEXT.get()
        attributes = (
            {"operation_id": current.operation_id, **current._context}
            if context is None
            else context
        )
        with context_scope(**attributes):
            yield current
    finally:
        if os.getpid() == entry_pid:
            with _LOCK:
                _REFERENCES -= 1
                if _REFERENCES == 0:
                    try:
                        current.close()
                    finally:
                        _ACTIVE = None


def local_reporting_operation(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Resolve local launch options without mutating the caller's environment."""

    @wraps(function)
    def invoke(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        supplied = kwargs.get("env_vars")
        if supplied is not None and (
            not isinstance(supplied, dict)
            or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in supplied.items()
            )
        ):
            return function(*args, **kwargs)
        config = (
            load_config(supplied)
            if supplied and any(k in supplied for k in KNOWN_ENV)
            else None
        )
        with session_scope(config, startup_environment=supplied):
            return function(*args, **kwargs)

    return invoke


def reporting_operation(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Initialize an owned entry point before business work without changing APIs."""

    @wraps(function)
    def invoke(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with session_scope():
            return function(*args, **kwargs)

    return invoke


@contextmanager
def reporting_context(
    snapshot: Mapping[str, Any] | None = None, /, **attributes: Any
) -> Iterator[None]:
    """Record-scoped correlation; inactive calls import no optional SDK."""
    session = active_session()
    if session is None or session._sender is None:
        yield
        return
    from tributo._common.reporting.sender import CONTEXT, safe_attributes

    current = CONTEXT.get()
    inherited = (
        snapshot
        if snapshot is not None
        else session._sender.attributes
        if current is None
        else current
    )
    token = CONTEXT.set({**inherited, **safe_attributes(attributes)})
    try:
        yield
    finally:
        CONTEXT.reset(token)


def capture_reporting_context() -> dict[str, Any] | None:
    """Copy the effective identity; an empty scope must remain distinct from None."""
    session = active_session()
    if session is None or session._sender is None:
        return None
    from tributo._common.reporting.sender import CONTEXT

    current = CONTEXT.get()
    return dict(session._sender.attributes if current is None else current)


@contextmanager
def producer_scope(
    policy: str | None = None,
    *,
    ray_worker: bool = False,
    _initialize: bool = True,
) -> Iterator[ReportingSession | None]:
    """Reuse transport while binding trusted identity to this invocation only."""
    token = _PRODUCER_ENTRY.set(True)
    try:
        with _producer_transport_scope(
            policy, ray_worker=ray_worker, _initialize=_initialize
        ) as session:
            yield session
    finally:
        _PRODUCER_ENTRY.reset(token)


@contextmanager
def _producer_transport_scope(
    policy: str | None, *, ray_worker: bool, _initialize: bool
) -> Iterator[ReportingSession | None]:
    value = policy or os.environ.get(POLICY_ENV)
    if not value:
        yield None
        return
    config, context = _decode_policy(value)
    session = (
        bootstrap_producer(value, ray_worker=ray_worker, _context_scoped=True)
        if _initialize
        else active_session()
    )
    if not _initialize and session is not None and session.sender_config != config:
        yield None
        return
    if session is None or session._sender is None:
        yield session
        return
    from tributo._common.reporting.sender import context_scope

    with context_scope(_POLICY_CONTEXT, **context):
        yield session


def reporting_worker(
    function: Callable[[Mapping[str, Any]], None],
) -> Callable[[Mapping[str, Any]], None]:
    """Initialize one Train producer and reuse public worker identity, without collectives."""

    @wraps(function)
    def invoke(config: Mapping[str, Any]) -> None:
        with producer_scope(
            config.get("_core_reporting_policy"), ray_worker=True
        ) as session:
            execute(config, session)

    def execute(config: Mapping[str, Any], session: ReportingSession | None) -> None:
        if session is None or session._sender is None:
            function(config)
            return
        from tributo._common.reporting.sender import context_scope

        identity: dict[str, int] = {}
        try:
            import ray
            from ray import train

            if ray.is_initialized():
                context = train.get_context()
                identity = {
                    "rank": context.get_world_rank(),
                    "world_size": context.get_world_size(),
                }
        except Exception:
            session.diagnose("sender.rank_unavailable")
        with context_scope(**identity):
            function(config)

    return invoke


def merge_runtime_env(
    runtime_env: Mapping[str, Any] | None = None, *, for_job: bool = False
) -> dict[str, Any]:
    """Copy, never mutate, caller options; preserve hook order and frozen policy."""
    result = dict(runtime_env or {})
    env = dict(result.get("env_vars", {}))
    session = active_session()
    if for_job:
        # A remote driver owns a separate execution domain. Never export the
        # submitting machine's borrowed loopback endpoint into another Job.
        if POLICY_ENV in env:
            raise ReportingError("reporting.policy.reserved")
        for key in KNOWN_ENV:
            if key not in env and key in os.environ:
                env[key] = os.environ[key]
        if env.get("TRIBUTO_LOG_REPORTING_ENABLED") == "1":
            hook = result.get("worker_process_setup_hook")
            reserved = env.get("_TRIBUTO_REPORTING_USER_HOOK")
            if reserved is not None and hook not in (reserved, WORKER_HOOK):
                raise ReportingError("reporting.policy.reserved_hook")
            if isinstance(hook, str) and hook != WORKER_HOOK:
                env["_TRIBUTO_REPORTING_USER_HOOK"] = hook
            if hook is None or isinstance(hook, str):
                result["worker_process_setup_hook"] = WORKER_HOOK
    elif session is not None:
        policy = session.producer_policy()
        previous = env.get(POLICY_ENV)
        if previous is not None and previous != policy:
            raise ReportingError("reporting.policy.conflict")
        if policy:
            if "_TRIBUTO_REPORTING_USER_HOOK" in env and (
                previous != policy
                or result.get("worker_process_setup_hook") != WORKER_HOOK
            ):
                raise ReportingError("reporting.policy.reserved_hook")
            env[POLICY_ENV] = policy
            if session.sender_config.headers_ref:
                key = session.sender_config.headers_ref.removeprefix("env://")
                if key not in env and key in session._environment:
                    env[key] = session._environment[key]
            hook = result.get("worker_process_setup_hook")
            if hook is None:
                hook = os.environ.get("_TRIBUTO_REPORTING_USER_HOOK")
            if hook != WORKER_HOOK:
                if isinstance(hook, str):
                    env["_TRIBUTO_REPORTING_USER_HOOK"] = hook
                elif hook is not None:
                    from tributo._common.reporting.worker import CombinedHook

                    result["worker_process_setup_hook"] = (
                        hook if isinstance(hook, CombinedHook) else CombinedHook(hook)
                    )
                if hook is None or isinstance(hook, str):
                    result["worker_process_setup_hook"] = WORKER_HOOK
    if env:
        result["env_vars"] = env
    return result


def serve_runtime_env(
    runtime_env: Mapping[str, Any] | None, app_name: str
) -> dict[str, Any] | None:
    """Require an actual persistent owner for managed detached serving."""
    session = active_session()
    if session is None:
        return dict(runtime_env) if runtime_env is not None else None
    if session.role != "owner" or session.config is None:
        raise ReportingError("reporting.serve.owner_required")
    if session.config.mode == "managed" and not session.service_lifetime:
        raise ReportingError("reporting.serve.service_lifetime_required")
    return merge_runtime_env(runtime_env)


def reserve_serve_app(app_name: str) -> None:
    session = active_session()
    if session is None or session.config is None or session.config.mode != "managed":
        return
    from ray import serve

    if app_name not in session._services and app_name in serve.status().applications:
        raise ReportingError("reporting.serve.application_not_owned")
    session._services.add(app_name)


def bootstrap_producer(
    policy: str | None = None,
    *,
    ray_worker: bool = False,
    _context_scoped: bool = False,
) -> ReportingSession | None:
    """Initialize transport; a changed identity requires producer_scope."""
    global _ACTIVE, _REFERENCES
    value = policy or os.environ.get(POLICY_ENV)
    if not value:
        return None
    with _LOCK:
        existing = active_session()
        config, context = _decode_policy(value)
        if existing:
            if existing.sender_config != config:
                raise ReportingError("reporting.policy.conflict")
            if (
                existing.role == "producer"
                and existing._context != context
                and not _context_scoped
            ):
                raise ReportingError("reporting.policy.context_requires_scope")
            if existing._sender is not None and ray_worker:
                existing._sender.ray_worker = True
            return existing
        session = ReportingSession(producer=config, producer_context=context)
        try:
            session.start(producer_best_effort=True)
        except Exception:
            session.state = "degraded"
            session.diagnose("producer.initialization_failed")
        _ACTIVE = session
        if session._sender is not None:
            session._sender.ray_worker = ray_worker
        _REFERENCES = 1  # the worker owns a process-lifetime producer reference
        atexit.register(session.close)
        return session


def _after_fork() -> None:
    global _ACTIVE, _LOCK, _REFERENCES
    _LOCK = threading.RLock()
    _ACTIVE, _REFERENCES = None, 0
    for name in logging.Logger.manager.loggerDict.copy():
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if getattr(handler, "_tributo_reporting_handler", False):
                logger.removeHandler(handler)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
