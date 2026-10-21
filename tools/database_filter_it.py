"""Owned single-node database filter validation using the shared IT runtime."""

from __future__ import annotations

import ipaddress
import json
import secrets
import signal
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from tools import tributo_it as it

_COMPOSE = it.ROOT / "tests/integrations/docker-compose.database-filter.yml"
_PREFIX = "tributo-database-filter-"


def _image_diagnostics(directory: Path, phase: str) -> None:
    commands = (
        ("dangling", ["docker", "image", "ls", "--filter", "dangling=true"]),
        ("images", ["docker", "image", "ls", "--no-trunc"]),
        ("disk", ["docker", "system", "df"]),
    )
    for name, command in commands:
        result = it._run(command, check=False)
        (directory / f"{name}-{phase}.log").write_text(result.stdout + result.stderr)


def _cleanup_built_runtime(prepared: it.PreparedRuntime) -> None:
    # Reused managed images belong to the shared runtime cache. Only a runtime
    # built by this invocation may be removed, after exact identity checks.
    if prepared.source != "build":
        return
    with it.runtime_lock(prepared.identity):
        image = it._image_inspect(prepared.identity.local_tag)
        if image is None:
            return
        if image["Id"] != prepared.image_id:
            raise it.TributoITError("Runtime tag changed; refusing image cleanup")
        users = it._run(
            [
                "docker",
                "ps",
                "--all",
                "--quiet",
                "--filter",
                f"ancestor={prepared.image_id}",
            ]
        )
        if users.stdout.strip():
            raise it.TributoITError("Runtime is in use; refusing image cleanup")
        it._run(["docker", "image", "rm", prepared.identity.local_tag])


def _available_subnet() -> ipaddress.IPv4Network:
    ids = it._run(["docker", "network", "ls", "--quiet"]).stdout.split()
    networks = (
        json.loads(it._run(["docker", "network", "inspect", *ids]).stdout)
        if ids
        else []
    )
    used = [
        ipaddress.ip_network(config["Subnet"])
        for network in networks
        for config in ((network.get("IPAM") or {}).get("Config") or [])
        if config.get("Subnet")
    ]
    for second in range(30, 15, -1):
        for third in range(256):
            candidate = ipaddress.ip_network(f"172.{second}.{third}.0/24")
            if not any(candidate.overlaps(existing) for existing in used):
                return candidate
    raise it.TributoITError("No isolated private subnet is available")


def validate_evidence(directory: Path, database: str) -> None:
    """Require executed tests without skips and independently captured SQL evidence."""
    report = directory / "junit.xml"
    tree = ET.parse(report)
    cases = list(tree.iter("testcase"))
    if not cases or any(
        list(tree.iter(tag)) for tag in ("skipped", "failure", "error")
    ):
        raise it.TributoITError(
            "Required database tests did not all execute successfully"
        )
    records = [
        json.loads(line)
        for line in (directory / "evidence.jsonl").read_text().splitlines()
        if line.strip()
    ]
    modes = {record["mode"] for record in records}
    expected = (
        {"single", "partition", "range"}
        if database == "clickhouse"
        else {"mysql", "flight"}
    )
    if not expected <= modes:
        raise it.TributoITError("Database execution evidence is missing required modes")
    for record in records:
        if (
            record["database"] != database
            or record["ray_nodes"] != 1
            or record["baseline_rows"] != 10_000
            or record["source_rows"] != 100
            or record["final_rows"] != 100
            or not record["queries"]
            or record["residual_ordinals"]
            or any(
                query.get("stage") != "real_driver_execute_succeeded"
                or "WHERE" not in query["sql"].upper()
                for query in record["queries"]
            )
        ):
            raise it.TributoITError("Source-side filter evidence is incomplete")


class _Tee:
    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def run_database_filter(database: str, *, log_dir: Path | None = None) -> None:
    """Retain lifecycle diagnostics even when preparation fails before startup."""
    if database not in {"clickhouse", "doris"}:
        raise it.TributoITError("database must be clickhouse or doris")
    directory = (
        log_dir
        or Path("/tmp/tributo-database-filter") / f"{database}-{uuid.uuid4().hex}"
    ).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise it.TributoITError("Evidence directory must be empty for a fresh run")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "lifecycle.log").open("w") as log:
        with (
            redirect_stdout(_Tee(sys.stdout, log)),
            redirect_stderr(_Tee(sys.stderr, log)),
        ):
            prepared = None
            try:
                _image_diagnostics(directory, "before")
                prepared = it.prepare_runtime(
                    it.load_profile("database-filter"), platform=it.docker_platform()
                )
                _run_database_filter(database, log_dir=directory, prepared=prepared)
            except BaseException as error:
                traceback.print_exc()
                if not (directory / "result.json").exists():
                    (directory / "result.json").write_text(
                        json.dumps(
                            {
                                "database": database,
                                "passed": False,
                                "phase": "preparation",
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                raise
            finally:
                try:
                    if prepared is not None:
                        _cleanup_built_runtime(prepared)
                except BaseException:
                    # A successful data test cannot hide unsuccessful cleanup.
                    report = directory / "result.json"
                    result = json.loads(report.read_text()) if report.exists() else {}
                    result.update(passed=False, cleanup_error=True)
                    report.write_text(json.dumps(result, indent=2) + "\n")
                    raise
                finally:
                    _image_diagnostics(directory, "after")


def _run_database_filter(
    database: str, *, log_dir: Path, prepared: it.PreparedRuntime
) -> None:
    """Own one isolated Compose lifecycle and submit tests through Ray Jobs."""
    if database not in {"clickhouse", "doris"}:
        raise it.TributoITError("database must be clickhouse or doris")
    project = f"{_PREFIX}{database}-{uuid.uuid4().hex[:12]}"
    it._assert_project_absent(project)
    images_before = it._capture_image_diagnostic_baseline()
    containers_before = it._capture_container_diagnostic_baseline()
    profile = it.load_profile("database-filter")
    definition = profile.definition
    image_fields = (
        ("clickhouse_image",)
        if database == "clickhouse"
        else ("doris_fe_image", "doris_be_image")
    )
    resolved = {}
    for field in image_fields:
        reference = str(definition[field])
        it.ensure_digest_image(reference)
        resolved[field] = it._local_digest_reference(reference)
    subnet = _available_subnet()
    credential = secrets.token_hex(24)
    env = it._docker_environment()
    env.update(
        {
            "COMPOSE_PROJECT_NAME": project,
            "COMPOSE_PROFILES": database,
            "TRIBUTO_FILTER_DATABASE": database,
            "TRIBUTO_FILTER_SOURCE": str(it.ROOT),
            "TRIBUTO_FILTER_SOURCE_SHA": it._run(
                ["git", "rev-parse", "HEAD"]
            ).stdout.strip(),
            "TRIBUTO_FILTER_RUNTIME": prepared.image_id,
            "TRIBUTO_FILTER_SUBNET": str(subnet),
            "TRIBUTO_FILTER_FE_IP": str(subnet.network_address + 250),
            "TRIBUTO_FILTER_BE_IP": str(subnet.network_address + 251),
            "TRIBUTO_CLICKHOUSE_PASSWORD": credential,
            "TRIBUTO_FILTER_CLICKHOUSE": resolved.get(
                "clickhouse_image", str(definition["clickhouse_image"])
            ),
            "TRIBUTO_FILTER_DORIS_FE": resolved.get(
                "doris_fe_image", str(definition["doris_fe_image"])
            ),
            "TRIBUTO_FILTER_DORIS_BE": resolved.get(
                "doris_be_image", str(definition["doris_be_image"])
            ),
        }
    )
    compose = ["docker", "compose", "--file", str(_COMPOSE), "--project-name", project]
    started = False
    primary_error: BaseException | None = None
    cleanup_errors = []
    summary: dict[str, Any] = {
        "database": database,
        "commit": env["TRIBUTO_FILTER_SOURCE_SHA"],
        "runtime_image": prepared.image_id,
        "runtime_key": prepared.identity.runtime_key,
        "database_images": resolved,
        "project": project,
        "passed": False,
    }

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        started = True
        services = (
            ["ray-head", "clickhouse"]
            if database == "clickhouse"
            else ["ray-head", "fe", "be"]
        )
        it._run([*compose, "up", "--detach", "--no-build", *services], env=env)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ready = it._run(
                [
                    *compose,
                    "exec",
                    "--no-TTY",
                    "ray-head",
                    "python",
                    "-c",
                    "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8265/api/version',timeout=2).read()",
                ],
                env=env,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(2)
        else:
            raise it.TributoITError(
                "Single-node Ray Jobs endpoint did not become ready"
            )
        it._run_streamed(
            [
                *compose,
                "exec",
                "--no-TTY",
                "ray-head",
                "ray",
                "job",
                "submit",
                "--address",
                "http://127.0.0.1:8265",
                "--submission-id",
                project,
                "--",
                "python",
                "-m",
                "pytest",
                "tests/integrations/test_database_filter_pushdown.py",
                "-o",
                "addopts=",
                "-o",
                "cache_dir=/tmp/pytest-cache",
                "-vv",
                "-ra",
                "--timeout=300",
                "--junitxml=/workspace/tributo-work/artifacts/junit.xml",
            ],
            env,
            log_dir / "tests.log",
        )
        it._run(
            [
                *compose,
                "cp",
                "ray-head:/workspace/tributo-work/artifacts/.",
                str(log_dir),
            ],
            env=env,
        )
        validate_evidence(log_dir, database)
        summary["passed"] = True
    except BaseException as error:
        primary_error = error
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if started:
            logs = it._run([*compose, "logs", "--no-color"], env=env, check=False)
            (log_dir / "services.log").write_text(
                (logs.stdout + logs.stderr).replace(credential, "<redacted>")
            )
            it._run(
                [
                    *compose,
                    "cp",
                    "ray-head:/workspace/tributo-work/artifacts/.",
                    str(log_dir),
                ],
                env=env,
                check=False,
            )
            if database == "doris":
                audit = it._run(
                    [
                        *compose,
                        "exec",
                        "--no-TTY",
                        "fe",
                        "bash",
                        "-c",
                        "cat /opt/apache-doris/fe/log/fe.audit.log*",
                    ],
                    env=env,
                    check=False,
                )
                (log_dir / "doris-audit.log").write_text(
                    (audit.stdout + audit.stderr).replace(credential, "<redacted>")
                )
            cleanup = it._run(
                [*compose, "down", "--volumes", "--remove-orphans"],
                env=env,
                check=False,
            )
            if cleanup.returncode:
                cleanup_errors.append("Compose cleanup failed")
            try:
                it._assert_project_absent(project)
            except it.TributoITError as error:
                cleanup_errors.append(str(error))
        it._report_external_container_activity(project, containers_before)
        it._report_new_image_artifacts(project, images_before)
        if primary_error or cleanup_errors:
            summary["passed"] = False
        (log_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    if primary_error or cleanup_errors:
        raise it.TributoITError(
            f"Database filter validation failed; evidence: {log_dir}; "
            f"cause: {primary_error}; cleanup: {cleanup_errors}"
        ) from primary_error
    print(f"Single-node {database} filter validation passed; evidence: {log_dir}")
