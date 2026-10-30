"""Run owned real Collector -> Doris and Ray reporting acceptance tests.

Usage: python -m tools.reporting_it --binary-dir /absolute/preprovisioned/dir
The directory must contain an executable otelcol-contrib for Docker's platform.
No Collector downloads or builds take place in the application or test container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from tools import tributo_it as it
from tools.database_filter_it import (
    _available_subnet,
    _cleanup_built_runtime,
    _image_diagnostics,
)


def run(binary_dir: Path, directory: Path, *, cases: str = "all") -> None:
    """Own the exact Compose project and retain evidence on success or failure."""
    binary = binary_dir.resolve() / "otelcol-contrib"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise it.TributoITError("Pre-provisioned executable is required")
    if directory.exists() and any(directory.iterdir()):
        raise it.TributoITError("Use an empty evidence directory for each state")
    directory.mkdir(parents=True, exist_ok=True)
    project = "tributo-reporting-" + uuid.uuid4().hex[:12]
    it._assert_project_absent(project)
    profile = it.load_profile("log-reporting")
    compose = [
        "docker",
        "compose",
        "--file",
        str(it.ROOT / "tests/integrations/docker-compose.log-reporting.yml"),
        "--project-name",
        project,
    ]
    env = it._docker_environment()

    def command(*arguments, check=True):
        return it._run([*compose, *arguments], env=env, check=check)

    prepared = None
    started = False
    summary: dict[str, object] = {"project": project, "passed": False, "cases": cases}
    errors: list[str] = []

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        _image_diagnostics(directory, "before")
        prepared = it.prepare_runtime(profile, platform=it.docker_platform())
        images = {}
        for name in ("doris_fe_image", "doris_be_image"):
            ref = str(profile.definition[name])
            it.ensure_digest_image(ref, use_mirror=True)
            images[name] = it._local_digest_reference(ref)
        subnet = _available_subnet()
        with binary.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        env.update(
            REPORTING_RUNTIME=prepared.image_id,
            REPORTING_SOURCE=str(it.ROOT),
            REPORTING_BINARY_DIR=str(binary_dir.resolve()),
            REPORTING_COLLECTOR_SHA=checksum,
            REPORTING_FE_IMAGE=images["doris_fe_image"],
            REPORTING_BE_IMAGE=images["doris_be_image"],
            REPORTING_SUBNET=str(subnet),
            REPORTING_FE_IP=str(subnet.network_address + 250),
            REPORTING_BE_IP=str(subnet.network_address + 251),
        )
        summary.update(
            runtime_image=prepared.image_id,
            runtime_key=prepared.identity.runtime_key,
            binary_sha256=checksum,
            binary_bytes=binary.stat().st_size,
            images=images,
            head=it._run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        )
        # The container snapshot manifest (also captured in versions.json)
        # includes unstaged/new source; HEAD alone is not a source identity.
        (directory / "state.json").write_text(json.dumps(summary, indent=2) + "\n")
        started = True
        services = ["driver", "fe", "be"]
        clustered = cases not in {"collector", "local"}
        if clustered:
            services.extend(["ray-head", "ray-worker"])
        command("up", "--detach", "--no-build", *services)
        deadline = time.monotonic() + 180
        while clustered and time.monotonic() < deadline:
            ready = command(
                "exec",
                "--no-TTY",
                "driver",
                "python",
                "-c",
                "import urllib.request;urllib.request.urlopen('http://ray-head:8265/api/version',timeout=2).read()",
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(2)
        if clustered and time.monotonic() >= deadline:
            raise it.TributoITError("Ray Jobs endpoint did not become ready")
        selection = (
            []
            if cases == "all"
            else [
                "-k",
                "local_parent or multinode or tributo_job"
                if cases == "ray"
                else "local_parent"
                if cases == "local"
                else "multinode"
                if cases == "matrix"
                else "multinode"
                if cases == "light"
                else "review_regressions"
                if cases == "regressions"
                else "not (local_parent or multinode or tributo_job or review_regressions)",
            ]
        )
        it._run_streamed(
            [
                *compose,
                "exec",
                "--no-TTY",
                "driver",
                "python",
                "-m",
                "pytest",
                "tests/integrations/test_log_reporting.py",
                "-o",
                "addopts=",
                "-o",
                "cache_dir=/tmp/reporting-pytest-cache",
                "-vv",
                "-ra",
                "--timeout=240",
                "--junitxml=/workspace/tributo-work/artifacts/junit.xml",
                *selection,
            ],
            env,
            directory / "tests.log",
        )
        command("cp", "driver:/workspace/tributo-work/artifacts/.", str(directory))
        tree = ET.parse(directory / "junit.xml")
        if not list(tree.iter("testcase")) or any(
            list(tree.iter(tag)) for tag in ("skipped", "error", "failure")
        ):
            raise it.TributoITError(
                "Required reporting tests did not execute successfully"
            )
        summary["passed"] = True
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        if started:
            logs = command("logs", "--no-color", check=False)
            (directory / "services.log").write_text(logs.stdout + logs.stderr)
            command(
                "cp",
                "driver:/workspace/tributo-work/artifacts/.",
                str(directory),
                check=False,
            )
            cleanup = command("down", "--volumes", "--remove-orphans", check=False)
            (directory / "cleanup.log").write_text(cleanup.stdout + cleanup.stderr)
            if cleanup.returncode:
                errors.append("Compose cleanup failed")
            try:
                it._assert_project_absent(project)
            except it.TributoITError:
                errors.append("Compose resources remain")
        try:
            if prepared is not None:
                _cleanup_built_runtime(prepared)
        except it.TributoITError:
            errors.append("Runtime cleanup failed")
        finally:
            _image_diagnostics(directory, "after")
            summary["cleanup_errors"] = errors
            if errors:
                summary["passed"] = False
            (directory / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
        if errors:
            raise it.TributoITError("; ".join(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        choices=("all", "ray", "local", "collector", "matrix", "light", "regressions"),
        default="all",
    )
    arguments = parser.parse_args()
    run(arguments.binary_dir, arguments.log_dir, cases=arguments.cases)
