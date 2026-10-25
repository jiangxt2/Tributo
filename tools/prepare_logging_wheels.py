"""Prepare hash-verified locked wheels outside Docker's download path."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def select_wheels(requirements: bytes, lock: dict, architecture: str) -> list[dict]:
    """Select Linux CPython 3.12 artifacts from the existing lock, without resolve."""
    platforms = [
        f"manylinux_2_{version}_{architecture}" for version in range(35, 16, -1)
    ]
    platforms += [f"manylinux2014_{architecture}", f"linux_{architecture}"]
    tags = list(cpython_tags((3, 12), platforms=platforms))
    tags += list(compatible_tags((3, 12), interpreter="cp312", platforms=platforms))
    rank = {tag: index for index, tag in enumerate(tags)}
    environment = {
        "os_name": "posix",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": architecture,
        "python_version": "3.12",
        "python_full_version": "3.12.12",
        "implementation_name": "cpython",
        "implementation_version": "3.12.12",
        "platform_python_implementation": "CPython",
        "extra": "",
    }
    selected = {}
    for line in requirements.decode().replace("\\\n", " ").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        request = Requirement(line.split("--hash=", 1)[0].strip())
        if request.marker is not None and not request.marker.evaluate(environment):
            continue
        packages = [
            p
            for p in lock["package"]
            if canonicalize_name(p["name"]) == canonicalize_name(request.name)
            and request.specifier.contains(p["version"])
        ]
        choices = []
        for package in packages:
            for wheel in package.get("wheels", []):
                filename = urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]
                _, _, _, wheel_tags = parse_wheel_filename(filename)
                priority = min(
                    (rank[tag] for tag in wheel_tags if tag in rank), default=None
                )
                if priority is not None:
                    choices.append((priority, filename, wheel))
        if not choices:
            raise ValueError(f"No locked target wheel for {request.name}")
        _, filename, wheel = min(choices, key=lambda item: item[0])
        selected[filename] = {**wheel, "filename": filename, "package": request.name}
    return [selected[name] for name in sorted(selected)]


def download_wheel(wheel: dict, directory: Path) -> dict:
    destination = directory / wheel["filename"]
    expected = wheel["hash"].removeprefix("sha256:")

    def valid(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != wheel["size"]:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected

    if valid(destination):
        return wheel
    source = urlsplit(wheel["url"])
    if source.scheme != "https" or source.netloc != "files.pythonhosted.org":
        raise ValueError("Only existing PyPI artifacts are accepted")
    mirror = "https://pypi.tuna.tsinghua.edu.cn" + source.path
    partial = destination.with_suffix(".part")
    env = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "2",
            "--retry-all-errors",
            "--connect-timeout",
            "15",
            "--max-time",
            "180",
            "-o",
            str(partial),
            mirror,
        ],
        env=env,
        check=True,
    )
    if not valid(partial):
        raise ValueError(f"Locked wheel checksum/size mismatch: {wheel['filename']}")
    partial.replace(destination)
    print("Verified locked wheel: " + wheel["filename"], flush=True)
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    from tools.tributo_it import ROOT, docker_platform, load_profile, runtime_identity

    target = docker_platform()
    architecture = "aarch64" if target.endswith("/arm64") else "x86_64"
    identity = runtime_identity(load_profile("debug-logging"), target)
    if identity.requirements_content is None:
        raise ValueError("A locked export is required")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    wheels = select_wheels(identity.requirements_content, lock, architecture)
    args.directory.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda wheel: download_wheel(wheel, args.directory), wheels))
    (args.directory / "manifest.json").write_text(
        json.dumps({"platform": target, "wheels": wheels}, indent=2)
    )
    print(f"Verified wheelhouse ready: {len(wheels)} artifacts ({target})", flush=True)


if __name__ == "__main__":
    main()
