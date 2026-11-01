"""Provision the minimal Collector before application startup, using official OCB.

Generated Go sources, go.sum, binary and checksum remain in the chosen output
directory. The task runtime never invokes this builder or downloads a binary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def build(output: Path, *, goos: str, goarch: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    builder_dir = output / "builder"
    builder_dir.mkdir()
    environment = dict(os.environ, GOBIN=str(builder_dir))
    # First build a host-native OCB. Only its generated Collector is cross-built.
    for name in ("GOOS", "GOARCH"):
        environment.pop(name, None)
    subprocess.run(
        ["go", "install", "go.opentelemetry.io/collector/cmd/builder@v0.163.0"],
        env=environment,
        check=True,
    )
    manifest = json.loads(
        Path(__file__).with_name("reporting-collector.json").read_text()
    )
    manifest["dist"]["output_path"] = str(output / "dist")
    path = output / "builder-config.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment.update(GOOS=goos, GOARCH=goarch, CGO_ENABLED="0")
    subprocess.run(
        [str(builder_dir / "builder"), "--config", str(path)],
        env=environment,
        check=True,
    )
    binary = output / "dist/tributo-otelcol"
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "goos": goos,
                "goarch": goarch,
                "sha256": digest,
                "bytes": binary.stat().st_size,
                "ocb": "0.163.0",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goos", choices=("linux", "darwin"), required=True)
    parser.add_argument("--goarch", choices=("arm64", "amd64"), required=True)
    arguments = parser.parse_args()
    build(arguments.output, goos=arguments.goos, goarch=arguments.goarch)
