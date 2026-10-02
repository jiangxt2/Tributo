"""Create immutable provenance for one distributed algorithm Gate run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SourceRevision:
    """One exact local source revision used to build Gate Wheels."""

    root: str
    commit: str
    dirty: bool


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def inspect_source(root: Path) -> SourceRevision:
    """Resolve a worktree root, commit, and complete dirty state."""
    resolved = root.resolve()
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel")).resolve()
    if top_level != resolved:
        raise ValueError(f"source root is not a Git worktree root: {resolved}")
    commit = _git(resolved, "rev-parse", "--verify", "HEAD^{commit}")
    dirty = bool(_git(resolved, "status", "--porcelain=v1", "--untracked-files=all"))
    return SourceRevision(root=str(resolved), commit=commit, dirty=dirty)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(values) != len(set(values)):
        raise ValueError("Gate selections must not contain duplicate names")
    return tuple(sorted(values))


def _configuration_sha256(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest")


def _gate_identity(
    *,
    project_name: str,
    scope: str,
    evidence_mode: Literal["certification", "diagnostic"],
    core: SourceRevision,
    algorithms: SourceRevision,
    categories: str,
    entry_points: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_categories = _selection(categories)
    selected_entry_points = _selection(entry_points)
    if evidence_mode == "certification":
        if core.dirty or algorithms.dirty:
            raise ValueError(
                "certification requires clean Core and algorithm worktrees"
            )
        if scope != "official" or selected_categories or selected_entry_points:
            raise ValueError(
                "certification requires the unfiltered official algorithm scope"
            )
    selection = {
        "categories": list(selected_categories),
        "entry_points": list(selected_entry_points),
    }
    sources = {
        "core": asdict(core),
        "algorithms": asdict(algorithms),
    }
    identity = {
        "schema_version": 1,
        "project_name": project_name,
        "scope": scope,
        "evidence_mode": evidence_mode,
        "certifying": evidence_mode == "certification",
        "selection": selection,
        "sources": sources,
    }
    source_configuration = {
        "schema_version": identity["schema_version"],
        "scope": scope,
        "evidence_mode": evidence_mode,
        "selection": selection,
        "sources": {
            "core": {"commit": core.commit, "dirty": core.dirty},
            "algorithms": {
                "commit": algorithms.commit,
                "dirty": algorithms.dirty,
            },
        },
    }
    return identity, source_configuration


def _recorded_document(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **identity,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recorder_python_version": platform.python_version(),
    }


def build_preflight_provenance(
    *,
    project_name: str,
    scope: str,
    evidence_mode: Literal["certification", "diagnostic"],
    core: SourceRevision,
    algorithms: SourceRevision,
    categories: str = "",
    entry_points: str = "",
) -> dict[str, Any]:
    """Bind Gate scope and source state before any Docker preparation."""
    identity, source_configuration = _gate_identity(
        project_name=project_name,
        scope=scope,
        evidence_mode=evidence_mode,
        core=core,
        algorithms=algorithms,
        categories=categories,
        entry_points=entry_points,
    )
    return _recorded_document(
        {
            "document_type": "preflight",
            **identity,
            "configuration_sha256": _configuration_sha256(source_configuration),
        }
    )


def load_preflight_provenance(path: Path) -> tuple[dict[str, Any], str]:
    """Load one preflight document and return its exact file digest."""
    if not path.is_file():
        raise ValueError(f"Gate preflight provenance is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gate preflight provenance is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Gate preflight provenance is not an object: {path}")
    return value, _sha256_file(path)


def build_provenance(
    *,
    project_name: str,
    scope: str,
    evidence_mode: Literal["certification", "diagnostic"],
    core: SourceRevision,
    algorithms: SourceRevision,
    wheel_paths: Sequence[Path],
    runtime_image: str,
    runtime_image_id: str,
    runtime_python_version: str,
    runtime_ray_version: str,
    preflight: Mapping[str, Any],
    preflight_sha256: str,
    categories: str = "",
    entry_points: str = "",
) -> dict[str, Any]:
    """Build a credential-free, digest-bound Gate provenance document."""
    identity, source_configuration = _gate_identity(
        project_name=project_name,
        scope=scope,
        evidence_mode=evidence_mode,
        core=core,
        algorithms=algorithms,
        categories=categories,
        entry_points=entry_points,
    )
    expected_preflight = {
        "document_type": "preflight",
        **identity,
        "configuration_sha256": _configuration_sha256(source_configuration),
    }
    mismatched_fields = sorted(
        key for key, value in expected_preflight.items() if preflight.get(key) != value
    )
    if mismatched_fields:
        raise ValueError(
            "Gate preflight provenance does not match final source configuration: "
            f"{mismatched_fields}"
        )
    _validate_sha256(preflight_sha256, name="preflight_sha256")
    resolved_wheels = tuple(path.resolve() for path in wheel_paths)
    if len(resolved_wheels) != len(set(resolved_wheels)):
        raise ValueError("Gate Wheel paths must be unique")
    if evidence_mode == "certification" and len(resolved_wheels) != 16:
        raise ValueError(
            "official certification requires Core plus 15 algorithm Wheels"
        )
    if (
        not runtime_image_id.startswith("sha256:")
        or len(runtime_image_id) != 71
        or any(
            character not in "0123456789abcdef"
            for character in runtime_image_id.removeprefix("sha256:")
        )
    ):
        raise ValueError("Gate runtime image ID must be a sha256 digest")
    wheels = []
    for path in sorted(resolved_wheels, key=lambda value: value.name):
        if not path.is_file() or path.suffix != ".whl":
            raise ValueError(f"Gate Wheel is unavailable: {path}")
        wheels.append(
            {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    final_identity = {
        "document_type": "final",
        **identity,
        "preflight": {
            "configuration_sha256": preflight["configuration_sha256"],
            "document_sha256": preflight_sha256,
        },
        "wheels": wheels,
        "runtime": {
            "image": runtime_image,
            "image_id": runtime_image_id,
            "python_version": runtime_python_version,
            "ray_version": runtime_ray_version,
        },
    }
    configuration = {
        **source_configuration,
        "preflight_configuration_sha256": preflight["configuration_sha256"],
        "wheels": wheels,
        "runtime": final_identity["runtime"],
    }
    return _recorded_document(
        {
            **final_identity,
            "configuration_sha256": _configuration_sha256(configuration),
        }
    )


def write_provenance(path: Path, provenance: dict[str, Any]) -> None:
    """Write a new provenance file without overwriting prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Gate provenance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(
                json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=True)
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
        Path(temporary_name).unlink()
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument(
        "--evidence-mode", choices=("certification", "diagnostic"), required=True
    )
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--algorithms-root", type=Path, required=True)
    parser.add_argument("--categories", default="")
    parser.add_argument("--entry-points", default="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    identities = commands.add_parser("export-identities")
    identities.add_argument("--manifest", type=Path, required=True)
    identities.add_argument("--output", type=Path, required=True)
    identities.add_argument("--expected-count", type=int, required=True)
    preflight = commands.add_parser("preflight")
    _add_source_arguments(preflight)
    final = commands.add_parser("final")
    _add_source_arguments(final)
    final.add_argument("--preflight", type=Path, required=True)
    final.add_argument("--wheel", type=Path, action="append", default=[])
    final.add_argument("--runtime-image", required=True)
    final.add_argument("--runtime-image-id", required=True)
    final.add_argument("--runtime-python-version", required=True)
    final.add_argument("--runtime-ray-version", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export-identities":
        if args.output.exists():
            raise FileExistsError(
                f"refusing to overwrite identity manifest: {args.output}"
            )
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = payload.get("entry_points") if isinstance(payload, Mapping) else None
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != 1
            or not isinstance(entries, Mapping)
            or len(entries) != args.expected_count
        ):
            raise ValueError("official algorithm identity manifest is malformed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(args.manifest.read_bytes())
        print(
            json.dumps(
                {
                    "entry_point_count": len(entries),
                    "sha256": _sha256_file(args.output),
                },
                sort_keys=True,
            )
        )
        return
    core = inspect_source(args.core_root)
    algorithms = inspect_source(args.algorithms_root)
    if args.command == "preflight":
        provenance = build_preflight_provenance(
            project_name=args.project_name,
            scope=args.scope,
            evidence_mode=args.evidence_mode,
            core=core,
            algorithms=algorithms,
            categories=args.categories,
            entry_points=args.entry_points,
        )
        write_provenance(args.output, provenance)
        print(json.dumps(provenance, sort_keys=True))
        return
    preflight, preflight_sha256 = load_preflight_provenance(args.preflight)
    provenance = build_provenance(
        project_name=args.project_name,
        scope=args.scope,
        evidence_mode=args.evidence_mode,
        core=core,
        algorithms=algorithms,
        wheel_paths=args.wheel,
        runtime_image=args.runtime_image,
        runtime_image_id=args.runtime_image_id,
        runtime_python_version=args.runtime_python_version,
        runtime_ray_version=args.runtime_ray_version,
        preflight=preflight,
        preflight_sha256=preflight_sha256,
        categories=args.categories,
        entry_points=args.entry_points,
    )
    write_provenance(args.output, provenance)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
