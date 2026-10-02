"""Source-free CLI for installed algorithm Wheel conformance."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tributo.algorithms.conformance import _run_installed_conformance


def _comma_separated(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution-prefix", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--identity-manifest", required=True, type=Path)
    parser.add_argument("--require-contracts", default="config,input,output,coverage")
    parser.add_argument("--forbid-import", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run installed-Wheel conformance and print one deterministic report."""
    args = _parser().parse_args(argv)
    reports = _run_installed_conformance(
        distribution_prefix=args.distribution_prefix,
        expected_count=args.expected_count,
        identity_manifest=args.identity_manifest,
        required_contracts=_comma_separated(args.require_contracts),
        forbidden_imports=_comma_separated(args.forbid_import),
    )
    print(
        json.dumps(
            {"count": len(reports), "reports": [asdict(item) for item in reports]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
