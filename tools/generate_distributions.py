"""Write the normalized installed-distribution inventory inside the image."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from packaging.utils import canonicalize_name

OUTPUT = Path("/opt/tributo-image/installed-distributions.json")


def main() -> None:
    distributions = {
        canonicalize_name(metadata.metadata["Name"]): metadata.version
        for metadata in importlib.metadata.distributions()
        if metadata.metadata.get("Name")
    }
    OUTPUT.write_text(
        json.dumps(dict(sorted(distributions.items())), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
