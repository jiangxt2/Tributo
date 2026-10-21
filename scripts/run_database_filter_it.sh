#!/usr/bin/env sh
set -eu
repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
exec uv run --no-project --python 3.12 python tools/tributo_it.py run-database-filter "$@"
