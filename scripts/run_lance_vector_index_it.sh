#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"
command -v uv >/dev/null 2>&1 || {
  echo "uv is required to run Tributo integration tests" >&2
  exit 1
}
exec uv run --no-project --python 3.12 python tools/tributo_it.py run-lance-vector-index "$@"
