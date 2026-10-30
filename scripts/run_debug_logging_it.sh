#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TRIBUTO_IT_VERIFIED_WHEELHOUSE="${TRIBUTO_IT_VERIFIED_WHEELHOUSE:-/tmp/tributo-debug-logging-wheels}"
cd "$ROOT"
exec uv run --no-project --python 3.12 python "$ROOT/tools/tributo_it.py" run-debug-logging "$@"
