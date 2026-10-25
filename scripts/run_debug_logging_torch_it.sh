#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
exec env TRIBUTO_LOGGING_TORCH_LOCAL=1 "$ROOT/.venv/bin/python" -m tests.integrations.test_debug_logging --channel torch
