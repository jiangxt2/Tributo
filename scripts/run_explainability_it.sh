#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${script_dir}/run_inference_it.sh" --suite explainability "$@"
