#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMPOSE_FILE="$ROOT/tests/integrations/docker-compose.runtime-image.yml"
PROJECT="tributo-runtime-image-it-$(date +%s)-$$"
OUTPUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tributo-runtime-image-it.XXXXXX")
IMAGE="tributo-runtime-full:local"
DOCKER_ENV=(
    env
    -u HTTP_PROXY
    -u HTTPS_PROXY
    -u ALL_PROXY
    -u http_proxy
    -u https_proxy
    -u all_proxy
)

docker_compose() {
    "${DOCKER_ENV[@]}" docker compose "$@"
}

export COMPOSE_PROJECT_NAME="$PROJECT"
export TRIBUTO_RUNTIME_IMAGE="$IMAGE"
export TRIBUTO_IMAGE_SOURCE_ROOT="$ROOT"

cleanup() {
    # Preserve the status that triggered EXIT/INT/TERM; cleanup is best effort.
    status=$?
    docker_compose --project-name "$PROJECT" --file "$COMPOSE_FILE" logs --no-color \
        >"$OUTPUT_DIR/compose.log" 2>&1 || true
    docker_compose --project-name "$PROJECT" --file "$COMPOSE_FILE" down \
        --volumes --remove-orphans || true
    echo "Runtime image gate artifacts: $OUTPUT_DIR" >&2
    exit "$status"
}
trap cleanup EXIT INT TERM

BUILD_COMMAND=(
    env
    -u HTTP_PROXY
    -u HTTPS_PROXY
    -u ALL_PROXY
    -u http_proxy
    -u https_proxy
    -u all_proxy
    "UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/tributo-image-uv-cache}"
    uv run
    --locked
    --no-sync
    python
    "$ROOT/tools/build_tributo_image.py"
    --repo-root
    "$ROOT"
    --config
    "$ROOT/tools/tributo-runtime-full.json"
    --output-dir
    "$OUTPUT_DIR/image"
)
if [[ -n "${TRIBUTO_IMAGE_PLATFORM:-}" ]]; then
    BUILD_COMMAND+=(--platform "$TRIBUTO_IMAGE_PLATFORM")
fi
"${BUILD_COMMAND[@]}"

runtime_platform="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["platform"])' "$OUTPUT_DIR/image/manifest.json")"
case "$runtime_platform" in
    linux/amd64|linux/arm64) ;;
    *)
        echo "Invalid runtime image platform in manifest: $runtime_platform" >&2
        exit 1
        ;;
esac
export TRIBUTO_RUNTIME_PLATFORM="$runtime_platform"

docker_compose --project-name "$PROJECT" --file "$COMPOSE_FILE" up \
    --detach --wait --wait-timeout 180
docker_compose --project-name "$PROJECT" --file "$COMPOSE_FILE" exec --no-TTY ray-head \
    ray job submit \
    --address http://127.0.0.1:8265 \
    --working-dir /workspace/tributo-src \
    -- python tests/integrations/jobs/runtime_image_gate_job.py
