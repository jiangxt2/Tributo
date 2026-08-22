#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

RUN_ID="${TRIBUTO_ALGORITHM_LOCAL_IT_RUN_ID:-$(date +%Y%m%d%H%M%S)-$$}"
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "TRIBUTO_ALGORITHM_LOCAL_IT_RUN_ID must be a unique safe identifier" >&2
  exit 2
fi
CONTAINER_NAME="tributo-algorithm-local-${RUN_ID}"
SOURCE_VOLUME_NAME="tributo-algorithm-local-source-${RUN_ID}"
WORK_VOLUME_NAME="tributo-algorithm-local-work-${RUN_ID}"
LOG_DIR="/tmp/tributo-algorithm-local-${RUN_ID}"
TEST_LOG="${LOG_DIR}/tests.log"
IMAGE_LOG="${LOG_DIR}/image-prepare.log"
PLUGIN_DIST_DIR="${LOG_DIR}/plugin-dist"
PLUGIN_WHEEL=""
TORCH_RECIPE_DIST_DIR="${LOG_DIR}/torch-recipe-dist"
TORCH_RECIPE_WHEEL=""
BASELINE_CONTAINERS="${LOG_DIR}/existing-containers.tsv"
BASELINE_IMAGES="${LOG_DIR}/baseline-dangling-images.txt"
FINAL_IMAGES="${LOG_DIR}/final-dangling-images.txt"
CONTAINER_CREATED=0
SOURCE_VOLUME_CREATED=0
WORK_VOLUME_CREATED=0
BASELINE_CAPTURED=0
REQUIRED_RUNTIME_IMAGE=""
REQUIRED_RUNTIME_IMAGE_ID=""

snapshot_images() {
  local destination="$1"
  {
    docker image ls --all --quiet --no-trunc --filter dangling=true
    docker image ls --all --no-trunc \
      --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' |
      awk -F '\t' '$1 == "<none>" || $2 == "<none>" {print $3}'
  } | sort -u >"${destination}"
}

snapshot_existing_containers() {
  docker ps --all \
    --format '{{.ID}}\t{{.State}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' \
    >"${BASELINE_CONTAINERS}"
  snapshot_images "${BASELINE_IMAGES}"
  BASELINE_CAPTURED=1
}

report_existing_container_changes() {
  local container_id expected_state container_name compose_project actual_state
  while IFS=$'\t' read -r \
    container_id expected_state container_name compose_project; do
    [[ -n "${container_id}" ]] || continue
    actual_state="$(docker inspect --format '{{.State.Status}}' "${container_id}" 2>/dev/null || true)"
    if [[ "${actual_state}" != "${expected_state}" ]]; then
      echo "Diagnostic: pre-existing container changed independently: ${container_name} (${container_id}, project=${compose_project:-none}) ${expected_state} -> ${actual_state:-missing}" >&2
    fi
  done <"${BASELINE_CONTAINERS}"
}

report_global_image_changes() {
  local added image_id compose_project
  snapshot_images "${FINAL_IMAGES}"
  added="$(comm -13 "${BASELINE_IMAGES}" "${FINAL_IMAGES}")"
  if [[ -n "${added}" ]]; then
    while IFS= read -r image_id; do
      [[ -n "${image_id}" ]] || continue
      compose_project="$(docker image inspect \
        --format '{{index .Config.Labels "com.docker.compose.project"}}' \
        "${image_id}" 2>/dev/null || true)"
      if [[ -z "${compose_project}" || "${compose_project}" == "<no value>" ]]; then
        compose_project="unattributed"
      fi
      echo "Diagnostic: Docker daemon gained dangling or <none> image ${image_id} (compose_project=${compose_project})" >&2
    done <<<"${added}"
    echo "Global Docker image changes are diagnostic only; this runner performs no image build after resolving its required content-addressed image" >&2
    return 0
  fi
  echo "Diagnostic: Docker daemon gained no dangling or <none> image IDs"
}

verify_required_runtime_image() {
  local actual_image_id
  [[ -n "${REQUIRED_RUNTIME_IMAGE}" && -n "${REQUIRED_RUNTIME_IMAGE_ID}" ]] || return 0
  actual_image_id="$(docker image inspect \
    --format '{{.Id}}' "${REQUIRED_RUNTIME_IMAGE}" 2>/dev/null || true)"
  if [[ "${actual_image_id}" != "${REQUIRED_RUNTIME_IMAGE_ID}" ]]; then
    echo "Required runtime image identity changed: ${REQUIRED_RUNTIME_IMAGE} expected=${REQUIRED_RUNTIME_IMAGE_ID} actual=${actual_image_id:-missing}" >&2
    return 1
  fi
  echo "Required runtime image verified: ${REQUIRED_RUNTIME_IMAGE_ID}"
}

cleanup() {
  local test_status=$?
  local cleanup_status=0
  trap - EXIT
  if [[ "${CONTAINER_CREATED}" -eq 1 ]]; then
    docker logs "${CONTAINER_NAME}" >"${LOG_DIR}/container.log" 2>&1 || true
    docker rm --force "${CONTAINER_NAME}" >/dev/null || cleanup_status=1
  fi
  if [[ "${SOURCE_VOLUME_CREATED}" -eq 1 ]]; then
    docker volume rm "${SOURCE_VOLUME_NAME}" >/dev/null || cleanup_status=1
  fi
  if [[ "${WORK_VOLUME_CREATED}" -eq 1 ]]; then
    docker volume rm "${WORK_VOLUME_NAME}" >/dev/null || cleanup_status=1
  fi
  if [[ "${BASELINE_CAPTURED}" -eq 1 ]]; then
    verify_required_runtime_image || cleanup_status=1
    report_global_image_changes
    report_existing_container_changes
  fi
  echo "Local algorithm IT log: ${TEST_LOG}"
  if [[ "${test_status}" -eq 0 && "${cleanup_status}" -eq 0 ]]; then
    echo "Result: PASS (owned container and volumes removed)"
    exit 0
  fi
  echo "Result: FAIL (test_status=${test_status}, cleanup_status=${cleanup_status})" >&2
  [[ "${test_status}" -ne 0 ]] && exit "${test_status}"
  exit "${cleanup_status}"
}

trap cleanup EXIT
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
command -v docker >/dev/null
command -v uv >/dev/null
docker info >/dev/null
snapshot_existing_containers

uv build \
  --wheel \
  --out-dir "${PLUGIN_DIST_DIR}" \
  --no-create-gitignore \
  tests/fixtures/distributed_algorithm_plugin \
  2>&1 | tee "${LOG_DIR}/plugin-wheel.log"
shopt -s nullglob
plugin_wheels=("${PLUGIN_DIST_DIR}"/*.whl)
shopt -u nullglob
if [[ "${#plugin_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one third-party plugin wheel, found ${#plugin_wheels[@]}" >&2
  exit 1
fi
PLUGIN_WHEEL="${plugin_wheels[0]}"

uv build \
  --wheel \
  --out-dir "${TORCH_RECIPE_DIST_DIR}" \
  --no-create-gitignore \
  tests/fixtures/torch_recipe_algorithm_plugin \
  2>&1 | tee "${LOG_DIR}/torch-recipe-wheel.log"
shopt -s nullglob
torch_recipe_wheels=("${TORCH_RECIPE_DIST_DIR}"/*.whl)
shopt -u nullglob
if [[ "${#torch_recipe_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one Torch recipe plugin wheel, found ${#torch_recipe_wheels[@]}" >&2
  exit 1
fi
TORCH_RECIPE_WHEEL="${torch_recipe_wheels[0]}"

prepare_args=(prepare-runtime --profile data-ingestion)
if [[ -n "${TRIBUTO_IT_RUNTIME_REGISTRY:-}" ]]; then
  prepare_args+=(--registry "${TRIBUTO_IT_RUNTIME_REGISTRY}")
fi
if [[ "${TRIBUTO_IT_ALLOW_LOCAL_BUILD:-1}" != "1" ]]; then
  prepare_args+=(--no-local-build)
fi
python3 tools/tributo_it.py "${prepare_args[@]}" 2>&1 | tee "${IMAGE_LOG}"
RUNTIME_IMAGE="$({
  python3 tools/tributo_it.py runtime-key --profile data-ingestion
} | python3 -c 'import json, sys; print(json.loads(sys.stdin.read().splitlines()[-1])["local_tag"])')"
REQUIRED_RUNTIME_IMAGE="${RUNTIME_IMAGE}"
REQUIRED_RUNTIME_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${RUNTIME_IMAGE}")"

if docker ps --all --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Refusing to replace existing container ${CONTAINER_NAME}" >&2
  exit 2
fi
for volume_name in "${SOURCE_VOLUME_NAME}" "${WORK_VOLUME_NAME}"; do
  if docker volume ls --format '{{.Name}}' | grep -Fxq "${volume_name}"; then
    echo "Refusing to replace existing volume ${volume_name}" >&2
    exit 2
  fi
done

docker volume create "${SOURCE_VOLUME_NAME}" >/dev/null
SOURCE_VOLUME_CREATED=1
docker volume create "${WORK_VOLUME_NAME}" >/dev/null
WORK_VOLUME_CREATED=1
docker run --rm \
  --user 0:0 \
  --volume "${PROJECT_ROOT}:/host-source:ro" \
  --volume "${SOURCE_VOLUME_NAME}:/workspace/tributo-src" \
  "${RUNTIME_IMAGE}" \
  /opt/tributo/.venv/bin/python \
  /host-source/tools/tributo_it.py create-source-snapshot \
  --source /host-source \
  --destination /workspace/tributo-src \
  --owner-uid 1000 \
  --owner-gid 100
docker run --rm \
  --user 0:0 \
  --volume "${WORK_VOLUME_NAME}:/workspace/tributo-work" \
  "${RUNTIME_IMAGE}" \
  sh -c 'mkdir -p /workspace/tributo-work/cache /workspace/tributo-work/tmp && chown -R 1000:100 /workspace/tributo-work'
docker create \
  --name "${CONTAINER_NAME}" \
  --init \
  --shm-size 2gb \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/workspace/tributo-plugin.whl:/workspace/tributo-torch-recipe.whl:/workspace/tributo-src/src:/workspace/tributo-src \
  --env TRIBUTO_DOCKER_ALGORITHM_LOCAL_IT=1 \
  --env "TRIBUTO_ALGORITHM_LOCAL_ONLY=${TRIBUTO_ALGORITHM_LOCAL_ONLY:-}" \
  --env TMPDIR=/workspace/tributo-work/tmp \
  --env XDG_CACHE_HOME=/workspace/tributo-work/cache \
  --volume "${SOURCE_VOLUME_NAME}:/workspace/tributo-src:ro" \
  --volume "${PLUGIN_WHEEL}:/workspace/tributo-plugin.whl:ro" \
  --volume "${TORCH_RECIPE_WHEEL}:/workspace/tributo-torch-recipe.whl:ro" \
  --volume "${WORK_VOLUME_NAME}:/workspace/tributo-work" \
  --workdir /workspace/tributo-src \
  "${RUNTIME_IMAGE}" \
  /opt/tributo/.venv/bin/python -m pytest \
  tests/integration/test_distributed_algorithm_local.py \
  -o addopts= \
  -o cache_dir=/workspace/tributo-work/cache/pytest-algorithm-local \
  -m integration -vv -rP --tb=short --timeout=1200 >/dev/null
CONTAINER_CREATED=1
docker start --attach "${CONTAINER_NAME}" 2>&1 | tee "${TEST_LOG}"
