#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/tests/integrations/docker-compose.data-ingestion.yml"
COMPOSE_OVERRIDE="${PROJECT_ROOT}/tests/integrations/docker-compose.distributed-algorithm.yml"
VERSIONS_FILE="${PROJECT_ROOT}/tests/integrations/component-versions.env"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tributo-distributed-algorithm-it-$(date +%Y%m%d%H%M%S)-$$}"
if [[ ! "${PROJECT_NAME}" =~ ^tributo-distributed-algorithm-it-[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "COMPOSE_PROJECT_NAME must be a unique tributo-distributed-algorithm-it-* identifier" >&2
  exit 2
fi
export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export OMP_NUM_THREADS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

LOG_DIR="/tmp/${PROJECT_NAME}"
BASELINE_CONTAINERS="${LOG_DIR}/existing-containers.tsv"
BASELINE_IMAGES="${LOG_DIR}/baseline-dangling-images.txt"
FINAL_IMAGES="${LOG_DIR}/final-dangling-images.txt"
TEST_LOG="${LOG_DIR}/tests.log"
SERVICE_LOG="${LOG_DIR}/services.log"
RAY_JOB_LOG="${LOG_DIR}/ray-jobs.log"
PLUGIN_DIST_DIR="${LOG_DIR}/plugin-dist"
PLUGIN_CONTAINER_DIR="/workspace/tributo-work/distributed-plugin"
PLUGIN_WHEEL=""
BASELINE_CAPTURED=0
COMPOSE_TOUCHED=0
RAY_NODE_WAIT_SECONDS=120
RAY_NODE_STABLE_SAMPLES=3
REQUIRED_RUNTIME_IMAGE=""
REQUIRED_RUNTIME_IMAGE_ID=""

cd "${PROJECT_ROOT}"
if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing to reuse existing IT log directory ${LOG_DIR}" >&2
  exit 2
fi
mkdir -p "${PLUGIN_DIST_DIR}"

compose() {
  docker compose \
    --env-file "${VERSIONS_FILE}" \
    --project-name "${PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    --file "${COMPOSE_OVERRIDE}" \
    "$@"
}

project_resource_ids() {
  local label="com.docker.compose.project=${PROJECT_NAME}"
  docker ps --all --quiet --filter "label=${label}"
  docker network ls --quiet --filter "label=${label}"
  docker volume ls --quiet --filter "label=${label}"
}

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
    echo "Global Docker image changes are diagnostic only; this runner uses Compose --no-build --pull never with its required content-addressed image" >&2
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

read_alive_ray_nodes() {
  compose exec -T ray-head \
    ray list nodes \
    --format json \
    --filter state=ALIVE \
    --limit 100 \
    --address http://127.0.0.1:8265 2>>"${LOG_DIR}/cluster-readiness.log" |
    python3 -c 'import json, sys; print(len(json.load(sys.stdin)))' \
      2>>"${LOG_DIR}/cluster-readiness.log"
}

wait_for_stable_ray_nodes() {
  local expected_nodes="$1"
  local deadline=$((SECONDS + RAY_NODE_WAIT_SECONDS))
  local observed_nodes="unavailable"
  local node_count
  local stable_samples=0
  while ((SECONDS < deadline)); do
    if node_count="$(read_alive_ray_nodes)" && [[ "${node_count}" =~ ^[0-9]+$ ]]; then
      observed_nodes="${node_count}"
      if ((node_count == expected_nodes)); then
        stable_samples=$((stable_samples + 1))
      else
        stable_samples=0
      fi
    else
      observed_nodes="unavailable"
      stable_samples=0
    fi
    printf '%s expected_nodes=%d observed_nodes=%s stable_samples=%d/%d\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "${expected_nodes}" \
      "${observed_nodes}" \
      "${stable_samples}" \
      "${RAY_NODE_STABLE_SAMPLES}" >>"${LOG_DIR}/cluster-readiness.log"
    if ((stable_samples >= RAY_NODE_STABLE_SAMPLES)); then
      echo "Ray cluster stable with ${expected_nodes} alive nodes"
      return 0
    fi
    sleep 2
  done
  echo "Ray cluster did not stabilize at ${expected_nodes} nodes (last observed: ${observed_nodes})" >&2
  return 1
}

cleanup() {
  local test_status=$?
  local cleanup_status=0
  trap - EXIT
  if [[ "${COMPOSE_TOUCHED}" -eq 1 ]]; then
    compose cp \
      "ray-head:/workspace/tributo-work/distributed-algorithm-ray-jobs.log" \
      "${RAY_JOB_LOG}" >/dev/null 2>&1 || true
    compose logs --no-color --timestamps >"${SERVICE_LOG}" 2>&1 || cleanup_status=1
    compose down --volumes --remove-orphans --timeout 30 || cleanup_status=1
  fi
  if project_resource_ids | grep -q .; then
    echo "Owned Compose resources remain after cleanup: ${PROJECT_NAME}" >&2
    cleanup_status=1
  fi
  if [[ "${BASELINE_CAPTURED}" -eq 1 ]]; then
    verify_required_runtime_image || cleanup_status=1
    report_global_image_changes
    report_existing_container_changes
  fi
  echo "Distributed algorithm IT project: ${PROJECT_NAME}"
  echo "Test log: ${TEST_LOG}"
  echo "Ray Job log: ${RAY_JOB_LOG}"
  echo "Service log: ${SERVICE_LOG}"
  if [[ "${test_status}" -eq 0 && "${cleanup_status}" -eq 0 ]]; then
    echo "Result: PASS (owned containers, network, and volumes removed)"
    exit 0
  fi
  echo "Result: FAIL (test_status=${test_status}, cleanup_status=${cleanup_status})" >&2
  if [[ "${test_status}" -ne 0 ]]; then
    exit "${test_status}"
  fi
  exit "${cleanup_status}"
}

trap cleanup EXIT

command -v docker >/dev/null
command -v uv >/dev/null
docker info >/dev/null
docker compose version >/dev/null
test -r "${VERSIONS_FILE}"
test -r "${COMPOSE_FILE}"
test -r "${COMPOSE_OVERRIDE}"
snapshot_existing_containers

if project_resource_ids | grep -q .; then
  echo "Refusing to take over existing Compose project resources: ${PROJECT_NAME}" >&2
  exit 2
fi

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

set -a
# shellcheck disable=SC1090
source "${VERSIONS_FILE}"
set +a

prepare_args=(prepare-runtime --profile data-ingestion)
if [[ -n "${TRIBUTO_IT_RUNTIME_REGISTRY:-}" ]]; then
  prepare_args+=(--registry "${TRIBUTO_IT_RUNTIME_REGISTRY}")
fi
if [[ "${TRIBUTO_IT_ALLOW_LOCAL_BUILD:-1}" != "1" ]]; then
  prepare_args+=(--no-local-build)
fi
python3 "${PROJECT_ROOT}/tools/tributo_it.py" "${prepare_args[@]}" \
  2>&1 | tee "${LOG_DIR}/runtime-prepare.log"

export TRIBUTO_IT_RUNTIME_IMAGE
TRIBUTO_IT_RUNTIME_IMAGE="$({
  python3 "${PROJECT_ROOT}/tools/tributo_it.py" runtime-key --profile data-ingestion
} | python3 -c 'import json, sys; print(json.loads(sys.stdin.read().splitlines()[-1])["local_tag"])')"
REQUIRED_RUNTIME_IMAGE="${TRIBUTO_IT_RUNTIME_IMAGE}"
REQUIRED_RUNTIME_IMAGE_ID="$(docker image inspect \
  --format '{{.Id}}' "${TRIBUTO_IT_RUNTIME_IMAGE}")"
export TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE}"
export TRIBUTO_IT_SOURCE_ROOT="${PROJECT_ROOT}"
export TRIBUTO_IT_TOOL_IMAGE="${TOOL_IMAGE}"

python3 -c \
  'from tools.tributo_it import ensure_digest_image, load_profile; p = load_profile("data-ingestion"); ensure_digest_image(p.tool_image); ensure_digest_image(p.minio_image)' \
  2>&1 | tee "${LOG_DIR}/infrastructure-images.log"

compose config --quiet
COMPOSE_TOUCHED=1
compose up \
  --detach \
  --no-build \
  --pull never \
  --wait \
  --wait-timeout "${RAY_NODE_WAIT_SECONDS}" \
  --scale ray-worker=2
wait_for_stable_ray_nodes 3

compose exec -T ray-head mkdir -p "${PLUGIN_CONTAINER_DIR}"
compose cp "${PLUGIN_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
PLUGIN_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${PLUGIN_WHEEL}")"

compose exec -T \
  --env TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT=1 \
  --env "TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL=${PLUGIN_CONTAINER_WHEEL}" \
  --env TRIBUTO_DISTRIBUTED_GATE_LOG=/workspace/tributo-work/distributed-algorithm-ray-jobs.log \
  ray-head \
  python -m pytest \
  tests/training/test_dnn_pu_training.py::test_formal_distributed_algorithms_complete_on_ray_cluster \
  -o addopts= \
  -o cache_dir=/workspace/tributo-work/cache/pytest-distributed-algorithm \
  -m integration -vv -rP --tb=short --timeout=1200 2>&1 | tee "${TEST_LOG}"
