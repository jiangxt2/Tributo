#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/tests/integrations/docker-compose.data-ingestion.yml"
VERSIONS_FILE="${PROJECT_ROOT}/tests/integrations/component-versions.env"

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tributo-tune-it-$(date +%Y%m%d%H%M%S)-$$}"
if [[ ! "${PROJECT_NAME}" =~ ^tributo-tune-it-[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "COMPOSE_PROJECT_NAME must be a unique tributo-tune-it-* identifier" >&2
  exit 2
fi
export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export OMP_NUM_THREADS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
LOG_DIR="/tmp/${PROJECT_NAME}"
BASELINE_FILE="${LOG_DIR}/existing-containers.tsv"
TEST_LOG="${LOG_DIR}/tests.log"
READINESS_LOG="${LOG_DIR}/cluster-readiness.log"
SERVICE_LOG="${LOG_DIR}/services.log"
BASELINE_CAPTURED=0
COMPOSE_TOUCHED=0
RAY_NODE_WAIT_SECONDS=60
RAY_NODE_STABLE_SAMPLES=3
RAY_NODE_SAMPLE_INTERVAL_SECONDS=2

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

compose() {
  docker compose \
    --env-file "${VERSIONS_FILE}" \
    --project-name "${PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    "$@"
}

read_alive_ray_nodes() {
  compose exec -T ray-head \
    ray list nodes \
    --format json \
    --filter state=ALIVE \
    --limit 100 \
    --address http://127.0.0.1:8265 \
    2>>"${READINESS_LOG}" |
    python3 -c 'import json, sys; print(len(json.load(sys.stdin)))' \
      2>>"${READINESS_LOG}"
}

wait_for_stable_ray_nodes() {
  local expected_nodes="$1"
  local deadline=$((SECONDS + RAY_NODE_WAIT_SECONDS))
  local observed_nodes="unavailable"
  local node_count
  local stable_samples=0

  while ((SECONDS < deadline)); do
    observed_nodes="unavailable"
    if node_count="$(read_alive_ray_nodes)" &&
      [[ "${node_count}" =~ ^[0-9]+$ ]]; then
      observed_nodes="${node_count}"
      if ((node_count == expected_nodes)); then
        stable_samples=$((stable_samples + 1))
      else
        stable_samples=0
      fi
    else
      stable_samples=0
    fi

    printf '%s expected_nodes=%d observed_nodes=%s stable_samples=%d/%d\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "${expected_nodes}" \
      "${observed_nodes}" \
      "${stable_samples}" \
      "${RAY_NODE_STABLE_SAMPLES}" >>"${READINESS_LOG}"

    if ((stable_samples >= RAY_NODE_STABLE_SAMPLES)); then
      echo "Ray cluster stable with ${expected_nodes} alive nodes"
      return 0
    fi
    sleep "${RAY_NODE_SAMPLE_INTERVAL_SECONDS}"
  done

  echo "Ray cluster did not stabilize at ${expected_nodes} alive nodes within ${RAY_NODE_WAIT_SECONDS}s (last observed: ${observed_nodes})" >&2
  return 1
}

scale_ray_workers() {
  local worker_replicas="$1"
  local expected_nodes=$((worker_replicas + 1))

  echo "Scaling Ray workers to ${worker_replicas} replica(s)"
  if ((worker_replicas == 1)); then
    compose up \
      --detach \
      --no-build \
      --pull never \
      --wait \
      --wait-timeout "${RAY_NODE_WAIT_SECONDS}" \
      --scale "ray-worker=${worker_replicas}"
  else
    compose up \
      --detach \
      --no-build \
      --pull never \
      --wait \
      --wait-timeout "${RAY_NODE_WAIT_SECONDS}" \
      --no-deps \
      --no-recreate \
      --scale "ray-worker=${worker_replicas}" \
      ray-worker
  fi
  wait_for_stable_ray_nodes "${expected_nodes}"
}

project_resource_ids() {
  local label="com.docker.compose.project=${PROJECT_NAME}"
  docker ps --all --quiet --filter "label=${label}"
  docker network ls --quiet --filter "label=${label}"
  docker volume ls --quiet --filter "label=${label}"
}

snapshot_existing_containers() {
  docker ps --all \
    --format '{{.ID}}\t{{.State}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' \
    >"${BASELINE_FILE}"
  BASELINE_CAPTURED=1
}

verify_existing_containers_unchanged() {
  local container_id expected_state container_name compose_project actual_state
  local changed=0
  while IFS=$'\t' read -r \
    container_id expected_state container_name compose_project; do
    [[ -n "${container_id}" ]] || continue
    actual_state="$(docker inspect --format '{{.State.Status}}' "${container_id}" 2>/dev/null || true)"
    if [[ "${actual_state}" != "${expected_state}" ]]; then
      echo "Concurrent container change: ${container_name} (${container_id}, project=${compose_project:-none}) ${expected_state} -> ${actual_state:-missing}" >&2
      changed=1
    fi
  done <"${BASELINE_FILE}"
  return "${changed}"
}

cleanup() {
  local test_status=$?
  local cleanup_status=0
  trap - EXIT

  if [[ "${COMPOSE_TOUCHED}" -eq 1 ]]; then
    compose logs --no-color >"${SERVICE_LOG}" 2>&1 || cleanup_status=1
    compose down --volumes --remove-orphans || cleanup_status=1

    if docker ps --all --quiet \
      --filter "label=com.docker.compose.project=${PROJECT_NAME}" | grep -q .; then
      echo "Owned Compose containers remain after cleanup" >&2
      cleanup_status=1
    fi
    if docker network ls --quiet \
      --filter "label=com.docker.compose.project=${PROJECT_NAME}" | grep -q .; then
      echo "Owned Compose networks remain after cleanup" >&2
      cleanup_status=1
    fi
    if docker volume ls --quiet \
      --filter "label=com.docker.compose.project=${PROJECT_NAME}" | grep -q .; then
      echo "Owned Compose volumes remain after cleanup" >&2
      cleanup_status=1
    fi
  fi
  if [[ "${BASELINE_CAPTURED}" -eq 1 ]]; then
    if ! verify_existing_containers_unchanged; then
      echo "Other Docker activity changed pre-existing containers; this IT only targeted project ${PROJECT_NAME}" >&2
    fi
  fi

  echo "Tune IT project: ${PROJECT_NAME}"
  echo "Test log: ${TEST_LOG}"
  echo "Cluster readiness log: ${READINESS_LOG}"
  echo "Service log: ${SERVICE_LOG}"
  if [[ "${test_status}" -eq 0 && "${cleanup_status}" -eq 0 ]]; then
    echo "Result: PASS (owned containers and volumes removed)"
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
docker info >/dev/null
docker compose version >/dev/null
test -r "${VERSIONS_FILE}"
test -r "${COMPOSE_FILE}"
snapshot_existing_containers

if project_resource_ids | grep -q .; then
  echo "Refusing to take over existing Compose project resources: ${PROJECT_NAME}" >&2
  exit 2
fi

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
python3 "${PROJECT_ROOT}/tools/tributo_it.py" "${prepare_args[@]}"

export TRIBUTO_IT_RUNTIME_IMAGE
TRIBUTO_IT_RUNTIME_IMAGE="$(
  python3 "${PROJECT_ROOT}/tools/tributo_it.py" runtime-key --profile data-ingestion |
    python3 -c 'import json, sys; print(json.loads(sys.stdin.read().splitlines()[-1])["local_tag"])'
)"
export TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE%%@*}"
export TRIBUTO_IT_SOURCE_ROOT="${PROJECT_ROOT}"
export TRIBUTO_IT_TOOL_IMAGE="${TOOL_IMAGE%%@*}"

python3 -c \
  'from tools.tributo_it import ensure_digest_image, load_profile; p = load_profile("data-ingestion"); ensure_digest_image(p.tool_image); ensure_digest_image(p.minio_image)' \
  2>&1 | tee "${LOG_DIR}/image-prepare.log"

compose config --quiet

COMPOSE_TOUCHED=1
scale_ray_workers 1
scale_ray_workers 2
scale_ray_workers 3

compose exec -T \
  --env TRIBUTO_DOCKER_TUNE_IT=1 \
  ray-head \
  python -m pytest \
  tests/integration/test_tune_ray_cluster.py \
  -o addopts= \
  -o cache_dir=/workspace/tributo-work/cache/pytest-tune \
  -m integration -vv -rP --tb=short --timeout=1200 2>&1 | tee "${TEST_LOG}"
