#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"
compose_file="${project_root}/tests/integrations/docker-compose.inference.yml"
versions_file="${project_root}/tests/integrations/inference-it-versions.conf"

set -a
# shellcheck disable=SC1090
. "${versions_file}"
set +a

if [ -z "${COMPOSE_PROJECT_NAME:-}" ]; then
  COMPOSE_PROJECT_NAME="tributo-explainability-$(date +%s)-$$"
fi
case "${COMPOSE_PROJECT_NAME}" in
  *[!a-zA-Z0-9_-]*|"")
    echo "COMPOSE_PROJECT_NAME contains unsafe characters" >&2
    exit 2
    ;;
esac
export COMPOSE_PROJECT_NAME
export TRIBUTO_INFERENCE_IMAGE_TAG="${TRIBUTO_INFERENCE_IMAGE_TAG:-${COMPOSE_PROJECT_NAME}}"

compose=(docker compose --project-name "${COMPOSE_PROJECT_NAME}" --file "${compose_file}")
test_log="/tmp/${COMPOSE_PROJECT_NAME}-test.log"
service_log="/tmp/${COMPOSE_PROJECT_NAME}-services.log"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  "${compose[@]}" logs --no-color >"${service_log}" 2>&1
  "${compose[@]}" down --volumes --remove-orphans
  if [ -n "$("${compose[@]}" ps -aq 2>/dev/null)" ]; then
    status=1
  fi
  echo "Explainability IT logs: ${test_log}"
  echo "Explainability service logs: ${service_log}"
  exit "${status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${project_root}"
"${compose[@]}" up --detach --build

ready=0
for attempt in $(seq 1 60); do
  if "${compose[@]}" exec -T ray-head python -c \
    "import ray; ray.init(address='auto'); assert ray.cluster_resources().get('CPU', 0) >= 4" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  echo "Waiting for Docker Ray worker (attempt ${attempt}/60)"
  sleep 2
done
if [ "${ready}" -ne 1 ]; then
  echo "Docker Ray cluster did not become ready" >&2
  exit 1
fi

"${compose[@]}" exec -T ray-head python -m tests.integrations.verify_inference_it_versions
set -o pipefail
"${compose[@]}" exec -T \
  -e TRIBUTO_DOCKER_EXPLAINABILITY_IT=1 \
  ray-head python -m pytest \
  tests/integration/test_explainability_ray_jobs.py \
  -o "addopts=" -m integration -v --tb=short --timeout=600 \
  2>&1 | tee "${test_log}"
