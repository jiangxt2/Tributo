#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/tests/integrations/docker-compose.data-ingestion.yml"
VERSIONS_FILE="${PROJECT_ROOT}/tests/integrations/component-versions.env"

tributo_it() {
  uv run --no-project --python 3.12 python \
    "${PROJECT_ROOT}/tools/tributo_it.py" "$@"
}

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-tributo-algorithm-contract-it-$(date +%Y%m%d%H%M%S)-$$}"
if [[ ! "${PROJECT_NAME}" =~ ^tributo-algorithm-contract-it-[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "COMPOSE_PROJECT_NAME must be a unique tributo-algorithm-contract-it-* identifier" >&2
  exit 2
fi
export COMPOSE_PROJECT_NAME="${PROJECT_NAME}"
export OMP_NUM_THREADS=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export TRIBUTO_IT_SOURCE_ROOT="${PROJECT_ROOT}"

LOG_DIR="/tmp/${PROJECT_NAME}"
RAY_NODE_COMPOSE_OVERRIDE="${LOG_DIR}/ray-node-tempdir-override.yml"
BASELINE_CONTAINERS="${LOG_DIR}/existing-containers.tsv"
BASELINE_IMAGES="${LOG_DIR}/baseline-images.tsv"
BASELINE_DANGLING_IMAGES="${LOG_DIR}/baseline-dangling-images.txt"
BASELINE_DOCKER_DF="${LOG_DIR}/baseline-docker-df.txt"
FINAL_IMAGES="${LOG_DIR}/final-images.tsv"
FINAL_DANGLING_IMAGES="${LOG_DIR}/final-dangling-images.txt"
FINAL_DOCKER_DF="${LOG_DIR}/final-docker-df.txt"
IMAGE_LOG="${LOG_DIR}/image-prepare.log"
TEST_LOG="${LOG_DIR}/tests.log"
SERVICE_LOG="${LOG_DIR}/services.log"
RAY_JOB_LOG="${LOG_DIR}/ray-jobs.log"
CORE_DIST_DIR="${LOG_DIR}/core-dist"
PLUGIN_DIST_DIR="${LOG_DIR}/plugin-dist"
TORCH_RECIPE_BUNDLE_DIR="${LOG_DIR}/torch-recipe-bundle"
OFFLINE_DIST_DIR="${LOG_DIR}/offline-dist"
OFFLINE_BUNDLE_DIR="${LOG_DIR}/offline-bundle"
OFFLINE_BUNDLE_ARCHIVE="${LOG_DIR}/offline-bundle.zip"
PLUGIN_CONTAINER_DIR="/workspace/tributo-work/algorithm-contract"
OFFLINE_CONTAINER_DIR="${PLUGIN_CONTAINER_DIR}/offline-bundle"
OFFLINE_ARCHIVE_CONTAINER_PATH="${PLUGIN_CONTAINER_DIR}/offline-bundle.zip"
OFFLINE_BUCKET="tributo-algorithm-contract-it"
OFFLINE_KEY="${PROJECT_NAME}/offline-bundle.zip"
OFFLINE_URI="s3://${OFFLINE_BUCKET}/${OFFLINE_KEY}"
TORCH_RECIPE_CONTAINER_DIR="${PLUGIN_CONTAINER_DIR}/torch-recipe-bundle"

DOCKER=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy docker)
COMPOSE_TOUCHED=0
BASELINE_CAPTURED=0
REQUIRED_RUNTIME_IMAGE=""
REQUIRED_RUNTIME_IMAGE_ID=""

compose() {
  "${DOCKER[@]}" compose \
    --env-file "${VERSIONS_FILE}" \
    --project-name "${PROJECT_NAME}" \
    --file "${COMPOSE_FILE}" \
    --file "${RAY_NODE_COMPOSE_OVERRIDE}" \
    "$@"
}

project_resources() {
  local label="com.docker.compose.project=${PROJECT_NAME}"
  "${DOCKER[@]}" ps --all --quiet --filter "label=${label}" || return $?
  "${DOCKER[@]}" network ls --quiet --filter "label=${label}" || return $?
  "${DOCKER[@]}" volume ls --quiet --filter "label=${label}" || return $?
}

snapshot_images() {
  "${DOCKER[@]}" image ls --all --no-trunc \
    --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' |
    sort -u >"$1"
}

snapshot_dangling_images() {
  {
    "${DOCKER[@]}" image ls --all --quiet --no-trunc --filter dangling=true
    "${DOCKER[@]}" image ls --all --no-trunc \
      --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' |
      awk -F '\t' '$1 == "<none>" || $2 == "<none>" {print $3}'
  } | sort -u >"$1"
}

snapshot_existing_state() {
  "${DOCKER[@]}" ps --all \
    --format '{{.ID}}\t{{.State}}\t{{.Names}}\t{{.Label "com.docker.compose.project"}}' \
    >"${BASELINE_CONTAINERS}"
  snapshot_images "${BASELINE_IMAGES}"
  snapshot_dangling_images "${BASELINE_DANGLING_IMAGES}"
  "${DOCKER[@]}" system df >"${BASELINE_DOCKER_DF}"
  BASELINE_CAPTURED=1
}

report_existing_container_changes() {
  local container_id expected_state container_name compose_project actual_state
  while IFS=$'\t' read -r \
    container_id expected_state container_name compose_project; do
    [[ -n "${container_id}" ]] || continue
    actual_state="$("${DOCKER[@]}" inspect \
      --format '{{.State.Status}}' "${container_id}" 2>/dev/null || true)"
    if [[ "${actual_state}" != "${expected_state}" ]]; then
      echo "Diagnostic: pre-existing container changed independently: ${container_name} (${container_id}, project=${compose_project:-none}) ${expected_state} -> ${actual_state:-missing}" >&2
    fi
  done <"${BASELINE_CONTAINERS}"
}

verify_required_runtime_image() {
  local actual_image_id
  [[ -n "${REQUIRED_RUNTIME_IMAGE}" && -n "${REQUIRED_RUNTIME_IMAGE_ID}" ]] || return 0
  actual_image_id="$("${DOCKER[@]}" image inspect \
    --format '{{.Id}}' "${REQUIRED_RUNTIME_IMAGE}" 2>/dev/null || true)"
  if [[ "${actual_image_id}" != "${REQUIRED_RUNTIME_IMAGE_ID}" ]]; then
    echo "Required runtime image identity changed: ${REQUIRED_RUNTIME_IMAGE} expected=${REQUIRED_RUNTIME_IMAGE_ID} actual=${actual_image_id:-missing}" >&2
    return 1
  fi
}

cleanup() {
  local test_status=$?
  local cleanup_status=0
  local added_images image_id compose_project resource_check_output
  local resource_check_status
  trap - EXIT INT TERM
  set +e
  if [[ "${COMPOSE_TOUCHED}" -eq 1 ]]; then
    compose cp \
      "ray-head:/workspace/tributo-work/algorithm-contract-ray-jobs.log" \
      "${RAY_JOB_LOG}" >/dev/null 2>&1 || true
    compose logs --no-color --timestamps >"${SERVICE_LOG}" 2>&1 || cleanup_status=1
    compose down --volumes --remove-orphans --timeout 30 || cleanup_status=1
  fi
  resource_check_output="$(project_resources 2>"${LOG_DIR}/cleanup-resource-check.log")"
  resource_check_status=$?
  if [[ "${resource_check_status}" -ne 0 ]]; then
    echo "Unable to verify owned Compose resources after cleanup: ${PROJECT_NAME}" >&2
    cat "${LOG_DIR}/cleanup-resource-check.log" >&2
    cleanup_status=1
  elif [[ -n "${resource_check_output}" ]]; then
    echo "Owned Compose resources remain after cleanup: ${PROJECT_NAME}" >&2
    printf '%s\n' "${resource_check_output}" >&2
    cleanup_status=1
  fi
  if [[ "${BASELINE_CAPTURED}" -eq 1 ]]; then
    verify_required_runtime_image || cleanup_status=1
    report_existing_container_changes
    snapshot_images "${FINAL_IMAGES}"
    snapshot_dangling_images "${FINAL_DANGLING_IMAGES}"
    "${DOCKER[@]}" system df >"${FINAL_DOCKER_DF}" || cleanup_status=1
    added_images="$(comm -13 "${BASELINE_DANGLING_IMAGES}" "${FINAL_DANGLING_IMAGES}")"
    if [[ -n "${added_images}" ]]; then
      while IFS= read -r image_id; do
        [[ -n "${image_id}" ]] || continue
        compose_project="$("${DOCKER[@]}" image inspect \
          --format '{{index .Config.Labels "com.docker.compose.project"}}' \
          "${image_id}" 2>/dev/null || true)"
        echo "Diagnostic: Docker daemon gained dangling image ${image_id} (compose_project=${compose_project:-unattributed})" >&2
      done <<<"${added_images}"
    fi
  fi
  echo "Algorithm contract Ray Jobs project: ${PROJECT_NAME}"
  echo "Test log: ${TEST_LOG}"
  echo "Ray Job log: ${RAY_JOB_LOG}"
  echo "Service log: ${SERVICE_LOG}"
  echo "Image logs and snapshots: ${LOG_DIR}"
  if [[ "${test_status}" -eq 0 && "${cleanup_status}" -eq 0 ]]; then
    echo "Result: PASS (owned Compose resources removed)"
    exit 0
  fi
  echo "Result: FAIL (test_status=${test_status}, cleanup_status=${cleanup_status})" >&2
  if [[ "${test_status}" -ne 0 ]]; then
    exit "${test_status}"
  fi
  exit "${cleanup_status}"
}

wait_for_stable_ray_nodes() {
  local expected_nodes="$1"
  local deadline=$((SECONDS + 120))
  local node_count stable_samples=0
  while ((SECONDS < deadline)); do
    if node_count="$(
      compose exec -T ray-head \
        ray list nodes --format json --filter state=ALIVE --limit 100 \
        --address http://127.0.0.1:8265 2>>"${LOG_DIR}/cluster-readiness.log" |
        python3 -c 'import json, sys; print(len(json.load(sys.stdin)))' \
          2>>"${LOG_DIR}/cluster-readiness.log"
    )" && [[ "${node_count}" =~ ^[0-9]+$ ]] && ((node_count == expected_nodes)); then
      stable_samples=$((stable_samples + 1))
    else
      stable_samples=0
    fi
    printf '%s expected_nodes=%d observed_nodes=%s stable_samples=%d/3\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
      "${expected_nodes}" "${node_count:-unavailable}" "${stable_samples}" \
      >>"${LOG_DIR}/cluster-readiness.log"
    if ((stable_samples >= 3)); then
      return 0
    fi
    sleep 2
  done
  echo "Ray cluster did not stabilize at ${expected_nodes} alive nodes" >&2
  return 1
}

if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing to reuse existing IT log directory ${LOG_DIR}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${CORE_DIST_DIR}" "${PLUGIN_DIST_DIR}" "${OFFLINE_DIST_DIR}"
cat >"${RAY_NODE_COMPOSE_OVERRIDE}" <<'YAML'
services:
  ray-head:
    command:
      - /bin/bash
      - -lc
      - >-
        exec ray start
        --head
        --port=6379
        --dashboard-host=0.0.0.0
        --num-cpus=${TRIBUTO_IT_RAY_HEAD_CPUS:-1}
        --temp-dir="/workspace/tributo-work/tmp/ray-$${HOSTNAME}"
        --block
  ray-worker:
    command:
      - /bin/bash
      - -lc
      - >-
        exec ray start
        --address=ray-head:6379
        --num-cpus=2
        --temp-dir="/workspace/tributo-work/tmp/ray-$${HOSTNAME}"
        --block
YAML
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v docker >/dev/null
command -v uv >/dev/null
"${DOCKER[@]}" info >/dev/null
"${DOCKER[@]}" compose version >/dev/null
test -r "${VERSIONS_FILE}"
test -r "${COMPOSE_FILE}"

snapshot_existing_state
if ! existing_resources="$(project_resources 2>"${LOG_DIR}/preflight-resource-check.log")"; then
  echo "Unable to verify whether Compose project resources already exist: ${PROJECT_NAME}" >&2
  cat "${LOG_DIR}/preflight-resource-check.log" >&2
  exit 1
fi
if [[ -n "${existing_resources}" ]]; then
  echo "Refusing to take over existing Compose project resources: ${PROJECT_NAME}" >&2
  printf '%s\n' "${existing_resources}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
set -a
# shellcheck disable=SC1090
source "${VERSIONS_FILE}"
set +a

uv build --wheel --out-dir "${CORE_DIST_DIR}" --no-create-gitignore \
  "${PROJECT_ROOT}" 2>&1 | tee "${LOG_DIR}/core-wheel.log"
shopt -s nullglob
core_wheels=("${CORE_DIST_DIR}"/tributo-*.whl)
if [[ "${#core_wheels[@]}" -ne 1 ]]; then
  echo "Expected one Tributo Core Wheel, found ${#core_wheels[@]}" >&2
  exit 1
fi
CORE_WHEEL="${core_wheels[0]}"

uv build --wheel --out-dir "${PLUGIN_DIST_DIR}" --no-create-gitignore \
  tests/fixtures/torch_recipe_algorithm_plugin \
  2>&1 | tee "${LOG_DIR}/torch-recipe-wheel.log"
plugin_wheels=("${PLUGIN_DIST_DIR}"/*.whl)
if [[ "${#plugin_wheels[@]}" -ne 1 ]]; then
  echo "Expected one TorchRecipe fixture Wheel, found ${#plugin_wheels[@]}" >&2
  exit 1
fi
TORCH_RECIPE_WHEEL="${plugin_wheels[0]}"
uv run --locked python tools/build_algorithm_bundle.py \
  --algorithm-wheel "${TORCH_RECIPE_WHEEL}" \
  --output "${TORCH_RECIPE_BUNDLE_DIR}" \
  --algorithm-id "torch_recipe.fixture" \
  --plugin-name "third_party_binary_linear" \
  --profile-id "data-ingestion.cpu.v1" \
  2>&1 | tee "${LOG_DIR}/torch-recipe-bundle.log"

uv build --wheel --out-dir "${OFFLINE_DIST_DIR}" --no-create-gitignore \
  tests/fixtures/offline_algorithm_dependency \
  2>&1 | tee "${LOG_DIR}/offline-dependency-wheel.log"
uv build --wheel --out-dir "${OFFLINE_DIST_DIR}" --no-create-gitignore \
  tests/fixtures/offline_algorithm_plugin \
  2>&1 | tee "${LOG_DIR}/offline-algorithm-wheel.log"
offline_dependency_wheels=("${OFFLINE_DIST_DIR}"/tributo_test_offline_dependency-*.whl)
offline_algorithm_wheels=("${OFFLINE_DIST_DIR}"/tributo_test_offline_algorithm-*.whl)
if [[ "${#offline_dependency_wheels[@]}" -ne 1 || "${#offline_algorithm_wheels[@]}" -ne 1 ]]; then
  echo "Expected one offline fixture Wheel and one dependency Wheel" >&2
  exit 1
fi
shopt -u nullglob
uv run --locked python tools/build_algorithm_bundle.py \
  --algorithm-wheel "${offline_algorithm_wheels[0]}" \
  --dependency-wheel "${offline_dependency_wheels[0]}" \
  --output "${OFFLINE_BUNDLE_DIR}" \
  --algorithm-id "offline.algorithm" \
  --profile-id "data-ingestion.cpu.v1" \
  2>&1 | tee "${LOG_DIR}/offline-bundle.log"

python3 - "${OFFLINE_BUNDLE_DIR}" "${OFFLINE_BUNDLE_ARCHIVE}" <<'PY'
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(root).as_posix())
PY
OFFLINE_ARCHIVE_SHA256="$(python3 - "${OFFLINE_BUNDLE_ARCHIVE}" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

prepare_args=(prepare-runtime --profile data-ingestion)
if [[ -n "${TRIBUTO_IT_RUNTIME_REGISTRY:-}" ]]; then
  prepare_args+=(--registry "${TRIBUTO_IT_RUNTIME_REGISTRY}")
fi
if [[ "${TRIBUTO_IT_ALLOW_LOCAL_BUILD:-1}" != "1" ]]; then
  prepare_args+=(--no-local-build)
fi
tributo_it "${prepare_args[@]}" 2>&1 | tee "${IMAGE_LOG}"

TRIBUTO_IT_RUNTIME_IMAGE="$(
  tributo_it runtime-key --profile data-ingestion |
    python3 -c 'import json, sys; print(json.loads(sys.stdin.read().splitlines()[-1])["local_tag"])'
)"
export TRIBUTO_IT_RUNTIME_IMAGE
TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE%%@*}"
export TRIBUTO_IT_MINIO_IMAGE
tributo_it prepare-infrastructure --profile data-ingestion \
  2>&1 | tee "${LOG_DIR}/infrastructure-prepare.log"
REQUIRED_RUNTIME_IMAGE="${TRIBUTO_IT_RUNTIME_IMAGE}"
REQUIRED_RUNTIME_IMAGE_ID="$("${DOCKER[@]}" image inspect \
  --format '{{.Id}}' "${REQUIRED_RUNTIME_IMAGE}")"

compose config --quiet
COMPOSE_TOUCHED=1
compose up --detach --no-build --pull never --wait --wait-timeout 120 \
  --scale ray-worker=1
wait_for_stable_ray_nodes 2
compose up --detach --no-build --pull never --wait --wait-timeout 120 \
  --no-deps --no-recreate --scale ray-worker=2 ray-worker
wait_for_stable_ray_nodes 3

RUNTIME_IMAGE_DIGEST="${REQUIRED_RUNTIME_IMAGE_ID#sha256:}"
compose exec -T ray-head mkdir -p \
  "${PLUGIN_CONTAINER_DIR}" \
  "${TORCH_RECIPE_CONTAINER_DIR}/wheelhouse" \
  "${OFFLINE_CONTAINER_DIR}/wheelhouse"
compose cp "${CORE_WHEEL}" \
  "ray-head:${PLUGIN_CONTAINER_DIR}/$(basename "${CORE_WHEEL}")"
CORE_WHEEL_CONTAINER_PATH="${PLUGIN_CONTAINER_DIR}/$(basename "${CORE_WHEEL}")"
for bundle_file in algorithm.whl requirements.lock manifest.json; do
  compose cp "${TORCH_RECIPE_BUNDLE_DIR}/${bundle_file}" \
    "ray-head:${TORCH_RECIPE_CONTAINER_DIR}/${bundle_file}"
done
for torch_recipe_wheel in "${TORCH_RECIPE_BUNDLE_DIR}"/wheelhouse/*.whl; do
  compose cp "${torch_recipe_wheel}" \
    "ray-head:${TORCH_RECIPE_CONTAINER_DIR}/wheelhouse/"
done
for bundle_file in algorithm.whl requirements.lock manifest.json; do
  compose cp "${OFFLINE_BUNDLE_DIR}/${bundle_file}" \
    "ray-head:${OFFLINE_CONTAINER_DIR}/${bundle_file}"
done
for offline_wheel in "${OFFLINE_BUNDLE_DIR}"/wheelhouse/*.whl; do
  compose cp "${offline_wheel}" \
    "ray-head:${OFFLINE_CONTAINER_DIR}/wheelhouse/"
done
compose cp "${OFFLINE_BUNDLE_ARCHIVE}" \
  "ray-head:${OFFLINE_ARCHIVE_CONTAINER_PATH}"

compose exec -T \
  --env "TRIBUTO_OFFLINE_BUNDLE_ARCHIVE=${OFFLINE_ARCHIVE_CONTAINER_PATH}" \
  --env "TRIBUTO_OFFLINE_BUNDLE_BUCKET=${OFFLINE_BUCKET}" \
  --env "TRIBUTO_OFFLINE_BUNDLE_KEY=${OFFLINE_KEY}" \
  ray-head python -c \
  'import boto3, os; from botocore.config import Config; from botocore.exceptions import ClientError; client = boto3.client("s3", endpoint_url=os.environ["TRIBUTO_MINIO_ENDPOINT"], aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], region_name=os.environ["AWS_DEFAULT_REGION"], config=Config(s3={"addressing_style": "path"})); bucket = os.environ["TRIBUTO_OFFLINE_BUNDLE_BUCKET"]; key = os.environ["TRIBUTO_OFFLINE_BUNDLE_KEY"]; archive = os.environ["TRIBUTO_OFFLINE_BUNDLE_ARCHIVE"];
try: client.head_bucket(Bucket=bucket)
except ClientError: client.create_bucket(Bucket=bucket)
client.upload_file(archive, bucket, key); print(f"uploaded s3://{bucket}/{key}")'

compose exec -T \
  --env TRIBUTO_DOCKER_ALGORITHM_CONTRACT_IT=1 \
  --env TRIBUTO_DOCKER_GRAPH_CORE_GATE=1 \
  --env "TRIBUTO_CORE_WHEEL=${CORE_WHEEL_CONTAINER_PATH}" \
  --env "TRIBUTO_TORCH_RECIPE_PLUGIN_BUNDLE=${TORCH_RECIPE_CONTAINER_DIR}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE=${OFFLINE_CONTAINER_DIR}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_URI=${OFFLINE_URI}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_SHA256=${OFFLINE_ARCHIVE_SHA256}" \
  --env "TRIBUTO_ALGORITHM_IMAGE_DIGEST=${RUNTIME_IMAGE_DIGEST}" \
  --env "TRIBUTO_IT_RUNTIME_IMAGE=${TRIBUTO_IT_RUNTIME_IMAGE}" \
  --env "TRIBUTO_DISTRIBUTED_GATE_LOG=/workspace/tributo-work/algorithm-contract-ray-jobs.log" \
  ray-head \
  python -m pytest \
  tests/training/test_dnn_pu_training.py \
  tests/integration/algorithm_jobs/test_portable_execution_jobs.py \
  tests/training/test_graph_core_gate.py \
  -o addopts= \
  -o cache_dir=/workspace/tributo-work/cache/pytest-algorithm-contract-ray-jobs \
  -m "integration and slow" -vv -rP --tb=short --timeout=7200 \
  2>&1 | tee "${TEST_LOG}"
