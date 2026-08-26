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
TORCH_RECIPE_DIST_DIR="${LOG_DIR}/torch-recipe-dist"
TORCH_RECIPE_WHEEL=""
OFFICIAL_DIST_DIR="${LOG_DIR}/official-dist"
OFFICIAL_ALGORITHMS_ROOT="${TRIBUTO_ALGORITHMS_ROOT:-${PROJECT_ROOT}/../tributo-algorithms}"
OFFICIAL_CLASSICAL_WHEEL=""
OFFICIAL_TIMESERIES_WHEEL=""
OFFICIAL_REPRESENTATION_WHEEL=""
OFFICIAL_GRAPH_PYG_WHEEL=""
OFFICIAL_TABULAR_TORCH_WHEEL=""
OFFICIAL_RECSYS_TORCH_WHEEL=""
OFFICIAL_TRANSFORMERS_NLP_WHEEL=""
OFFICIAL_CAUSAL_CORE_WHEEL=""
OFFICIAL_CAUSAL_DISCOVERY_WHEEL=""
OFFICIAL_MULTISTAGE_TORCH_WHEEL=""
OFFICIAL_BOOSTING_WHEEL=""
OFFICIAL_CAUSAL_XLEARNER_WHEEL=""
OFFICIAL_CAUSAL_DR_WHEEL=""
OFFICIAL_CAUSAL_DOWHY_WHEEL=""
TRIBUTO_CORE_WHEEL=""
OFFLINE_DIST_DIR="${LOG_DIR}/offline-dist"
OFFLINE_BUNDLE_DIR="${LOG_DIR}/offline-bundle"
OFFLINE_BUNDLE_ARCHIVE="${LOG_DIR}/offline-bundle.zip"
OFFLINE_BUNDLE_CONTAINER_DIR="/workspace/tributo-work/offline-bundle"
OFFLINE_BUNDLE_ARCHIVE_CONTAINER="/workspace/tributo-work/offline-bundle.zip"
OFFLINE_BUNDLE_BUCKET="tributo-algorithm-it"
OFFLINE_BUNDLE_KEY="${PROJECT_NAME}/offline-bundle.zip"
OFFLINE_BUNDLE_URI="s3://${OFFLINE_BUNDLE_BUCKET}/${OFFLINE_BUNDLE_KEY}"
BASELINE_CAPTURED=0
COMPOSE_TOUCHED=0
RAY_NODE_WAIT_SECONDS=120
RAY_NODE_STABLE_SAMPLES=3
REQUIRED_RUNTIME_IMAGE=""
REQUIRED_RUNTIME_IMAGE_ID=""

cd "${PROJECT_ROOT}"
if [[ -z "${TRIBUTO_ALGORITHMS_ROOT:-}" ]]; then
  echo "TRIBUTO_ALGORITHMS_ROOT is required; refusing the ambiguous default path" >&2
  exit 2
fi
if ! OFFICIAL_ALGORITHMS_COMMIT="$(git -C "${OFFICIAL_ALGORITHMS_ROOT}" rev-parse --verify HEAD^{commit} 2>/dev/null)"; then
  echo "TRIBUTO_ALGORITHMS_ROOT must be a Git checkout with a committed HEAD: ${OFFICIAL_ALGORITHMS_ROOT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${OFFICIAL_ALGORITHMS_ROOT}" status --porcelain=v1)" ]]; then
  echo "Official algorithm source has uncommitted changes; Wheels will be built from the worktree" >&2
  OFFICIAL_ALGORITHMS_WORKTREE_STATE="dirty"
else
  OFFICIAL_ALGORITHMS_WORKTREE_STATE="clean"
fi
for required_package in \
  boosting classical causal-core causal-dr causal-xlearner; do
  if [[ ! -f "${OFFICIAL_ALGORITHMS_ROOT}/packages/${required_package}/pyproject.toml" ]]; then
    echo "Official algorithm checkout is missing packages/${required_package}: ${OFFICIAL_ALGORITHMS_ROOT}" >&2
    exit 2
  fi
done
echo "Official algorithm source: ${OFFICIAL_ALGORITHMS_ROOT} (HEAD ${OFFICIAL_ALGORITHMS_COMMIT}, worktree ${OFFICIAL_ALGORITHMS_WORKTREE_STATE})"
if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing to reuse existing IT log directory ${LOG_DIR}" >&2
  exit 2
fi
mkdir -p "${PLUGIN_DIST_DIR}"
mkdir -p "${TORCH_RECIPE_DIST_DIR}"
mkdir -p "${OFFICIAL_DIST_DIR}"
mkdir -p "${OFFLINE_DIST_DIR}"

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

if [[ ! -r "${OFFICIAL_ALGORITHMS_ROOT}/pyproject.toml" ]]; then
  echo "Official algorithm Monorepo is unavailable: ${OFFICIAL_ALGORITHMS_ROOT}" >&2
  exit 1
fi
uv build --wheel --out-dir "${OFFICIAL_DIST_DIR}" --no-create-gitignore "${PROJECT_ROOT}" \
  2>&1 | tee "${LOG_DIR}/core-wheel.log"
(
  cd "${OFFICIAL_ALGORITHMS_ROOT}"
  uv build --package tributo-algorithms-classical --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-timeseries --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-representation --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-graph-pyg --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-tabular-torch --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-recsys-torch --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-transformers-nlp --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-causal-core --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-causal-discovery --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-multistage-torch --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-boosting --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-causal-xlearner --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-causal-dr --out-dir "${OFFICIAL_DIST_DIR}"
  uv build --package tributo-algorithms-causal-dowhy --out-dir "${OFFICIAL_DIST_DIR}"
) 2>&1 | tee "${LOG_DIR}/official-wheel.log"
shopt -s nullglob
official_classical_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_classical-*.whl)
official_timeseries_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_timeseries-*.whl)
official_representation_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_representation-*.whl)
official_graph_pyg_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_graph_pyg-*.whl)
official_tabular_torch_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_tabular_torch-*.whl)
official_recsys_torch_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_recsys_torch-*.whl)
official_transformers_nlp_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_transformers_nlp-*.whl)
official_causal_core_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_causal_core-*.whl)
official_causal_discovery_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_causal_discovery-*.whl)
official_multistage_torch_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_multistage_torch-*.whl)
official_boosting_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_boosting-*.whl)
official_causal_xlearner_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_causal_xlearner-*.whl)
official_causal_dr_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_causal_dr-*.whl)
official_causal_dowhy_wheels=("${OFFICIAL_DIST_DIR}"/tributo_algorithms_causal_dowhy-*.whl)
core_wheels=("${OFFICIAL_DIST_DIR}"/tributo-*.whl)
shopt -u nullglob
if [[ "${#core_wheels[@]}" -ne 1 || "${#official_classical_wheels[@]}" -ne 1 || "${#official_timeseries_wheels[@]}" -ne 1 || "${#official_representation_wheels[@]}" -ne 1 || "${#official_graph_pyg_wheels[@]}" -ne 1 || "${#official_tabular_torch_wheels[@]}" -ne 1 || "${#official_recsys_torch_wheels[@]}" -ne 1 || "${#official_transformers_nlp_wheels[@]}" -ne 1 || "${#official_causal_core_wheels[@]}" -ne 1 || "${#official_causal_discovery_wheels[@]}" -ne 1 || "${#official_multistage_torch_wheels[@]}" -ne 1 || "${#official_boosting_wheels[@]}" -ne 1 || "${#official_causal_xlearner_wheels[@]}" -ne 1 || "${#official_causal_dr_wheels[@]}" -ne 1 || "${#official_causal_dowhy_wheels[@]}" -ne 1 ]]; then
  echo "Expected all official algorithm Wheels" >&2
  exit 1
fi
TRIBUTO_CORE_WHEEL="${core_wheels[0]}"
OFFICIAL_CLASSICAL_WHEEL="${official_classical_wheels[0]}"
OFFICIAL_TIMESERIES_WHEEL="${official_timeseries_wheels[0]}"
OFFICIAL_REPRESENTATION_WHEEL="${official_representation_wheels[0]}"
OFFICIAL_GRAPH_PYG_WHEEL="${official_graph_pyg_wheels[0]}"
OFFICIAL_TABULAR_TORCH_WHEEL="${official_tabular_torch_wheels[0]}"
OFFICIAL_RECSYS_TORCH_WHEEL="${official_recsys_torch_wheels[0]}"
OFFICIAL_TRANSFORMERS_NLP_WHEEL="${official_transformers_nlp_wheels[0]}"
OFFICIAL_CAUSAL_CORE_WHEEL="${official_causal_core_wheels[0]}"
OFFICIAL_CAUSAL_DISCOVERY_WHEEL="${official_causal_discovery_wheels[0]}"
OFFICIAL_MULTISTAGE_TORCH_WHEEL="${official_multistage_torch_wheels[0]}"
OFFICIAL_BOOSTING_WHEEL="${official_boosting_wheels[0]}"
OFFICIAL_CAUSAL_XLEARNER_WHEEL="${official_causal_xlearner_wheels[0]}"
OFFICIAL_CAUSAL_DR_WHEEL="${official_causal_dr_wheels[0]}"
OFFICIAL_CAUSAL_DOWHY_WHEEL="${official_causal_dowhy_wheels[0]}"

uv build \
  --wheel \
  --out-dir "${OFFLINE_DIST_DIR}" \
  --no-create-gitignore \
  tests/fixtures/offline_algorithm_dependency \
  2>&1 | tee "${LOG_DIR}/offline-dependency-wheel.log"
uv build \
  --wheel \
  --out-dir "${OFFLINE_DIST_DIR}" \
  --no-create-gitignore \
  tests/fixtures/offline_algorithm_plugin \
  2>&1 | tee "${LOG_DIR}/offline-algorithm-wheel.log"
shopt -s nullglob
offline_dependency_wheels=("${OFFLINE_DIST_DIR}"/tributo_test_offline_dependency-*.whl)
offline_algorithm_wheels=("${OFFLINE_DIST_DIR}"/tributo_test_offline_algorithm-*.whl)
shopt -u nullglob
if [[ "${#offline_dependency_wheels[@]}" -ne 1 || "${#offline_algorithm_wheels[@]}" -ne 1 ]]; then
  echo "Expected one offline algorithm Wheel and one dependency Wheel" >&2
  exit 1
fi
uv run --locked --no-sync python "${PROJECT_ROOT}/tools/build_algorithm_bundle.py" \
  --algorithm-wheel "${offline_algorithm_wheels[0]}" \
  --dependency-wheel "${offline_dependency_wheels[0]}" \
  --output "${OFFLINE_BUNDLE_DIR}" \
  --algorithm-id "offline.algorithm" \
  --profile-id "data-ingestion.cpu.v1" \
  2>&1 | tee "${LOG_DIR}/offline-bundle.log"

uv run --locked --no-sync python -c \
  'import pathlib, sys, zipfile; root = pathlib.Path(sys.argv[1]); output = pathlib.Path(sys.argv[2]); archive = zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED); [archive.write(path, path.relative_to(root).as_posix()) for path in root.rglob("*") if path.is_file()]; archive.close()' \
  "${OFFLINE_BUNDLE_DIR}" "${OFFLINE_BUNDLE_ARCHIVE}"
OFFLINE_BUNDLE_ARCHIVE_SHA256="$(shasum -a 256 "${OFFLINE_BUNDLE_ARCHIVE}" | awk '{print $1}')"

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
uv run --locked --no-sync python "${PROJECT_ROOT}/tools/tributo_it.py" "${prepare_args[@]}" \
  2>&1 | tee "${LOG_DIR}/runtime-prepare.log"

export TRIBUTO_IT_RUNTIME_IMAGE
TRIBUTO_IT_RUNTIME_IMAGE="$({
  uv run --locked --no-sync python "${PROJECT_ROOT}/tools/tributo_it.py" runtime-key --profile data-ingestion
} | python3 -c 'import json, sys; print(json.loads(sys.stdin.read().splitlines()[-1])["local_tag"])')"
REQUIRED_RUNTIME_IMAGE="${TRIBUTO_IT_RUNTIME_IMAGE}"
REQUIRED_RUNTIME_IMAGE_ID="$(docker image inspect \
  --format '{{.Id}}' "${TRIBUTO_IT_RUNTIME_IMAGE}")"
export TRIBUTO_IT_SOURCE_ROOT="${PROJECT_ROOT}"

uv run --locked --no-sync python -c \
  'from tools.tributo_it import ensure_digest_image, load_profile; p = load_profile("data-ingestion"); ensure_digest_image(p.tool_image); ensure_digest_image(p.minio_image)' \
  2>&1 | tee "${LOG_DIR}/infrastructure-images.log"
export TRIBUTO_IT_TOOL_IMAGE="${TOOL_IMAGE%%@*}"
export TRIBUTO_IT_MINIO_IMAGE="${MINIO_IMAGE%%@*}"

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
compose cp "${TORCH_RECIPE_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
TORCH_RECIPE_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${TORCH_RECIPE_WHEEL}")"
compose cp "${OFFICIAL_CLASSICAL_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CLASSICAL_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CLASSICAL_WHEEL}")"
compose cp "${OFFICIAL_TIMESERIES_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_TIMESERIES_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_TIMESERIES_WHEEL}")"
compose cp "${OFFICIAL_REPRESENTATION_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_REPRESENTATION_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_REPRESENTATION_WHEEL}")"
compose cp "${OFFICIAL_GRAPH_PYG_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_GRAPH_PYG_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_GRAPH_PYG_WHEEL}")"
compose cp "${OFFICIAL_TABULAR_TORCH_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_TABULAR_TORCH_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_TABULAR_TORCH_WHEEL}")"
compose cp "${OFFICIAL_RECSYS_TORCH_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_RECSYS_TORCH_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_RECSYS_TORCH_WHEEL}")"
compose cp "${OFFICIAL_TRANSFORMERS_NLP_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_TRANSFORMERS_NLP_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_TRANSFORMERS_NLP_WHEEL}")"
compose cp "${OFFICIAL_CAUSAL_CORE_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CAUSAL_CORE_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CAUSAL_CORE_WHEEL}")"
compose cp "${OFFICIAL_CAUSAL_DISCOVERY_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CAUSAL_DISCOVERY_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CAUSAL_DISCOVERY_WHEEL}")"
compose cp "${OFFICIAL_MULTISTAGE_TORCH_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_MULTISTAGE_TORCH_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_MULTISTAGE_TORCH_WHEEL}")"
compose cp "${OFFICIAL_BOOSTING_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_BOOSTING_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_BOOSTING_WHEEL}")"
compose cp "${OFFICIAL_CAUSAL_XLEARNER_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CAUSAL_XLEARNER_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CAUSAL_XLEARNER_WHEEL}")"
compose cp "${OFFICIAL_CAUSAL_DR_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CAUSAL_DR_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CAUSAL_DR_WHEEL}")"
compose cp "${OFFICIAL_CAUSAL_DOWHY_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
OFFICIAL_CAUSAL_DOWHY_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${OFFICIAL_CAUSAL_DOWHY_WHEEL}")"
compose cp "${TRIBUTO_CORE_WHEEL}" "ray-head:${PLUGIN_CONTAINER_DIR}/"
TRIBUTO_CORE_CONTAINER_WHEEL="${PLUGIN_CONTAINER_DIR}/$(basename "${TRIBUTO_CORE_WHEEL}")"
compose exec -T ray-head mkdir -p \
  "${OFFLINE_BUNDLE_CONTAINER_DIR}/wheelhouse"
compose cp "${OFFLINE_BUNDLE_DIR}/algorithm.whl" \
  "ray-head:${OFFLINE_BUNDLE_CONTAINER_DIR}/algorithm.whl"
compose cp "${OFFLINE_BUNDLE_DIR}/requirements.lock" \
  "ray-head:${OFFLINE_BUNDLE_CONTAINER_DIR}/requirements.lock"
compose cp "${OFFLINE_BUNDLE_DIR}/manifest.json" \
  "ray-head:${OFFLINE_BUNDLE_CONTAINER_DIR}/manifest.json"
compose cp "${OFFLINE_BUNDLE_ARCHIVE}" \
  "ray-head:${OFFLINE_BUNDLE_ARCHIVE_CONTAINER}"
for offline_wheel in "${OFFLINE_BUNDLE_DIR}"/wheelhouse/*.whl; do
  compose cp "${offline_wheel}" \
    "ray-head:${OFFLINE_BUNDLE_CONTAINER_DIR}/wheelhouse/"
done

compose exec -T \
  --env "TRIBUTO_OFFLINE_BUNDLE_ARCHIVE=${OFFLINE_BUNDLE_ARCHIVE_CONTAINER}" \
  --env "TRIBUTO_OFFLINE_BUNDLE_BUCKET=${OFFLINE_BUNDLE_BUCKET}" \
  --env "TRIBUTO_OFFLINE_BUNDLE_KEY=${OFFLINE_BUNDLE_KEY}" \
  ray-head \
  python -c \
  'import boto3, os; from botocore.config import Config; from botocore.exceptions import ClientError; client = boto3.client("s3", endpoint_url=os.environ["TRIBUTO_MINIO_ENDPOINT"], aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"], region_name=os.environ["AWS_DEFAULT_REGION"], config=Config(s3={"addressing_style": "path"})); bucket = os.environ["TRIBUTO_OFFLINE_BUNDLE_BUCKET"]; key = os.environ["TRIBUTO_OFFLINE_BUNDLE_KEY"]; archive = os.environ["TRIBUTO_OFFLINE_BUNDLE_ARCHIVE"];
try: client.head_bucket(Bucket=bucket)
except ClientError: client.create_bucket(Bucket=bucket)
client.upload_file(archive, bucket, key); print(f"uploaded s3://{bucket}/{key}")'

test_targets=(
  tests/training/test_dnn_pu_training.py::test_out_of_tree_torch_recipe_completes_on_ray_cluster
  tests/training/test_dnn_pu_training.py::test_official_algorithm_wheels_complete_on_ray_cluster
  tests/training/test_dnn_pu_training.py::test_offline_wheelhouse_installs_unique_dependency_on_driver_and_workers
  tests/training/test_dnn_pu_training.py::test_remote_offline_wheelhouse_archive_installs_on_driver_and_workers
)
if [[ "${TRIBUTO_DISTRIBUTED_ALGORITHM_SCOPE:-all}" == "priority" ]]; then
  test_targets=(
    tests/training/test_dnn_pu_training.py::test_priority_algorithm_wheels_complete_on_ray_cluster
  )
elif [[ "${TRIBUTO_DISTRIBUTED_ALGORITHM_SCOPE:-all}" != "all" ]]; then
  echo "TRIBUTO_DISTRIBUTED_ALGORITHM_SCOPE must be all or priority" >&2
  exit 2
elif [[ "${TRIBUTO_DISTRIBUTED_ALGORITHM_RERUN_FAILED_ONLY:-0}" == "1" ]]; then
  test_targets=(
    tests/training/test_dnn_pu_training.py::test_official_algorithm_wheels_complete_on_ray_cluster
  )
elif [[ "${TRIBUTO_DISTRIBUTED_ALGORITHM_RERUN_FAILED_ONLY:-0}" == "official" ]]; then
  test_targets=(
    tests/training/test_dnn_pu_training.py::test_official_algorithm_wheels_complete_on_ray_cluster
  )
elif [[ "${TRIBUTO_DISTRIBUTED_ALGORITHM_RERUN_FAILED_ONLY:-0}" != "0" ]]; then
  echo "TRIBUTO_DISTRIBUTED_ALGORITHM_RERUN_FAILED_ONLY must be 0, 1, or official" >&2
  exit 2
fi

compose exec -T \
  --env TRIBUTO_DOCKER_DISTRIBUTED_ALGORITHM_IT=1 \
  --env "TRIBUTO_DISTRIBUTED_PLUGIN_WHEEL=${PLUGIN_CONTAINER_WHEEL}" \
  --env "TRIBUTO_TORCH_RECIPE_PLUGIN_WHEEL=${TORCH_RECIPE_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CLASSICAL_WHEEL=${OFFICIAL_CLASSICAL_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_TIMESERIES_WHEEL=${OFFICIAL_TIMESERIES_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_REPRESENTATION_WHEEL=${OFFICIAL_REPRESENTATION_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_GRAPH_PYG_WHEEL=${OFFICIAL_GRAPH_PYG_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_TABULAR_TORCH_WHEEL=${OFFICIAL_TABULAR_TORCH_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_RECSYS_TORCH_WHEEL=${OFFICIAL_RECSYS_TORCH_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_TRANSFORMERS_NLP_WHEEL=${OFFICIAL_TRANSFORMERS_NLP_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CAUSAL_CORE_WHEEL=${OFFICIAL_CAUSAL_CORE_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CAUSAL_DISCOVERY_WHEEL=${OFFICIAL_CAUSAL_DISCOVERY_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_MULTISTAGE_TORCH_WHEEL=${OFFICIAL_MULTISTAGE_TORCH_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_BOOSTING_WHEEL=${OFFICIAL_BOOSTING_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CAUSAL_XLEARNER_WHEEL=${OFFICIAL_CAUSAL_XLEARNER_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CAUSAL_DR_WHEEL=${OFFICIAL_CAUSAL_DR_CONTAINER_WHEEL}" \
  --env "TRIBUTO_OFFICIAL_CAUSAL_DOWHY_WHEEL=${OFFICIAL_CAUSAL_DOWHY_CONTAINER_WHEEL}" \
  --env "TRIBUTO_CORE_WHEEL=${TRIBUTO_CORE_CONTAINER_WHEEL}" \
  --env TRIBUTO_EXPECTED_ALGORITHM_MODE=image_py_modules \
  --env "TRIBUTO_ALGORITHM_IMAGE_DIGEST=${REQUIRED_RUNTIME_IMAGE_ID}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE=${OFFLINE_BUNDLE_CONTAINER_DIR}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_URI=${OFFLINE_BUNDLE_URI}" \
  --env "TRIBUTO_OFFLINE_ALGORITHM_BUNDLE_SHA256=${OFFLINE_BUNDLE_ARCHIVE_SHA256}" \
  --env TRIBUTO_DISTRIBUTED_GATE_LOG=/workspace/tributo-work/distributed-algorithm-ray-jobs.log \
  ray-head \
  python -m pytest \
  "${test_targets[@]}" \
  -o addopts= \
  -o cache_dir=/workspace/tributo-work/cache/pytest-distributed-algorithm \
  -m integration -vv -rP --tb=short --timeout=1200 2>&1 | tee "${TEST_LOG}"
