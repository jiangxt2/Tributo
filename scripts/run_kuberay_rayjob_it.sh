#!/usr/bin/env bash

# Heavyweight manual external IT. Do not run this runner during normal
# development or pre-checks unless the user explicitly requests the KubeRay IT.
# It creates a disposable four-node Kind cluster and loads a large runtime image.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CLUSTER_CONFIG="${PROJECT_ROOT}/tests/integrations/kind-kuberay-it.yaml"
KIND_STORAGE_HOST_PATH="/tmp/tributo-kuberay-shared"
KUBERAY_VERSION="${TRIBUTO_KUBERAY_VERSION:-1.6.0}"
KUBERAY_CHART_ARCHIVE="${TRIBUTO_KUBERAY_CHART_ARCHIVE:-}"
RUN_ID="${TRIBUTO_KUBERAY_IT_RUN_ID:-$(date +%Y%m%d%H%M%S)-$$}"
CLUSTER_NAME="tributo-kuberay-it-${RUN_ID}"
NAMESPACE="${TRIBUTO_KUBERAY_NAMESPACE:-tributo-kuberay-${RUN_ID}}"
KIND_NODE_IMAGE="${TRIBUTO_KIND_NODE_IMAGE:?TRIBUTO_KIND_NODE_IMAGE must be a pinned kindest/node image}"
RUNTIME_IMAGE="${TRIBUTO_KUBERAY_RUNTIME_IMAGE:?TRIBUTO_KUBERAY_RUNTIME_IMAGE must name a local Tributo runtime image}"
LOG_DIR="/tmp/${CLUSTER_NAME}"
KUBECONFIG_PATH="${LOG_DIR}/kubeconfig"
TEST_LOG="${LOG_DIR}/tests.log"
CLUSTER_LOG="${LOG_DIR}/cluster.log"
BASELINE_CONTAINERS="${LOG_DIR}/baseline-containers.tsv"
BASELINE_IMAGES="${LOG_DIR}/baseline-dangling-images.tsv"
POST_IMAGES="${LOG_DIR}/post-dangling-images.tsv"
KIND_NODE_IMAGE_ID=""
CLUSTER_CREATED=0

mkdir -p "${LOG_DIR}"

docker_clean=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
  docker
)
kind_clean=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
  kind
)
kubectl_cmd=(kubectl --kubeconfig "${KUBECONFIG_PATH}")
helm_cmd=(helm)

snapshot_baseline() {
  "${docker_clean[@]}" ps --all \
    --format '{{.ID}}\t{{.State}}\t{{.Names}}\t{{.Label "io.tributo.it.compose-project"}}' \
    >"${BASELINE_CONTAINERS}"
  "${docker_clean[@]}" image ls --all --no-trunc \
    --filter dangling=true \
    --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' \
    | LC_ALL=C sort \
    >"${BASELINE_IMAGES}"
}

verify_baseline_containers() {
  local container_id expected_state container_name label actual_state
  while IFS=$'\t' read -r container_id expected_state container_name label; do
    [[ -n "${container_id}" ]] || continue
    actual_state="$("${docker_clean[@]}" inspect --format '{{.State.Status}}' "${container_id}" 2>/dev/null || true)"
    if [[ "${actual_state}" != "${expected_state}" ]]; then
      echo "Pre-existing container changed: ${container_name} (${container_id}) ${expected_state} -> ${actual_state:-missing}" >&2
      return 1
    fi
  done <"${BASELINE_CONTAINERS}"
}

verify_baseline_images() {
  "${docker_clean[@]}" image ls --all --no-trunc \
    --filter dangling=true \
    --format '{{.Repository}}\t{{.Tag}}\t{{.ID}}' \
    | LC_ALL=C sort \
    >"${POST_IMAGES}"
  local comparable_images="${LOG_DIR}/comparable-dangling-images.tsv"
  if [[ -n "${KIND_NODE_IMAGE_ID}" ]]; then
    rg -v -F "${KIND_NODE_IMAGE_ID}" "${POST_IMAGES}" >"${comparable_images}" || true
    local comparable_baseline="${LOG_DIR}/comparable-baseline-dangling-images.tsv"
    rg -v -F "${KIND_NODE_IMAGE_ID}" "${BASELINE_IMAGES}" >"${comparable_baseline}" || true
  else
    cp "${POST_IMAGES}" "${comparable_images}"
    local comparable_baseline="${BASELINE_IMAGES}"
  fi
  if ! cmp -s "${comparable_baseline}" "${comparable_images}"; then
    echo "Dangling image set changed; leaving images untouched for diagnosis" >&2
    diff -u "${BASELINE_IMAGES}" "${POST_IMAGES}" || true
    return 0
  fi
}

cleanup() {
  local test_status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  if [[ "${CLUSTER_CREATED}" -eq 1 ]]; then
    "${kubectl_cmd[@]}" get rayjob,raycluster,pods -A -o wide >"${CLUSTER_LOG}" 2>&1 || true
    "${kind_clean[@]}" delete cluster --name "${CLUSTER_NAME}" --kubeconfig "${KUBECONFIG_PATH}" || cleanup_status=1
  fi
  if ! verify_baseline_containers; then
    cleanup_status=1
  fi
  if ! verify_baseline_images; then
    cleanup_status=1
  fi
  echo "KubeRay IT cluster: ${CLUSTER_NAME}"
  echo "KubeRay IT namespace: ${NAMESPACE}"
  echo "KubeRay IT test log: ${TEST_LOG}"
  echo "KubeRay IT cluster log: ${CLUSTER_LOG}"
  if [[ "${test_status}" -eq 0 && "${cleanup_status}" -eq 0 ]]; then
    echo "Result: PASS (Kind cluster destroyed; node image cache retained; pre-existing containers unchanged)"
    exit 0
  fi
  echo "Result: FAIL (test_status=${test_status}, cleanup_status=${cleanup_status})" >&2
  if [[ "${test_status}" -ne 0 ]]; then
    exit "${test_status}"
  fi
  exit "${cleanup_status}"
}

trap cleanup EXIT INT TERM

[[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
[[ "${NAMESPACE}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]
[[ -r "${CLUSTER_CONFIG}" ]]
mkdir -p "${KIND_STORAGE_HOST_PATH}"
chmod 777 "${KIND_STORAGE_HOST_PATH}"
command -v kind >/dev/null
command -v kubectl >/dev/null
command -v helm >/dev/null
command -v uv >/dev/null
"${docker_clean[@]}" info >/dev/null
snapshot_baseline

if "${kind_clean[@]}" get clusters | rg -Fxq "${CLUSTER_NAME}"; then
  echo "Refusing to replace existing Kind cluster ${CLUSTER_NAME}" >&2
  exit 2
fi
if "${docker_clean[@]}" image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  :
else
  echo "Runtime image is not available locally: ${RUNTIME_IMAGE}" >&2
  exit 2
fi

RUNTIME_PLATFORM="$("${docker_clean[@]}" image inspect --format '{{.Os}}/{{.Architecture}}' "${RUNTIME_IMAGE}")"
DOCKER_OS="$("${docker_clean[@]}" info --format '{{.OSType}}')"
DOCKER_ARCHITECTURE="$("${docker_clean[@]}" info --format '{{.Architecture}}')"
case "${DOCKER_ARCHITECTURE}" in
  aarch64) DOCKER_ARCHITECTURE=arm64 ;;
  x86_64) DOCKER_ARCHITECTURE=amd64 ;;
esac
DOCKER_PLATFORM="${DOCKER_OS}/${DOCKER_ARCHITECTURE}"
if [[ "${RUNTIME_PLATFORM}" != "${DOCKER_PLATFORM}" ]]; then
  echo "Runtime image platform ${RUNTIME_PLATFORM} does not match Docker platform ${DOCKER_PLATFORM}" >&2
  exit 2
fi

CLUSTER_CREATED=1
"${kind_clean[@]}" create cluster \
  --name "${CLUSTER_NAME}" \
  --image "${KIND_NODE_IMAGE}" \
  --config "${CLUSTER_CONFIG}" \
  --kubeconfig "${KUBECONFIG_PATH}" \
  --wait 5m
KIND_NODE_IMAGE_ID="$("${docker_clean[@]}" image inspect "${KIND_NODE_IMAGE}" --format '{{.Id}}')"
export KUBECONFIG="${KUBECONFIG_PATH}"
export HELM_CONFIG_HOME="${LOG_DIR}/helm/config"
export HELM_CACHE_HOME="${LOG_DIR}/helm/cache"
export HELM_DATA_HOME="${LOG_DIR}/helm/data"
mkdir -p "${HELM_CONFIG_HOME}" "${HELM_CACHE_HOME}" "${HELM_DATA_HOME}"

if [[ -n "${KUBERAY_CHART_ARCHIVE}" ]]; then
  [[ -r "${KUBERAY_CHART_ARCHIVE}" ]]
  CHART_VERSION="$("${helm_cmd[@]}" show chart "${KUBERAY_CHART_ARCHIVE}" | sed -n 's/^version: //p' | head -1)"
  if [[ "${CHART_VERSION}" != "${KUBERAY_VERSION}" ]]; then
    echo "KubeRay chart version ${CHART_VERSION:-unknown} does not match ${KUBERAY_VERSION}" >&2
    exit 2
  fi
  HELM_CHART_SOURCE="${KUBERAY_CHART_ARCHIVE}"
else
  "${helm_cmd[@]}" repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null
  "${helm_cmd[@]}" repo update >/dev/null
  HELM_CHART_SOURCE="kuberay/kuberay-operator"
fi
if [[ -n "${KUBERAY_CHART_ARCHIVE}" ]]; then
  "${helm_cmd[@]}" upgrade --install kuberay-operator "${HELM_CHART_SOURCE}" \
    --namespace kuberay-system \
    --create-namespace \
    --wait \
    --timeout 5m
else
  "${helm_cmd[@]}" upgrade --install kuberay-operator "${HELM_CHART_SOURCE}" \
    --version "${KUBERAY_VERSION}" \
    --namespace kuberay-system \
    --create-namespace \
    --wait \
    --timeout 5m
fi

"${kubectl_cmd[@]}" create namespace "${NAMESPACE}"
"${kind_clean[@]}" load docker-image "${RUNTIME_IMAGE}" --name "${CLUSTER_NAME}"

TRIBUTO_KUBERAY_IT=1 \
TRIBUTO_KUBERAY_NAMESPACE="${NAMESPACE}" \
TRIBUTO_KUBERAY_RUNTIME_IMAGE="${RUNTIME_IMAGE}" \
KUBECONFIG="${KUBECONFIG_PATH}" \
UV_CACHE_DIR="${LOG_DIR}/uv-cache" \
uv run --locked --extra dev --extra kuberay --no-sync python -m pytest \
  tests/integration/test_kuberay_rayjob_resources.py \
  -o addopts= \
  -p no:cacheprovider \
  -m integration \
  -vv --tb=short --timeout=1800 \
  2>&1 | tee "${TEST_LOG}"
