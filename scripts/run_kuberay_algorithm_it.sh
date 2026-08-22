#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/tests/integration/kuberay/provision-gate.yaml"
RUN_ID="${TRIBUTO_KUBERAY_GATE_RUN_ID:-$(date +%Y%m%d%H%M%S)-$$}"
CLUSTER_NAME="tributo-ray-native-gate-${RUN_ID}"
KUBE_CONTEXT="kind-${CLUSTER_NAME}"
NAMESPACE="tributo-ray-native-gate"
RAYJOB="tributo-provision-gate"
KUBERAY_VERSION="1.6.0"
KIND_NODE_IMAGE="docker.m.daocloud.io/kindest/node:v1.32.11@sha256:5fc52d52a7b9574015299724bd68f183702956aa4a2116ae75a63cb574b35af8"
OPERATOR_IMAGE="quay.io/kuberay/operator:v1.6.0"
RUNTIME_IMAGE="tributo-runtime-full:local"
LOG_DIR="/tmp/tributo-kuberay-gate-${RUN_ID}"
RAYJOB_LOG="${LOG_DIR}/rayjob.log"
STATUS_LOG="${LOG_DIR}/status.txt"
CLUSTER_CREATED=0

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "TRIBUTO_KUBERAY_GATE_RUN_ID must be a unique safe identifier" >&2
  exit 2
fi
if [[ -e "${LOG_DIR}" ]]; then
  echo "Refusing to reuse existing Gate directory ${LOG_DIR}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"

docker_cli() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    docker "$@"
}

cleanup() {
  status=$?
  cleanup_status=0
  if [[ "${CLUSTER_CREATED}" -eq 1 ]]; then
    kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
      get rayjobs,rayclusters,pods,jobs -o wide \
      >"${LOG_DIR}/resources.txt" 2>&1 || true
    kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
      logs "job/${RAYJOB}" >"${RAYJOB_LOG}" 2>&1 || true
    kind delete cluster --name "${CLUSTER_NAME}" \
      >"${LOG_DIR}/kind-delete.log" 2>&1 || cleanup_status=$?
  fi
  echo "KubeRay provision Gate artifacts: ${LOG_DIR}" >&2
  if [[ "${status}" -eq 0 && "${cleanup_status}" -ne 0 ]]; then
    exit "${cleanup_status}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
for command in docker kind kubectl helm uv rg; do
  command -v "${command}" >/dev/null
done
test -r "${MANIFEST}"
docker_cli info >/dev/null

if kind get clusters | grep -Fxq "${CLUSTER_NAME}"; then
  echo "Refusing to take over existing kind cluster ${CLUSTER_NAME}" >&2
  exit 2
fi

if ! docker_cli image inspect "${KIND_NODE_IMAGE}" >/dev/null 2>&1; then
  docker_cli pull "${KIND_NODE_IMAGE}"
fi
if ! docker_cli image inspect "${OPERATOR_IMAGE}" >/dev/null 2>&1; then
  docker_cli pull "${OPERATOR_IMAGE}"
fi

env \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  "UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/tributo-image-uv-cache}" \
  uv run --locked --no-sync python tools/build_tributo_image.py \
    --repo-root "${PROJECT_ROOT}" \
    --config "${PROJECT_ROOT}/tools/tributo-runtime-full.json" \
    --output-dir "${LOG_DIR}/runtime-image" \
    2>&1 | tee "${LOG_DIR}/runtime-image.log"

CLUSTER_CREATED=1
kind create cluster \
  --name "${CLUSTER_NAME}" \
  --image "${KIND_NODE_IMAGE}" \
  --wait 180s \
  2>&1 | tee "${LOG_DIR}/kind-create.log"

kind load docker-image --name "${CLUSTER_NAME}" \
  "${OPERATOR_IMAGE}" "${RUNTIME_IMAGE}"

helm install kuberay-operator kuberay-operator \
  --repo https://ray-project.github.io/kuberay-helm/ \
  --version "${KUBERAY_VERSION}" \
  --kube-context "${KUBE_CONTEXT}" \
  --namespace kuberay-system \
  --create-namespace \
  2>&1 | tee "${LOG_DIR}/helm-install.log"
kubectl --context "${KUBE_CONTEXT}" --namespace kuberay-system \
  wait deployment/kuberay-operator \
  --for=condition=Available --timeout=180s

kubectl --context "${KUBE_CONTEXT}" apply -f "${MANIFEST}"
kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
  wait "rayjob/${RAYJOB}" \
  --for=jsonpath='{.status.jobDeploymentStatus}'=Complete \
  --timeout=600s

deployment_status="$(kubectl --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" get "rayjob/${RAYJOB}" \
  -o jsonpath='{.status.jobDeploymentStatus}')"
job_status="$(kubectl --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" get "rayjob/${RAYJOB}" \
  -o jsonpath='{.status.jobStatus}')"
ray_cluster_name="$(kubectl --context "${KUBE_CONTEXT}" \
  --namespace "${NAMESPACE}" get "rayjob/${RAYJOB}" \
  -o jsonpath='{.status.rayClusterName}')"
printf 'deployment=%s\njob=%s\nray_cluster=%s\n' \
  "${deployment_status}" "${job_status}" "${ray_cluster_name}" \
  | tee "${STATUS_LOG}"
if [[ "${deployment_status}" != "Complete" || "${job_status}" != "SUCCEEDED" ]]; then
  echo "KubeRay RayJob did not succeed" >&2
  exit 1
fi
if [[ -z "${ray_cluster_name}" ]]; then
  echo "KubeRay RayJob did not report its RayCluster identity" >&2
  exit 1
fi

kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
  logs "job/${RAYJOB}" | tee "${RAYJOB_LOG}"
for evidence in \
  '"status": "succeeded"' \
  '"execution_profile": "cluster"' \
  '"input_complete": true' \
  '"runtime_owned": false' \
  '"resource_preflight": "deferred_to_ray"' \
  '"bundle_uri"'; do
  if ! rg -Fq "${evidence}" "${RAYJOB_LOG}"; then
    echo "RayJob log is missing evidence: ${evidence}" >&2
    exit 1
  fi
done

kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
  wait "raycluster/${ray_cluster_name}" --for=delete --timeout=180s
if kubectl --context "${KUBE_CONTEXT}" --namespace "${NAMESPACE}" \
  get "raycluster/${ray_cluster_name}" >/dev/null 2>&1; then
  echo "KubeRay did not remove the RayCluster natively" >&2
  exit 1
fi

echo "Result: PASS (KubeRay provisioned and natively cleaned the RayCluster)"
