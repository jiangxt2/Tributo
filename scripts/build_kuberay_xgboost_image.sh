#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
ALGORITHMS_ROOT="${TRIBUTO_ALGORITHMS_ROOT:?TRIBUTO_ALGORITHMS_ROOT must point to a committed tributo-algorithms checkout}"
IMAGE="${TRIBUTO_KUBERAY_RUNTIME_IMAGE:-tributo-kuberay-xgboost:it-local}"
PLATFORM="${TRIBUTO_KUBERAY_PLATFORM:-linux/arm64}"
BUILD_ROOT="$(mktemp -d /tmp/tributo-kuberay-xgboost-build.XXXXXX)"
TRIBUTO_WHEELHOUSE="${BUILD_ROOT}/tributo"
BOOSTING_WHEELHOUSE="${BUILD_ROOT}/boosting"

mkdir -p "${TRIBUTO_WHEELHOUSE}" "${BOOSTING_WHEELHOUSE}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tributo-kuberay-uv-cache}" \
  uv build --wheel --out-dir "${TRIBUTO_WHEELHOUSE}" --project "${PROJECT_ROOT}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tributo-kuberay-uv-cache}" \
  uv build --wheel --out-dir "${BOOSTING_WHEELHOUSE}" \
  --package tributo-algorithms-boosting --project "${ALGORITHMS_ROOT}"

docker buildx build \
  --load \
  --platform "${PLATFORM}" \
  --file "${PROJECT_ROOT}/tests/integrations/Dockerfile.kuberay-it" \
  --build-context "tributo-wheelhouse=${TRIBUTO_WHEELHOUSE}" \
  --build-context "boosting-wheelhouse=${BOOSTING_WHEELHOUSE}" \
  --build-context "xgboost-data=${PROJECT_ROOT}/tests/integrations" \
  --tag "${IMAGE}" \
  "${PROJECT_ROOT}"

echo "KubeRay XGBoost runtime image: ${IMAGE}"
echo "Build artifacts: ${BUILD_ROOT}"
