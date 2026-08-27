#!/usr/bin/env bash

set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/.." && pwd -P)"
compose_file="${project_root}/tests/integrations/docker-compose.inference.yml"
versions_file="${project_root}/tests/integrations/inference-it-versions.conf"
suite="inference"

usage() {
  echo "Usage: $0 [--suite inference|explainability|all]" >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --suite)
      [ "$#" -ge 2 ] || {
        usage
        exit 2
      }
      suite="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
case "$suite" in
  inference|explainability|all) ;;
  *)
    usage
    exit 2
    ;;
esac

load_versions() {
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ""|\#*) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      TRIBUTO_INFERENCE_*|TRIBUTO_EXPECTED_*) ;;
      *)
        echo "Unexpected key in inference IT version contract: ${key}" >&2
        return 1
        ;;
    esac
    if [ -z "$value" ] || [ "$line" = "$key" ]; then
      echo "Invalid inference IT version contract entry: ${key}" >&2
      return 1
    fi
    export "${key}=${value}"
  done < "$versions_file"
}

load_versions

if [ -z "${COMPOSE_PROJECT_NAME:-}" ]; then
  COMPOSE_PROJECT_NAME="tributo-inference-$(date +%s)-$$"
fi
case "$COMPOSE_PROJECT_NAME" in
  *[!a-zA-Z0-9_-]*|"")
    echo "COMPOSE_PROJECT_NAME contains unsafe characters" >&2
    exit 2
    ;;
esac
export COMPOSE_PROJECT_NAME

if [ -z "${TRIBUTO_INFERENCE_IMAGE_TAG:-}" ]; then
  TRIBUTO_INFERENCE_IMAGE_TAG="$COMPOSE_PROJECT_NAME"
fi
case "$TRIBUTO_INFERENCE_IMAGE_TAG" in
  *[!a-zA-Z0-9_.-]*|"")
    echo "TRIBUTO_INFERENCE_IMAGE_TAG contains unsafe characters" >&2
    exit 2
    ;;
esac
export TRIBUTO_INFERENCE_IMAGE_TAG
TRIBUTO_INFERENCE_RUNTIME_IMAGE="tributo-inference-it:${TRIBUTO_INFERENCE_IMAGE_TAG}"
export TRIBUTO_INFERENCE_RUNTIME_IMAGE
docker_clean=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
  docker
)

compose=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
  docker compose
  --project-name "$COMPOSE_PROJECT_NAME"
  --file "$compose_file"
)

test_log="/tmp/${COMPOSE_PROJECT_NAME}-test.log"
service_log="/tmp/${COMPOSE_PROJECT_NAME}-services.log"
baseline_file="/tmp/${COMPOSE_PROJECT_NAME}-baseline-containers.tsv"
image_log="/tmp/${COMPOSE_PROJECT_NAME}-image-build.log"
baseline_images="/tmp/${COMPOSE_PROJECT_NAME}-baseline-dangling-images.txt"
final_images="/tmp/${COMPOSE_PROJECT_NAME}-final-dangling-images.txt"
baseline_df="/tmp/${COMPOSE_PROJECT_NAME}-baseline-docker-df.txt"
final_df="/tmp/${COMPOSE_PROJECT_NAME}-final-docker-df.txt"
requirements_dir="/tmp/${COMPOSE_PROJECT_NAME}-requirements"
project_wheel_dir="/tmp/${COMPOSE_PROJECT_NAME}-project-wheel"
image_created=0

docker ps -a --format '{{.ID}}' | while IFS= read -r container_id; do
  [ -n "$container_id" ] || continue
  state="$(docker inspect --format '{{.State.Status}}' "$container_id")"
  printf '%s\t%s\n' "$container_id" "$state"
done > "$baseline_file"

verify_existing_containers() {
  local container_id expected_state actual_state
  while IFS=$'\t' read -r container_id expected_state; do
    [ -n "$container_id" ] || continue
    if ! actual_state="$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null)"; then
      echo "Pre-existing container disappeared during inference IT: ${container_id}" >&2
      return 1
    fi
    if [ "$actual_state" != "$expected_state" ]; then
      echo "Pre-existing container state changed: ${container_id} ${expected_state} -> ${actual_state}" >&2
      return 1
    fi
  done < "$baseline_file"
}

project_resources() {
  docker ps -aq --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}"
  docker network ls -q --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}"
  docker volume ls -q --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}"
}

cleanup_transient_path() {
  local path="$1"
  [ -e "$path" ] || return 0
  if command -v trash >/dev/null 2>&1; then
    trash "$path"
  else
    python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1])' "$path"
  fi
}

if [ -n "$(project_resources)" ]; then
  echo "Inference IT Compose project already owns Docker resources: ${COMPOSE_PROJECT_NAME}" >&2
  project_resources >&2
  exit 2
fi

cleanup() {
  local status=$?
  local down_status=0
  trap - EXIT INT TERM
  set +e
  "${compose[@]}" logs --no-color > "$service_log" 2>&1
  "${compose[@]}" down --volumes --remove-orphans
  down_status=$?
  if [ "$down_status" -ne 0 ]; then
    echo "Scoped inference IT cleanup failed with status ${down_status}" >&2
    status=1
  fi
  if [ -n "$(project_resources)" ]; then
    echo "Scoped inference IT resources remain after cleanup" >&2
    project_resources >&2
    status=1
  fi
  if ! verify_existing_containers; then
    status=1
  fi
  if [ "$image_created" -eq 1 ]; then
    "${docker_clean[@]}" image rm \
      "$TRIBUTO_INFERENCE_RUNTIME_IMAGE" >/dev/null 2>&1 || status=1
    if "${docker_clean[@]}" image inspect \
      "$TRIBUTO_INFERENCE_RUNTIME_IMAGE" >/dev/null 2>&1; then
      echo "Run-scoped inference IT image remains after cleanup" >&2
      status=1
    fi
  fi
  "${docker_clean[@]}" image ls \
    --all --no-trunc --filter dangling=true > "$final_images"
  "${docker_clean[@]}" system df > "$final_df"
  if ! cleanup_transient_path "$requirements_dir" ||
    ! cleanup_transient_path "$project_wheel_dir"; then
    echo "Failed to clean inference IT transient build contexts" >&2
    status=1
  fi
  echo "Inference IT image log: ${image_log}"
  echo "Docker image snapshots: ${baseline_images}, ${final_images}"
  echo "Docker disk summaries: ${baseline_df}, ${final_df}"
  echo "Inference IT logs: ${test_log}"
  echo "Inference service logs: ${service_log}"
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$project_root"
"${docker_clean[@]}" image ls \
  --all --no-trunc --filter dangling=true > "$baseline_images"
"${docker_clean[@]}" system df > "$baseline_df"
if "${docker_clean[@]}" image inspect \
  "$TRIBUTO_INFERENCE_RUNTIME_IMAGE" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing run-scoped image: ${TRIBUTO_INFERENCE_RUNTIME_IMAGE}" >&2
  exit 2
fi

ray_image_ref="$TRIBUTO_INFERENCE_RAY_IMAGE"
minio_image_ref="$TRIBUTO_INFERENCE_MINIO_IMAGE"
command -v uv >/dev/null
mkdir -p "$requirements_dir" "$project_wheel_dir"
uv run --no-project --python 3.12 python tools/tributo_it.py \
  export-requirements \
  --output-file "$requirements_dir/requirements.txt" \
  --uv-version "$TRIBUTO_EXPECTED_UV_VERSION" \
  --extra dev \
  --extra test-integration \
  --extra explainability \
  --extra model-export-torch
uv build --wheel --out-dir "$project_wheel_dir" --no-create-gitignore
project_wheel_count="$(find "$project_wheel_dir" -maxdepth 1 -type f -name 'tributo-*.whl' | wc -l | tr -d ' ')"
if [ "$project_wheel_count" -ne 1 ]; then
  echo "Expected exactly one Tributo project wheel, found ${project_wheel_count}" >&2
  exit 1
fi
uv run --no-project --python 3.12 python -c \
  'from tools.tributo_it import ensure_digest_image; import os; [ensure_digest_image(os.environ[name]) for name in ("TRIBUTO_INFERENCE_RAY_IMAGE", "TRIBUTO_INFERENCE_MINIO_IMAGE")]'
export TRIBUTO_INFERENCE_MINIO_IMAGE="${minio_image_ref%@*}"

image_created=1
"${docker_clean[@]}" buildx build \
  --load \
  --file tests/integrations/Dockerfile.inference \
  --tag "$TRIBUTO_INFERENCE_RUNTIME_IMAGE" \
  --label "io.tributo.it.compose-project=${COMPOSE_PROJECT_NAME}" \
  --build-arg "BASE_IMAGE=${ray_image_ref%@*}" \
  --build-context "locked-requirements=${requirements_dir}" \
  --build-context "project-wheelhouse=${project_wheel_dir}" \
  . 2>&1 | tee "$image_log"

"${compose[@]}" config --quiet
"${compose[@]}" up --detach --no-build --pull never

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
if [ "$ready" -ne 1 ]; then
  echo "Docker Ray cluster did not become ready" >&2
  exit 1
fi

"${compose[@]}" exec -T ray-head \
  python -m tests.integrations.verify_inference_it_versions
"${compose[@]}" exec -T minio minio --version

set -o pipefail
if [ "$suite" = "inference" ] || [ "$suite" = "all" ]; then
  "${compose[@]}" exec -T ray-head \
    python -m pytest \
    tests/integration/test_lance_result_sink_ray.py \
    -o "addopts=" \
    -m integration \
    -v --tb=short --timeout=600 \
    2>&1 | tee "$test_log"
fi
if [ "$suite" = "explainability" ] || [ "$suite" = "all" ]; then
  "${compose[@]}" exec -T \
    -e TRIBUTO_DOCKER_EXPLAINABILITY_IT=1 \
    ray-head python -m pytest \
    tests/integration/test_explainability_ray_jobs.py \
    -o "addopts=" -m integration -v --tb=short --timeout=600 \
    2>&1 | tee -a "$test_log"
fi
