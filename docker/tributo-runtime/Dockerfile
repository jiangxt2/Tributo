ARG BASE_IMAGE=tributo-ray-base:2.55.1-py312
ARG UV_IMAGE=tributo-uv:0.11.23
ARG TRIBUTO_BASE_IMAGE=rayproject/ray:2.55.1-py312@sha256:911245f2478ad2e9f67ac13978dc2a75bcae0498b9f188b10bba703324b78379
ARG TRIBUTO_PLATFORM

FROM ${UV_IMAGE} AS uv
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG TRIBUTO_BASE_IMAGE
ARG TRIBUTO_PLATFORM
ARG TRIBUTO_MANIFEST_SHA256=unsealed
ARG TRIBUTO_RUNTIME_EXTRAS
ARG TRIBUTO_VERSION=1.0.0

LABEL org.opencontainers.image.title="tributo-runtime-full" \
      org.opencontainers.image.version="${TRIBUTO_VERSION}" \
      org.tributo.base-image="${TRIBUTO_BASE_IMAGE}" \
      org.tributo.ray-version="2.55.1" \
      org.tributo.python-version="3.12" \
      org.tributo.platform="${TRIBUTO_PLATFORM}" \
      org.tributo.runtime-extras="${TRIBUTO_RUNTIME_EXTRAS}" \
      org.tributo.manifest-sha256="${TRIBUTO_MANIFEST_SHA256}"

COPY --from=uv /uv /uvx /usr/local/bin/

USER root
RUN mkdir -p /opt/tributo /opt/tributo-image /workspace && \
    chown -R ray:users /opt/tributo /opt/tributo-image /workspace

# Some locked Alpha dependencies, notably econml on linux/arm64, do not ship
# a compatible wheel and must build their C extensions in the image.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

USER ray

ENV PATH="/opt/tributo/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/tributo/.venv

COPY --chown=ray:users pyproject.toml uv.lock README.md /opt/tributo/
COPY --chown=ray:users src /opt/tributo/src

WORKDIR /opt/tributo

RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=100 \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
    uv sync \
        --extra data \
        --extra data-daft \
        --extra hive-ray \
        --extra vector-index \
        --extra postgresql \
        --extra clickhouse \
        --extra mysql \
        --extra doris-flight \
        --extra s3 \
        --extra model-export \
        --extra model-export-torch \
        --extra hf \
        --extra model-export-hf \
        --extra training \
        --extra tune \
        --extra explainability \
        --extra identity \
        --extra streaming \
        --extra grpc \
        --extra registry \
        --extra graph \
        --extra causal \
        --extra streaming-inference \
        --no-default-groups \
        --no-dev \
        --locked

# Ray runtime_env may need pip when an algorithm uses the offline wheelhouse
# distribution mode. Bootstrap it in the image without contacting an index.
RUN python -m ensurepip --upgrade

# The builder always supplies this named context, including an empty directory
# when no external connector wheels were requested.
COPY --from=external-wheelhouse / /opt/tributo-image/external-wheelhouse/

# Keep the manifest value in a RUN instruction as well as an OCI label. This
# makes the sealing value part of the BuildKit cache key for the second build.
RUN printf '%s\n' "${TRIBUTO_MANIFEST_SHA256}" > /opt/tributo-image/manifest-seal

RUN if find /opt/tributo-image/external-wheelhouse -type f -name '*.whl' -print -quit | grep -q .; then \
        python -m pip install --no-index --no-deps \
            /opt/tributo-image/external-wheelhouse/*.whl; \
    fi && \
    pip_check_output=""; \
    pip_check_status=0; \
    pip_check_output="$(python -m pip check 2>&1)" || pip_check_status=$?; \
    if [ "${pip_check_status}" -ne 0 ]; then \
        if [ "${TRIBUTO_PLATFORM}" != "linux/arm64" ] || \
            ! printf '%s\n' "${pip_check_output}" | grep -Eq \
                '^nvidia-cusparselt-cu13 0[.]8[.]1 is not supported on this platform[[:space:]]*$'; then \
            printf '%s\n' "${pip_check_output}"; \
            exit "${pip_check_status}"; \
        fi; \
    fi

COPY --chown=ray:users tools/generate_distributions.py /opt/tributo-image/

RUN python /opt/tributo-image/generate_distributions.py

WORKDIR /workspace

# Keep an explicit final non-root boundary if later instructions add a root step.
USER ray
