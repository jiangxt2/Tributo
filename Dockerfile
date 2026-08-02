# Tributo Docker image — Ray 2.55.1 + Tributo (source-only)
#
# Build (BuildKit required for cache mounts):
#   DOCKER_BUILDKIT=1 docker build -t tributo:latest .
#
# Build with custom base image (e.g. GPU):
#   docker build --build-arg BASE_IMAGE=rayproject/ray:2.55.1-py312-gpu -t tributo:gpu .
#
# Build for specific platform:
#   docker build --platform linux/amd64 -t tributo:latest .

ARG BASE_IMAGE=rayproject/ray:2.55.1-py312
ARG TORCH_VERSION=2.12.1
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TZ=UTC

FROM ${BASE_IMAGE}

ARG EXTRAS="training,embeddings,identity,streaming,registry"
ARG TORCH_VERSION
ARG PIP_INDEX_URL
ARG TZ

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${TZ}

# System dependencies (apt cache persists across builds)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl wget git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /workspace && chown ray:users /workspace

USER ray

# Pre-install CPU-only torch (pip cache persists across builds).
# TORCH_VERSION pins the reproducible default image; pyproject.toml keeps a
# loose floor (torch>=2.5.0) so GPU users can layer CUDA wheels on top.
# CUDA wheels live under a different index (whl/cuXXX), so a GPU build must
# override both TORCH_VERSION and PIP_INDEX_URL, e.g.:
#   docker build --build-arg TORCH_VERSION=2.12.1+cu121 \
#                --build-arg PIP_INDEX_URL=https://download.pytorch.org/whl/cu121 ...
RUN --mount=type=cache,target=/home/ray/.cache/pip \
    /home/ray/anaconda3/bin/pip install \
        --timeout 600 --retries 10 \
        -i ${PIP_INDEX_URL} \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VERSION}

# Copy only source code + build files (see .dockerignore)
COPY . /opt/tributo

# Install Tributo with all extras (pip cache persists across builds).
# Dependencies are resolved by pip from pyproject.toml; the reproducible
# default comes from the pinned BASE_IMAGE tag (Ray 2.55.1 + Python 3.12)
# and TORCH_VERSION. The repo's uv.lock gates CI reproducibility — Docker
# builds are intentionally not lock-bound.
RUN --mount=type=cache,target=/home/ray/.cache/pip \
    /home/ray/anaconda3/bin/pip install \
        --timeout 600 --retries 10 \
        -i ${PIP_INDEX_URL} \
        -e "/opt/tributo[${EXTRAS}]"

WORKDIR /workspace
USER ray

EXPOSE 8265 10001 8000 8001
