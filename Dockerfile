# Tributo Docker image — Ray 2.55.0 + Tributo (source-only)
#
# Build (BuildKit required for cache mounts):
#   DOCKER_BUILDKIT=1 docker build -t tributo:latest .
#
# Build with custom base image (e.g. GPU):
#   docker build --build-arg BASE_IMAGE=rayproject/ray:2.55.0-py312-gpu -t tributo:gpu .
#
# Build for specific platform:
#   docker build --platform linux/amd64 -t tributo:latest .

ARG BASE_IMAGE=rayproject/ray:2.55.0-py312
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TZ=UTC

FROM ${BASE_IMAGE}

ARG EXTRAS="training,embeddings,identity,streaming,registry"
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

# Pre-install CPU-only torch (pip cache persists across builds)
RUN --mount=type=cache,target=/home/ray/.cache/pip \
    /home/ray/anaconda3/bin/pip install \
        --timeout 600 --retries 10 \
        -i ${PIP_INDEX_URL} \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1

# Copy only source code + build files (see .dockerignore)
COPY . /opt/tributo

# Install Tributo with all extras (pip cache persists across builds)
RUN --mount=type=cache,target=/home/ray/.cache/pip \
    /home/ray/anaconda3/bin/pip install \
        --timeout 600 --retries 10 \
        -i ${PIP_INDEX_URL} \
        -e "/opt/tributo[${EXTRAS}]"

WORKDIR /workspace
USER ray

EXPOSE 8265 10001 8000 8001
