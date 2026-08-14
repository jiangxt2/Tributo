ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
COPY --chown=1000:100 pyproject.toml README.md /workspace/tributo-src/
COPY --chown=1000:100 src /workspace/tributo-src/src
COPY --chown=1000:100 tests/training/jobs/distributed_algorithm_gate_job.py /workspace/tributo-src/tests/training/jobs/distributed_algorithm_gate_job.py
COPY --chown=1000:100 tests/fixtures/distributed_algorithm_plugin /workspace/distributed-algorithm-plugin
RUN /bin/uv pip install \
      --python /opt/tributo/.venv/bin/python \
      --no-deps \
      /workspace/tributo-src \
      /workspace/distributed-algorithm-plugin
USER 1000:100
