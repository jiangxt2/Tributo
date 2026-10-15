# Build the Tributo runtime image

Tributo provides one full runtime image definition targeting CPU
execution with the Ray baseline in the
[installation requirements](../getting-started/installation.md): Linux,
Python 3.12, and every first-party runtime
extra, including the Alpha modules. By default the builder selects the host's
native architecture. The image is dependency-closed and does not install
development or test dependencies. On Linux, the locked PyTorch closure may
include transitive CUDA/NVIDIA distributions; their presence does not make
this a GPU image or a GPU support claim.

This development revision uses stable PyPI connector packages with versions
and hashes selected by `uv.lock`. The standard CPU image gate targets Linux
arm64 and covers dependency imports, manifest sealing and the Ray Jobs worker
probe. Native `amd64`, GPU and external algorithm execution require their
own validation. The focused source compatibility gate remains a separate check.

## Prerequisites

Run the commands from the repository root with Docker Buildx, Python 3.12, and
the repository's locked `uv` environment available. The pinned Ray base image
and host `uv` export baseline are declared in
[`tools/tributo-runtime-full.json`](../../tools/tributo-runtime-full.json).

The default target is the native host architecture (`linux/arm64` on Apple
Silicon and `linux/amd64` on an x86_64 host). To build a different target,
specify it explicitly, for example `--platform linux/amd64`. A cross-platform
build is not a native runtime validation; its Ray Jobs gate must run on a
matching native host.

The builder downloads the digest-pinned Ray base through the DaoCloud mirror
recorded in the JSON configuration, verifies its digest, and gives the image a
local tag before BuildKit starts. Dependency export uses the host `uv` executable.
Docker image operations explicitly
remove both upper- and lower-case `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`,
so PandaFan is not used for registry traffic. The locked Python wheels keep
their `uv.lock` URLs and are not rewritten to a different index.

## Build and attest the image

```bash
uv run --locked --no-sync python tools/build_tributo_image.py \
  --config tools/tributo-runtime-full.json \
  --output-dir dist/tributo-runtime-full
```

The builder performs two Buildx builds. The first discovers the installed
distribution closure. The second seals a canonical SHA-256 manifest digest in
the image label `org.tributo.manifest-sha256` and verifies that the closure did
not change. It also runs `pip check`, imports every declared runtime/Alpha
module for the selected architecture, and checks `tributo --help`.

The pinned `linux/arm64` Ray base exposes one known pip metadata baseline:
`nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`.
The builder records that exact line in the manifest and accepts no other
`pip check` error; all other dependency failures stop the build.

The output directory contains:

- `manifest.json`: pinned image, base image, runtime extras, dependency
  closure, mirror/local image source records, Alpha capability list, and image
  digest;
- `image-profile.json`: the validated `ImageProfile` consumed by algorithm
  artifact compatibility and preflight checks; it does not change an existing
  Ray cluster's image;
- `installed-distributions.json`: the normalized image dependency inventory;
- `capabilities.json`: the Alpha capability declaration;
- `build.log`: the command/evidence log for this build.

The builder never pushes an image. Publishing requires a separate reviewed
workflow. The development configuration resolves the stable connector
packages declared in `pyproject.toml`, including Ray ClickHouse through the
`clickhouse` extra. The committed lock selects the exact image inputs.

The standard host-export builder rejects local dependency paths and exports
PyPI connector versions with SHA-256 hashes from `uv.lock`. Exporting the
closure does not by itself certify the full runtime image.

The `external_wheelhouse` option remains available for packages outside the lock:
it is copied into a named build context and installed with
`pip --no-index --no-deps`; every wheel is recorded by filename,
package/version, size, and SHA-256. Such a variant is an attested,
image-specific extension and does not change the Tributo lockfile.

The connector extras can also be installed from the source checkout outside
Docker. These commands are alternatives; run the single profile you need. To
combine profiles, pass every required `--extra` to one `uv sync` command because
each sync reconciles the environment to its selected extras:

```bash
uv sync --locked --no-dev --extra clickhouse
# The clickhouse extra includes the fixed Ray ClickHouse source.
uv sync --locked --no-dev --extra mysql          # Daft Doris + Ray Doris over MySQL
uv sync --locked --no-dev --extra doris-flight   # Daft/Ray Doris Flight dependencies
uv sync --locked --no-dev --extra hive-ray       # Ray HiveServer2 connector package
```

A Doris test that selects
`engine="tributo.daft"` needs `daft-doris`; a Ray Doris or generic training
path that selects `engine="tributo.ray_data"` needs `ray-doris`.
The Ray ClickHouse route uses `ray-clickhouse` installed by the
existing `clickhouse` extra; it does not add a Daft ClickHouse route.
`ray-hive` backs the built-in `tributo.hive` to `tributo.ray.hive` route. The
request must still select Ray explicitly; no Daft, native ORC/HDFS, or automatic
fallback route is added.

## Run the image gate

The gate builds the image for the native host architecture, starts a unique
two-node Docker Ray cluster, and submits
`tests/integrations/jobs/runtime_image_gate_job.py` through the Ray Jobs API:

```bash
bash scripts/run_runtime_image_it.sh
```

An explicit target is available when the test host matches it:

```bash
TRIBUTO_IMAGE_PLATFORM=linux/amd64 bash scripts/run_runtime_image_it.sh
```

The job imports all runtime and Alpha modules on the driver and a Ray worker,
checks Ray Data materialization, and verifies the Ray/Tributo versions. The
runner writes compose logs and attestation files under a unique temporary
directory, then removes only that Compose project and its volumes.

## Runtime boundary

The full image includes the first-party data, model export, training,
Tune, serving, streaming, registry, vector-index, explainability, graph, and
causal dependency closure, plus `ray-hive` for the built-in Ray
HiveServer2 Binding. It does not include development/test dependencies, GPU
compatibility or a GPU execution gate, HDFS/ORC file readers, a Daft Hive
Binding, or a KubeRay control plane. Linux PyTorch dependencies may still contain
CUDA/NVIDIA distributions; GPU drivers, GPU scheduling, and NCCL validation
are outside this image contract.
Kubernetes deployment remains the responsibility of the Ray/KubeRay
environment. Tributo selects and validates an immutable image profile; it does
not change a Ray cluster's image per submitted job.
