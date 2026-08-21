# Build the Tributo runtime image

Tributo provides one directly buildable full runtime image validated for CPU
execution with Ray 2.55.1: Linux, Python 3.12, and every first-party runtime
extra, including the Alpha modules. By default the builder selects the host's
native architecture. The image is dependency-closed and does not install
development or test dependencies. On Linux, the locked PyTorch closure may
include transitive CUDA/NVIDIA distributions; their presence does not make
this a GPU image or a GPU support claim.

## Prerequisites

Run the commands from the repository root with Docker Buildx, Python 3.12, and
the repository's locked `uv` environment available. The pinned Ray and `uv`
base images are declared in
[`tools/tributo-runtime-full.json`](../../tools/tributo-runtime-full.json).

The default target is the native host architecture (`linux/arm64` on Apple
Silicon and `linux/amd64` on an x86_64 host). To build a different target,
specify it explicitly, for example `--platform linux/amd64`. A cross-platform
build is not a native runtime validation; its Ray Jobs gate must run on a
matching native host.

The builder downloads the digest-pinned Ray and `uv` inputs through the DaoCloud
mirrors recorded in the JSON configuration, verifies the digest, and gives each
image a local tag before BuildKit starts. Docker image operations explicitly
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
workflow. The canonical configuration resolves `daft-clickhouse==1.0`,
`daft-doris==1.0`, `ray-doris==1.0`, and `ray-hive==1.0` through the locked
Tributo extras, so the normal image build does not require a local connector
wheelhouse. The
`external_wheelhouse` option remains available for packages outside the lock:
it is copied into a named build context and installed with
`pip --no-index --no-deps`; every wheel is recorded by filename,
package/version, size, and SHA-256. Such a variant is an attested,
image-specific extension and does not change the Tributo lockfile.

The connector extras can also be installed outside Docker:

```bash
pip install "tributo[clickhouse]"
pip install "tributo[mysql]"          # Daft Doris + Ray Doris over MySQL
pip install "tributo[doris-flight]"   # Daft/Ray Doris Flight dependencies
pip install "tributo[hive-ray]"       # Ray HiveServer2 connector package
```

The equivalent uv commands are `uv sync --extra clickhouse`,
`uv sync --extra mysql`, `uv sync --extra doris-flight`, and
`uv sync --extra hive-ray`. A Doris test that selects
`engine="tributo.daft"` needs `daft-doris`; a Ray Doris or generic training
path that selects `engine="tributo.ray_data"` needs `ray-doris`.
`ray-hive` is packaged for the Ray-native runtime, but Tributo does not yet
register a Hive Provider or Binding and does not silently add a Hive route.

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
causal dependency closure, plus the external `ray-hive==1.0` HiveServer2
connector package. It does not include development/test dependencies, GPU
compatibility or a GPU execution gate, a Tributo-native Hive Provider/Binding,
HDFS/ORC file readers, or a KubeRay control plane. Linux PyTorch dependencies may still contain
CUDA/NVIDIA distributions; GPU drivers, GPU scheduling, and NCCL validation
are outside this image contract.
Kubernetes deployment remains the responsibility of the Ray/KubeRay
environment. Tributo selects and validates an immutable image profile; it does
not change a Ray cluster's image per submitted job.
