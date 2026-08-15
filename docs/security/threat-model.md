# Security threat model

This document describes the principal security boundaries for model bundles,
plugins, data references, explainability, vector-index jobs, and stream
offsets. It complements the implementation-specific validation and tests.

## Threat categories

### Untrusted checkpoints (pickle deserialization)

Threat: A malicious checkpoint could contain a pickled object that
executes arbitrary code when deserialized.

Mitigations:
- ``SourceProvider`` implementations declare their trust level explicitly.
- Pickle-based deserialization is only used when the source is trusted
  (e.g. a Ray checkpoint produced by the same user in the same cluster).
- Third-party sources (HuggingFace, external URLs) must use safe formats
  (Safetensors, JSON configs).

### HuggingFace ``trust_remote_code``

Threat: Loading a Hugging Face model with ``trust_remote_code=True``
allows the model repository to execute arbitrary Python code.

Mitigations:
- ``trust_remote_code`` defaults to ``False`` in all Tributo loaders.
- Users must explicitly opt in via configuration.
- Documentation warns about the risks.

### Third-party plugin code injection

Threat: A third-party ``ModelExporter`` or ``ExportSourceProvider`` plugin
could contain malicious code that runs in the main process.

Mitigations:
- Plugins are installed via standard Python packaging (pip/wheel).
- Trust is delegated to the installation mechanism — the plugin runs
  with the same privileges as the Tributo process.
- ``TRIBUTO_PLUGINS`` environment variable allows filtering which
  plugins are loaded.
- Plugin diagnostics log all loaded plugins at startup.

### Symlink / path traversal

Threat: An exporter could write files outside the designated
``artifact_dir``, or a malicious bundle could use symlinks to escape
the extraction directory.

Mitigations:
- ``ExportManager._materialize_artifact()`` verifies that every file
  is within the artifact staging directory (``is_relative_to`` check).
- ``BundleReader`` and ``ResolvedArtifact.path_for()`` enforce path
  containment.
- ``StructureValidator`` rejects symlinks and absolute paths.
- Manifest declares only POSIX relative paths.

### Credential leakage to manifest

Threat: Cloud credentials, API keys, or internal URLs could leak
into the manifest and be persisted to storage.

Mitigations:
- ``ManifestSourceInfo`` only stores source kind, fingerprint,
  framework name/version, architecture id, and task type.
- No credentials, environment variables, or internal paths are
  included in the manifest.
- ``StorageProfile`` is only resolved at runtime and never serialized.
- ``FailureInfo`` truncates messages to 4096 characters and excludes
  tracebacks.

### Bundle tampering detection

Threat: A bundle's artifacts could be modified after commit,
causing silent data corruption.

Mitigations:
- Every file in a bundle has a SHA-256 hash recorded in the manifest.
- ``LogicalArtifact.tree_digest`` is a Merkle root of all file hashes
  in the artifact.
- ``BundleReader`` verifies per-file hashes + tree digest on read.
- Manifest SHA-256 is stored in the alias, creating a tamper-evident
  chain from alias → manifest → artifact tree → individual files.
- ``bundle_digest`` (content-addressable identity) detects duplicate
  content across different export runs.

### Concurrent publish conflicts

Threat: Two processes publishing the same bundle simultaneously
could produce corrupted or inconsistent state.

Mitigations:
- Local: atomic ``os.rename()`` (same filesystem guarantee).
- S3: lease protocol (5-minute TTL, renewable) with If-None-Match
  on manifest write; If-None-Match on artifact uploads.
- Alias updates: CAS via If-Match on ETag.
- Idempotency: same content → same result.

### Explainability reference and result exposure

Threat: A reference dataset or feature-attribution result can expose sensitive
training or subject-level data.

Mitigations:

- Reference bindings declare privacy level, optional digest, row count, and
  time-to-live metadata.
- Reference URIs reject credentials, query strings, and fragments.
- NPY reference loading disables pickle.
- Result policies bound rows and bytes and record access, privacy, and
  retention declarations in the receipt.
- Each leased attempt writes to a distinct directory, so a replacement driver
  cannot overwrite another attempt.

### Vector-index credential and result leakage

Threat: A dataset reference, query vector, filter, or materialized Top-K result
can reveal credentials or sensitive vector data.

Mitigations:

- Lance references accept only local, `file://`, S3, or explicit namespace
  modes and reject URI user information, queries, and fragments.
- Storage and namespace credentials resolve through named worker-side
  environment references.
- Query vectors are hidden from request representations and have finite-value,
  dimension, and serialized-size bounds.
- Inline results enforce row and byte limits; materialized results require an
  explicit Parquet destination.

### Kafka offset loss or premature commit

Threat: Committing offsets before inference and output succeed can lose
messages. Polling past an uncommitted batch can hide replay responsibility.

Mitigations:

- `KafkaStreamSource` disables automatic commit.
- An uncommitted batch blocks the next poll.
- Commit failure preserves pending offsets for retry.
- Tombstones, malformed JSON, consumer errors, and non-object records fail
  closed as poison messages.
