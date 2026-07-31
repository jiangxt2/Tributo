# Tributo Model Export — Security Threat Model

This document enumerates security threats for the model export subsystem
and documents the mitigations in place as of P0.

## Threat categories

### Untrusted checkpoints (pickle deserialization)

**Threat**: A malicious checkpoint could contain a pickled object that
executes arbitrary code when deserialized.

**Mitigation**:
- ``SourceProvider`` implementations declare their trust level explicitly.
- Pickle-based deserialization is only used when the source is trusted
  (e.g. a Ray checkpoint produced by the same user in the same cluster).
- Third-party sources (HuggingFace, external URLs) must use safe formats
  (Safetensors, JSON configs).

### HuggingFace ``trust_remote_code``

**Threat**: Loading a HuggingFace model with ``trust_remote_code=True``
allows the model repository to execute arbitrary Python code.

**Mitigation**:
- ``trust_remote_code`` defaults to ``False`` in all Tributo loaders.
- Users must explicitly opt in via configuration.
- Documentation warns about the risks.

### Third-party plugin code injection

**Threat**: A third-party ``ModelExporter`` or ``SourceProvider`` plugin
could contain malicious code that runs in the main process.

**Mitigation**:
- Plugins are installed via standard Python packaging (pip/wheel).
- Trust is delegated to the installation mechanism — the plugin runs
  with the same privileges as the Tributo process.
- ``TRIBUTO_PLUGINS`` environment variable allows filtering which
  plugins are loaded.
- Plugin diagnostics log all loaded plugins at startup.

### Symlink / path traversal

**Threat**: An exporter could write files outside the designated
``artifact_dir``, or a malicious bundle could use symlinks to escape
the extraction directory.

**Mitigation**:
- ``ExportManager._materialize_artifact()`` verifies that every file
  is within the artifact staging directory (``is_relative_to`` check).
- ``BundleReader`` and ``ResolvedArtifact.path_for()`` enforce path
  containment.
- ``StructureValidator`` rejects symlinks and absolute paths.
- Manifest declares only POSIX relative paths.

### Credential leakage to manifest

**Threat**: Cloud credentials, API keys, or internal URLs could leak
into the manifest and be persisted to storage.

**Mitigation**:
- ``ManifestSourceInfo`` only stores source kind, fingerprint,
  framework name/version, architecture id, and task type.
- No credentials, environment variables, or internal paths are
  included in the manifest.
- ``StorageProfile`` is only resolved at runtime and never serialized.
- ``FailureInfo`` truncates messages to 4096 characters and excludes
  tracebacks.

### Bundle tampering detection

**Threat**: A bundle's artifacts could be modified after commit,
causing silent data corruption.

**Mitigation**:
- Every file in a bundle has a SHA-256 hash recorded in the manifest.
- ``LogicalArtifact.tree_digest`` is a Merkle root of all file hashes
  in the artifact.
- ``BundleReader`` verifies per-file hashes + tree digest on read.
- Manifest SHA-256 is stored in the alias, creating a tamper-evident
  chain from alias → manifest → artifact tree → individual files.
- ``bundle_digest`` (content-addressable identity) detects duplicate
  content across different export runs.

### Concurrent publish conflicts

**Threat**: Two processes publishing the same bundle simultaneously
could produce corrupted or inconsistent state.

**Mitigation**:
- Local: atomic ``os.rename()`` (same filesystem guarantee).
- S3: lease protocol (5-minute TTL, renewable) with If-None-Match
  on manifest write; If-None-Match on artifact uploads.
- Alias updates: CAS via If-Match on ETag.
- Idempotency: same content → same result.
