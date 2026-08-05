# Overview

Tributo is a Ray-native machine learning SDK for organizing data access,
distributed training, model bundles, and batch or online inference.

## Runtime boundary

Ray provides distributed execution through Ray Jobs, Ray Data, Ray Train,
Ray Tune, and Ray Serve. Tributo provides the configuration, extension,
validation, and model lifecycle contracts around those services.

Tributo is not a Kubernetes control plane or a multi-tenant ML platform. It
does not provide cluster provisioning, RBAC, quota management, or approval
workflows.

## End-to-end workflow

```text
canonical data source -> Ray Dataset -> trainer -> model bundle -> inference
```

Use the [support matrix](../reference/support-matrix.md) to distinguish stable
features from alpha contracts and planned extension points.

## Start here

- Follow [Getting Started](../quickstart.md) for the shortest runnable path.
- Review [Installation](../installation.md) for optional dependency groups.
- Select a workflow from [Use Cases](../user-guide/index.md).
- Use [Reference](../reference/index.md) for API and CLI contracts.
