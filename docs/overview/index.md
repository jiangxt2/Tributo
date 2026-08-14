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
bounded source -> Ray Data or Daft -> algorithm -> Bundle -> inference
                                                -> explainability
Lance vector dataset -> vector-index job -> search or maintenance receipt
```

Use the [support matrix](../reference/support-matrix.md) to distinguish stable
features from alpha contracts and planned extension points.

## Start here

- Follow [Getting started](../getting-started/index.md) for the shortest path.
- Select a component from the [documentation home](../index.md).
- Use [Reference](../reference/index.md) for API and CLI contracts.
