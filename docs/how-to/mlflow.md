# Track published bundles with MLflow

Tributo's MLflow publication Hook records a committed bundle as provenance in
an MLflow tracking run. It is opt-in: exporting without `hooks` neither imports
MLflow nor creates a receipt.

Install the optional integration:

```bash
pip install 'tributo[registry]'
```

Configure a new MLflow run by naming an experiment:

```python
from tributo.exporting.models import BundleOutputConfig, ExportTarget, HookBinding

output = BundleOutputConfig(
    bundle_uri="s3://models/customer-risk",
    request_id="training-run-2026-08-06",
    targets=[ExportTarget(name="onnx", format="onnx")],
    hooks=(
        HookBinding(
            hook_id="mlflow-log-artifacts-v1",
            required=False,
            options={
                "tracking_uri": "http://mlflow.example:5000",
                "experiment_name": "customer-risk",
                "run_name": "bundle-publication",
                "tags": {"environment": "staging"},
            },
        ),
    ),
)
```

To log into a run owned by another workflow, provide `run_id` instead.
`experiment_name` and `run_name` must then be omitted. Tributo does not
terminate a caller-owned run or write Tributo's immutable Bundle parameters
into it. A caller-owned run may represent only one Tributo delivery identity:
replaying that delivery is safe, while attempting to attach a different Bundle
fails before any artifact or provenance value is changed.

The CLI accepts the same binding as repeatable JSON:

```bash
tributo export \
  --source /checkpoints/model \
  --targets onnx \
  --output /models/customer-risk \
  --hook '{"hook_id":"mlflow-log-artifacts-v1","required":false,"options":{"tracking_uri":"http://127.0.0.1:8050","experiment_name":"customer-risk"}}'
```

## Delivery behavior

The service validates the requested entry point and options before export
planning. After the bundle is committed, it derives a stable
`bundle.published` event from the committed manifest, claims a local delivery
lease, and verifies the committed manifest. Local and `file://` bundles are
materialized and uploaded beneath the MLflow `bundle/` artifact path. Remote
bundles such as S3 upload only the exact committed bytes of
`bundle/manifest.json`; hashing the downloaded MLflow copy therefore yields the
recorded manifest digest. Their canonical URI remains the durable reference to
the complete bundle, avoiding downloading the full remote bundle and uploading
it again through the publisher process.

MLflow 2.x resolves tracking-server proxy artifact URIs through its process
tracking URI. Tributo confines that compatibility switch to the artifact call,
serializes it, and restores the previous URI even when upload fails. Tracking
API calls otherwise use the Hook's explicit client URI.

Receipts use these states:

| Status | Meaning |
| --- | --- |
| `succeeded` | MLflow accepted the complete delivery |
| `skipped` | The same idempotency key already succeeded or is in progress |
| `retryable_failed` | A tracking operation failed and a later delivery may retry |
| `terminal_failed` | Configuration, dependency, integrity, or identity is invalid |
| `accepted` | Reserved for a future asynchronous dispatcher; not returned inline |

The idempotency key includes the Hook ID, canonical bundle URI, manifest
digest, and MLflow target options. For a new run, the key is attached in the
same `create_run` request. A retry recovers the single matching run; multiple
matches are a terminal conflict. Local delivery claims are scoped by event,
Bundle digest, Hook ID, and idempotency key, so a weak third-party key cannot
make a different Bundle appear delivered.

The run records the canonical Bundle URI, Bundle and event IDs, manifest
digest, source kind, Tributo version, Hook identity, and stable run, request,
and execution correlation IDs when available. Delivery attempt IDs remain in
the receipt and OperationStore; they are not written as publication facts.

`required=False` returns a failed receipt while preserving the successful
bundle result. `required=True` raises `PostPublishCallbackError`, whose
`bundle_result` and `receipts` still identify the committed bundle. Neither
mode rolls back publication.

For an `in_progress` receipt, `completed_at` is the time the current caller
observed the active lease; it is not evidence that the remote delivery
completed. Terminal receipts use the persisted completion time.

This Hook logs ordinary artifacts and provenance only. It does not generate an
`MLmodel`, create a Model Version, assign an Alias, or implement an Outbox or
asynchronous worker. Training metrics remain the responsibility of
`MLflowTrackingCallback`; model governance remains a separate concern.

## Credentials

Do not put tokens, passwords, access keys, or signed query strings in the Hook
binding or user tags. Embedded credentials in `tracking_uri` and tag names that
identify common secret fields are rejected. Supply authentication through the
MLflow process environment or the deployment's credential provider.

## Beta migration

The previous Beta path automatically ran every installed Hook. The current
contract runs only bindings listed in `BundleOutputConfig.hooks` or passed with
`--hook`. The built-in entry point is now `mlflow-log-artifacts-v1` instead of
`mlflow-log-v1`.

Move `required` out of the MLflow-specific options and onto `HookBinding`.
Creating a run now requires `experiment_name`; reusing `run_id` forbids both
`experiment_name` and `run_name`.

Receipt values also changed: old `success` maps to `succeeded`; old `failed`
is split into `retryable_failed` and `terminal_failed`; and a configured Hook
without the MLflow extra is now `terminal_failed` rather than `skipped`.

If `TRIBUTO_PLUGINS` is set, every explicitly configured Hook must also be in
that list. Tributo fails closed instead of silently dropping a configured
external side effect. Legacy `PublicationAttempt` records remain readable for
Beta compatibility, but the current dispatcher writes delivery records and no
longer creates new publication-attempt records.
