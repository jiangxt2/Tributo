# Broker provider boundary

## Status

Accepted for the Alpha API.

## Context

Tributo needs optional message-broker integrations without becoming a message
queue platform. A broker task is a control-plane admission request; bounded
data ingestion, unbounded inference streams, training, inference, Bundle
publication, and result sinks retain their existing Tributo contracts.

Transport clients and external wire protocols must remain independently
installable. Core must be usable and testable without Redis, Kafka, RabbitMQ,
or another provider dependency.

## Decision

Core owns a deliberately small Broker API v1 with Alpha stability:

- `BrokerPlugin` discovery, structural version checks, capability metadata,
  stability metadata, explicit config validation, and runtime construction;
- opaque `Message` payloads with a delivery token and restricted string
  metadata;
- `TaskConsumer`, `BrokerRuntime`, `TaskDisposition`, and a minimal
  `TaskOutcome` with an optional credential-safe `BrokerError`;
- workload-neutral `RayJobSubmission` identity and deterministic submission
  IDs derived from an operation namespace, `run_id`, and `attempt_id`;
- ambiguous Ray submission reconciliation plus status and stop operations
  keyed by `submission_id`.

`BROKER_API_VERSION = 1` checks structural compatibility; it does not imply a
Beta or long-term compatibility promise. Discovery is lazy and fail-open with
diagnostics. Explicit resolution and configuration validation fail closed.
Discovery never instantiates a provider or performs connectivity checks.

The following concerns belong to provider packages:

- broker connections, polling, acknowledgments, re-delivery, recovery, dead
  letters, cancellation watchers, and the production consume CLI/runtime;
- external request and event schemas, operation mapping, capability profiles,
  credential references, error mapping, redaction, and event durability;
- structured terminal-event publication from existing `TrainingResult`,
  `InferenceResult`, Bundle, and result-sink receipts.

Core does not define a workload registry or an external operation schema.
Providers submit one thin execution-driver Ray Job through the generic helper;
that driver calls existing in-process training or batch-inference APIs. Worker
side broker cancellation, arbitrary execution context, a generic Core consume
loop, and durable workflow semantics are outside the Alpha contract.

Trusted deployment configuration may supply an execution-driver package through
the generic Ray Jobs runtime-environment arguments on `submit_ray_job`. Core
only forwards explicitly declared modules and requirements; it does not inspect
broker payloads, discover provider installations, or resolve dependencies. This
deployment mechanism is separate from Broker API v1 and does not change
`BROKER_API_VERSION`.

`submission_id` is the primary Ray Jobs identity for admission, status, logs,
and stop. `ray_job_id` is optional execution metadata and is populated only
from a real Ray `JobDetails.job_id`; Core never substitutes `submission_id` for
it. An optional credential-free `request_digest` may be recorded as Ray
metadata, but Core does not persist it or promise cross-restart conflict
detection.

## Consequences

Normal Tributo installations remain free of broker dependencies, and a
provider can evolve transport and protocol behavior independently. Providers
must own their infrastructure and healthy-path tests. The first release does
not promise exactly-once execution, durable terminal events, high availability,
or complete pending-message recovery.
