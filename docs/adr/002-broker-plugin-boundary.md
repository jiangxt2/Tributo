# Broker plugin boundary

## Status

Accepted

## Context

Tributo needs an optional control-plane integration for internal Redis
Streams tasks, lifecycle events, and cooperative cancellation. Redis Streams
is not a bounded data source or a streaming inference input. Future Kafka and
RabbitMQ providers should be able to use the same extension mechanism without
adding their client libraries to Tributo Core.

## Decision

Tributo Core owns only a transport-neutral, beta Broker SPI, lazy entry-point
discovery, explicit provider resolution, a generic runner, and JSON-safe Ray
execution-context plumbing. Redis and KnoVa protocol models live in an
independently installable provider wheel.

The Core contract has these safety rules:

- ordinary Tributo startup and execution never import or connect to a broker;
- discovery is fail-open with diagnostics, while explicitly selected brokers
  fail closed when missing, disabled, or invalid;
- broker configuration is passed as provider-owned JSON; Core does not define
  Redis/Kafka/RabbitMQ fields or perform network probes implicitly;
- cancellation checkers are reconstructed in Ray workers from serializable
  specs; clients, sockets, pools, and secrets are never serialized;
- provider submission must bind the business task ID to `run_id` and use a
  deterministic submission ID per execution attempt. Transport delivery
  retries must not become new execution attempts: the first Redis provider
  reuses `attempt-1` and the same submission ID until an explicit business
  retry is authorized after a terminal execution failure;
- temporary transport/submission failures retain the task for recovery;
  permanently invalid messages are best-effort reported as FAILED and then
  acknowledged even if reporting is unavailable;
- a missing or invalid outer `job_id` is permanently invalid and never becomes
  a shared sentinel identity. Its FAILED event goes to a provider-owned
  invalid-event stream and carries the delivery ID instead;
- provider reporters implement the Core `EventReporter` method signatures;
  provider-specific fields such as KnoVa error codes use explicit extension
  methods. Reporter warnings are time-window rate limited;
- reporter failure cannot turn successful training or Bundle publication into
  a failed computation.

The first Redis provider supports training tasks only. It reports lifecycle
events from the Ray Job driver and replays metrics history after training;
real-time metric sinks are a later extension.

## Consequences

The Core public surface can evolve independently of transport implementations,
and normal Tributo installations remain free of Redis dependencies. Provider
packages must publish their own protocol and infrastructure contract tests,
and a provider wheel must be installed in the Ray runtime when worker-side
cancellation or provider entrypoints are used.
