# Inference and serving key concepts

## Resolve before execution

`InferenceRequest` combines a model reference, bounded input, named tensor
bindings, result sink, and Ray execution policy. Resolution freezes the Bundle,
model role, source identity, binding, run identity, and submission identity
before distributed work begins.

## Bind columns to tensors

Input and output bindings connect table columns to named model tensors. The
runtime validates names, dtypes, rank, fixed dimensions, and row-preserving
batch dimensions. Null and NaN policies remain explicit.

## Separate source, model, and result credentials

Each storage domain resolves its own profile. Tributo does not copy source
credentials to a Bundle repository or result sink. Serialized plans and
receipts contain only credential-free references and digests.

## Choose batch or online execution

Batch inference uses Ray Data `map_batches` and an actor pool so each actor can
reuse a model. Formal requests require a verified Bundle. A compatibility API
can accept raw ONNX or a caller-provided predictor.

Online inference uses Ray Serve. HTTP and gRPC are Beta transports. The LLM
Server-Sent Events path is Alpha. A raw model path is a compatibility entry;
prefer a Bundle URI and explicit artifact role.

## Run explainability separately

Explainability loads a declared Bundle role and materializes long-format
Parquet attributions. It is not an extra prediction column. Its request,
limits, lease, partial-result semantics, receipt, and retention policy form an
independent batch contract.
