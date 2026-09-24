# Distributed graph training path

## Scope

Tributo Core provides an Alpha data path for static homogeneous node tasks. It
keeps node features and edge rows in disjoint Ray Data partitions, then gives
each `RayTorchAdapter` worker a typed `GraphReadHandle`. The worker requests a
bounded local batch from its seed IDs and receives `GraphBatch` values with
local edge IDs and seed rows first.

This Core path does not add PyG or DGL dependencies or model code. The existing
GraphSAGE and R-GCN Wheel still uses the full-batch route until it adopts the
Core graph input contract in a separate change. A passing Core graph gate does
not change their support status.

## Input and execution path

`TorchPolicy.graph_input` names the node, edge, and seed roles. Node and edge
roles remain Core-owned inputs; only the seed role is passed to `TorchTrainer`
as a split dataset. Core materializes the node and edge Ray Data datasets in the
cluster object store, reads their row counts, and uses row boundaries to create
balanced, disjoint splits. Each split is sent to one graph owner actor, and the
actor handles reach each worker through
`TorchWorkerCheckpointContext.graph_reader`. The driver handles Dataset
references and scalar row counts; it does not collect graph rows.

The graph roles use the existing ordered `InputBinding.feature_names` columns:
the node role lists the node ID first, followed by numeric feature columns; the
edge role lists source ID, destination ID, and an optional relation ID; the seed
role lists only the seed ID and sets `label_name`. Core does not infer these
columns from the schema.

Each graph owner stores only its node feature rows and edge rows. A sampling
request goes to every owner. Each owner returns at most the requested fanout
per queried node; Core merges the candidates, applies a stable priority derived
from the request seed, builds the local node map, and fetches features for the
selected nodes. Missing seed features, duplicate sampled node IDs, inconsistent
feature width, and malformed relation IDs fail closed. Seed labels come from
the split `train` dataset and remain separate from graph storage.

The public contract contains:

- `GraphInputSpec`: role names and partition count for a one-stage Adapter plan;
- `GraphSamplingSpec`: per-hop fanout, seed batch size, random seed, and
  incoming or outgoing direction;
- `GraphBatch`: original node IDs, local edge indices, node features, seed
  count, seed labels, and optional edge types;
- `GraphWorkerEvidence` and `GraphPartitionEvidence`: worker sample counts,
  sampler profiles (fanouts, batch size, direction, request count, and a digest
  of the random-seed sequence), touched partitions, graph version digest, and
  owner row counts.

Core validates graph seed coverage against the existing Torch role receipt and
attaches partition ownership to `TorchExecutionEvidence`. Graph handles and
actor references stay out of JSON algorithm configuration. Existing Torch
Adapters receive no graph handle unless their `TorchPolicy` declares a
`GraphInputSpec`.

## Resource and performance limits

V1 requires at least two owners and fails if one owner holds every node or edge
row. Each owner builds an in-memory Python feature and adjacency index from
only its own row split. Total graph rows are never collected into one owner
actor. Ray Data keeps the materialized source blocks in the distributed object
store during the handoff, so peak memory includes those blocks and the owner
indexes while loading.
Sampling currently queries all owners for each hop, so request traffic grows
with owner count. No throughput or maximum graph size is claimed until a
separate benchmark records owner memory, worker RSS, sampling traffic, and
batch wait time.

V1 supports integer node IDs, numeric node features, optional integer edge
types, static graphs, and node-seeded sampling without replacement. It does
not provide link prediction, graph-level batches, heterogeneous or temporal
graphs, GraphAr, or a remote GraphStore. Future algorithms that use the
versioned Core contract can add model and loss logic in their Wheel without
adding an algorithm-name branch to Core.

Duplicate edge rows are preserved as parallel edges; each row is independently
eligible for fanout. Graph owner actors request zero logical CPU so they do not
reserve capacity outside the declared Torch worker budget. Their Python work
still competes for physical CPU, so this path makes no throughput claim.

## Validation

Unit tests cover role policy, seed placement, local ID mapping, incoming and
outgoing fanout, empty neighborhoods, repeated seeds, missing sampled features,
parallel duplicate edge rows, numeric feature and relation validation, sampler
evidence, and graph-version consistency. The Docker Ray Jobs Gate uses two
workers on separate nodes and a graph with cross-partition neighbors; its
receipt proves exact seed coverage and checks that no owner or worker receives
the complete graph inputs and that the driver materializes no training rows.
Historical full-batch GraphSAGE/R-GCN Gate results do not certify this new path.
