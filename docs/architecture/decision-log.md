# Architecture Decision Log

Purpose: Record every Go/No-Go decision for condition tracks, including timing,
evidence, and maintainer determination. Each track is re-evaluated when its
trigger condition is met.

## How Decisions Are Made

| Attribute | Value |
|-----------|-------|
| Decider | Tributo maintainers (currently @jiangxt2) |
| Evidence required | Measurable signal, not opinion (see per-track criteria) |
| Decision format | `GO` / `NO-GO` with date, evidence summary, and link to supporting data |
| Re-evaluation | At trigger events listed below; also on request from any contributor |
| Record location | This file, in the track's section |

## Condition Tracks

### Transform Compiler (D4)

**Trigger**: D1+D2 is merged AND at least one production `DataSourceProvider`
has a benchmarked pushdown path.

**Decision criteria**:
- A benchmark script exists that measures (a) full Ray Data read, (b) Daft
  pushdown + residual, against the same dataset.
- The pushdown path shows ≥ 20% wall-clock improvement on ≥ 10 GB datasets, OR
  ≥ 2× reduction in data transferred.
- Residual transform correctness is verified by output equivalence test.

| Date | Decision | Evidence | Decider |
|------|----------|----------|---------|
| 2026-08-02 | **NO-GO** | D1+D2 not yet merged; no production Provider exists | @jiangxt2 |

### Data-volume / Multi-worker (T3 Distributed extension)

T3 Core is an unconditional reliability baseline and does not require a
Go/No-Go decision. It owns single-worker batch/worker memory budgets,
input-size validation and fail-fast before unbounded materialization. This
decision track applies only to the distributed extension.

**Trigger**: D1+D2, T1 and T3 Core are merged AND at least one real task exceeds
the declared single-worker safe capacity.

**Decision criteria**:
- A specific task OOMs on a single worker, OR a user reports a training job
  that requires > 16 GB worker memory, OR multi-GPU user requirement is
  documented with a concrete workload.
- The safe capacity is `min(worker_memory * 0.7, 16 GB)` for single-worker
  training per the T3 Core memory budget config.
- `pd.concat` of full `iter_batches` exceeding this limit is the
  signal — not speculation about future scale.

| Date | Decision | Evidence | Decider |
|------|----------|----------|---------|
| 2026-08-02 | **NO-GO** | No reported OOM; no multi-GPU user requirement documented | @jiangxt2 |

### Plugin Platform (PL1+PL2)

**Trigger**: A1 AND D1+D2 AND E1 are merged AND at least one third-party
package registers a Tributo entry point outside the `tributo` package itself.

**Decision criteria**:
- A package published on PyPI (not a local editable install) declares
  `[project.entry-points."tributo.*"]`.
- OR multiple independent teams maintain Tributo extensions in separate repos.
- The existing `discover_*_plugins()` functions in `plugin.py` are NOT
  considered third-party consumers — they are built-in discovery for the
  framework's own integrations.

| Date | Decision | Evidence | Decider |
|------|----------|----------|---------|
| 2026-08-02 | **NO-GO** | Zero third-party packages registered on PyPI with Tributo entry points | @jiangxt2 |

### Streaming (S0/S1/S2)

**S0 status**: The unconditional fail-closed safety baseline is delivered
(2026-08-03): commit failures retain pending offsets for retry, poisoned
records stop the source instead of being skipped, and an uncommitted batch
blocks further polling. The `StreamSource` contract tests are added in
`tests/streaming/`. S1/S2 remain NO-GO (see table below).

**Trigger**: A0 is complete AND a Kafka source runs continuously for ≥ 24 hours
in a production or equivalent pre-production environment.

**Decision criteria**:
- A `StreamSource` consumer runs for ≥ 24 hours without manual intervention.
- OR a user reports a specific Kafka consumption issue (offset loss, poison
  message, commit failure) that requires S1/S2 fixes.
- The existing `kafka_source.py` unit tests passing is NOT sufficient — the
  Go requires demonstrated production load.

| Date | Decision | Evidence | Decider |
|------|----------|----------|---------|
| 2026-08-02 | **NO-GO** | No production Kafka workload; existing tests are unit-level only | @jiangxt2 |

## Re-evaluation Log

When a condition track's trigger fires, update the track's table above with a
new row and document the evidence here.

### Template

```markdown
## YYYY-MM-DD: <Track> Re-evaluation

**Trigger event**: <description>
**Evidence**: <link to benchmark results, user report, or package>
**Decision**: GO / NO-GO
**Rationale**: <1-3 sentences>
**Next steps**: <if GO: create PR; if NO-GO: what would change the decision>
```

<!-- END -->
