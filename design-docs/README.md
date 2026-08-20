# Design proposals

This directory contains technical proposals for changes that need architectural
review before implementation. It is part of the contributor workflow, not the
user documentation or the source of truth for implemented capabilities.

## Document responsibilities

| Location | Responsibility |
| --- | --- |
| GitHub issues | Establish the problem, use cases, and whether Tributo should address it |
| `design-docs/` | Compare options and review a proposed technical design |
| `docs/adr/` | Record an accepted, durable architectural decision |
| `docs/architecture/` | Describe the architecture implemented by the repository |
| `docs/reference/support-matrix.md` | Record support claims backed by implementation, configuration, and tests |

## When a proposal is required

Write a design proposal before implementation when a change does one or more of
the following:

- adds or changes a public API, persisted configuration, Bundle Manifest,
  storage schema, wire protocol, or extension contract;
- changes an execution chain or ownership boundary across Tributo components;
- introduces consistency, concurrency, retry, recovery, streaming-offset, or
  credential-security semantics;
- introduces an execution engine, runtime, extension point, or substantial
  third-party dependency;
- requires a compatibility, deprecation, or migration plan.

A proposal is normally unnecessary for a contract-preserving bug fix, local
refactor, test addition, or documentation correction. Maintainers may still
request one when the architectural effect is unclear.

## Proposal workflow

1. Open a feature issue that states the problem, use cases, affected components,
   and alternatives. Use a `[SCOPE]` issue when the proposal changes the product
   boundary defined in
   [`docs/architecture/product-scope.md`](../docs/architecture/product-scope.md).
2. Copy [`template.md`](template.md) to
   `YYYY-MM-DD-<descriptive-topic>.md` and open a draft pull request containing
   the proposal and any supporting images. Do not include production code in
   that pull request.
3. Keep the proposal status `Draft` while reviewers discuss the design in pull
   request comments. Resolve or explicitly defer every open question.
4. A maintainer records the outcome. Change the status to `Accepted` only after
   the maintainer explicitly approves the design. If the proposal establishes a
   durable public contract or product boundary, add a concise ADR that links to
   the detailed proposal before merging the design pull request.
5. Close rejected or withdrawn proposal pull requests instead of merging their
   draft documents. Record an important Go/No-Go outcome in
   [`docs/architecture/decision-log.md`](../docs/architecture/decision-log.md)
   when it governs future reconsideration.
6. Implement an accepted design in a separate pull request. Link the accepted
   proposal and update architecture, support, API, and user documentation only
   when the corresponding behavior is implemented and verified.

Acceptance authorizes the design direction. It does not mean the feature is
implemented, available, or supported. The support matrix remains authoritative
for capability claims.

## Status values

| Status | Meaning |
| --- | --- |
| `Draft` | Under review in an open draft pull request; must not be merged |
| `Accepted` | Explicitly approved by a maintainer and eligible to merge |
| `Superseded` | Replaced by a linked, accepted proposal |

Rejected and withdrawn drafts remain discoverable through their closed pull
requests and related issues. They do not enter the default branch.

## Review and evolution

Use pull request line comments for design details and the related issue for
problem-scope discussion. The decider named in the proposal owns the final
outcome. Acceptance requires concrete compatibility and verification criteria,
not only agreement on the happy path.

An accepted proposal is a historical design record. If implementation requires
a material change to its public contracts, component boundaries, failure
semantics, or acceptance criteria, update the proposal in a new design pull
request and obtain approval before changing the implementation plan. Mark a
replaced proposal `Superseded` and link its successor instead of rewriting the
original rationale without explanation.

All proposals use English, sentence-case headings without numeric prefixes, and
credential-safe examples. Store images under `design-docs/images/<topic>/` to
avoid filename collisions.
