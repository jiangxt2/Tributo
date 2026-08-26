# Algorithm Ecosystem Test Ledger

This ledger tracks non-duplicated evidence for the algorithm decomposition and
official Wheel delivery change. A passed long-running suite is not repeated
unless relevant production code, test code, infrastructure, configuration, or
the execution environment changes.

| Suite | Scope | Code state | Reason | Result |
| --- | --- | --- | --- | --- |
| Core decomposition contracts | Strategy serialization, Builder, Planner, Receipt, plugin ownership | Current worktree | Fast contract baseline | Passed: 58 tests plus support-evidence and conformance contract coverage |
| Core legacy compatibility | Existing Torch recipe, collective, plugin, input lifecycle | Current worktree | Protect existing public paths | Passed: 40 recipe/plugin tests and 29 input tests |
| Decomposition local Ray | Joblib estimator tasks, ensemble units, iterative rounds/checkpoint, resume, corruption rejection | Current worktree | Development diagnosis before cluster IT | Passed: 5 tests with local Ray; resume and fail-closed corruption paths included |
| Official Wheel unit/static/build | All 14 packages, 27 official Entry Points, public SPI, Descriptor-only discovery, mypy, Ruff, independent Wheel build | Current Core plus official Monorepo | Prove package boundaries before cluster installation | Passed: 67 tests; 27-entry-point Conformance; Ruff and mypy(packages); all Wheels built |
| Official Wheel local end-to-end | Ray Data, decomposition runtimes, RecipeV2 DDP, Graph/XGBoost Framework Native, causal MapReduce, finite staged Framework Native, checkpoint, ONNX/Safetensors/report Bundle | Current Core plus installed fixed Wheels | Diagnose runtime and exporter boundaries | Passed prior category paths plus out-of-tree XGBoost, five-stage X-Learner with typed safe batch Flavor, and three-stage AIPW/DR with distributed nuisance models, finite ATE, stage evidence, report and ONNX Bundle |
| Distributed algorithm Docker IT | All official categories, RF Joblib/Native, LR iterative, cross-node Bundle/inference, Tune, recovery, failure injection | Final candidate before Gate | Required cross-node execution proof | Blocked after final attempts: official Ray Job exceeded pytest 1200s while serially running the 27-record gate; every run cleaned its Compose project. Earlier targets and individual local proofs passed. |
| Final static and unit suite | Cumulative Ruff, mypy, Core regression, official Wheel tests, Conformance | Final implementation candidate | Final merge-readiness evidence | Core default `3467 passed, 20 skipped, 177 deselected`; official `67 passed`; static checks and `pr-precheck` API layer passed; Docker Gate remains blocked by timeout |
