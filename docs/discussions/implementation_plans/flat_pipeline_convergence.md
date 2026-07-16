# Making the Flat-Array Pipeline First-Class (Convergence Epic)

> Decision on [#197](https://github.com/hiraditya/Vx/issues/197): **Option A — converge.** The
> flat-array parallel pipeline becomes the production compile path, because that is the core claim of
> the Vx architecture ([`../../parallel_compiler_architecture.md`](../../parallel_compiler_architecture.md)).
> This is a multi-stage effort; this doc is the roadmap. Narrative log:
> [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md).

## North star

`vxc` drives `src/pipeline.rs::compile_pipeline`; codegen consumes the flat GID / HIR streams
(`LocalWorkerState::{local_type_stream, local_hir_stream}`) and the frozen registry; the sequential
AST middle/back-end is retired.

## Guiding principle — keep-green via differential testing

The sequential AST path (`driver.rs::execute` → `run_codegen`) **remains the production path and the
test oracle** until the flat path passes the *entire* suite. We never ship a half-converted compiler:

1. Each milestone lands with the full suite green.
1. The flat codegen is brought up **behind a flag**, run **alongside** the AST codegen, and
   **differential-tested** — emitted MLIR (or JIT results) from the flat path must match the AST path
   across the whole `tests/` corpus — *before* it becomes default.
1. `vxc` flips to the flat path only after parity; the AST path stays behind `--legacy-codegen` for a
   soak, then is removed.

This makes the AST path the oracle that proves the flat path correct, and means a regression in the
flat path is caught as a diff, not as a broken release.

## Milestones

### C0 — Coherent flat infrastructure (prerequisites)

Behaviour-preserving; no change to the production path. Unblocks everything downstream.

- **C0.1 — Word-2 codec unification. ✅ done** (`cf9d5c4` + `a947b60`).
  [#193](https://github.com/hiraditya/Vx/issues/193), plan in [`gid_word2_codec.md`](./gid_word2_codec.md).
  One codec owns word 2 so generic + lifetime GIDs stop colliding — required the moment the HIR
  stream mixes them. W1–W4 landed; W5 (borrowed-generic composite) deferred until borrowed generics
  are emitted. Journal: [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md) Entry 5.
- **C0.2 — Cross-module GID resolution. ✅ done** (`ce246d7`).
  [#194](https://github.com/hiraditya/Vx/issues/194). `resolve_names` now attaches the *defining*
  module's GID for qualified (`A::Foo`) and imported references via a `ResolutionScope`, and the
  registry exposes `resolve_in_module` (the `module_indices` read path), exercised end-to-end.
  Journal: [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md) Entry 6.
  Deferred: `import … as` renames, glob imports, method/associated paths.
- **C0.3 — Stable, project-controlled hash. ✅ done** (`3973058`).
  [#195](https://github.com/hiraditya/Vx/issues/195). GID hashes are now hand-rolled FNV-1a
  (reproducible across toolchains), with a deterministic duplicate-GID guard at registry-build time
  and the doc claims corrected. Journal:
  [`../parallel_pipeline_convergence.md`](../parallel_pipeline_convergence.md) Entry 7.

### C1 — HIR lowering (populate `local_hir_stream`) — [#198](https://github.com/hiraditya/Vx/issues/198)

The long pole. An instruction-selection pass lowering each type-checked function body from the AST to
a flat `Vec<HirInstruction>` (`src/hir/bytecode.rs`: `{opcode, operand1, operand2, type_idx, imm}`),
with `type_idx` indexing the (now populated) `local_type_stream`. Runs in `type_check_phase` next to
`emit_function_type_gids`. Grow it by the corpus:

- C1.1 — literals, locals, arithmetic, `return`. **✅ done** (`1af30aa`, `src/hir/flatten.rs`;
  atomic per-function lowering, journal Entry 9).
- C1.2 — value ops + control flow (`if`/`else`, `loop`/`for` with break/continue). **✅ done**
  (`0c687a7`, `dee88b4`, `7d792d6`). Calls deferred (function-symbol resolution + variadic args).
- C1.3 — the `vx`-dialect surface. `spawn` region **✅ done** (`77f66d3`); `transfer`/tensor/struct
  **blocked** on the non-scalar type + layout prerequisite
  ([#199](https://github.com/hiraditya/Vx/issues/199)).

Verified structurally (well-formed stream, in-bounds `type_idx`) and — where feasible — by
re-execution parity against the AST path.

### C2 — Flat codegen (consume the streams) — [#200](https://github.com/hiraditya/Vx/issues/200)

A backend that lowers `local_hir_stream` + the type stream + the registry to MLIR — reusing the
existing melior emission (`src/codegen/`) at the leaves where possible, but driven by the flat
instruction array instead of an AST walk. Brought up behind `--flat-codegen`, differential-tested
against the AST path across `tests/` (MLIR text and/or JIT results). This is where "O(1) array
codegen" (doc Phase 7) becomes real.

### C3 — Switch `vxc` — [#201](https://github.com/hiraditya/Vx/issues/201)

Once C2 is at parity on the full suite: `vxc` drives `compile_pipeline`; the AST path moves behind
`--legacy-codegen`; the parallel verification hooks (`parallel_architecture_verifier`) run in debug;
after a soak, remove the AST middle/back-end.

## Also on the path (fold in)

- [#196](https://github.com/hiraditya/Vx/issues/196) — assert stream **order** determinism (codegen
  will index the stream, so order becomes contractual).
- The `vx` dialect (`vx.spawn`/`vx.transfer`/topology/seams) and all the front-end work already done
  (memory spaces, slice ops, sub-space scheduling) are **orthogonal and preserved** — C2 must emit
  the same `vx` dialect ops the AST codegen does; the differential tests enforce that.

## Sequencing

C0 (parallelizable: #193 → then #194, #195 alongside) → C1 (incremental by corpus) → C2 (behind a
flag, differential) → C3 (flip + soak + remove). **C0 is complete** — C0.1 (#193 word-2 codec),
C0.2 (#194 cross-module resolution), and C0.3 (#195 stable hash) have all landed. Next is
**C1 — HIR lowering** (populate `local_hir_stream`), the long pole, grown incrementally by corpus.

## Risk register

| Risk | Mitigation |
|---|---|
| HIR lowering is a large surface (C1) | Grow by corpus; the AST path is the oracle; parity-test per subset |
| Flat codegen regressions | Behind a flag + differential testing before default |
| Losing the `vx`-dialect / front-end semantics | C2 emits the same dialect; diff tests enforce identical output |
| Determinism of stream order | #196, assert before codegen depends on it |
| Word-2 / identity incoherence | C0.1/C0.2/C0.3 land first |
