# Journal: Converging the Parallel Compiler Pipeline

> A running log of closing the gaps between the *target* architecture in
> [`parallel_compiler_architecture.md`](../parallel_compiler_architecture.md) and the
> implementation. Each entry: what was scaffolding, what we did, why, code pointers, tests, and
> what remains. The status table in that doc's §9.3 is the summary; this file is the narrative.

## Context

The parallel pipeline (`src/pipeline.rs::compile_pipeline`, `rayon`) had all the *infrastructure* —
256-bit GIDs (`gid.rs`), the frozen session + per-worker arenas (`session.rs`), the phase functions,
and the debug-only verification hooks (`parallel_architecture_verifier.rs`) — but several pieces of
the **data flow** were not connected: the flat type stream was never emitted, and the frozen
registry was a mock. So the dedup / SIMD-patch / verification phases ran on empty data. These
entries connect that flow, one gap at a time, each behind tests.

______________________________________________________________________

## Entry 1 — Flat GID-stream lowering (Phase 3)

**Commit:** `15e4114`

**Gap.** `LocalWorkerState::local_type_stream` (the flat `Vec<TypeId>` that replaces the AST for the
backend) was never populated — the type checker didn't emit GIDs. So `extract_type_streams`,
`simd_patch_phase`, and `verify_phase_6_simd_patch` operated on empty vectors (trivially passing).

**Key realization.** `resolve_names` (`src/syntax/resolve.rs`) *already* attaches the resolved GID
onto the AST: `Type::{Struct,Enum,Generic}(name, Option<TypeId>)`. So lowering doesn't need to
recompute anything — it *harvests* the attached GIDs.

**What we did.** After each per-function worker runs `check_function`, it lowers the function's type
references into GIDs on its `local_type_stream` (`src/pipeline.rs`):

- a resolved nominal type → the **settled** GID (words 0/1 = module/symbol hash);
- a generic instantiation `Foo<Bar>` → a **deferred** GID: the argument GIDs go into the worker's
  `local_generics_arena`, word 2 becomes that local offset index with
  `LOCAL_DEFERRED_BIT | IS_GENERIC_INST_FLAG` set. Phase 5 (`deduplication_phase`) interns the arena
  and Phase 6 (`simd_patch_phase`) remaps word 2 local→global and clears the deferred bit — the
  escape-hatch local→global handoff.

**Bug found + fixed.** `simd_patch_phase` read *both* the slow-path and generics mappings at the
deferred index and predicated-selected between them. Those arenas have **independent index spaces**,
so the cross-index read panics in general. Fixed to index only the mapping for the GID's kind.
(A branchless variant would need the two mappings padded to a shared index space.)

**Code pointers.** `emit_function_type_gids`, `emit_type_gid`, `nominal_gid`,
`mint_deferred_generic`, `simd_patch_phase`, `compile_pipeline_type_stream` (all `src/pipeline.rs`).

**Tests.** `deferred_generic_gid_is_interned_and_patched`,
`emit_type_gid_harvests_nominal_and_generic` (unit, `pipeline.rs`);
`compile_pipeline_gid_stream_is_deterministic` (integration) now runs over the real, non-empty,
patched stream.

**Scope / not yet.** Only function *signature* type references are harvested; body-expression
inferred types are not AST-attached and are not yet lowered.

______________________________________________________________________

## Entry 2 — Freeze the real registry + cycle detection (Phase 2)

**Commit:** `10b1658`

**Gap.** `GlobalSession.registry` was a **mock** empty struct (`session.rs`, `ImmutableGlobalRegistry {}`). The real one — `src/registry.rs` (`layouts: TypeId→TypeDefinition`, `module_indices: module_hash→symbol→TypeId`, petgraph cycle detection via `build_and_validate`) — was implemented and
unit-tested but never built from the modules nor stored in the session. So the Phase 2 "freeze
point" did nothing, and infinite-sized recursive structs were not caught on the parallel path.

**What we did.**

- `session.rs` re-exports the real `ImmutableGlobalRegistry` (drops the mock). `GlobalSession::new`
  stays (empty registry) for the sequential driver and the many unit-test call sites;
  `GlobalSession::with_registry` carries the frozen one.
- `pipeline.rs::build_frozen_registry` collects every module's top-level structs/enums into
  `TypeDefinition`s: the GID from the symbol map (identical to what `resolve_names` attached, so
  dependency GIDs resolve) plus **by-value dependency edges** — `by_value_nominal_gid` treats a
  nominal type held by value as a dependency but stops at `Ref`/`Pointer`/`Borrow` (indirection
  breaks containment and any cycle). `build_and_validate` then runs `toposort`; a cycle is a
  `PipelineError::Semantic`.
- Wired into `compile_pipeline` (Phase 2) and `compile_pipeline_type_stream`.

**Why this matters.** It activates a real correctness feature — infinite-sized recursive layout
detection — on the parallel path, and gives worker threads a real `module_indices` for future
cross-module layout lookup (the registry is `Arc`-shared, read-only, no lock).

**Code pointers.** `build_frozen_registry`, `by_value_nominal_gid` (`src/pipeline.rs`);
`ImmutableGlobalRegistry::build_and_validate` (`src/registry.rs`); `GlobalSession::with_registry`
(`src/session.rs`).

**Tests.** `frozen_registry_detects_infinite_recursion` (`struct List { next: List }` → error),
`frozen_registry_accepts_indirection_and_registers_types` (`&Node` breaks the cycle; both types
registered) — unit, `pipeline.rs`.

**Scope / not yet.** By-value cycle detection does not yet follow *generic instantiations*, and
cross-module by-value deps only resolve when `resolve_names` attached the cross-module GID.

______________________________________________________________________

## Entry 3 — Correction: the GID borrow check was already wired (+ tests)

**Commit:** `50ef7e0`

**Correction.** §9.3 had listed "`resolve_lifetime` / borrow over the GID stream" as *scaffolding,
not driven*. That was **wrong**. The 256-bit-GID fast-path borrow checker is real and *driven*:
`TypeChecker::is_assignable` (`src/hir/expr.rs`), for `Borrow`/`Pointer` subtyping, lowers both
sides to lifetime GIDs via `lower_to_type_id` (packing the borrow's region + variance into word 2)
and calls `borrow::verify_subtyping_bounds`, which decodes the fast-path bitfield via
`session.rs::resolve_lifetime` and applies invariance / covariance / contravariance rules. This runs
in *both* compile paths (the checker is shared).

**What was actually missing:** tests. `borrow.rs` had none.

**What we did.** Added unit tests for the fast-path comparison: covariance (source lifetime must
outlive-or-equal target; region 0 = `'static`, so a smaller region id lives longer), invariance
(exact region), and variance mismatch. Corrected the §9.3 row.

**Scope / not yet.** The slow-path (>4 params, or arena-backed) variance evaluation is a stub
(`evaluate_slow_path_variance` in `borrow.rs`).

______________________________________________________________________

## Remaining gaps (next entries)

- **`local_hir_stream`** — `HirInstruction` (`src/hir/bytecode.rs`: `{opcode, operand1, operand2, type_idx→LOCAL_TYPE_STREAM, imm}`) is a defined flat bytecode, but the stream is never populated;
  lowering function bodies to it is a real instruction-selection pass (and codegen would need to
  consume it — currently codegen is AST-based).
- **Slow-path variance** — `evaluate_slow_path_variance` (`borrow.rs`) is a stub.
- **Path convergence** — `vxc` still runs the sequential driver (`driver.rs::execute` →
  `run_codegen`); making it drive `compile_pipeline` (and codegen consume the flat streams) is the
  end goal that turns all of the above from "exercised by tests" into "the production compile".
