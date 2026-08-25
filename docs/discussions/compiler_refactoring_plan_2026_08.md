# Compiler Refactoring Plan (August 2026)

This plan covers the code in the repository only: `src/`, `runtime/`, `include/`,
`build.rs`, `Cargo.toml`, and the test harnesses under `tests/`. It does not
propose language changes, and it does not propose anything that changes what the
compiler accepts or what MLIR it emits.

The goal is narrow: after hundreds of feature commits on memory algebra, fleet
configs, aborts, and transfer lowerings, make the next round of features cheap to
add. Today a new language feature costs roughly two implementations and two
review passes, and the second one is invisible until a corpus number moves.

## Where the code stands

| Area | Lines | Notes |
| ---- | ----- | ----- |
| `src/` total | 63,311 | 89 files, one crate |
| `src/hir/` | 23,493 | includes `check/` at 9,281 |
| `src/codegen/` | 12,579 | two independent backends |
| `src/parser/` | 5,025 | |
| `src/syntax/` | 4,147 | the AST |
| `src/pipeline.rs` | 2,905 | 1,716 code + 1,189 test |
| `src/driver.rs` | 1,513 | |
| `runtime/` C++ | ~4,900 | 13 header-only files + 3 `.cpp` |
| `tests/` Rust | 9,973 | |
| `tests/**/*.vx` | 514 files | |

The largest single functions in the compiler:

| Lines | Location | Function |
| ----- | -------- | -------- |
| 1,413 | `src/codegen/flat.rs:1372` | `emit_function_mlir` |
| 750 | `src/hir/check/transfer.rs:995` | `check_transfer_expr` |
| 556 | `src/codegen/lower/expr.rs:2145` | `FunctionCallExpr::lower` |
| 515 | `src/codegen/lower/expr.rs:651` | `BinaryOpExpr::lower` |
| 511 | `src/codegen/lower/control_flow.rs:267` | `lower` |
| 509 | `src/hir/flatten.rs:528` | `Lowerer::lower_expr` |
| 399 | `src/codegen/lower/stmt.rs:223` | `lower` |
| 378 | `src/hir/check/calls.rs:398` | `check_functioncall_expr` |
| 354 | `src/syntax/expr.rs:827` | `Expr::substitute` |

## Findings

### 1. Two compilers share one repository

This is the deepest problem and the reason most of the others are hard to fix.

The **frontend** walks the AST twice with two unrelated pieces of code:

- `TypeChecker` (`hir/env.rs` plus `hir/check/*`, about 10,700 lines) resolves
  names, checks types, and produces diagnostics.
- `hir/flatten.rs` (3,668 code lines) walks the same AST again and re-derives
  types, layouts, element kinds, and enum instance shapes from scratch:
  `lowered_ty`, `scalar_of`, `sizeof_bytes`, `agg_gid_of_ty`,
  `substitute_generics`, `parse_enum_instance`, `element_mlir`.

The **backend** emits MLIR twice with two unrelated mechanisms:

- `codegen/generator.rs` plus `codegen/lower/*` (7,700 lines) build MLIR through
  melior's typed builder, walking the AST.
- `codegen/flat.rs` (2,785 code lines) builds MLIR as **text strings** from the
  flat HIR bytecode.

The duplication is acknowledged in the source. `element_mlir` in `flatten.rs`
carries the comment "the flat-lowerer counterpart of codegen's `mlir_scalar`".

The two halves are joined by a decline-and-fall-back contract. `flatten.rs` has
**72 bare `return None` sites** and `flat.rs` has about **96 `None` returns**.
None of them carries a reason. The only way to find out why a program declined is
to set `VX_FLAT_DBG=1` and read `[flat-dbg] fn X declined at stmt #N`.
`tests/integration_test/flat_corpus_sweep.rs` tracks the result as a hand-written
list of 35 `KNOWN_DECLINES` filenames.

Cost of this shape: every feature is written twice, the second write is optional,
and the gap between them is measured by a filename list rather than by anything
the compiler can explain.

### 2. A dead third IR

`hir/arena.rs` (456 lines) and `hir/lower_ast.rs` (465 lines) define an
`id_arena`-based tree HIR (`HirExpr`, `HirStmt`, `HirArena`). Nothing outside
those two files references any of those types. About 920 lines plus their tests
are dead.

### 3. God objects

`TypeChecker` has **43 fields**. It fuses at least seven separate concerns into
one AST walk:

- scope and name resolution (`scopes`, `used_vars`, `declared_vars`)
- borrow checking (`borrow: BorrowCx`)
- compile-time evaluation (`eval_env`, `constraints`)
- monomorphization (`monomorphized_functions`, `pending_topo_vars`,
  `pending_topo_bindings`, `closure_signatures`, `generated_structs`)
- memory algebra and seams, including a live z3 subprocess handle
  (`seam_solver`, `seam_contracts`, `seam_check_time`, `solver_init_time`)
- traffic and capacity accounting (`staging_routes`, `resident_sets`,
  `spawn_regions`, `memory_placements`)
- transfer-lowering context (`transfer_lowering_edge`, `transfer_lowering_params`)

Any new analysis lands here, which is why the struct keeps growing.

`emit_function_mlir` at 1,413 lines is one `match` over opcodes with the whole
MLIR text emitter inlined. `check_transfer_expr` at 750 lines is the memory
algebra's entry point.

### 4. Two orchestrators

`driver.rs::CompilerDriver` is the shipping `vxc` path. It does semantic analysis
inline in `run_semantic_analysis` (200 lines) and codegen in `run_codegen` (243
lines).

`pipeline.rs::compile_pipeline*` is the parallel path with named phases
(`parse_phase`, `macro_expansion_phase`, `name_resolution_phase`,
`type_check_phase`, `deduplication_phase`, `simd_patch_phase`,
`codegen_mlir_phase`).

They have different phase orders and different error types, and they meet only at
`build_frozen_registry`. The benchmarks, the determinism gates, and the scale
gates all exercise `pipeline`. Actual users get `driver`. A bug fixed in one is
not fixed in the other.

### 5. Layer inversions that block a workspace split

Measured by import direction:

- `codegen/flat.rs` imports `crate::pipeline::Schedule` at 11 sites, and
  `pipeline` imports `codegen`. That is a cycle between the backend and the
  orchestrator.
- `diagnostics_json.rs` imports `hir::env::{ResidentSet, StagingRoute, SpawnRegionTraffic, SpaceTraffic, Traffic, CostSource}`. The report format
  depends on the type checker's internals.
- `syntax/macro_expand.rs` constructs `crate::parser::Parser`. The AST layer
  re-enters the parser.
- `arch` imports `hir`; `registry` imports `hir`; `session` imports `hir`,
  `registry`, and `gid`, while `gid` imports `session`.

The result is a single 63,000-line crate that links MLIR. Touching one file
rebuilds and relinks everything.

### 6. Types identified by formatted strings

`Expr::substitute` builds instantiated names with
`format!("{}<{}>{}", base, args, remainder)` and special-cases the `Tensor_`
prefix. Downstream, `flatten::parse_enum_instance` splits that string back on
`<`, `,`, and `>`, and `parse_scalar_type_arg` maps `"i32"` back to
`ElementType::I32`. The identity round-trips through text.

There are **85 `starts_with("…")` sites** in `src/`. Twenty-seven of them match
MLIR type text (`memref<`, `memref<memref<`, `!llvm.ptr`, `vector<`, `tensor<`,
`struct<`), which is how `memref_elem`, `memref_lead_dims_and_elem`,
`slice_vec_len`, `is_slice_operand`, and `parse_llvm_struct_name` work. The rest
probe `Closure_`, `Tensor_`, `Option<`, and `Math::`.

A real identity system already exists in `gid.rs`. It is not what these seams use.

### 7. Five error channels

| Channel | Where | Shape |
| ------- | ----- | ----- |
| `Diagnostic` / `DiagnosticsVec` | checker | 92 stable codes, spans, notes, fix-its |
| `error::Error` | parser | newtype over `String` |
| `Result<_, String>` | driver | 71 sites |
| `PipelineError` | pipeline | three string variants |
| `LowerError` | codegen | wraps melior errors and strings |

The structured one is good. Anything that fails in the driver or the backend
reaches the user as a bare string with no code and no span, so it cannot appear
in `--diagnostics-json` and cannot be tested by code.

### 8. Misfiled modules

- `hir/check/mod.rs` documents the directory as holding "the checks for one
  family of expressions". `hir/check/transfer.rs` (1,895 lines) instead holds
  whole-program declaration validation: `check_topology_coherence`,
  `check_memory_coherence`, `check_declaration_conflicts`, `check_transfer_impls`.
- `hir/` holds an AST checker. The actual HIR is the bytecode in
  `hir/bytecode.rs`. The name says the opposite of what the directory contains.
- `src/scratch.rs` is a scratchpad experiment declared as `pub mod scratch` in
  `lib.rs` and compiled into the shipping library.
- `src/test_parent.rs` has a `fn main()` and is referenced by nothing.
- `lib.rs` carries an orphaned comment about an EVAL-ONLY `#295` module that no
  longer follows it.

### 9. Duplicated solver plumbing

`hir/prover.rs` spawns `z3` and drives it over stdin/stdout. `hir/seam.rs`
does the same again in `run_z3`, with a comment saying it "mirrors `prover.rs`".
Two subprocess lifecycles, two output parsers, two failure modes.

### 10. Configuration sprawl

Thirteen environment variables are read directly from analysis and codegen code:
`VX_FLAT_DBG` (10 sites), `VX_PIPELINE_QUIET`, `VX_STD_PATH`, `VX_DISPATCH_LIB`,
`ENZYME_LIB`, `OPT_PATH`, `LLC_PATH`, `CLANG_PATH`, `MLIR_TRANSLATE_PATH`,
`LLVM_CONFIG_PATH`, `RUST_BACKTRACE`, and the solver's allow-unverified switch.

`DriverOptions` is both the clap CLI definition and the compiler's configuration
object. It carries an `Action` enum plus seven boolean aliases for the same
choice, reconciled with `overrides_with_all`, and a stringly-typed
`intern_mode: String`.

`Cargo.toml` declares no features. CUDA, NPU, Enzyme, and z3 support are all
unconditional; platform selection happens with `cfg!(target_os)` inside
`build.rs`, which also shells out to `python3` during the build.

### 11. Test and CI gaps

- `cargo clippy --all-targets --all-features -- -D warnings` runs in
  `scripts/pre-commit.sh` but **not** in CI.

- CI runs `cargo test --release` only. Every `#[cfg(debug_assertions)]` block
  never executes in CI, including the whole `parallel_architecture_verifier` and
  the 10 verification hooks in `pipeline.rs`.

- Inline test coverage is inverted against risk:

  | File | Code lines | Test lines |
  | ---- | ---------- | ---------- |
  | `arch.rs` | 600 | 1,403 |
  | `hir/memory.rs` | 594 | 688 |
  | `pipeline.rs` | 1,716 | 1,189 |
  | `hir/flatten.rs` | 3,668 | 2,879 |
  | `codegen/lower/expr.rs` | 3,649 | 0 |
  | `hir/check/calls.rs` | 2,050 | 0 |
  | `hir/check/raw.rs` | 1,955 | 0 |
  | `hir/check/transfer.rs` | 1,895 | 0 |
  | `codegen/generator.rs` | 1,897 | 0 |

  The four largest untested files are the two backends and the two heaviest
  checkers.

### 12. Runtime C++

Thirteen files under `runtime/` are headers with the implementation inline:
`vx_dispatch_plan.h` (547), `vx_wire.h` (530), `vx_remote_routing.h` (520),
`vx_remote_client.h` (475), `vx_manifest.h` (362), `host_dispatch_common.h`
(349). `cuda_dispatch.cpp` is 1,198 lines. There is no include/src split and no
library target; `build.rs` (586 lines) compiles the set directly, and the
`rerun-if-changed` list is maintained by hand. A comment in `build.rs` records
that a missing entry cost an hour of debugging twice.

## The plan

Phases are ordered by dependency. Each phase should land behind the gates listed
in "Invariants" below, and each is small enough to be several commits.

### Phase 0 — Clear the floor

Low risk, no behaviour change, do this first because later phases are easier
once it lands.

1. Delete `hir/arena.rs`, `hir/lower_ast.rs`, `src/scratch.rs`,
   `src/test_parent.rs`, and the orphaned EVAL-ONLY comment in `lib.rs`.
1. Move `Schedule` from `pipeline.rs` to `session.rs` (or a new `config.rs`).
   This alone breaks the `codegen` to `pipeline` cycle.
1. Move `ResidentSet`, `StagingRoute`, `SpawnRegionTraffic`, `SpaceTraffic`,
   `Traffic`, and `CostSource` out of `hir/env.rs` into a new `src/report.rs`.
   `diagnostics_json.rs` then depends on `report`, not on the checker.
1. Move `syntax/macro_expand.rs` to `parser/macro_expand.rs`, so the AST layer
   stops depending on the parser.
1. Move the scratch artifacts out of the repository root (`*.mlir`, `*.ll`,
   `*.o`, `*.log`, `dump.mlir`, `out.mlir`, `raw.mlir`, `temp.*`, `test_*.mlir`)
   into a git-ignored directory.
1. Add to CI: `cargo clippy --all-targets -- -D warnings`, and one debug-profile
   test job so `#[cfg(debug_assertions)]` code actually runs.

Result: the cycle is gone, ~1,000 dead lines are gone, and CI runs the checks
that pre-commit already runs.

### Phase 1 — Name the decline

The flat path's coverage is the single most important number for the next round
of features, and today it is opaque.

1. Introduce `enum Decline { UnsupportedOpcode(Opcode), UnmodelledType(..), UnknownLayout(TypeId), GenericNotInstantiated(..), ControlFlowShape(..), … }`
   with a span.
1. Change `flatten.rs` from `Option<T>` to `Result<T, Decline>` at all 72 sites,
   and `flat.rs` likewise.
1. Surface the reason: `vxc --explain-decline`, and a reason histogram in
   `flat_corpus_sweep`.
1. Replace `KNOWN_DECLINES` (35 filenames) with a per-reason count so the sweep
   reports *what* is missing instead of *which files* are missing it.

Result: "what does the flat path not do yet" becomes a query rather than a
reading exercise, which makes Phase 4 plannable.

### Phase 2 — One type identity

1. Stop mangling and re-parsing instantiated names. `Expr::substitute` should
   carry `TypeId` plus structured type arguments; delete
   `parse_enum_instance` and `parse_scalar_type_arg`.
1. Introduce a small `MlirTy` value type (scalar, memref, vector, llvm-ptr,
   llvm-struct) with `Display`, and replace the 27 `starts_with("memref<")`-class
   string probes with matches on it.
1. Fold `flatten::element_mlir` and `flat::mlir_scalar` into one function.

Result: the type seams stop being text, and Phase 4's shared emitter becomes
possible.

### Phase 3 — Split the god objects

1. `emit_function_mlir` (1,413 lines): split by opcode family into
   `flat/emit/{arith, control, memory, call, aggregate, tensor, transfer}.rs`,
   with the dispatch left in place. Mechanical, no logic change.
1. `TypeChecker`: extract a `CheckCx` holding the shared context (scopes, env,
   worker, diagnostics, active topology and memory), then move each analysis into
   its own struct that borrows it — `BorrowChecker`, `ConstEval`,
   `Monomorphizer`, `SeamProver`, `TrafficAccountant`. The dispatch stays where it
   is; only ownership of state moves.
1. Move the declaration checks out of `hir/check/transfer.rs` into
   `hir/decl_check/{topology, memory, conflicts}.rs`, matching what
   `hir/check/mod.rs` says the directory is for.
1. Split `check_transfer_expr` (750 lines) along its own phases: resolve the
   edge, route the staging hops, cost them, then discharge the obligations.

Do this after Phase 1 and 2 so the split does not have to preserve string-parsing
subtleties.

### Phase 4 — One backend

This is the payoff phase, and it needs Phases 1–3 first.

1. Define a `MlirBuilder` trait with the operations both backends need. Implement
   it twice: over melior (typed) and over the current text emitter.
1. Port `flat.rs` to emit through the trait. Its output must stay byte-identical;
   the differential test already checks that.
1. Use the reason histogram from Phase 1 to close the remaining flat gaps, worst
   reason first.
1. When the histogram is empty, demote the AST path to test-only: keep it as the
   oracle behind `--legacy-codegen` and the differential test, delete it from the
   shipping path, and remove the fall-back branch in `driver::run_codegen`.

Result: one implementation per language feature.

### Phase 5 — One orchestrator

1. Give `driver` and `pipeline` the same phase functions. `pipeline`'s phase
   decomposition is the better one; `driver::run_semantic_analysis` and
   `run_codegen` should call into it rather than reimplement it.
1. `CompilerDriver` becomes a thin CLI adapter: parse `DriverOptions`, translate
   to a plain `CompileConfig`, call the pipeline, format the result.
1. Split `DriverOptions` into the clap struct and the internal `CompileConfig`.
   Collapse the seven boolean action aliases into `Action`. Make `intern_mode` an
   enum. Route the 13 environment variables through `CompileConfig` so they are
   read in one place and testable.
1. Unify the error channels: `PipelineError`, `LowerError`, `error::Error`, and
   the 71 `Result<_, String>` sites all become `Diagnostic` with a code.

Result: the path users take and the path CI measures are the same path.

### Phase 6 — Workspace split

Only possible once Phase 0 and Phase 5 have removed the inversions.

```
vx-syntax     AST, spans, symbols, types           (no deps on parser/sema)
vx-lex-parse  lexer, parser, macro expansion
vx-sema       resolver, checker, borrow, memory algebra
vx-hir        bytecode, flatten
vx-codegen    MLIR emission + melior
vx-runtime-sys build.rs, C++ runtime, dialect
vx-driver     CLI (the vxc binary)
```

Add cargo features: `cuda`, `npu`, `enzyme`, `z3`, so a frontend-only change does
not rebuild the dispatch libraries. Expected payoff is on incremental build time,
which is the thing that most limits iteration on a 63,000-line crate that links
MLIR.

### Phase 7 — Runtime C++

1. Split `runtime/` into `runtime/include/` (declarations) and `runtime/src/`
   (definitions). The 13 header-only files are the reason `build.rs` needs a
   hand-maintained `rerun-if-changed` list.
1. Split `cuda_dispatch.cpp` (1,198 lines) by concern: device management, memory,
   kernel launch, collectives.
1. Generate the `rerun-if-changed` list from a glob instead of by hand.
1. Move the build-time `python3` call for ANE primitives behind the `npu`
   feature.

## Invariants

None of the above may regress these. All of them already have gates in CI, and
every phase should be landed behind them:

- Byte-identical MLIR across thread counts
  (`pipeline_emits_byte_identical_mlir_across_thread_counts` and the three
  sibling gates).
- Compilation isolation
  (`concurrent_compilations_have_isolated_topologies`).
- Parity with the AST oracle
  (`pipeline_emits_mlir_that_matches_the_ast_oracle`).
- Flat coverage does not shrink
  (`flat_path_coverage_of_the_backend_corpus_holds`).
- Determinism at benchmark scale, 16,000 functions
  (`pipeline_emits_identical_mlir_at_benchmark_scale`).
- No locks, no thread parking, no process-global atomics (the two grep gates).

Two more should be added as part of Phase 0:

- clippy clean, as a CI gate rather than a pre-commit convention.
- a debug-profile test run, so the architecture verifier runs somewhere.

## Suggested order and effort

| Phase | Risk | Blocks | Notes |
| ----- | ---- | ------ | ----- |
| 0 Clear the floor | very low | 5, 6 | mostly deletions and moves |
| 1 Name the decline | low | 4 | mechanical `Option` to `Result` |
| 2 One type identity | medium | 4 | touches 85 string sites |
| 3 Split god objects | medium | 4 | moves only, no logic change |
| 4 One backend | high | — | the payoff; needs 1–3 |
| 5 One orchestrator | medium | 6 | |
| 6 Workspace split | medium | — | build-time payoff |
| 7 Runtime C++ | low | — | independent of 0–6 |

Phase 0 and Phase 1 are worth doing immediately and independently: they are
cheap, they are reversible, and they turn the flat path's coverage into a number
the team can steer by. Phase 7 is independent of everything else and can run in
parallel.

## What not to do

- Do not rewrite the AST into an arena or an index-based IR. The last attempt is
  still sitting in `hir/arena.rs` unused. The AST's shape is not what is slowing
  development down; the duplication is.
- Do not split the crate before Phase 0 and 5. The import cycles make it fail,
  and a half-done workspace split is worse than none.
- Do not delete the AST codegen path before the decline histogram is empty. It is
  the oracle the differential test compares against, and it is what keeps the
  flat path honest.
- Do not merge `driver` and `pipeline` by deleting one. Their phase orders differ
  for reasons that need to be checked case by case against the gates.

## Corrections found while executing (Vx#387)

Checked against the tree as Phase 0 was carried out. The findings above hold except
for these.

**Phase 0 item 5 was already done.** No scratch artifact in the repository root is
tracked, and `.gitignore` lines 10-14 already cover `*.o`, `*.log`, `*.ll`, and
`*.mlir`. The files sitting there are untracked local clutter. A smaller real
problem remains behind it: tests and scripts write scratch output to the working
directory instead of a temp directory.

**Phase 0 item 3 named six types; the set is eight.** `BufferTraffic` and
`TrafficSource` sit in the same block, and the other six reach them --
`SpawnRegionTraffic` holds a `Vec<BufferTraffic>`, `Traffic` holds a
`TrafficSource`. Moving only the six named would have split types from their own
fields. All eight are in `src/report.rs`.

**There is no `gid`/`session` cycle in shipping code.** `gid.rs` names
`crate::session` at three places, all after the `#[cfg(test)]` at line 417, and it
has no top-level import of it. The dependency runs one way, `session` to `gid`.
The remaining inversions in that finding are one site each: `arch.rs:669` calls
`hir::memory::MemoryHierarchy::build`, and `registry.rs:93` holds a
`Vec<hir::bytecode::HirInstruction>`.

**Phase 1 is larger than "mechanical `Option` to `Result`".** The 72 and 96 counts
do not match the tree: `flatten.rs` has 39 bare `return None;` and `flat.rs` has 31.
The 96 looks like a count of `None` tokens (98 in `flat.rs`), most of which are
ordinary values rather than declines. What sizes the phase is the signature change:
61 `Option`-returning functions in `flatten.rs` and 19 in `flat.rs`, with about 350
`?` sites propagating through them. `?` works on `Result` unchanged, so propagation
is free, and the work is choosing a reason at each origin and moving 80 signatures
with their call sites.

**Phase 0 leaves one edge in each direction it claimed to cut.**
`codegen/flat.rs:2844` still calls `pipeline::build_frozen_registry`, inside
`#[cfg(test)] mod tests`. Shipping code no longer crosses. That test wants to be an
integration test, which is Phase 6's business.

### The proposed crate list needs two changes (Vx#387)

Found by cutting the cycles rather than by reading the module names.

`vx-hir  bytecode, flatten` cannot hold both. `registry`, `session`, and `metadata`
all store `HirInstruction`, and `flatten` depends on `registry` and `session`, so
the instruction format has to sit below all three while the flattener sits above
them. `bytecode.rs` is now `src/bytecode.rs` and imports nothing at all, which is
what let the move be a rename.

`vx-lex-parse  lexer, parser, macro expansion` puts the lexer above the AST, and
the edge runs the other way: `syntax/macro_rules.rs` stores `OwnedToken`, because a
macro's rules are tokens. `lexer` never reaches `syntax`, so this is a layout
choice rather than a cycle -- the token types belong below `vx-syntax`, either in
their own crate or inside it.

`arch` is not in the list at all and should be part of `vx-syntax`. It is the
vocabulary the AST is written in (`MemorySpace`, `TopologyDescriptor`), `syntax`
holds one field of it (`Program.topologies`), and `arch` reads `syntax` at 23
places. In one crate that mutual reference costs nothing; split apart it is a cycle
with no clean cut.

One cycle is left standing on purpose. `parser` no longer reaches `borrow`, but
`hir` still reaches `parser`, through `hir::env::parse_ty_str` re-parsing a type
from its printed form. That is finding 6 above, and Phase 2 deletes it. Cutting it
any sooner would mean working around the string round-trip instead of removing it.

### What Phase 3 got wrong, checked against the tree (Vx#387)

**`emit_function_mlir` is 1,721 lines, not 1,413.** The plan's figure was taken on
a pre-#386 tree; the function grew before anyone split it, which is the argument
for splitting it rather than against.

**Seven emit families should be eight.** `flat/emit/` came out as `arith`,
`control`, `memory`, `call`, `aggregate`, `tensor`, `parallel`, `io`. Printing and
aborting are their own family rather than a misc pile, and `transfer` is really
`parallel` — it also holds `Spawn`, `SpawnEnd` and `Barrier`.

**`TypeChecker` has 42 fields, not 43.** Two of them, `closure_depths` and
`closure_captures_stack`, carry a stale `#[allow(dead_code)]`; both are live, read
from `check/access.rs`, `check/control.rs` and `check/raw.rs`.

**The declaration checks are five functions, not a line range.** `check/transfer.rs`
does not divide into "declaration checks above, expression checks below".
`check_capacity`, `check_type_placement`, `space_is_cached` and
`value_memory_space` sit among them and read like declaration checks, but each is
shared: `check_capacity` is called from `check_transfer_expr`,
`check_type_placement` from `hir/stmt.rs` per statement, and the other two from
`check/access.rs` and `check/region_traffic.rs`. What moved to `hir/decl_check/`
is the five functions reachable only from `check_whole_program_declarations`.

**Splitting `check_transfer_expr` is not code motion.** The plan asks to split its
749 lines "along its own phases: resolve the edge, route the staging hops, cost
them, then discharge the obligations". Those phases are not sequential statements
that can be lifted — 628 of the 749 lines are a single `if let Expr::...` arm, and
the phases are nested inside it. Doing this means restructuring the memory
algebra's entry point, with the behaviour risk that implies. It should be scoped as
its own piece of work, not run as the tail of a mechanical phase.

**A note on method, from Phase 3.** The mechanical rewrites in this phase were
checked before they were run, and three of the four problems found would have
compiled cleanly: 22 of the 23 `func.` occurrences in the emitter's match arms are
emitted MLIR text (`func.call`, `func.return`) rather than field accesses, a
closure took `body` as a parameter whose name a naive rewrite captured, and prose
inside comments was rewritten as if it were code. A refactor over a text emitter
cannot be validated by the type checker alone, because its output is strings.

### Phase 3 step 2 was done differently, and why (Vx#387)

The plan asks for a `CheckCx` holding the shared context, with `BorrowChecker`,
`ConstEval`, `Monomorphizer`, `SeamProver` and `TrafficAccountant` borrowing it.
What landed instead groups the same state into four structs owned *by*
`TypeChecker` — `ConstEvalState`, `MonoState`, `SeamState`, `TrafficState` in
`hir/check_state.rs` — taking the struct from 42 fields to 22.

The measurement is the reason. Of 443 methods across `hir/`, **399 touch no
analysis-specific field at all**; 33 touch exactly one group, and 11 span more
than one. A shared context object pays its cost in the 399: each would have to
reach `self.cx.scopes` instead of `self.scopes`, roughly a thousand sites
rewritten so that the 44 methods holding real analysis state could be separated.
Grouping cost 137 sites and separates the same state.

It also follows something already in the tree. `borrow: BorrowCx` is this exact
pattern, introduced to replace three loose fields, and its module documents why
the boundary earns its keep: `active_borrows` is private, so a new access path
cannot skip the dead-borrow sweep. The four new structs have no invariant to
protect yet and keep their fields reachable; when one grows an invariant it
should grow methods and hide the field, as `BorrowCx` did. Extending one pattern
seemed better than standing up a second one beside it.

The eleven methods that span groups are worth naming, since they are where a
future split would have to do real thinking rather than mechanical work:
`check_function`, `instantiate_generic_function_call`, `check_identifier_expr`,
`check_memberaccess_expr`, `check_static_method_call`,
`instantiate_method_call_rewrite`, `check_raw_primitive`, `check_transfer_expr`,
`push_scope`, `pop_scope`, and `check_let_decl_stmt`.

One trap worth recording: `eval_env` was initialized as `vec![HashMap::new()]`,
one open scope, so `ConstEvalState` needs a hand-written `Default`. It was the
only one of the twenty-four moved fields whose initializer differed from
`Default::default()`, and deriving it would have started compile-time evaluation
with no scope to bind into.

### Phase 3 step 4, and what made it possible (Vx#387)

The earlier note in this document said splitting `check_transfer_expr` was not
code motion, because 628 of its 749 lines sat inside one `if let Expr::Transfer`
arm with the phases nested rather than sequential. That was true of the shape,
and wrong about the conclusion. The phases the plan named are all there; what
held them together was not logic but scope. Two facts did it:

- `inner_ty` and `target_mem` were declared uninitialized at the top of the
  function and assigned inside the arm, so both had to outlive it.
- The staged-route rewrite ran *after* the arm and needed `expr` back, so the
  arm could not simply own the borrow.

Naming the arm's output fixed both. `ResolvedEdge` carries the two endpoints, the
route, and the four cost figures out of the resolver; the deferred `let`s go
away, the `if let` becomes a `let ... else`, and the staged case becomes an early
return. What is left is 37 lines that read as the sequence the plan described:

| function | lines | what it answers |
| --- | --- | --- |
| `resolve_transfer_edge` | 173 | where does this go, and how does the hardware get there |
| `select_transfer_lowering` | 159 | whose lowering runs on this edge |
| `record_emittable_lowering` | 132 | does its declared tile match the one being moved |
| `record_staging_route` | 139 | what did this route cost, and what traffic did it move |
| `discharge_seam_obligation` | 35 | is the buffer crossing the seam described well enough |
| `stage_multi_hop` | 40 | rewrite a staged route into single hops and re-check |
| `transfer_result_type` | 34 | what type does the value have now |
| `check_transfer_expr` | 37 | the order the above run in |

Two things came out of it that were not the point. The space-to-topology map was
written twice, twenty-six identical lines each, and is now `pinned_topology_for`.
And E6023's source-versus-destination message carries a run of embedded spaces
from a string continuation that rustfmt folded onto one line; no corpus program
reaches that arm, which is why it survived.

**On verifying this one.** The type checker cannot tell a faithful move from an
unfaithful one here, the same way it could not for the emitter in step 1 — the
output is diagnostics, and a dropped check is silence. So the oracle was the
compiler's own output: every `.vx` file under `tests/` and `examples/`, 514 of
them, compiled to MLIR with stderr and exit status captured, snapshotted before
the first edit and compared after every increment. The snapshot was itself
checked for stability first, since a nondeterministic baseline would have proved
nothing. All 514 stayed byte-identical through all six extractions.
