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

## Entry 4 — Decision: make the flat pipeline first-class (#197)

**Decision.** Per [#197](https://github.com/hiraditya/Vx/issues/197): **converge** — the flat-array
pipeline becomes the production compile path, because that is the core claim of the Vx architecture.
The staged roadmap and keep-green strategy are in
[`../implementation_plans/flat_pipeline_convergence.md`](../implementation_plans/flat_pipeline_convergence.md);
#197 is the tracking epic. Prerequisites first (C0: word-2 codec #193, cross-module resolution #194,
stable hash #195), then HIR lowering (C1), flat codegen behind a flag with differential testing (C2),
then flip `vxc` (C3). The AST path stays production + oracle throughout.

## Entry 5 — C0.1: word-2 codec unification, end to end (#193)

**Commits:** `cf9d5c4` (W1 codec) → `a947b60` (W2/W3/W4).

**Gap.** Word 2 of the 256-bit GID was *triple-booked*: a fast-path lifetime bitfield, a lifetime
slow-path arena index, and a generic deferred arena index — signalled by **two disagreeing** flags
(`ESCAPE_HATCH_MASK` on word 2 vs `LOCAL_DEFERRED_BIT` on word 3) and read with **two index
conventions** (masked vs raw). The generic path even left the escape-hatch bit *clear* while storing
a raw index, so the moment the HIR stream mixes generic and lifetime GIDs they collide (design review
H1, [`parallel_pipeline_design_review.md`](./parallel_pipeline_design_review.md)). This is the first
C0 prerequisite for making the flat pipeline first-class (#197).

**What we did.** One codec now owns word 2 (`gid.rs`): `classify_word2` / `set_arena_index` over a
`Word2 { FastLifetime(u64) | Index { index, arena: {Generics,SlowMeta}, scope: {Local,Global} } }`.
The discriminant is a *single* bit — `ESCAPE_HATCH_MASK` — and the word-3 flags only pick
*arena*/*scope*, never "is this an index". All three touch points were routed through it:

- `mint_deferred_generic` → `set_arena_index(offset, Generics, Local)` (now sets the escape-hatch bit
  even for index 0 — that is exactly what distinguishes "arena index 0" from an empty bitfield);
- `simd_patch_phase` → `classify_word2` → remap local→global → `set_arena_index(global, arena, Global)`
  (the patched GID stays an *index*, not a bitfield);
- `resolve_lifetime` → `classify_word2`: `FastLifetime → FastPath`, pure `Generics → FastPath(0)`
  (lifetime-unconstrained), `SlowMeta → SlowPath` in the local/global arena by scope.

W2 reserved word-2 bit 63 by narrowing `FAST_PARAM_VARIANCE_MASK` `0x000F → 0x0007`, so a fully
packed 4-param fast GID can never set the escape-hatch bit (variance only needs 2 bits).

**Invariant established.** Word 2 is always *exactly one of* {inline lifetime bitfield, one arena
index}. A type that is both borrowed and generic will ride the `SlowMeta` composite arena
(`UnboundedFunctionMetadata` already carries `type_arguments` + `lifetime_regions`) — that is W5,
deferred until borrowed generics are actually emitted.

**Tests.** `fast_param_never_touches_escape_hatch_bit` (W2), `generic_deferred_gid_is_not_misread_as_lifetime`
(W4, the H1 symptom), and the interning/patch end-to-end test now asserts via the codec. Full suite
green (274 lib + 49 integration). Behaviour-preserving for the current surface.

**Scope / not yet.** W5 (borrowed-generic composite). Next C0 gaps: cross-module GID resolution
(#194) and a stable, project-controlled hash (#195).

## Entry 6 — C0.2: cross-module GID resolution (#194)

**Commit:** `ce246d7`.

**Gap.** The 256-bit GID exists for *cross-module* identity (word 0 = defining-module hash), but name
resolution was intra-module only: `Type::resolve_names` looked a nominal type up **only** in the
current module's `SymbolTable` (`mod_syms.and_then(|m| m.get(name))`), never in the full
`symbol_map`. A reference to a type in another module got `id = None` — or, worse, was silently
shadowed by a same-named local. So the flat type stream + frozen registry carried only within-module
identity, and the registry's `module_indices` (built at the Phase 2 freeze) was never read (design
review H2, [`parallel_pipeline_design_review.md`](./parallel_pipeline_design_review.md)).

**What we did.**

- **Parser** (`parser/types.rs`): `parse_named_type` now parses a `::`-qualified nominal path
  (`A::Foo`) and keeps the whole path as the nominal's name `Symbol`, so the AST shape
  (`Type::Struct(Symbol, Option<TypeId>)`) is unchanged. Builtins (`Tensor`, scalars) are never
  qualified, so the branch only fires when a `::` actually follows.
- **Resolution** (`syntax/resolve.rs`): a new `ResolutionScope` is threaded as *one* value through
  the recursive walk (replacing the `(mod_syms, symbol_map)` pair — a mechanical, contained
  signature change, all internal to `resolve.rs`; the public `Program::resolve_names(&symbol_map)`
  entry is unchanged). It centralizes nominal resolution in `resolve_nominal`:
  - a **qualified** `Mod::…::Name` splits at the last `::`; the module part resolves to a real
    `symbol_map` key either literally (`crate::a`) or by expanding a leading segment through an
    `import a::b;` alias; the leaf is looked up in *that* module. A qualified reference **never**
    falls back to a same-named local type.
  - an **unqualified** name takes local definitions first (as before), then an `import a::b::Name;`
    brings `Name` in from module `a::b`.
- **Registry** (`registry.rs`): `resolve_in_module(module_hash, symbol)` — the frozen-registry
  counterpart of the symbol-map lookup, i.e. the `module_indices` read path the design (§7 Step 3)
  describes. It is now the queried path in the end-to-end pipeline test.

**Why the `ResolutionScope` bundle.** The alternative (adding a third `imports` param to every
`resolve_names` arm) spreads the cross-module policy across ~30 call sites. Bundling
current-module + symbol map + a pre-built import index behind one reference keeps the policy in one
place and threads a single value. The scope borrows only `symbol_map` and *owns* an index cloned
from the imports, so building it before the mutable walk sidesteps a self-borrow conflict.

**Tests.** Qualified cross-module resolution attaches A's GID (word 0 = A's module hash); a qualified
ref is not shadowed by a same-named local; imported unqualified names resolve cross-module;
`resolve_in_module` reads `module_indices` with module isolation; and an **end-to-end pipeline** test
(`frozen_registry_carries_cross_module_by_value_identity`) proves the frozen registry resolves a
cross-module *by-value dependency* (`struct Bar { f: A::Foo }` in B edges to A's `Foo` node). Full
suite green (276 lib + 52 integration).

**Scope / not yet.** Method/associated-path resolution (`a::b::method`) beyond nominal types; use
aliases with renaming (`import a::b as c;`) — the parser has no `as` form yet; and glob imports.
Next C0 gap: a stable, project-controlled hash (#195) so this identity is reproducible across runs
and toolchains.

## Entry 7 — C0.3: project-controlled stable hash + collision guard (#195)

**Commit:** `3973058`.

**Gap.** GID word 0/1 content hashes used `FxHasher` — non-cryptographic **and not stable across
`rustc_hash` versions** — while the architecture doc claimed a "cryptographic" hash "rendering
collisions mathematically impossible." The reproducible-build and cross-crate-identity guarantees
both rest on a *stable, project-owned* algorithm; the doc claim was also indefensible for a paper
(design review M1). Ironically `arch.rs` had *already* hand-rolled FNV-1a for dispatch ids
"specifically to avoid unstable std hashing."

**What we did.**

- `src/hash.rs`: replaced `FxHasher` with a hand-rolled **64-bit FNV-1a** (standard offset basis
  `0xcbf29ce484222325` + prime `0x100000001b3`). It is streaming, so a multi-field `DefPath` folds
  as one FNV stream, with the two `u64` fields of `Anonymous` written in a fixed little-endian byte
  order (architecture-independent). GIDs now reproduce byte-for-byte across toolchains and machines.
- `src/registry.rs`: `build_and_validate` now **deterministically rejects a GID collision** — two
  *distinct* symbols landing on the same 256-bit identity is a hard error, not a silent conflation.
  (Re-listing the same type — same id *and* same name — stays allowed.) This backs the honest
  framing: FNV is stable but not cryptographic, so we *catch* the improbable collision instead of
  asserting it away.
- Docs: `parallel_compiler_architecture.md` §2 and the zero-swizzle section now say
  "project-controlled stable content hash (FNV-1a), negligible-but-not-impossible collisions caught
  at registry-build time," and note a keyed 128-bit hash (SipHash-1-3 / truncated BLAKE3) is a
  layout-compatible future upgrade.

**Tests.** FNV-1a **golden values** (pin the exact algorithm + constants so a silent swap back to a
std hasher is caught), empty-input = offset basis, `build_rejects_gid_collision_between_distinct_types`,
`build_allows_same_type_listed_twice`. Full suite green (280 lib + 52 integration) — changing the
hash broke nothing, confirming no code depended on the old `FxHasher` values.

**This closes C0.** The flat pipeline now has coherent identity: one word-2 codec (#193),
cross-module resolution (#194), and a stable, collision-checked hash (#195). Next is **C1 — HIR
lowering**: populating `local_hir_stream` so the flat streams carry function *bodies*, not just
signature type references.

## Entry 8 — Hardening symbol resolution for recursion + corner cases (pre-C1)

**Commits:** `544d566` (local fast-path), `4083a10` (corner-case suite + two fixes).

**Why now.** C1 (HIR lowering) builds directly on the GIDs that resolution attaches, so before
lowering we stress-tested the resolver against recursion and self-reference — the classic places a
naive resolver loops or mis-binds. Resolution walks the AST and **never follows a type's
definition**, so recursive types can't loop it; the value was in confirming the *right* GID lands,
and the suite surfaced two real gaps.

**Fast-path (`544d566`).** A qualified reference to the *current* module (`A::Foo` inside `A`) took
the two-level `symbol_map` lookup even though `current` *is* `symbol_map[current_path]`. The scope
now carries the current module path (owned `Arc<str>`) and short-circuits self-references to the
local table — module-local refs dominate. Correctness-equivalent (same GID).

**Two gaps fixed (`4083a10`).**

1. **Generic parameters bound to same-named nominals.** A parser-produced `Type::Generic` is a type
   *variable* (types.rs classifies it from the in-scope generic list), yet it shared the nominal
   resolution arm, so a param `T` would bind to a `struct T`. Dropped `Type::Generic` from the arm —
   its `id` is dead (every consumer binds it `_`), so this is pure correctness, no behaviour change.
2. **Enum variant payloads never resolved.** `EnumDecl::resolve_names` was a no-op, so
   `enum Tree { Node(Tree) }` left payload GIDs `None`. Fine for the AST checker (it re-resolves) but
   the **flat pipeline reads these GIDs**, so `build_frozen_registry` missed enum by-value deps —
   recursive / cross-module enum cycles went undetected. Now it walks the payload types.

**Coverage added.** Integration: self-recursive struct, mutually recursive structs (same-module +
cross-module), recursion through a generic instance, recursion under `&`, qualified self-recursion,
multi-segment module paths, unresolved qualified module → `None` (no panic), recursive-function
signature types, generic-param non-binding, recursive + cross-module enum payloads. Pipeline/registry:
cross-module by-value recursion detected, broken by indirection, recursive enum detected. Full suite
green (283 lib + 65 integration). Resolution is now trusted enough to lower on top of.

## Entry 9 — C1.1: flatten scalar bodies to HIR bytecode (#197)

**Commit:** `1af30aa`. Plan: [`../implementation_plans/hir_flattening.md`](../implementation_plans/hir_flattening.md).

**Gap.** `local_hir_stream` (`hir/bytecode.rs`) was defined but never populated — the flat pipeline
carried signature type-refs (`emit_function_type_gids`) but no function *bodies*, so codegen still had
to walk the AST. C1 is the instruction-selection pass that fills it; this entry starts it.

**What we did (`src/hir/flatten.rs`).** A new pass, `lower_function_to_hir(func, worker)`, lowers a
function body to flat `Vec<HirInstruction>`. Conventions it establishes (C2 must honor): **SSA by
position** — instruction at index `i` defines `Register(i)`, operands name earlier instructions by
index, no destination field; `type_idx` indexes `local_type_stream`; **params** become `Load`
instructions at the top; **locals** are pure SSA aliases (`let x = e` binds `x` to `e`'s register).

*C1.1 subset:* scalar params, int/float literals (`Const` + raw `imm`), identifier reads,
arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Matmul`), `return` (`Ret`), expression statements. Everything
else (calls, control flow, structs, non-scalar params, generics) is out of scope.

**Keep-green invariant.** Lowering is **atomic per function**: any unsupported construct aborts and
the worker is left untouched, so `local_hir_stream` is *only ever a complete, correct lowering or
empty*. This lets the corpus grow safely and lets C2's differential testing trust every non-empty
stream. Distinct from `hir/lower_ast.rs` (arena *tree* HIR) — this is the flat bytecode.

**Coherence.** The primitive-scalar GID is now one shared `scalar_gid` (pipeline's `nominal_gid`
reuses it), so `i32` has identical identity in a signature and in a lowered body — required once the
type stream mixes both.

**Wiring + verification.** Runs in `type_check_phase` next to `emit_function_type_gids` (function and
impl-method loops), with a debug `verify_hir_stream` hook (type_idx in-bounds; SSA operand
dominance). Tests: six unit (arithmetic/return, let reuse, int/float immediates, atomic abort on
unsupported + on non-scalar param) plus an end-to-end pipeline test proving a scalar body lowers
through the real parallel phase. Full suite green (290 lib + 65 integration).

**Next.** C1.2 — control flow (`if`/loops → branch opcodes), calls (`Call`), mutable locals
(`Store`/reload), comparisons; then C1.3 — struct/field/tensor/`vx`-dialect surface.

## Entry 10 — C1.2: value ops + control flow in the flat HIR (#197)

**Commits:** `0c687a7` (value ops), `dee88b4` (if/else), `7d792d6` (loops).

**What we did.** Grew `flatten.rs` from the C1.1 scalar core to full intra-function control flow:

- **Value ops (model-agnostic):** comparisons → `Cmp` (relation in `imm`, result `bool`), `as` →
  `Cast` (`type_idx` = target), unary → `Neg`/`Not`.
- **Control flow:** `if`/`else` and both loops (`loop`, `for i in a..b`) lower to basic blocks
  (`BlockStart`) + branches (`Br`/`CondBr`, targets in `imm`), with `break`/`continue` driven by a
  per-function `loop_stack` of `(continue, break)` block ids. `for` puts the induction variable and
  the once-evaluated bound in slots and routes `continue` to the increment *latch* so it can't skip
  the step.

**Key decision — the local-variable model.** The existing AST codegen lowers control flow to the
*unstructured* `cf` dialect with `alloca`-backed locals (the -O0 memory model), so the flat HIR
mirrors that rather than SSA-block-args or structured `scf`: a **two-tier** scheme selected per
function — straight-line functions stay pure-SSA (locals alias value registers, cheap), functions
with control flow switch to the **memory model** (named locals get an `Alloca` slot; reads
`SlotLoad`, writes `Store`), so values cross blocks without phi/block-args. This makes C2 codegen a
near-direct translation and keeps differential parity with the AST path tractable. Effect
instructions (`Store`/`Br`/`CondBr`/`BlockStart`) carry a `NO_TYPE` `type_idx` sentinel — value and
type index spaces are now decoupled.

**Not here: calls.** Function/method calls were deliberately deferred. They need (a) function-symbol
resolution wired into the flat path (a separate C0.2-style pass — `resolve_names` resolves *types*,
not call targets), and (b) a variadic-argument representation the 2-operand instruction can't hold
directly. Kept out so control flow could land clean.

**Verification.** `verify_hir_stream` now also validates the sentinel, the new opcodes' operand
dominance (temporaries are block-local, so the linear check still captures SSA dominance), and that
every branch targets a declared block. The atomic-abort invariant is unchanged — any unsupported
construct still discards the whole function's stream. Full suite green (302 lib + 65 integration).

**Next.** C1.3 — the memory / `vx`-dialect surface (struct & field access, tensor/slice ops,
`spawn`/`transfer`).

## Entry 11 — C1.3 start: `spawn` region, and the non-scalar blocker (#197)

**Commit:** `77f66d3`.

**What we did.** Started the `vx`-dialect surface with `spawn on (<topology>) { body }` — the
signature Vx op. It lowers to a `Spawn`/`SpawnEnd`-delimited region in the flat HIR carrying the
topology dispatch id (`arch::topology_dispatch_id`, the same id the AST codegen stamps on
`vx.spawn`'s `topology` attribute); the body lowers inline between the markers. First cut is narrow
(statement-form, straight-line body/function) so the region is a linear instruction range; codegen
reconstructs `vx.spawn` from the marker pair.

**The blocker for the rest of C1.3.** `transfer`, tensor/slice ops, and struct/field access all need
the flat HIR to carry **non-scalar values and types**, which the C1.1–C1.2 model doesn't have:

- `transfer`'s operand is a tensor/`Ref`, not a scalar — `lower_expr` only produces scalar `Val`s.
- struct field access needs **field offsets**, but `ImmutableGlobalRegistry` layouts are built with
  `size_bytes = 0` / `align_bytes = 0` (`build_frozen_registry`) — layout/offset computation is a
  prerequisite that does not exist yet.
- tensor ops need tensor types (shape + element) in the type stream and index/elementwise opcodes.

So the next sub-project is **extending the type/value model beyond scalars**: aggregate + tensor GIDs
in the stream, `Alloca` sizing from real layouts, and field/index opcodes — which also unblocks
`transfer`. This is a meaningful new design surface (layout computation, tensor type modelling) and a
natural checkpoint.

## Entry 12 — C2 start: flat MLIR emitter for scalars (#200)

**Commit:** `01fb868`. `src/codegen/flat.rs`.

**What we did.** First codegen driven by the *flat* stream instead of an AST walk:
`emit_function_mlir(func, hir, types)` lowers a function's `local_hir_stream` + `local_type_stream`
to a `func.func`. SSA name of a value = its producing instruction's register; params are block
arguments (`Load imm=i` → `%argi`); `Const`/`Add`/`Sub`/`Mul`/`Div`/`Ret` map to `arith`/`func` ops.
Element types are recovered **from the type-stream GIDs** by inverting `scalar_gid` over the finite
scalar set — the flat-driven counterpart of an AST type walk. This is the first time the whole
pipeline (parse → resolve → GID → flat HIR → MLIR) runs end to end.

**Scope + keep-green.** Straight-line scalar arithmetic only; anything else (control flow, spawn,
memory, matmul, non-scalar types) returns `None`, so the AST path stays the oracle and nothing
half-lowered is emitted. Tests parse **and `verify()`** the emitted MLIR in a real melior context
(integer add, float mul+add, constant + signed div), and confirm a control-flow function is declined.

**Design note — recovering types from GIDs.** The type stream stores content-hash GIDs, not
element types, so codegen must invert them. For scalars a finite reverse scan works; for nominals
(later) the registry maps GID → layout. This is the same "GID is the identity, look up the rest"
pattern the architecture uses everywhere.

**Next (C2.1).** A real differential harness — JIT-run the flat- and AST-emitted functions and
compare numeric results — then wire the flat path behind `--flat-codegen` so it runs alongside the
AST oracle across the corpus. That turns "emits verifiable MLIR" into "provably equal to the AST
path," the actual convergence criterion.

## Remaining gaps (next entries)

- **`local_hir_stream`** — `HirInstruction` (`src/hir/bytecode.rs`: `{opcode, operand1, operand2, type_idx→LOCAL_TYPE_STREAM, imm}`) is a defined flat bytecode, but the stream is never populated;
  lowering function bodies to it is a real instruction-selection pass (and codegen would need to
  consume it — currently codegen is AST-based).
- **Slow-path variance** — `evaluate_slow_path_variance` (`borrow.rs`) is a stub.
- **Path convergence** — `vxc` still runs the sequential driver (`driver.rs::execute` →
  `run_codegen`); making it drive `compile_pipeline` (and codegen consume the flat streams) is the
  end goal that turns all of the above from "exercised by tests" into "the production compile".
