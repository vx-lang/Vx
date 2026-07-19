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
1. **Enum variant payloads never resolved.** `EnumDecl::resolve_names` was a no-op, so
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

## Entry 13 — C1.3a: real struct/enum layouts at the freeze (#199)

**Commit:** `3a19fa0`. `src/layout.rs`.

**Gap.** `build_frozen_registry` built every `TypeDefinition` with `size_bytes = 0 / align_bytes = 0`
— a stub. The flat HIR (C1.3) needs real sizes to size an `Alloca` and field offsets to lower struct
field access, so the non-scalar surface (`transfer`, tensor, struct) was blocked on layout not
existing (the #199 prerequisite).

**What we did.** New `LayoutComputer` resolves nested by-value nominals **by GID** (cross-module
correct — name resolution already ran) and computes, per nominal:

- struct size/align/field-offsets with C-like natural-alignment packing (fields in decl order, each
  at the next multiple of its align; struct align = max field align; size rounded up). Matches the
  AST codegen's non-packed `!llvm.struct` lowering.
- scalar sizes mirroring `SizeOfExpr` (i8/bool=1, f16/i16=2, i32/f32=4, i64/f64=8, i128=16);
  pointer-like fields (`*T`/`&T`/`Ref`) = 8/8; `Pinned`/`Verified` pass through; a payload-free enum
  = an i32 discriminant (4/4).

**Conservative by design.** Incomputable cases stay at the 0/0 stub rather than guessing: a generic
param field, a not-yet-modelled field type (tensor, closure, generic instance), a payload-carrying
enum (tagged-union layout must match codegen exactly — deferred), or a by-value cycle (`layout_of`
returns `None` on the back-edge, so it never loops; the registry's `toposort` still reports the cycle).
`TypeDefinition` gains `fields: Vec<FieldLayout>`.

**Tests.** 10 unit tests (scalar table, padding/alignment, nested struct, pointer field, C-like vs
payload enum, generic field, by-value cycle, pointer-breaks-cycle) + end-to-end
`frozen_registry_computes_real_layouts` (Pair=8/4, Wrap nesting Pair=12/4, enum=4/4). Full suite green
(321 lib + 67 integration).

**Next (C1.3b).** Model aggregate + tensor GIDs in the flat type stream and size `Alloca` from these
layouts; add field/index opcodes + non-scalar `Val`s in the lowerer; then `transfer`/tensor/slice
lowering (the rest of #199).

## Entry 14 — C1.3b: aggregate values in the flat HIR — struct params + field reads (#199)

**Commits:** `f55466d` (aggregate value type + sized `Alloca`), `a90828a` (`FieldLoad`).

**Gap.** The flat lowerer modelled only scalars — `Val`/`Binding` carried an `ElementType`, every
type-stream entry was a `scalar_gid` — so a struct parameter or field access couldn't lower. With
real layouts now at the freeze (Entry 13), the lowerer can carry aggregates.

**What we did.**

- `LoweredTy { Scalar(ElementType) | Aggregate(TypeId) }` threaded through `Val`/`Binding` and the
  emitters. `emit_typed` is the general emitter; `emit_value` stays the scalar convenience so scalar
  GIDs are byte-identical and existing call sites are untouched.
- The lowerer holds the frozen registry. A struct/enum parameter with a *computed* layout binds as an
  aggregate and forces the memory model (an aggregate must sit in an addressable slot); `emit_alloca`
  sizes the slot from `layout.size_bytes` (in the `Alloca`'s `imm`). Its GID flows through
  `Load`/`Alloca`/`Store`/`SlotLoad`/`Ret`. `lowered_ty` declines (atomic) for an aggregate still on
  the 0/0 stub — generic, tensor, or otherwise unmodelled field types.
- `FieldLayout` gains `ty: FieldTy { Scalar | Nominal | Opaque }` so a field read recovers its value
  type. New `Opcode::FieldLoad` (slot handle + byte offset → field value); `base.member` lowers to it
  for scalar fields (nested-aggregate/pointer fields declined until addressed sub-views land).

**Tests.** `struct_param_lowers_as_aggregate_slot` (Point param → memory slot, `Alloca.imm == 8`,
aggregate GID in the type stream), `unmodelled_aggregate_param_is_declined` (tensor-field struct),
`struct_field_read_lowers_to_field_load` (`p.y` → FieldLoad, imm 4), and
`field_read_offset_honours_alignment_padding` (`Rec{a:i8,b:i32}`.b at padded offset 4). Full suite
green (325 lib + 67 integration).

**Next (C1.3c).** Struct *construction* (`S { .. }` → `Alloca` + per-field `FieldStore`) — needs a
name→GID resolution for the `StructInit` expression (the expression-level analogue of #194). Then
tensor GIDs + index opcodes, and `transfer`.

## Entry 15 — C1.3c: struct construction + the StructInit GID annotation (#199)

**Commits:** `fe4b414` (sema annotation), `9f19d01` (`FieldStore` + construction lowering).

**Gap.** Struct *construction* (`let s = S { .. }`) needs the struct's GID to reach its registry
layout, but a `StructInit` expression carried only the struct *name* — unlike a parameter or field,
whose GID rides on a resolved type. (Design choice, taken with the user: annotate the expression in
the type checker, the cleanest of the three options.)

**What we did.**

- **Sema.** `StructInitExpr` gains `type_id: Option<TypeId>`, set by `check_structinit_expr` from a
  new `GlobalAstEnv::struct_gids` (the resolver's exact GID formula, so it matches the registry key).
  The expression's *returned* type keeps GID `None` — struct type-identity comparisons are unchanged,
  the GID is a pure side channel. A struct name defined in two modules is left unannotated (the
  name-keyed env can't disambiguate; a wrong GID is worse than none — the lowerer then declines).
- **Flatten.** New `Opcode::FieldStore`. `let x = S { .. }` constructs in place: `Alloca` the
  aggregate (sized), a `FieldStore` per field at its layout offset, and the local binds directly to
  that slot. Declines (atomic) when the GID is absent/uncomputed or a field is non-scalar. The
  flatten test harness now type-checks before lowering (so the annotation exists).

**Tests.** `structinit_is_annotated_with_struct_gid` (sema), `struct_construction_lowers_to_alloca_ and_field_stores` (`let p = Point{..}; return p.y` → 8-byte Alloca + two FieldStores @0/@4 + a
FieldLoad). Full suite green (327 lib + 67 integration).

**Where #199 stands.** Structs are now first-class in the flat HIR — params, construction, field
read, all layout-driven. Remaining: tensor GIDs + index opcodes (the tensor path), nested-aggregate /
pointer field access, and `transfer` lowering.

## Entry 16 — C1.3d: the tensor path — identity + indexing (#199)

**Commits:** `f0aa1c2` (tensor GIDs), `0ac1bf5` (`TensorIndex`).

**Gap.** Tensors had no identity in the flat type stream (`emit_type_gid`/`nominal_gid` fell through
for `Type::Tensor`; the lowerer declined any tensor), and no way to index them.

**What we did.**

- **Identity.** `flatten::tensor_gid(elem, shape)` / `tensor_gid_of(ty)`: a stable GID (module 0 =
  builtin) content-hashed from element + canonical shape — same source of truth as `scalar_gid`, so a
  tensor has one identity in a signature and a lowered body. `pipeline::{emit_type_gid, nominal_gid}`
  now emit it. Declines a generic element or a non-literal/name dim.
- **Values.** New `LoweredTy::Tensor { elem, shape }` (shape carried, not just the GID, so indexing
  can rank-reduce). A tensor is a reference (memref), so a tensor *parameter* binds as an SSA
  register — never `Alloca`'d.
- **Indexing.** New `Opcode::TensorIndex` (base + scalar index → rank-reduced result). `base[index]`
  drops the outermost dim: a remaining shape → a row/sub-view tensor, an empty one → the scalar
  element. `q[i][j]` recurses the nested `IndexAccess` (`[2,4] → [4] → f32`).

**Tests.** `tensor_param_binds_as_ssa_reg_with_tensor_gid`, `tensor_gid_distinguishes_element_and_ shape`, `tensor_signature_emits_the_tensor_gid` (signature==body), `tensor_full_index_yields_scalar_ element`, `tensor_partial_index_yields_row_view`. Full suite green (332 lib + 67 integration).

**Next.** Tensor elementwise ops + reductions (the slice-ops surface `dot`/`sum` → `vector.reduction`
in the flat HIR), tensor allocation/`transfer` lowering, and nested-aggregate/pointer field access.

## Entry 17 — C1.3e: slice reductions + elementwise (the FA payoff) (#199)

**Commits:** `51f7b6f` (`Reduce`), `6c77f29` (elementwise).

**Gap.** The tensor path had identity + indexing (Entry 16) but no way to *compute* on slices — the
FlashAttention `dot`/softmax surface.

**What we did.**

- **Reductions.** New `Opcode::Reduce`: a rank-1 slice → a scalar. `operand1` the slice (`operand2` a
  second slice for `dot`, else `Register(0)`), `imm` the kind (0 dot / 1 sum / 2 max / 3 min),
  `type_idx` the scalar element. `lower_expr` intercepts the `dot`/`sum`/`max`/`min` `FunctionCall`s;
  each operand must be a rank-1 tensor slice (`q` or an indexed row `q[i]`). Mirrors the AST codegen's
  `vector.reduction` (with a fused `mulf` for `dot`).
- **Elementwise.** No new opcode: an elementwise tensor op is an arith opcode (`Mul`/`Add`/…) whose
  *result type* is a tensor — the MLIR convention (`arith.mulf` on a vector). The `BinaryOp` arm
  already flowed tensors through; the fix is the result-type pick — when either operand is a tensor
  the result is the tensor type, so `s * a` (scalar on the left) broadcasts to a tensor.

**Tests.** dot/sum/max reductions, `dot(q[0], k[0])` on rank-2 tensors (index→reduce), elementwise
mul + scalar broadcast, and a capstone `flashattention_score_expression_composes`
(`dot(q[i], k[j]) * scale` → index → reduce → scalar multiply, end to end). Full suite green
(338 lib + 67 integration).

**Where #199 stands.** The flat HIR now models the whole non-scalar *value* surface — structs
(layout/params/construction/field access) and tensors (identity/indexing/reductions/elementwise),
enough to express the FA inner loop. Remaining: `transfer` and tensor *allocation/store* lowering (so
a produced tensor can be written back), nested-aggregate/pointer field access, and then C2 codegen
consuming these opcodes.

## Entry 18 — C1.3f: tensor allocation, store, and transfer (the write side) (#199)

**Commit:** `97626bf`.

**Gap.** Everything tensor so far was read-only (index/reduce/elementwise into a scalar or a value);
a *produced* tensor couldn't be allocated, written back, or moved. The user's constraint: the
receiving side of a store must have enough storage.

**What we did.**

- `Opcode::TensorAlloc` — `Tensor<T>([..])`; `imm` is the static byte size (elem size × Πdims, via
  `hir::memory::static_tensor_bytes`), so the buffer holds every element. A tensor local binds as an
  SSA register (a memref descriptor); a dynamic/symbolic shape declines.
- `Opcode::TensorStore` — `o[i] = <slice>` stores through the row/sub-view place (the assignment's
  left side lowers to a `TensorIndex`; scalar-element stores aren't modelled yet).
- `Opcode::Transfer` — `transfer(src, Memory::X)`; `imm` is the space's dispatch id, and the result
  keeps element + shape, so the destination is sized to hold the source.

**Tests.** alloc sizes the buffer (`Tensor<f32>([2,4])` → 32 bytes), a row store (alloc + index +
store), a transfer to `Memory::NPU_HBM` (dispatch id 100), and a write-path capstone
`flashattention_write_path_composes` (`let o = Tensor<f32>([2,4]); o[0] = v * (dot(q[0],k[0]) * scale); return o` → alloc + 3 index + reduce + 2 mul + store). Full suite green (342 lib + 67 integration).

**Where #199 stands.** The flat HIR now models the whole non-scalar surface end to end — read *and*
write — for structs and tensors: layouts, params, construction, field access; tensor identity,
indexing, reductions, elementwise, allocation, store, transfer. The FlashAttention inner loop lowers
in both directions. The remaining work is a different phase — **C2 codegen** consuming these opcodes
(so the flat path emits MLIR, not just a structurally-verified stream) — plus nested-aggregate/pointer
field access and scalar-element tensor stores.

## Entry 19 — C1 Calls: fixed-arity function calls (#198)

**Commit:** `08c8d3a`. The last C1 item.

**Gap.** Ordinary calls `f(a, b)` didn't lower: a call needs the callee's identity + result type, and
its N arguments don't fit a two-operand instruction.

**What we did.**

- **Callee resolution via the registry.** `ImmutableGlobalRegistry` gains `fn_sigs` (name → `FnSig { gid, ret_ty }`), minted in `build_frozen_registry` with the resolver's GID formula; a name defined
  in >1 module (distinct GIDs) is dropped (name-keyed, so ambiguous → decline, mirroring the
  struct-GID policy). The lowerer already holds the registry, so no name→AST walk.
- **N-ary args.** New `Opcode::Arg` marks each argument's value register; N `Arg`s immediately precede
  the `Call`, in order. This is the *ordinary-call* encoding for a fixed two-operand instruction —
  **not** language-level varargs (evaluated separately, #213).
- **`Call`.** `type_idx` = the callee's GID (C2 resolves name + result type from it), `imm` = arg
  count. Value-returning calls only for now — a void or unmodelled return declines.

**Tests.** `fixed_arity_call_lowers_to_args_and_call` (`add(3,4)` → 2 `Arg`s + `Call`, callee GID in
`type_idx`, count in `imm`), `call_to_unknown_fn_declines`. Full suite green (344 lib + 71 integration).

**C1 is now complete** (C1.1–C1.3 + Calls). Follow-ups: void/non-scalar-return calls, and the declined
edge cases in #212. Next is **C2** (#200) — the flat codegen consuming these opcodes.

## Entry 20 — C2.1: the differential harness + the attention corpus (#200)

**Commits:** `bfe5156` (harness), `9398c4d` (corpus). (Landed alongside Entry 19; grouped here.)

**What we did.**

- **Differential harness** (`tests/integration_test/flat_codegen_differential.rs`). For a function the
  flat HIR lowers today (straight-line scalar arithmetic), the flat emitter's MLIR is run through the
  *same* production `lower_to_llvm` + JIT as the AST path, and the two **process exit codes** must
  match each other and the expected value. `ast_exit_code` / `flat_exit_code` / `assert_parity`.
  Cases: `3+4*5`=23, `100-84/2`=58, `(2+3)*(10-3)`=35; plus a control-flow case where the flat path
  *declines* (outside the C2.0 subset) so the AST path stays the sole oracle — no false parity. This
  is the first time the flat emitter runs through the real lowering + JIT, not just an arith-level
  parse+verify. Parity here is the criterion that lets `vxc` eventually flip (C3).
- **Attention corpus** (`tests/backend/pass/{full_softmax,multi_query,grouped_query,linear,sparse_ local}_attention.vx`). Five hand-checkable, JIT-verified attention variants — a real-workload target
  for the differential harness as C2 grows, and standalone backend coverage now (they run through the
  AST oracle, like `attention_reference.vx`). Each has `EXPECT` matched to JIT output and hand-derived
  math in the header.

## Entry 21 — C2.2: control flow in the flat emitter (brick 1, #200)

**Commit:** _this session_. The first C2 brick after the harness — the flat emitter now lowers
intra-function control flow, so an `if`/`for`/`loop` `main` JIT-matches the AST oracle.

**Gap.** `flat.rs::emit_function_mlir` handled only straight-line scalar arithmetic (`Load`/`Const`/
`Add`/`Sub`/`Mul`/`Div`/`Ret`); a control-flow stream (`BlockStart`/`Br`/`CondBr`/`Alloca`/`Store`/
`SlotLoad`, plus `Cmp` for the condition) returned `None` and stayed on the AST path.

**What we did.**

- **Per-register element types.** Added a parallel `etypes[reg]` recovered as the stream is walked —
  the flat-driven stand-in for reading an operand's type off the AST. Needed because `Cmp` (result
  `bool`) and `Store` (effect sentinel `type_idx`) carry no usable type of their own: a compare reads
  its operand's tracked type to pick `cmpi`/`cmpf` + predicate; a store reads its *slot*'s (the
  `Alloca`'s recorded) element type to print `memref<T>`.
- **Control-flow opcodes → `cf` + rank-0 `memref`.** `BlockStart b` opens block `b` (`b=0` is the
  func's implicit entry, no label; else `^bbN:`); `Br`/`CondBr` → `cf.br`/`cf.cond_br` (targets
  unpacked from `imm`'s `then | else<<32`); `Alloca` → `%s = memref.alloca() : memref<T>`, `Store`/
  `SlotLoad` → `memref.store`/`memref.load %s[]` — matching the AST codegen's scalar locals (rank-0
  memref) so the shared `lower_to_llvm` + JIT behave identically. `Cmp` → `arith.cmpi/​cmpf`.
  Terminators are emitted **inline** (not deferred), so each block closes correctly; an unterminated
  final block falls through to `return` (void) or declines (scalar — unreachable/ill-typed, no value).
- **Keep-green preserved.** Anything still outside the subset (calls, cast, the non-scalar surface)
  returns `None`. A scalar `as` cast is the new decline witness (was: the control-flow decline).

**Tests.** `flat.rs`: `emits_verifiable_if_else`, `emits_verifiable_for_loop` (emit + melior verify),
`declines_out_of_subset_scalar_op`. Differential harness: four **JIT-parity** cases through the real
`lower_to_llvm` + JIT — `if` no-else (=105), `if/else` (=2), `for` accumulator 0..5 (=10), `loop` +
`break` (=3) — each `flat == ast == expected`. Full suite green (346 lib + 75 integration).

**Next (brick 2).** Calls — a *module-level* flat emitter (all functions, not one) + callee GID→name
(reverse of registry `fn_sigs`) + call-signature types.

## Entry 22 — C2.3: calls in the flat emitter (brick 2, #200)

**Commit:** _this session_. The second C2 brick — the flat emitter now lowers fixed-arity scalar
calls, and gained a **module-level** entry point so a callee's `func.func` is present for the call.

**Gap.** `emit_function_mlir` did one function and declined `Arg`/`Call`. A call needs (a) the whole
program in one module (the callee `func.func`), (b) the callee's *name* (the `Call` carries only its
GID), and (c) the arg + return types for the call signature.

**What we did.**

- **Module-level emitter.** New `emit_module_mlir(funcs, registry)` emits every function's `func.func`
  and concatenates, declining the whole module if *any* function is outside the subset (keep-green at
  the module level). The differential harness's flat path now lowers **all** functions (each into its
  own worker) through the frozen registry and emits one module.
- **Callee resolution (GID→name).** `build_callee_map(registry)` inverts the registry's name-keyed
  `fn_sigs` into `GID → Callee { name, ret }` (the reverse of C1's minting). A `Call`'s `type_idx`
  resolves through it to the `func.call @name` symbol and the result type.
- **`Arg`/`Call` emission.** `Arg` records its value register into a `pending_args` stack; `Call`
  consumes its `imm` trailing entries (a nested inner call sits between its own `Arg`s and the outer
  ones, so each call's args are exactly the tail — no bookkeeping beyond a `split_off`). Emits
  `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`, arg types from the per-register `etypes[]`, the
  return type from the callee's `fn_sig`. Scalar-returning calls only; a void/non-scalar return
  declines (still #198 / brick 3).
- **Plumbing.** `build_frozen_registry` is now `pub` (the external differential harness — and later
  C3 — builds the registry to resolve callees).

**Tests.** `flat.rs`: `emits_verifiable_scalar_call` (module emit + melior verify, asserts the
`func.call` + signature). Differential harness: four **JIT-parity** cases — a scalar helper (`add`=42),
a call inside an expression (`mul(6,7)-2`=40), **nested** calls (`dbl(inc(9))`=20), and a call from
inside an `if` (bricks 1+2 together, `sq(5)`=25) — each `flat == ast == expected`. Full suite green
(347 lib + 79 integration).

**Filed.** #214 (flat emitter scalar unary ops: `as` cast, `-x`, `!x` — the remaining declined scalar
opcodes; the cast-decline harness test guards it).

**Next (brick 3).** The non-scalar surface — `FieldLoad`/`FieldStore`/`TensorIndex`/`Reduce`/
`TensorAlloc`/`TensorStore`/`Transfer` — so the attention corpus runs through the flat path. Needs
`etypes[]` extended to tensor/aggregate recovery (currently scalar-only).

## Entry 23 — C2.4a: structs in the flat emitter (non-scalar surface, part 1, #200)

**Commit:** _this session_. The start of the non-scalar surface — the flat emitter now lowers
all-scalar-field struct construction + field access, so a `main` that builds a struct locally and
reads its fields JIT-matches the AST oracle.

**Gap.** The emitter declined every aggregate opcode (`Alloca` of a struct, `FieldLoad`, `FieldStore`)
and tracked only scalar register types, so structs stayed on the AST path.

**What we did.**

- **Aggregate layout recovery.** New `build_agg_map(registry)` → `GID → AggLayout { struct_ty, offsets }` from the frozen registry's nominal layouts, for structs whose fields are all scalar (the
  `!llvm.struct<(...)>` type + each field's byte offset). Bundled with the callee map into a new
  `EmitCtx` (one resolution-context param, so the emitter signature stays stable as more non-scalar
  families land).
- **Struct opcodes → the `llvm` dialect.** An aggregate `Alloca` → `llvm.alloca` of the `!llvm.struct`
  type (its pointer tracked in a new `agg_of[reg]`, the aggregate analogue of `etypes`). `FieldStore`/
  `FieldLoad` → `llvm.getelementptr %slot[0, idx]` (the field index recovered by matching the
  instruction's byte offset against the layout) + `llvm.store`/`llvm.load`. Field/value scalar types
  come from the instruction's own `type_idx` (load) or the stored register's `etypes` (store). Mixes
  cleanly with `arith`/`memref`/`cf`/`func`, exactly as the AST path does.
- **Harnesses type-check first.** Struct construction needs the type checker's `StructInit` GID
  annotation, so the differential harness + the unit-test module helper now run a scratch type-check
  pass before lowering (mirroring `flatten`'s `lower_with_registry`).

**Tests.** `flat.rs`: `emits_verifiable_struct_construct_and_field_read` (module emit + melior verify).
Differential harness: two **JIT-parity** cases — a struct field sum (`Point{3,4}` → `p.x+p.y` = 7) and
struct fields feeding an `if` (structs + control flow, = 7) — each `flat == ast == expected`. Full
suite green (348 lib + 81 integration).

**Filed.** #215 (struct params, returns, struct-to-struct copy — still declined; the AST path stays
the oracle). Nested-aggregate/pointer fields remain #212.

**Next (part 2).** Tensors — read first (`TensorIndex`/`Reduce`/elementwise), then write
(`TensorAlloc`/`TensorStore`/`Transfer`), matching the AST's memref/vector lowering, so the attention
corpus runs through the flat path. Needs the per-register type tracking extended to tensor types.

## Entry 24 — Flat HIR: scalar-element tensor stores (#212, unblocks tensor codegen)

**Commit:** _this session_. A flat-HIR (lowering) change, taken as a prerequisite for tensor codegen:
without it there is no self-contained flat-lowerable tensor program (tensor data can't be
initialized), so tensor codegen couldn't be JIT-differential-tested the way scalars/calls/structs
were.

**Gap.** `q[i][j] = <scalar>` didn't lower — the assignment path only modelled row/sub-view stores
(`o[i] = <slice>`) and declined a scalar-element place. So a program couldn't fill a tensor with
known values, which is exactly what a differential `main` needs (the JIT runs a param-less `main` and
reads its exit code; there's no way to pass tensor buffers in).

**What we did.**

- **`lower_place`** (an lvalue lowering for tensor stores). The outer indices produce sub-view tensors
  as a read does, but the *final scalar* index yields an element **place** — a `TensorIndex` with
  `imm = 1` — so codegen addresses the element and stores into it rather than loading its value. A
  still-nonempty shape yields a row/sub-view place (`imm = 0`, identical to the existing read/row-store
  form). The `Assign` path now uses `lower_place` and lets `TensorStore` carry either a scalar-element
  or a row place; the store kind is recovered from the place type in codegen.
- The two-operand instruction shape is preserved: each index is one `TensorIndex` (base, index), and
  the `imm = 0`/`1` flag distinguishes value vs. place — no new opcode. `verify_hir_stream` already
  covers it (both operands dominated; `imm` isn't structural).

**Tests.** `scalar_element_store_into_rank1_marks_a_place` (`q[0] = 1.0` → one place index + one
store), `scalar_element_store_rank2_indexes_row_then_element_place` (`q[0][0] = 1.0` → row value index

- element place). Existing row-store / FA-write-path tests unchanged (they take the `imm = 0` path).
  Full suite green (350 lib + 81 integration).

**Scope.** The scalar-element-store half of #212; the nested-aggregate/pointer field-access half
remains open. Codegen for this (the `imm = 1` place + scalar `TensorStore`) lands with the tensor
emitter next.

## Entry 25 — C2.4b: the tensor-type side table + first tensor codegen (#200)

**Commit:** _this session_. The tensor GID isn't invertible (it's a content hash) and tensor types are
structural (never in the nominal registry), so the emitter couldn't reconstruct a tensor's memref type
from the stream — most acutely for `TensorAlloc`. This adds the recovery mechanism and the first
tensor codegen on top of it. (Design rationale: the C2 plan's "tensor-type recovery" note.)

**What we did.**

- **Tensor-type side table.** The lowerer records `GID → (element, shape)` for every tensor-typed
  value it emits (in `emit_typed`), onto `LocalWorkerState::local_tensor_types`; `commit` transfers it
  (keyed by the content-hash GID, so no rebasing). The emitter merges each function's table into
  `EmitCtx.tensors` — the tensor analogue of the registry's struct `layouts`, but sourced from
  lowering rather than the registry.
- **First tensor codegen (a self-contained i32 program).** `TensorAlloc` → `memref.alloc()` of a
  static `memref<NxT>` (shape recovered by GID from the side table; parity is the JIT result, so a
  static memref stands in for the AST's dynamic one). Scalar-element `TensorIndex` → `arith.index_cast`
  the index to `index` + `memref.load` (read, `imm = 0`) or a recorded element **place** (`imm = 1`);
  scalar-element `TensorStore` → `memref.store` into the place. Tensor registers are tracked in
  `mem_of` (memref type) and `place_of` (a pending element address).
- Reductions stay f32-only in the AST oracle, so the first differential case reads a scalar element
  (`return q[2]`) rather than summing — an i32 exit code, no cast.

**Tests.** `flat.rs`: `emits_verifiable_tensor_alloc_store_read` (module emit + melior verify).
Differential harness: `flat_matches_ast_tensor_element_read` — allocate `Tensor<i32>([4])`, fill it by
scalar-element stores, read one element back (= 7) — `flat == ast == expected` through the real JIT.
This is the first tensor program lowered flat-vs-AST to parity. Full suite green (351 lib + 82
integration).

**Next.** The rest of the tensor surface: `TensorIndex` sub-views (rows) → `memref.reinterpret_cast`,
`Reduce` → `vector.load`/`vector.reduction`, tensor elementwise → `vector` ops, row `TensorStore` →
`vector.store`, `Transfer`, and tensor params (memref in the signature). Then the attention corpus can
be JIT-compared through the flat path.

## Entry 26 — C2.4c: tensor row sub-views in the flat emitter (#200)

**Commit:** _this session_. Builds on Entry 25 — the flat emitter now rank-reduces a tensor to a row,
so rank-2 element access (`q[i][j]`) works.

**What we did.** A `TensorIndex` with a tensor (sub-view) result → `memref.reinterpret_cast` of the
contiguous base to the row at flat offset `index * product(row dims)`, with row-major strides —
`memref<3xi32, strided<[1], offset: ?>>`, matching the AST's S1 lowering. The row's memref type is
tracked in `mem_of`, so the following scalar index reads/stores through it (`memref.load`/`store` on a
strided memref). A sub-view of an already-strided row (a deeper chain, rank ≥ 3) is deferred.

**Tests.** `flat.rs`: `emits_verifiable_tensor_row_subview_read`. Differential harness:
`flat_matches_ast_tensor_row_element_read` — fill a `Tensor<i32>([2,3])` by element and read `q[1][2]`
(= 6) — `flat == ast == expected`. Full suite green (353 lib + 83 integration).

**Next.** `Reduce` (`vector.load`/`vector.reduction`), tensor elementwise (`vector` ops), row
`TensorStore` (`vector.store`), `Transfer`, tensor params.

## Entry 27 — C2.4d: tensor params + tensor call arguments (#200)

**Commit:** _this session_. The flat emitter now takes a tensor as a function parameter and passes one
as a call argument, so a tensor built in `main` can be handed to a helper.

**What we did.** The signature builder recovers a tensor param's `memref` type by GID from the side
table (`tensor_gid_of(ty)` → `ctx.tensors`), so `%arg0: memref<4xi32>`. The param's `Load` records the
memref type in `mem_of` (so later index/store address it). The `Call` handler prints a tensor arg's
type from `mem_of` (a scalar arg still from `etypes`). The callee's memref param type and the caller's
arg type resolve to the same static `memref` — the tensor GID is a content hash, so both sides agree.

**Tests.** `flat.rs`: `emits_verifiable_tensor_param_and_call`. Differential harness:
`flat_matches_ast_tensor_param_passed_to_helper` — `main` fills a `Tensor<i32>([4])` and calls
`get(q, 2)` returning `q[2]` (= 7) — `flat == ast == expected`. Full suite green (354 lib + 84
integration).

**Next.** `Reduce` (`vector.load`/`vector.reduction`) — JIT-testable by feeding the f32 result into a
comparison (`if sum(q) > 9.0 …`) so the exit code is an i32 — then tensor elementwise, row
`TensorStore`, `Transfer`.

## Entry 28 — C2.4e: tensor slice reductions in the flat emitter (#200)

**Commit:** _this session_. `sum`/`dot`/`max`/`min` over a rank-1 float slice now lower, so a reduction
result can drive an i32 exit code (via a compare).

**What we did.** `Reduce` → `arith.constant 0 : index` + `vector.load %slice[c0] : {memref}, vector<Nxf32>`
per operand (`N` parsed from the slice's memref type), `arith.mulf` for `dot`, then
`vector.reduction <add|maximumf|minimumf>, %v : vector<Nxf32> into f32` — matching the AST's S2 lowering
(float only; a non-float reduction declines, as the AST also lowers only f32).

**Two constraints hit.**

- **`if` as a value.** A trailing `if … { return 1 } else { return 0 }` parses as
  `return (if-expression)`, and the flat HIR lowers `if` only as a *statement* — filed #216. The tests
  use the statement form (`let mut r = 0; if c { r = 1; } return r;`).
- **The AST reduces only static-sized slices.** `lower_slice_reduction` parses the slice length from
  the memref type, so it can't reduce a whole dynamically-allocated tensor (`memref<?xf32>`) — only a
  row sub-view (`memref<4xf32, strided<…>>`). The differential case therefore reduces `q[0]` (a row),
  matching how the corpus reduces.

**Tests.** `flat.rs`: `emits_verifiable_tensor_sum_reduction`. Differential harness:
`flat_matches_ast_tensor_row_sum_reduction` — `sum(q[0])` over a filled row, `> 9` sets `r = 1` — `flat == ast == expected` through the real JIT (reinterpret_cast + vector.load + vector.reduction). Full suite
green (355 lib + 85 integration).

**Next.** Tensor elementwise (`vector.load` + `arith.*` + `vector.store`), row `TensorStore`
(`vector.store`), `Transfer`.

## Status (2026-07-18) — C0 + C1 done, C2 in progress

**Done.**

- **C0** — GID word-2 codec (#193), cross-module resolution (#194), stable FNV hash + collision guard
  (#195). *All closed.*
- **C1** (#198) — the flat HIR lowers the whole language surface a function body uses:
  - **C1.1/C1.2** scalar core, value ops, control flow (`if`/`loop`/`for`, break/continue), the
    two-tier SSA/memory-slot model.
  - **C1.3** (#199, closed) the non-scalar value surface — struct layouts/params/construction/field
    read (`FieldLoad`/`FieldStore`), tensor identity/indexing/reductions/elementwise/alloc/store/
    transfer (`TensorIndex`/`Reduce`/`TensorAlloc`/`TensorStore`/`Transfer`). Read **and** write; the
    FlashAttention inner loop lowers in both directions.
  - **Calls** — fixed-arity, value-returning (`Arg` + `Call`, callee via registry `fn_sigs`).
  - Verified structurally by `verify_hir_stream`; the `flatten` unit tests + `lower_with_registry`
    exercise every opcode family.
- **C2 (start + bricks 1–3a + first tensor)** — the differential harness proves flat==AST (Entry 20);
  the flat emitter now lowers intra-function **control flow** (Entry 21), fixed-arity **scalar calls**
  via a module-level emitter (Entry 22), **all-scalar-field structs** (Entry 23), and a first **tensor**
  program — alloc + scalar-element store/read via the tensor-type side table (Entry 25) — all to JIT
  parity.

**Open.**

- **C2** (#200) — the flat emitter (`src/codegen/flat.rs`) now covers scalar arithmetic + control
  flow + calls + structs + tensor alloc/element access; the rest of the **tensor surface** (sub-views,
  reductions, elementwise, row store, transfer, tensor params) remains, plus scalar unary ops (#214)
  and struct params/returns (#215). See the C2 roadmap:
  [`implementation_plans/c2_flat_codegen.md`](./implementation_plans/c2_flat_codegen.md).
- **C1 follow-ups** — void/non-scalar-return calls; the declined flat-HIR edge cases (#212); varargs
  evaluation (#213).
- **C3** (#201) — flip `vxc` from the sequential AST driver to `compile_pipeline` + flat codegen, once
  C2 reaches parity across the corpus.
- **Other** — slow-path variance (`evaluate_slow_path_variance` in `borrow.rs`) is still a stub; the
  parallel-test struct/tensor coverage is #211.

## Next: C2 (the flat codegen)

The flat emitter grows opcode-family by opcode-family, each verified by extending the differential
harness, in this order (details + MLIR mappings in `implementation_plans/c2_flat_codegen.md`):

1. ~~**Control flow** — `BlockStart`/`Br`/`CondBr`/`Alloca`/`Store`/`SlotLoad` (+`Cmp`) → `cf` +
   `memref.alloca`.~~ **Done** (Entry 21): `if`/`for`/`loop` `main`s JIT-match the AST oracle.
1. ~~**Calls** — a *module-level* flat emitter (all functions) + callee GID→name (reverse of registry
   `fn_sigs`) + call-signature types.~~ **Done** (Entry 22): fixed-arity scalar calls JIT-match.
1. **Non-scalar** ← *next* — `FieldLoad`/`FieldStore`/`TensorIndex`/`Reduce`/`TensorAlloc`/
   `TensorStore`/`Transfer`, matching the AST codegen's memref/vector lowering so the attention corpus
   runs through the flat path and differentially checks. Needs `etypes[]` extended to tensor/aggregate
   recovery.

Then **C3**: `--flat-codegen` flag → flip `vxc`.
