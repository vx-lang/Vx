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

## Entry 29 — C2.4f: tensor elementwise + row store (the write side) (#200)

**Commit:** _this session_. The FlashAttention write-path shape now lowers: an elementwise op over a
row, stored back into a row.

**What we did.**

- **Elementwise** (`Add`/`Sub`/`Mul`/`Div` with a tensor-GID result): each operand is coerced to a
  `vector<Nxf32>` (`coerce_vector`, matching the AST's `to_vector` — a vector passes through, a rank-1
  memref is `vector.load`ed, a scalar is `vector.broadcast`ed), then `arith.{addf,subf,mulf,divf}`
  yields a vector tracked in `vec_of`. Float only. The arith arm now dispatches on scalar-GID vs
  tensor-GID result.
- **Row `TensorStore`** (an `imm = 0` `TensorIndex` place, i.e. a row memref) writes an elementwise
  vector back with `vector.store %v, %row[c0]` — the counterpart of the scalar-element `memref.store`.

**Tests.** `flat.rs`: `emits_verifiable_tensor_elementwise_and_row_store`. Differential harness:
`flat_matches_ast_tensor_elementwise_row_store` — `o[0] = q[0] * 2.0` then read `o[0][1]` (= 4 > 3.5 →
`r = 1`) — `flat == ast == expected` through the real JIT (vector.load/broadcast + arith.mulf +
vector.store). Full suite green (356 lib + 86 integration). The tensor read **and** write surface now
lowers to parity; only `Transfer` (a memory-space move) remains.

## Entry 30 — C2 capstone: the FlashAttention write path composes flat-vs-AST (#200)

**Commit:** _this session_. A single differential case proving the tensor pieces compose end to end.

`flat_matches_ast_flashattention_write_path`: `o[0] = v[0] * (dot(q[0], k[0]) * scale)` — a
reduction (`dot`) → scalar multiply → elementwise scalar-broadcast → row `vector.store`, over rows
built by scalar-element stores — then read `o[0][0]` and compare for an i32 exit. `dot([1,2,3,4], [1,1,1,1]) = 10`; `* 0.5 = 5`; `v[0] * 5 = [10,…]`; `o[0][0] = 10 > 9 → 1`. `flat == ast == expected`
through the real JIT. The FlashAttention inner write path (the workload the flat HIR was designed
around) now lowers through the flat codegen to parity with the AST oracle. Full suite green (356 lib +
87 integration).

## Entry 31 — C2.4g: `vx.transfer` — the tensor opcode surface is complete (#200)

**Commit:** _this session_. The last tensor opcode. Every tensor op the flat HIR emits now lowers to
JIT parity.

**What we did.** `Transfer` → `vx.transfer` (generic form) carrying `target_topology` (the memory-space
dispatch id, straight from the instruction's `imm`); the vx→standard lowering (`VxLowering.cpp`) turns
it into an alloc + `memref.copy` + scope-end dealloc. The `Vx_TransferOp` definition requires only
`src` + `target_topology`, and the lowering reads only those — so the `space`/`granule`/… attributes
the AST adds are discardable scheduling metadata the flat HIR doesn't need. The result is a plain
`memref<NxT>` (shape from the side table). The unit-test context now registers the vx dialect.

**Tests.** `flat.rs`: `emits_verifiable_tensor_transfer`. Differential harness:
`flat_matches_ast_tensor_transfer` — `let o = transfer(q, Memory::NPU_HBM); … o[0][2] > 2.5 → r = 1` —
`flat == ast == expected` through the real JIT. Full suite green (357 lib + 88 integration).

**Tensor surface complete.** alloc, scalar element read/store, row sub-views, params + call args,
reductions (`sum`/`dot`/`max`/`min`), elementwise, row store, and transfer — all at JIT parity, with
the FlashAttention write path composing end to end (Entry 30). The differential harness now has **21**
parity cases across the whole subset.

**Remaining for C2/C3.** Deeper index chains (rank ≥ 3); the scalar unary ops (#214) and struct
params/returns (#215); the `if`-as-value gap (#216); the full attention **corpus** as an end-to-end
differential target (needs the harness to compare `print` output, not just exit codes). Then **C3**
(#201): the `--flat-codegen` flag to run this subset through `vxc`.

## Entry 32 — C2.5: `print` + stdout-output differential (toward the corpus) (#200)

**Commit:** _this session_. The differential harness could only compare exit codes; the attention
corpus verifies via `print` output. This adds `print` to the flat path and a stdout-comparison mode.

**What we did.**

- **Flat HIR `print`.** New `Opcode::Print` (arg reg + type in `type_idx`); `flatten` lowers `print(x)`
  as a statement-level effect.
- **Emitter.** A tensor `Print` → `memref.cast` to an unranked memref + `func.call @printMemref{F32,…}`;
  a scalar → `func.call @print_{f32,…}`. `emit_module_mlir` prepends `private` declarations for the
  runtime helpers a module calls (the JIT links their implementations, exactly as the AST path does).
- **Harness.** `flat_llvm`/`ast_llvm` (the lowered module text) split out of the exit-code helpers;
  `run_output` runs the JIT and returns **stdout**, `normalize` strips the non-deterministic
  `printMemref` base pointer (`0x…`), and `assert_output_parity` compares the flat vs AST print output.

**Tests.** `flat.rs`: `emits_verifiable_tensor_print`. Differential harness:
`flat_matches_ast_tensor_print_output` — print a filled `Tensor<f32>([2,2])`, the `printMemrefF32` dump
(shape/strides/data) matches flat-vs-AST after normalization. Full suite green (358 lib + 89
integration).

**Next.** Run the actual attention corpus (`tests/backend/pass/*_attention.vx`) through
`assert_output_parity`, iterating on any surface the flat path still declines.

## Entry 33 — C2.6: real corpus programs run through the flat path (`+=`, #200)

**Commit:** _this session_. Two real backend-corpus programs now lower end to end through the flat
codegen and match the AST oracle's printed output.

**What we did.**

- **`slice_reductions.vx`** passes as-is — tensor alloc + scalar-element stores + typed row bindings +
  `dot`/`sum`/`max`/`min` + a `for`-loop scalar oracle + scalar-element stores of the results +
  `print(o)`.
- **`linear_attention.vx`** (an attention corpus program, no softmax) needed one missing construct:
  **compound assignment** `+=`. Added `Statement::CompoundAssign` to `flatten` — `lhs op= rhs` desugars
  to read the place, combine, and store back (scalar identifier or tensor place). With that it passes:
  nested `for` loops, variable-index scalar stores (`ss[a][b] = …`), `+=`, `num / den`, and `print(o)`.

**Tests.** `flat.rs`/`flatten`: `compound_assign_desugars_to_op_and_store`. Differential harness:
`flat_matches_ast_corpus_slice_reductions`, `flat_matches_ast_corpus_linear_attention` (read the real
`.vx` files, `assert_output_parity`). Full suite green (359 lib + 91 integration).

**Remaining corpus.** The softmax attention files (`full_softmax`/`multi_query`/`grouped_query`/
`sparse_local`) use `exp` — a math intrinsic the flat HIR doesn't lower yet. That's the next corpus
step.

## Entry 34 — stdlib decoupling Step 2: `ModuleInterface` + in-situ dual-run gate (#219)

**Commit:** _this session_. The registry-backed import oracle (`docs/discussions/implementation_plans/ stdlib_decoupling_protocol.md` §4) gets a named query surface, and the type checker's method resolution
is now checked against it at every real resolution site — the keep-green gate before we retire the
borrowed-AST env for imported symbols.

**What we did.**

- **`ModuleInterface` trait** (`registry.rs`), implemented for `ImmutableGlobalRegistry`:
  `resolve_type` (`module_indices`), `layout_of` (`layouts`), `resolve_fn` (`fn_sigs`), `resolve_method`
  (`methods`, from Step 1/#218). This is the one surface the frontend consults for anything defined
  *outside the current module* — identical whether the target was just compiled (in-memory registry) or
  loaded from a cached artifact. `resolve_trait_impl`/`body_of` are intentionally deferred (they need a
  `trait_impls` table + GID-indexed HIR store — #220/#221) rather than stubbed to always-`None`.
- **In-situ dual-run gate** in `check_methodcall_expr`: after the AST `impls`-walk resolves a method on
  a *concrete* receiver via a *non-generic* impl, a `debug_assert!` requires the registry-backed
  `&dyn ModuleInterface` to resolve the same `(receiver GID, method)`. This proves the registry is a
  sufficient method oracle wherever a frozen registry is actually in use.

**What the gate caught.** Running it across the backend corpus immediately surfaced that the legacy
AST-only harness (`compile_test.rs::run_backend_test`, and the sequential driver) type-checks against
an **empty** registry (`GlobalSession::new`, no `build_frozen_registry`, no name-resolution phase) —
so every stdlib math method (`f32.exp`, `f64.sqrt`, `i32.expect_eq`, …) tripped the assert. That is the
gate working, not a bug in the table: those methods *are* concrete `impl`s the registry captures when a
registry is built. The fix is a precondition — the gate only runs when `registry.methods` is non-empty,
i.e. when this compilation actually froze a registry. Retrofitting the legacy harness to freeze one is
out of scope (invasive to 89 passing tests; needs a name-resolution pass first) and is exactly what
later convergence steps do.

**Tests.** `pipeline::gid_stream_tests`: `module_interface_serves_registry_backed_resolution` (drives
all four queries through `&dyn ModuleInterface`) and `type_checker_method_resolution_agrees_with_registry`
(freezes a registry over a concrete `impl Math for f32 { fn exp }` and type-checks a `x.exp()` caller
against it — the in-situ gate fires and holds). Full suite green (361 lib + 91 integration).

**Remaining for #219 (the flip).** Point imported-name/type resolution at `ModuleInterface` and stop
stashing borrowed AST for imported modules. Blocked in part on cross-boundary generic bodies (imported
generic fns still monomorphize from AST) — that wants `body_of` over flat HIR (#220/#221). The gate now
guards each increment.

## Entry 35 — stdlib decoupling Step 2, first flip: `StructInit` GID via the registry (#219)

**Commit:** _this session_. First piece of state actually *retired* from the borrowed-AST env and
served by `ModuleInterface` instead: the `StructInit` GID.

**What we did.**

- **Deleted `GlobalAstEnv::struct_gids`** — the bare-name→GID side map (and its cross-module ambiguity
  bookkeeping) that the env kept purely to annotate `StructInit` expressions. It re-minted the *exact*
  GID formula the frozen registry already computes, so it was redundant state the registry subsumes.
- **`resolve_unique_nominal(name)`** on the registry + `ModuleInterface`: resolve a bare nominal name
  to its unique GID, `None` when two modules define it with distinct GIDs (same "decline rather than
  guess" policy the old map used). Scans `module_indices`; not a hot path.
- **Routed the one consumer** (`check_structinit`, `expr.rs`) through `&dyn ModuleInterface`. The GID is
  read only by the flat-HIR lowerer (`flatten.rs`: `si.type_id?` → registry layout), which always has a
  real registry; the AST codegen never reads it, so the empty-registry driver/legacy paths (now
  annotating `None`) are unaffected.

**Why this one first.** It is the only imported-symbol resolution that is *fully* serviceable by the
registry today with **no** dependency on later steps: pure identity, single consumer, graceful `None`.
Imported **fn/method** AST can't be dropped yet — the driver re-checks and re-emits imported non-generic
bodies for codegen (`driver.rs`), which is exactly what flat-HIR body linking (#220/#221) unblocks. And
`FnSig` still lacks parameter types, and `FieldTy::Opaque` loses tensor/pointer field types — both
needed before imported fn/struct *uses* can be type-checked off the registry.

**Tests.** `resolve_unique_nominal_declines_cross_module_ambiguity` (two modules define `Point` →
`None`; a unique struct resolves); `module_interface_serves_registry_backed_resolution` extended;
`structinit_is_annotated_with_struct_gid` rebuilt to freeze a real registry and take the registry lookup
as its oracle. The flat struct differential corpus (`flat_matches_ast_struct_*`) exercises the whole
path — registry-resolved GID → flat lowering → JIT parity with the AST oracle. Full suite green
(361 lib + 91 integration).

## Entry 36 — stdlib decoupling Step 3 (stage 1): the `.vxlib` interface codec (#220)

**Commit:** _this session_. The serialization substrate for a precompiled module interface — the
foundation for loading the stdlib from an artifact instead of re-parsing it every compile.

**What we did.**

- **Hand-rolled binary codec** in `metadata.rs` (no serde in the tree — only `bytemuck` for the POD GID
  arrays): a little-endian `Writer`/`Reader` with bounds-checked reads, `VXLB` magic + an FNV format
  stamp (`hash.rs`) so a stale artifact is *detected*, not misread. `serialize_registry_interface` /
  `deserialize_registry_interface` cover the **closed** part of the frozen registry — `module_indices`
  (identity) and `layouts` (structural layout: `TypeDefinition` / `FieldLayout` / `FieldTy` /
  `ElementType`), i.e. the `resolve_type` / `layout_of` surface. Keys are emitted sorted, so the same
  registry yields byte-identical output.
- **Filled the `interface_data` slot.** Renamed `VxMetadata::ast_data` → `interface_data` (the design's
  reserved "write the interface here" slot) and added `save_with_interface` — the container now carries
  dictionary + interface. A round-trip *through a file* rebuilds a queryable `ImmutableGlobalRegistry`.

**Why only identity + layouts this stage.** `FnSig.ret_ty` is a `syntax::Type`, and `Type::Tensor`
carries `Vec<Expr>` dimension trees (and `Type::Const` a `Box<Expr>`) — serializing it drags in the
whole expression AST, exactly the explosion the decoupling avoids. That `Type` encoder (for
`fn_sigs`/`methods`) and the per-function flat HIR **body** store (the deferred `body_of`) are the next
stages of #220; each is independently round-trip-tested.

**Tests.** `registry_interface_round_trips_through_a_file` (serialize → file → load → deserialize;
`resolve_type`/`layout_of` and every field's offset/size/type match the original; serialization is
stable) and `deserialize_rejects_corrupt_or_stale_buffers` (bad magic / bumped format stamp / truncated
buffer all error, none panic). Full suite green (364 lib + 91 integration).

## Entry 37 — stdlib decoupling Step 3 (stage 2): signatures — the `Type` codec (#220)

**Commit:** _this session_. The interface now carries `fn_sigs` + `methods`, so `resolve_fn` /
`resolve_method` survive the artifact round trip — the signature half of the import oracle.

**What we did.**

- **Recursive `syntax::Type` codec** (`metadata.rs`): all the *closed* variants round-trip faithfully
  (scalar, struct/enum/generic nominal + optional GID, ref/pointer/borrow with memory space,
  pinned/verified, generic-instance, function/closure, simd, matrix, unknown), plus `MemorySpace` and
  the data-free `Topology` variants.
- **fn_sigs / methods sections**, keyed as in the registry (fn name; `(receiver GID, method name)`),
  each carrying the callee GID + encoded `ret_ty`. Format bumped to `v2`.
- **Fail-closed on the `Expr` surface.** A `ret_ty` reaching a symbolic tensor dimension, a `Const`
  expression, or a topology carrying a count is *not* misencoded — `write_type` returns `Err` and that
  one signature is **skipped** (so `resolve_*` declines it, the registry's existing policy for
  ambiguous names). Motivated by the data: stdlib return types are overwhelmingly scalars, generics
  (`Self`/`T`), nominals, and `Vec<T>` / `Tensor<T>` with **empty** dimension lists — the Expr-bearing
  paths don't occur in return positions, so nothing real is dropped. Those paths (a bounded type-level
  `Expr` encoder) are a later refinement.

**Tests.** `registry_interface_round_trips_functions_and_methods` — freeze a registry with functions
returning a scalar, a struct, and a (dimensionless) `Tensor<f32>`, plus struct- and scalar-receiver
methods; after serialize → deserialize, every `resolve_fn` / `resolve_method` returns an identical GID
and return type. Full suite green (365 lib + 91 integration).

**Remaining for #220.** The per-function flat HIR **body** store (`body_of`) + stream serialization
(stage 3), then precompiling `stdlib/std` → `std.vxlib` and teaching the loader to consume it with no
parse/typecheck of the stdlib (stage 4 — the payoff, and where loader/build architecture decisions land).

## Entry 38 — stdlib decoupling Step 3 (stage 3a): flat-HIR bodies serialize (`body_of`) (#220)

**Commit:** _this session_. The `.vxlib` payload now carries function **bodies**, and `body_of` — the
`ModuleInterface` query #219 deferred — is live. Design pinned first in
[`vxlib_bodies_and_loader.md`](./implementation_plans/vxlib_bodies_and_loader.md) (`0c8f94a`).

**What we did.**

- **`FnBody { name, params, ret_ty, hir, types }`** on `ImmutableGlobalRegistry.bodies` (keyed by fn
  GID) + `body_of` on the registry and `ModuleInterface`. Self-contained (the flat codegen reads
  `params`/`ret_ty` off it) so linking is one lookup, not a join. **Empty in a from-scratch compile**;
  populated only on artifact deserialization — which sidesteps the freeze/lower ordering.
- **`Opcode::from_u32`** (a checked match over the `0..=30` discriminants — no `transmute`) + a
  fixed-record `HirInstruction` codec; the `bodies` section appends after `methods`, format → **v3**.
- **Fail-closed portability gate.** A body is serialized only if every GID in its type stream is global
  (`!is_local_deferred()`); a generic-instantiation body is *skipped* so `body_of` declines it rather
  than the artifact linking a body it can't resolve. Non-generic bodies (the stdlib's concrete methods)
  have already-global streams — the local→global patch only touches generic deferred GIDs.

**Tests.** `registry_interface_round_trips_a_flat_hir_body` — lower a real `fn add(..)` through
`hir::flatten`, stash the `(hir, types)` as a `FnBody`, serialize → deserialize, assert `body_of`
returns byte-identical instruction + type streams and signature; `non_portable_generic_body_is_skipped`
(a body with a deferred GID is dropped, `body_of` is `None`). Full suite green (367 lib + 91 integration).

**Remaining for #220.** stage 3b — enrich `fn_sigs`/`methods` with `params` at freeze time (shared with
the #219 imported-call flip); stage 4 — body harvester + `vxc --emit-interface`, precompile `std.vxlib`
in the build, loader artifact path + registry merge, codegen links `body_of` (end-to-end
`import std::math`, unblocking the softmax corpus #217).

## Entry 39 — stdlib decoupling Step 3 (stage 4): a program links a body from a `.vxlib` (#220)

**Commit:** _this session_. The decoupling endgame in miniature: a program links an imported function's
**body from a precompiled `.vxlib` artifact**, its source never parsed in the consumer compile — the
whole producer → deserialize → merge → link → run chain, proven end to end through the flat path.

**What we did.**

- **Producer** (`pipeline::emit_module_interface`): build the frozen registry, then *harvest* each
  non-generic free function — lower it to flat HIR, and if it lowers completely and portably (no
  deferred GID), stash a `FnBody` in `registry.bodies` — then `serialize_registry_interface`.
- **`registry.merge_from`**: fold a deserialized interface into a compile's registry (own entries win;
  the import fills in what the compilation didn't build). This is how a downstream compile gains the
  library's types/sigs/bodies with no AST.
- **`vxc --emit-interface`** (`driver`): resolve the loaded modules and write the `.vxlib` (verified by
  hand: `triple.vx` → a 470-byte artifact, `VXLB` magic + interface).
- **The link:** the flat lowerer resolves a call via the registry `fn_sigs` (not the AST env), and
  `emit_function_mlir` reads only `params`/`ret_ty` — so an imported body links by synthesizing a
  *signature-only* `Function` from its `FnBody` (from `body_of`) and handing it to `emit_module_mlir`.
  No AST `Function`, no #219 type-checker flip, no `vxc` driver flip required.

**Test.** `program_links_a_function_body_from_a_vxlib_artifact` — a `mathlib` module is compiled to
`.vxlib` bytes; a separate `app` (`fn main() { return double(21); }`) is compiled *without the lib's
source*, merging the deserialized interface; the flat path links `double`'s body from the artifact and
JIT-returns **42**. Full suite green (367 lib + 91 integration; one transient JIT flake on an unrelated
generic-Vec backend test, green on re-run).

**What's left of #220 (a genuine C-dependency, not a serialization gap).** Precompiling the *real*
`std.vxlib` in the build and having the loader auto-use it for `import std::math` needs (a) the flat
codegen to be `vxc`'s production path (C3) — the AST codegen can't consume AST-free artifact bodies —
and (b) flat-codegen coverage of externs / math intrinsics so real stdlib bodies actually lower (today
they fail-closed and skip). The **mechanism** is done and proven; wiring it to the real stdlib rides on
finishing convergence.

## Entry 40 — convergence finish (E1): extern calls lower + link through the flat path

**Commit:** _this session_. The first of the two blockers to compiling a real stdlib program through the
flat path (plan: [`convergence_finish_externs_methods_c3.md`](./implementation_plans/convergence_finish_externs_methods_c3.md)).
A stdlib math body is `impl Math for f32 { fn exp(self) { return expf(self); } }` — a method whose body
calls a libm **extern**; externs were entirely invisible to the flat path.

**What we did.**

- **Register `module.externs` in `fn_sigs`** (`build_frozen_registry`): same GID formula + cross-module
  ambiguity drop as functions. This alone unblocks the flat HIR — `lower_call` resolves the callee via
  `fn_sigs` — and the emitter's callee map, for scalar-returning externs (libm is all `f32→f32`).
- **The emitter declares called-but-undefined callees.** `emit_function_mlir` records each `func.call`'s
  `(name, arg types, ret type)`; `emit_module_mlir` emits a `func.func private @name(...)->ret` for any
  callee that has no `func.func` body in the module (i.e. an extern) — signature taken from the emitted
  call, so they match by construction. The JIT links the symbol (libm via `-lm`). Generalizes the old
  hardcoded print-helper declaration list.

**Test.** `flat_matches_ast_extern_call` — `extern { safe fn sqrtf(x: f32) -> f32; } fn main() { print(sqrtf(16.0)); return 0; }` lowers + emits through the flat path and its stdout (`4`) matches the
AST oracle. (`safe` so the call needs no `unsafe` block — flat unsafe-block lowering is separate.) Full
suite green (367 lib + 93 integration).

**Next (E2).** Method calls — the type checker already rewrites `x.exp()` → `f32$exp(x)`; the emitter's
callee map must include the emitted monomorph bodies (#217). Then E3: the `--flat-codegen` driver path.

## Entry 41 — convergence finish (E3): `--flat-codegen` is a production path (C3 mechanism)

**Commit:** _this session_. The flat codegen leaves the test harness: `vxc --flat-codegen` now compiles
and runs a program through `flat::emit_module_mlir` instead of the AST-walk `MeliorGenerator` — the C3
mechanism (#201), the piece that lets a downstream compile consume AST-free `.vxlib` bodies (#220).

**What we did.**

- **`build_flat_module`** (`driver.rs`): resolve names → freeze the registry → lower every non-generic
  function across all modules (deduped by name) to flat HIR → `emit_module_mlir` → parse into a melior
  `Module`. Returns `None` if any function is outside the flat subset.
- **`run_codegen` branch**: with `--flat-codegen`, produce the module via `build_flat_module`, else via
  `MeliorGenerator`; **everything downstream is shared** (verify → the same pass pipeline → `RunJit` /
  `EmitMlir` / `EmitObj`). A `None` from the flat build **falls back to the AST path**, so the flag
  never regresses — the AST path stays the oracle for anything outside the subset.
- The driver's optimization pipeline already runs `convert-math-to-libm`, so an emitted `func.call @sqrtf` lowers + links exactly as the AST path's does.

**Verified.** By hand: `vxc --flat-codegen --run` on scalar arithmetic (exit **23**) and an extern-math
program (`print(sqrtf(16.0))` → **4**) both take the flat path and are correct; a struct program prints
`program outside the flat subset; using the AST path` and returns the right value (**22**) via fallback.
Test `flat_module_built_for_in_subset_declined_otherwise` locks in Some (scalar / helper-call / extern)
vs None (struct → fallback). Full suite green (368 lib + 93 integration).

**Remaining to flip the default.** Widen the emitter so the corpus is *total* under `--flat-codegen`
(casts / `neg` / `not` #214, struct returns #215, and method-call dispatch #217 — the type checker
rewrites `x.exp()` → `f32$exp(x)` but the monomorph isn't in the frozen `fn_sigs`); then flip default,
AST behind `--legacy-codegen`. With that, `import std::math` compiles from `std.vxlib` end to end.

## Entry 42 — convergence finish (E2): a real stdlib method call runs through the flat path (#217)

**Commit:** _this session_. `vxc --flat-codegen --run` now compiles + runs `import std::math; (16.0f32).sqrt()` **entirely through the flat codegen** → `4`; `(1.0f32).exp()` → `2.7182817`. The
method-dispatch blocker is closed.

**The chain, and the two fixes that closed it.**

1. **Reachability in `build_flat_module`.** The driver's `monomorphized_ast` already carries the method
   monomorph (`f32$sqrt`, the type checker's `x.sqrt()` → `f32$sqrt(x)` rewrite) in its `functions`. But
   lowering *every* function of *every* imported module sank the program — std::math has ~40, some
   outside the flat subset. Now `build_flat_module` lowers only what's **reachable from the main
   module** (roots = main's functions incl. monomorphs; a BFS over each body's called names pulls in
   transitively-called imported functions on demand). Externs aren't functions → skipped here, declared
   `func.func private` at emit. For `sqrt`, the reachable set is just `main` + `f32$sqrt` + the `sqrtf`
   extern.
1. **`unsafe` blocks lower.** The stdlib wrapper body is `return unsafe { sqrtf(self) }`. Safety is
   checked upstream, so `unsafe` is transparent to lowering: an `Expr::UnsafeBlock` arm runs the block's
   statements and yields its trailing value. That was the actual decline point (`f32$sqrt` declined
   before this).

With E1 (externs) already in, the extern call inside the wrapper links via `-lm`.

**Tests.** `unsafe_block_lowers_transparently` (flatten unit); differential `flat_matches_ast_unsafe_ extern_call` (the wrapper shape) and `flat_matches_ast_scalar_method_call` (method dispatch flat-vs-AST
— the harness now collects the checker's monomorphizations, appends + re-registers them, mirroring the
driver). Full suite green (369 lib + 95 integration).

**Where the flip stands.** `--flat-codegen` now handles the scalar / control-flow / call / extern /
tensor / **method-dispatch** subset (AST fallback for the rest). Remaining for the *default* flip:
emitter breadth — `Cast`/`Neg`/`Not` (#214, note `Neg` HIR-lowers but the emitter still declines) and
struct returns (#215) — then flip default, AST behind `--legacy-codegen`.

## Entry 43 — convergence finish (E4a/b): scalar unary + `as` casts in the emitter (#214)

**Commit:** _this session_. Widening the flat emitter toward total corpus coverage (the prerequisite to
flipping `--flat-codegen` to the default). Three opcode families that HIR-lowered but the emitter
declined now emit + verify:

- **`Neg`** — float `arith.negf`; integers `0 - x` via `arith.subi` (no `negi`).
- **`Not`** — `x ^ all-ones` (`arith.xori` with `1` for a bool `i1`, `-1` for ints).
- **`Cast`** (`as`) — a `cast_op` picks the right `arith` conversion by source/target kind + width:
  `extsi`/`extui`/`trunci` (int↔int), `sitofp`/`uitofp` (int→float), `fptosi`/`fptoui` (float→int),
  `extf`/`truncf` (float↔float); a same-type cast aliases (no op).

**Tests.** Emitter units `emits_verifiable_unary_neg_and_not`, `emits_verifiable_scalar_casts`;
differential `flat_matches_ast_integer_negation`, `flat_matches_ast_float_negation`,
`flat_matches_ast_scalar_casts` (JIT parity). The two `flat_declines_scalar_cast…` tests became parity
cases. Full suite green (370 lib + 97 integration).

**Remaining before the flip:** struct returns (#215), then a corpus-wide flat-vs-AST validation, then
flip the default (AST behind `--legacy-codegen`).

## Entry 44 — C3: the flat codegen is the production default (#201)

**Commit:** _this session_. `vxc` now compiles through the **flat codegen by default**; the AST-walk
`MeliorGenerator` moves behind `--legacy-codegen`. This is the C3 flip — the flat-array pipeline is the
production path, the core claim of the convergence.

**What we did.**

- Inverted the driver flag: flat is the default (`build_flat_module` first), `--legacy-codegen` forces
  the AST path. Per-program **AST fallback** is unchanged — a module outside the flat subset silently
  compiles through AST, so output never changes. The `[flat-codegen] …` markers moved to stderr.
- `test_optimizations` (FileCheck over `--emit-mlir`, pinned to the AST codegen's module structure) now
  passes `--legacy-codegen`, so it keeps validating the AST path; the flat path's emission is validated
  by the flat-vs-AST differential harness.

**Validation — the whole reason this is safe.** Swept the **entire backend corpus** (~115 programs)
through both paths, normalized (stderr excluded, pointers canonicalized): **every program the flat path
accepts (~27) produces byte-identical stdout to the AST path.** The only 4 residual diffs are
pre-existing *AST-path* non-determinism (timing benchmarks, `unwind`) in programs that don't even use
flat; `llama2_v2`'s `malloc_N` link error is pre-existing in the AST path (flat declines it). **Zero
flat accept-but-miscompile cases.** Full suite green (370 lib + 97 integration).

**What flows through flat now:** scalar arith + control flow + calls + **externs** + **method dispatch**
(a real `import std::math; x.sqrt()`) + tensors (reductions/elementwise/row store/transfer) + `print` +
scalar unary/`as` casts. Outside the subset (struct returns #215, spawn, richer generics, …) →
AST fallback.

**Remaining (post-flip polish, not blockers):** widen the emitter so more of the corpus takes the flat
path (struct returns #215, …); a soak; then eventually retire the AST back end. The convergence's
production-path flip itself is **done**.

## Entry 45 — convergence finish (E4c): struct returns through the flat path (#215)

**Commit:** _this session_. A function can now **return a struct** through the flat codegen, and — as a
bonus — *all* structs now take the flat production path via `vxc` (previously they fell back to AST).

**What we did.**

- **Emitter — struct by value (`!llvm.struct`).** The signature emits `-> !llvm.struct<(...)>`; the
  `Ret` arm loads the struct value from its slot pointer and returns it (or returns a call result value
  directly); a struct-returning `Call` spills the returned value into a slot (`llvm.store`) so field ops
  address it; `Callee` gained `ret_agg` (the return struct's GID).
- **HIR — struct value positions.** `lower_expr` gained an `Expr::StructInit` arm (a `return P { .. }`,
  not just `let x = P { .. }`); `bind_local` always slot-allocates an aggregate (so a bound
  struct-returning call result is addressable even in a straight-line function).
- **Driver — annotate `StructInit` GIDs.** `build_flat_module` re-runs the type checker against the
  *frozen registry* purely to settle each `StructInit`'s GID (the driver's own semantic analysis ran
  against an empty registry, so those were `None` and the flat lowerer declined every struct). This is
  what lets structs — not just returns — flow through the flat path via `vxc`; method dispatch is
  unaffected.

**Tests + validation.** Differential `flat_matches_ast_struct_return` / `_struct_return_used_directly`
(JIT parity). By hand: `vxc --run` takes the flat path for a struct return (7), a struct field-sum (22),
and still the stdlib method (`x.sqrt()` → 4). Re-swept the whole backend corpus: **zero new mismatches**
(the 3 residual diffs remain the pre-existing non-deterministic benchmarks / `unwind`). Full suite green
(370 lib + 99 integration).

**Convergence status.** The production-path flip (C3) is done and the flat subset now spans scalars,
control flow, calls, externs, method dispatch, tensors, print, unary/casts, and **structs incl.
returns**. Remaining is ongoing emitter breadth (struct params, `spawn`, richer generics/tensors) +
a soak before retiring the AST back end — all behind the safe AST fallback.

## Entry 46 — emitter widening: value-position `if` + a decline-diagnostics feature

**Commit:** _this session_. Growing the flat subset by the remaining tractable constructs, driven by a
survey of *why* corpus programs fall back to AST.

**Decline diagnostics (`VX_FLAT_DBG`).** A new env-gated feature: the flat lowerer/driver report *why* a
program declines (which function fails to lower, the first unsupported `Expr`/`Statement` variant, or an
emit decline). Ran it across the backend corpus to prioritise: the top tractable buckets are `Assert`
(8), `MethodCall` (5, mostly NPU device methods), `StringLiteral` (4), value-`If` (4), `EnumVariant`
(3), `ComptimeBlock` (2).

**Value-position `if` (#201).** `let v: T = if c { .. } else { .. }` — the attention corpus's
`let m_new = if tm > m { tm } else { m }` shape. The flat model is slot-based, so this maps cleanly to a
**result slot**: each branch stores its trailing (semicolon-less) value into the slot; the merge block
loads it — no block arguments. The slot type comes from the `let`'s annotation; the memory-mode trigger
now fires on a value-`if` in a `let`/`return`/`assign`. Nested value-`if`s (a branch value that is itself
an `if`) still decline (they need value-position `if` in `lower_expr`, not just the annotated-`let`
path).

**Tests.** Differential `flat_matches_ast_if_expression` (a plain `if a > b { a } else { b }` and one
with a side-effecting leading statement in a branch) — JIT parity. Corpus value-`If` declines dropped
4 → 1. Full suite green (370 lib + 100 integration).

## Entry 47 — emitter widening: `assert` is a runtime no-op (matches the AST)

**Commit:** _this session_. The single biggest tractable decline bucket (8 corpus programs) closed by a
one-line lowering — because the AST codegen already treats `assert(cond, msg)` as a **runtime no-op**
(`generator.rs`, `Statement::Assert`: it records the fact for seam certificates but emits *no* runtime
check). So the flat path lowers `Statement::Assert` to nothing too, and flat == AST at runtime by
construction.

**Validation.** Differential `flat_matches_ast_assert_is_a_noop`; re-swept the corpus — the 5 residual
flat-vs-AST diffs are all pre-existing *AST-path* non-determinism (timing benchmarks, `unwind`, and
`llama2_v2`'s non-deterministic `malloc_N` mangling — flat *declines* it and falls back), **zero flat
miscompiles**. flat-used across the corpus rose 27 → **31**. Full suite green (370 lib + 101 integration).

## Entry 48 — emitter widening: the `print!` macro form (`Expr::Print`)

**Commit:** _this session_. The flat path handled `print(x)` (a function call) but not the `print!`
**macro** form, which expands to a dedicated `Expr::Print { args }` — the biggest chunk of the survey's
"other-expr" bucket (loop/scalar/math programs that `print!` their results). Now `Expr::Print` lowers by
emitting a `Print` for each argument in sequence (matching the AST's per-arg `print_*` calls, no
separators); a `StringLiteral` argument still declines (no string support yet), so `print!("label", x)`
falls back to AST. Refactored the shared emission into `emit_print`.

**Validation.** Corpus parity re-swept (only the pre-existing non-deterministic diffs remain, zero flat
miscompiles); flat-used 31 → 32. Full suite green.

**Remaining emitter gaps — filed as issues** (each falls back to AST safely meanwhile; #200 tracks):
strings / `print!("…", x)` (#225, 11 programs — the biggest unlock, also softmax #217), NPU/device
method calls (#226, 5), enum construction (#227, 3), comptime blocks (#228, 2), value-`if` in
expression/nested position (#229, 1), borrows / pointer values (#230, 1).

## Entry 49 — emitter widening: print-position string literals (#225)

**Commit:** `8fdc86b` _(this session, 2026-07-25)_. Entry 48 left a `StringLiteral` argument of
`print!`/`println!` declining — the largest remaining decline bucket. Now it lowers through the flat
path:

- **`PrintStr` opcode** (`bytecode.rs`), `imm` = index into a new per-function **string side table**
  (`LocalWorkerState.local_string_table`, the string analogue of `local_tensor_types`). A `PrintStr`
  can't carry the bytes inline, so the lowerer records them and codegen emits a global.
- **flatten** (`Expr::Print`/`Expr::Println`): a `StringLiteral` arg → record bytes + emit `PrintStr`;
  any other arg keeps the existing `Print`. `println!` reuses `PrintStr` for the trailing newline (a
  `"\n"` string is byte-identical to the AST path's `println()` runtime call — no new helper).
- **flat emitter**: `PrintStr` → `llvm.mlir.addressof @".str.N"` + `func.call @print_str(!llvm.ptr) -> i32`; `emit_module_mlir` emits an `llvm.mlir.global internal constant` per string (MLIR-escaped,
  null-terminated) numbered from a running module base, and declares `@print_str` `private`. String
  tables thread parallel to `funcs` via a new `emit_module_mlir` / `emit_function_mlir` param.

**Scope.** Print-position strings only (the common case). General string *values* (`let s = "…"`,
passed to a fn — e.g. `ffi_stdio.vx`'s `vx_stdout_write(msg, 14)`) still decline: they need a
`LoweredTy::Ptr` and stay follow-up work under #225.

**Validation.** Three new differential tests (bare string, string+scalar, `println`) JIT-match the AST
oracle; the harness `parse` now runs macro expansion so `print!`/`println!` desugar as in the real
pipeline. Full backend-corpus flat-vs-legacy sweep: **flat-used 32 → 38, zero miscompiles** (the one
output diff is `cpu_fusion_overhead`'s wall-clock timing, which varies legacy-vs-legacy too). Full
suite green (371 lib + 104 integration).

## Entry 50 — device-placement methods + `vx.spawn` region emitter (#226), tensor-store coercion (#232)

**Commits:** `ac70ffd` (device/spawn), `0a37119` (coercion) _(this session, 2026-07-25)_. Device programs
(`spawn on(Topology::NPU[..]) { .. }`, `.with_memory(..)`) now lower through the flat path.

- **`with_memory`** is transparent to lowering (it only annotates a tensor's home memory for the
  seam/type analysis): `flatten` lowers the receiver tensor and drops the memory-space argument, like
  the AST codegen. The transfer *methods* (`to_device`/`to_host`/…) are already rewritten to
  `Expr::Transfer` by the type checker, so `with_memory` is the only device method reaching `flatten`.
- **`vx.spawn` region emitter.** `lower_spawn` no longer declines memory-mode functions or
  control-flow spawn bodies. The emitter's `Spawn`/`SpawnEnd` arms materialize the body as the
  `vx.spawn` op's nested MLIR region (generic form, inline in the enclosing block which continues after
  it): the region's entry block holds the body setup, nested `^bb` blocks carry a `for`/`if`, and the
  last block is terminated with `vx.yield`. `topology` is the dispatch id (identical to the AST's
  attribute). A value-producing spawn still declines.
- **Tensor-store element coercion (#232).** A separate pre-existing bug surfaced: a default-`f32` float
  literal stored into a `bf16` tensor (`a[i] = 1.0`) emitted an ill-typed `memref.store`. The
  `TensorStore` arm now coerces the value to the element type (`truncf`/`extf`/`trunci`/…) via the
  existing `cast_op` — mirroring the AST's `coerce_type`. This unblocked the bf16 test siblings that
  gate the whole module.

**Validation.** A flat.rs unit test checks a `spawn { for .. }` `vx.spawn` region parses + verifies;
differential tests cover `with_memory` and the bf16 store (read back via `as f32`). End to end,
`npu_float_scalar` computes an identical `c0=3` flat-vs-legacy; **all 5 named NPU programs
(`npu_vector_add`/`npu_matmul`/`npu_matrix_transpose`/`npu_float_scalar`/`npu_large_matmul_tiled`) now
flat-lower**. Full backend-corpus sweep: **flat-used 38 → 54, zero miscompiles** (every flagged program
is byte-identical once the runtime dispatcher's non-deterministic logging + benchmark wall-clock timing
are stripped). #226 closed; #232 fixed.

## Entry 51 — payload-free enum construction + `match` (#227)

**Commit:** `f2b4ec6` _(this session, 2026-07-25)_. C-like (payload-free) enum programs
(`enum Color { Red, Green, Blue }` + `match`) now lower through the flat path. A payload-free enum
value is a bare `i32` discriminant (as in the AST codegen), so nothing new is needed at the opcode
level — just the ordinal and the block chain:

- **registry** `enum_variants`: a payload-free enum's name → variant names in declaration order
  (populated by `build_frozen_registry`), so the lowerer resolves `Color::Green` → ordinal `1`. A
  data-carrying enum is omitted (→ declines). Threaded through `merge_from`; `.vxlib` deserialize
  defaults it.
- **flatten**: `lowered_ty` maps a payload-free enum (spelled `Type::Enum` *or* `Type::Struct` — the
  resolver uses both) to `Scalar(I32)`; `Expr::EnumVariant` → an `i32` `Const` of the ordinal;
  `lower_match` lowers a statement-form `match` to an eq-compare + conditional-branch chain (a
  `Wildcard` arm is the unconditional default); `body_has_control_flow` detects `match`.
- **flat emitter**: an enum-typed param/return emits as `i32` (`enum_scalar`, backed by
  `EmitCtx.enums`).

**Scope.** Data-carrying (tagged-union) enums — payload variants, generic `Option<T>`,
`Vec<Option<..>>` (`option_unwrap.vx`) — plus value-producing `match` and payload/literal/binding
patterns still decline to the AST path. Split out to **#233**.

**Validation.** Two differential tests (a 4-arm enum match returning 7; a wildcard default returning
99\) JIT-match the AST oracle; `option_unwrap` still declines cleanly. Full backend-corpus sweep:
**flat-used 54 → 56** (`match_simple`, `match_runtime`), zero miscompiles. #227 closed (payload-free
deliverable).

## Entry 52 — comptime blocks + `sizeof`, and return-value coercion (#228, #234)

**Commits:** `9541ba2` (comptime), `517e1ec` (return coercion) _(this session, 2026-07-25)_.

- **comptime blocks (#228).** The AST codegen lowers `comptime { .. }` *transparently* — `sizeof<T>()`
  folds to a constant and `assert`s are runtime no-ops, so at runtime a compile-time block has no
  observable effect. `flatten` mirrors that: `Expr::ComptimeBlock` lowers the inner statements then its
  trailing value (or a discarded dummy); `Expr::SizeOf` → an `i64` `Const` of `T`'s byte size
  (`sizeof_bytes`, scalar/pointer sizes only — a struct/enum `sizeof` declines). `sizeof_feature.vx`
  now flat-lowers.
- **return-value coercion (#234).** `assert_comptime_func.vx`'s comptime `assert`s reference
  `foo`/`bar`/`baz` (`-> i64 { return <i32-literal> }`), pulled into the flat path for the first time —
  exposing a latent bug: the `Ret` arm emitted `func.return %v : i32` for an `-> i64` signature. Now it
  coerces the returned scalar to the declared return element type (`arith.extsi`/`extf`/… via
  `cast_op`), mirroring the AST's `coerce_type` at return. `assert_comptime_func.vx` now flat-lowers.

**Validation.** Three differential tests (sizeof value → 9; a comptime block → 0; `return 7` from an
`-> i64` fn → 7) JIT-match the AST oracle. Full backend-corpus sweep: **flat-used 56 → 59** (both
comptime programs + a third unblocked by the return coercion), zero miscompiles. #228 closed; #234
fixed.

## Entry 53 — value-position `if` in expression context (#229)

**Commit:** `067bf38` _(this session, 2026-07-25)_. Follow-up to the annotated-`let` value-`if`
(`ba564b1`): a value-`if` in a general *expression* position now lowers through the flat path — a
nested if-expr as a branch value, a call argument, an implicit return, or a compound-assign RHS
(`expr_assignment.vx`).

`flatten::lower_expr` gains an `Expr::If` arm that infers the result type from the then-branch's
trailing value (`infer_expr_ty` / `infer_block_ty` — reads the AST + scope *without emitting*,
covering number / identifier / binop / unary / relational / cast / call / nested-if / unsafe /
comptime forms), allocates a result slot, stores each branch's value into it (reusing
`lower_if_into_slot`), and loads the result. The annotated `let v: T = if ..` path is unchanged (its
annotation is a more precise type source). `else if` chains fall out for free (the `else` branch's
trailing value is itself an `Expr::If`), and the `if`'s blocks are self-contained, so a value-`if`
also lowers in a pure-SSA function with no other control flow (the pre-branch ops sit in the implicit
entry block).

**Validation.** Four differential tests (nested → 100; implicit-return → 12; call-argument → 42;
else-if compound-assign → 51) JIT-match the AST oracle. Full backend-corpus sweep: **flat-used
59 → 60** (`expr_assignment.vx`), zero miscompiles. #229 closed.

## Entry 54 — transparent tensor borrow `&t` (#230)

**Commit:** `99eace3` _(this session, 2026-07-25)_. `print(&t)` where `t` is a tensor
(`npu_lowering_execution.vx`) now lowers through the flat path. A tensor is a memref — already a
reference value — so borrowing it is transparent: `flatten`'s new `Expr::Borrow` arm yields the tensor
itself, matching the AST codegen (`BorrowExpr` returns the memref for an allocated tensor identifier).

**Scope.** The general pointer ABI — `&x`/`*p` for a scalar/aggregate (a real `!llvm.ptr`), pointer
params/returns, and `void`/`ptr` extern returns — still declines. No corpus program drives it alone
(the pointer-heavy FFI programs also need string values #231 + structs), so it is split out to **#235**
rather than built speculatively.

**Validation.** A `print(&t)` differential test's `printMemref` dump JIT-matches the AST oracle;
`npu_lowering_execution.vx`'s output is byte-identical flat-vs-legacy (pointers + dispatcher logging
normalized). Full backend-corpus sweep: **flat-used 60 → 61**, zero miscompiles. #230 closed
(the corpus `Expr::Borrow` decline).

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
