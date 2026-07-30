# Design: Scalar References (`&i32`) on the Flat Path

**Status:** **immutable slice (§9), mutable slice (§10), and the step-2 per-local memory rule (§11) implemented** — `&x` / `&mut x` / `*r` / `*p = v` / `&i32` params + returns lower on the flat path and run (locally and *across a module boundary*), the immutable forms verified against the AST oracle and the mutable forms against the value-semantics equivalent (§6.2); a control-flow function now slots only the locals that need it. Deferred: §5 places/projections ([#275](https://github.com/hiraditya/Vx/issues/275)).
**Relates to:** [#230](https://github.com/hiraditya/Vx/issues/230) (borrows / pointer values, closed for the aggregate subset) · [#197](https://github.com/hiraditya/Vx/issues/197) (flat pipeline epic) · [#275](https://github.com/hiraditya/Vx/issues/275) (§5 places / projections, the next direction)
**Companion:** [`hir_flattening.md`](hir_flattening.md) — the SSA/instruction conventions this builds on

______________________________________________________________________

## 1. What declines today

Three related gaps, all around a reference whose pointee is a scalar:

1. **`&i32` as a parameter.** `lower_ty_synth` (`src/hir/flatten.rs`) maps scalars, tensors and
   aggregates; `is_ptr_to_agg` (`src/hir/flatten.rs`) covers `&mut Vec` but not `&i32`. So
   `fn pick(a : &i32, b : &i32) -> &i32` never lowers.
1. **`&x` on a scalar local** explicitly declines (the `Expr::Borrow` arm in `src/hir/flatten.rs`).
   The local is an SSA register with no address; it would need promoting to an `Alloca` slot.
1. **Returning a reference, and `*r`** need the reference threaded through as a pointer value.

The AST path also fails these (an unresolved `unrealized_conversion_cast`), so there is currently no
differential oracle for the construct — see §6.

## 2. Diagnosis: the predicate asks the wrong question

```rust
fn is_ptr_to_agg(ty: &Type, registry: &ImmutableGlobalRegistry) -> bool {
    matches!(ty, Type::Borrow { .. } | Type::Pointer(..)) && agg_gid_of_ty(ty, registry).is_some()
}
```

This reads: *"a reference is real only if its pointee has a layout GID."* Scalars are not being
excluded deliberately — they fall out because the predicate tests **pointee shape** when the question
is **does this value need an address**. Aggregates pass incidentally: they already live in slots for
unrelated reasons (they must be addressable to be GEP'd), so a reference to one already has something
to point at.

Fixing this by "adding scalars to `is_ptr_to_agg`" would paper over it. The pointee's shape is not
what determines whether a reference is representable.

## 3. The principle: register by default, demote on address-taken

> **clang allocas everything and promotes; Vx registers everything and demotes the few that need it.**

This is the inverse of LLVM's `mem2reg`. Clang emits an `alloca` for every local and relies on
`mem2reg` to promote the ones whose address is never taken. The flat path starts from the other end —
every local is an SSA register (`Binding::Reg`) — so the corresponding pass is a **demotion**: find
the locals whose address *is* taken and give those, and only those, a slot (`Binding::Slot`).

Both representations already exist (`Binding::{Reg, Slot}` in `src/hir/flatten.rs`). What is missing
is the analysis that chooses between them **per local**.

### 3.1 Why today's policy is too coarse

Memory mode is currently a **function-global flag**: on when the function has control flow, or has any
aggregate local; off otherwise. So a function with a single `if` puts *every* local in memory, and a
function without one cannot address *any* local. Both directions are wrong:

- over-allocating: locals that are never addressed still get `Alloca`/`Store`/`SlotLoad`;
- under-allocating: `&x` in a straight-line function has nowhere to point, hence the decline.

A per-local decision fixes both at once, and is strictly less memory traffic than the status quo.

### 3.2 The analysis

One pre-pass over the HIR body before lowering:

```
address_taken = { sym | Expr::Borrow(Expr::Identifier(sym)) appears anywhere in the body }
```

A local gets `Binding::Slot` iff **any** of:

- it is in `address_taken` — *(new)*
- it is an aggregate — *(today: must be addressable to GEP)*
- it is mutated and the function has control flow — *(today: must survive block boundaries)*

Everything else stays `Binding::Reg`. This is a single walk, monotone, and needs no fixpoint: taking
an address is a syntactic property of the body.

> [!NOTE]
> **Parameters need no demotion.** A `&i32` parameter arrives already materialized — it crossed a
> call boundary, so an address exists by construction and it is simply a `Ptr` on entry. Only *locals*
> are ever demoted. This is why gap (1) and gap (2) in §1 have different fixes despite looking alike.

### 3.3 Consequence for the decline site

The `Expr::Borrow` arm stops being a decline and becomes a lookup: the local is already a `Slot` by
the time lowering reaches the borrow, so `&x` yields the slot's register with `LoweredTy::Ptr` —
exactly what the aggregate arm does today, minus the aggregate-specific reasoning.

### 3.4 Parallel-architecture safety

This was checked rather than assumed, because "add an analysis pass" is exactly the kind of change
that can quietly acquire a cross-function dependency and need a new phase boundary.

**The analysis is body-local, and that is a property of the language, not a coincidence.** The rule in
§3.2 is only complete if `&x` is the *sole* way a local's address is taken. It is: Vx does not
auto-borrow, so passing a scalar local to a `&i32` parameter is a type error, not an implicit
address-of.

```console
$ vxc ab1.vx --action emit-mlir      # fn takes(r : &i32); let x = 5; takes(x)
Error[E3003]: Type mismatch in argument 1 for function 'takes'.
  Expected Borrow { inner: Scalar(I32), .. }, got Scalar(I32)
```

Had scalars auto-borrowed, `address_taken` would need the *callee's* signature at every call site and
the analysis would stop being body-local. It does not, so the syntactic rule stands.

**It satisfies Phase 3's isolation invariants trivially.** `verify_phase_3_isolation`
(`src/parallel_architecture_verifier.rs`) asserts exactly two things per worker — every worker's
`Arc` points at the same frozen `GlobalSession` (the aliasing proof), and local deferred type indices
stay inside the worker's arena. The pass reads one function body and writes a `HashSet<Symbol>`
consumed by that same function's lowering: no new global, no `Arc`, no atomic, nothing that could make
a worker diverge from the frozen session.

**No fixpoint, so no new phase boundary.** This is the part worth contrasting. Return-provenance
(`0a42af51`, #243) is *genuinely* inter-procedural — it needs an SCC fixpoint over the call graph —
which is why it had to be computed and **frozen into the env** as its own step. Address-taken needs
none of that: one walk, monotone, per function. It runs inside Phase 3 with no additional barrier.

Two consequences worth recording:

- §4's typed load/store uses `type_idx` into the per-worker `local_type_stream`, which is already the
  deferred-interning mechanism — so it *inherits* the arena-bounding invariant rather than
  side-stepping it.
- The per-local rule is **narrower** than what it replaces. Function-global memory mode makes the
  `Reg`/`Slot` choice depend on a whole-function property (does this function have control flow);
  per-local makes it depend on a per-symbol property. Strictly less coupling, same isolation.

The one place this could come back is §5. If projections ever needed cross-function reasoning, places
would acquire the same shape as return-provenance and want the same freeze-point treatment. That is a
second, independent reason to keep §5 out of scope.

## 4. Keep `Ptr` opaque; type the load and store instead

The obvious move is `LoweredTy::Ptr(Box<LoweredTy>)` so a deref knows what to load. **Don't.**

- MLIR and LLVM both moved *to* opaque pointers; carrying a pointee re-litigates that.
- A recursive variant complicates every `match` on `LoweredTy`, which is currently flat by design.
- The pointee type is available at every deref site from the HIR anyway.

Instead put the type on the **instruction** — `Load { ty }` / `Store { ty }`, which is what LLVM does
post-opaque-pointers. `LoweredTy` stays flat, and `is_ptr_to_agg` does not get extended: it collapses
to "is this a reference or pointer type," with the aggregate clause **deleted** rather than widened.

## 5. Places and projections — the direction, not now

**Tracked by [#275](https://github.com/hiraditya/Vx/issues/275).**

The principled long-term representation is MIR-style `Place = local + projection path`, which would
subsume `&outer.inner`, reborrows, and nested references under one model. It also lines up with what
the borrow checker already tracks — `BorrowRecord` stores `path: Vec<String>`, which is a projection
path under another name (see [`borrow_checker_architecture.md`](../borrow_checker_architecture.md)).

This is deliberately **out of scope**. §3 and §4 unblock `&i32` without it. Recorded here so the
choice in §4 (typed instructions rather than typed pointers) is understood as *compatible* with a
later move to places, not as a step away from it.

## 6. The oracle question

`src/codegen/flat.rs`'s header states the invariant the whole flat effort rests on:

> *"The AST path stays the oracle: the emitter returns `None` for any stream (or module) using opcodes
> outside this subset, so nothing half-lowered is ever emitted."*

Implementing scalar references on the flat path **only** breaks that invariant permanently for this
construct: flat becomes the sole path, with no differential to catch regressions in any future change
to scalar references.

Recommended order:

1. **Fix the AST path first.** Its failure is an unresolved `unrealized_conversion_cast` — it is
   producing the cast and failing to *resolve* it, not failing to model the construct. That is closer
   to working than it looks, and fixing it restores the differential oracle for free.
1. **If flat-only is accepted anyway**, JIT verification has to be stronger than "it runs" to
   compensate: differential against the value-semantics equivalent (`let r = &x; *r` versus plain
   `x`), which both paths already handle.

Either way the borrow-checker fixtures give cheap extra coverage — `bc10`/`bc11`/`bc22` in
[`borrow_checker_precision_analysis.md`](../borrow_checker_precision_analysis.md) exercise these
shapes over `&Map`; the same programs over `&i32` have known-correct expected verdicts.

## 7. Why this matters beyond the flat path

`fn pick(a : &i32, b : &i32) -> &i32` is one line away from the motivating example in
[`borrow_checker_parameter_provenance.md`](../borrow_checker_parameter_provenance.md) §1, which leads
with `fn pick(a : &Map, b : &Map) -> &i32` — the signature rustc cannot compile without a lifetime
annotation.

That example works **because `&Map` is an aggregate and gets a slot for unrelated reasons.** The
scalar variant — the first thing anyone will try — does not lower on either path. Worth closing before
that claim is written up.

## 8. Implementation sketch

| Step | Change | Where |
|---|---|---|
| 1 | Address-taken pre-pass; `Binding::Slot` per local | `src/hir/flatten.rs` (before the body walk) |
| 2 | Replace the function-global memory-mode flag with the per-local rule (§3.2) | `src/hir/flatten.rs` (`Lowerer`) |
| 3 | `Expr::Borrow` on a scalar local → slot register + `LoweredTy::Ptr` | `src/hir/flatten.rs`, the `Expr::Borrow` arm |
| 4 | Type the load/store instructions; drop the aggregate clause from `is_ptr_to_agg` | `src/hir/flatten.rs`, `src/hir/bytecode.rs` |
| 5 | `&i32` params in `lower_ty_synth`; deref + reference returns | `src/hir/flatten.rs`, `src/codegen/flat.rs` |
| 6 | Fix the AST path's `unrealized_conversion_cast` so the differential oracle applies | `src/codegen/lower/` |

Steps 1–2 are worth doing on their own merits even if scalar references slip: they strictly reduce
memory traffic in every function that has control flow but no address-taken locals, which today is
the common case.

## 9. Status: the immutable slice, as implemented

Landed as an **additive** change (not the full step-2 replacement of the function-global memory flag —
that optimization is deferred, since it would perturb currently-working functions and the differential
oracle is cheaper to trust than to re-establish):

- **Address-taken pre-pass** (`body_address_taken` in `src/hir/flatten.rs`): a syntactic walk collecting
  every local named by a `&x`, run once before the body lowers. Body-local, monotone, no fixpoint —
  §3.4 holds as written.
- **Per-local demotion** (`bind_local`): a scalar in that set gets a `Binding::Slot` *in addition to*
  the existing `memory || aggregate` rule. Straight-line functions with no `&scalar` are untouched.
- **`Expr::Borrow` on any slot** yields the slot register as `LoweredTy::Ptr` (the aggregate-only match
  relaxed to any scalar/aggregate slot — §3.3).
- **Codegen** (`src/codegen/flat.rs`): an address-taken scalar slot is an `llvm.alloca` of its element
  (flagged `imm = 1` on the `Alloca`, tracked in `sslot_of`), so `&x` is a real `!llvm.ptr` and `*r`
  GEPs cleanly — exactly the shape the AST path emits for a mutable scalar. A never-borrowed
  memory-mode scalar keeps its rank-0 `memref` (§4's typed-instruction principle: the pointer stays
  opaque; the load/store carry the element type).

**Correction to §1/§6.** The claim that "the AST path also fails these" is only true for the
**mutable** case. Empirically the AST path compiles and runs the **immutable** forms correctly
(`&x`/`*r`/`&i32` params/returns over a `let x`), so a differential oracle *did* exist for the slice
that shipped — §6 step 6 (fix the AST path first) was unnecessary for it. The `unrealized_conversion_cast`
is specifically the `let mut x` alloca path (a mutable scalar local materialized as `memref<i32>`, whose
`&x` never resolves to `!llvm.ptr`); that remains the blocker for the mutable slice and is unrelated to
the flat work here.

**Verified:** `driver_import_runs_cross_module_scalar_references` runs the `fn pick(a : &i32, b : &i32) -> &i32` showcase (§7) *across a `.vxlib` boundary* — the signature rustc cannot compile without a
lifetime — plus flat unit tests (`address_taken_scalar_demotes_to_a_flagged_slot`,
`non_address_taken_scalar_stays_a_register`, `reference_param_derefs_without_a_slot`) and JIT
differentials against the value-semantics equivalent.

**Deferred (from the immutable slice):** §5 places/projections ([#275](https://github.com/hiraditya/Vx/issues/275)). The mutable slice (§10) and the step-2
memory-flag replacement (§11) are no longer deferred.

## 10. Status: the mutable slice, as implemented

The mutable forms turned out to be **almost entirely already covered** by the immutable slice plus one
orthogonal gap. `&mut x` is an `Expr::Borrow` with `is_mut: true`, which the address-taken pre-pass and
the `Expr::Borrow` arm already handle mutability-agnostically; and store-through-a-pointer (`*p = v`)
was the existing `PtrStore` path (#242). So `bump(p : &mut i32) -> i32 { *p = *p + 1; return *p; }`
called as `return bump(&mut x)` already ran on the flat path after §9.

The one missing piece was **void-returning calls** — a general flat-codegen gap, *not* a reference
feature: the canonical mutators (`increment`/`swap`) return `void` and are called in statement
position (`bump(&mut x);`), then the local is read back. `lower_call` declined because
`lower_ty_synth(void)` is `None`; codegen's `Callee`/`Call` had no void case. The fix:

- `is_void_ty` (`void` is `Type::Struct("void", _)`); `Callee.ret_void`; `lower_call` gives a void
  callee a discarded placeholder result instead of declining.
- codegen emits `func.call @f(..) : (..) -> ()` with **no** result binding (MLIR forbids
  `%v = … -> ()`), and declares a void `extern` as `(args)` (empty return).

Read-back-after-mutation is correct because an address-taken scalar is now an `llvm.alloca` slot
(§9): `&mut x` passes that `!llvm.ptr`, the callee `llvm.store`s through it, and `return x` `llvm.load`s
the same slot — the aliasing holds by construction.

**Oracle.** The AST path cannot compile a borrowed mutable scalar local (the `memref -> !llvm.ptr`
cast, §1), so there is no *direct* AST oracle for the reference form. Per §6.2 the result is grounded
on the **value-semantics equivalent** (the inlined mutation), which both paths compile: `inc(&mut x)`
twice equals `x = x + 1` twice (43), and a full `swap` reads the swapped values apart from a no-op.

**Verified:** `flat_runs_mutation_through_a_reference` and `flat_runs_swap_through_mutable_references`
(value-semantics differentials); `void_call_lowers_and_mutates_through_a_reference` (unit); the
`borrow_semantics.vx` showcase (`increment` + `swap`) runs to 31 on the flat path; and a corpus
differential over `tests/backend/pass` confirmed the broad void-call change regresses nothing (the JIT
corpus has no `-> void` functions, so no program changed path).

**Still deferred:** taking the address of a mutable local *and reading it while borrowed* is an
E4002 borrow error (correctly rejected, not a codegen gap); and reference-typed struct fields / returns
of borrowed locals remain out of scope (§5).

## 11. Status: the step-2 per-local memory rule, as implemented

The §9 slice left the function-global memory flag in place (address-taken demotion was *additive* over
it). Step 2 (§3.2) replaces that coarse flag for scalars with a per-local decision. A scalar local now
gets a slot iff:

- its **address is taken** (`&x`, §9), or
- it is **reassigned in a control-flow function** (`x = ..`/`x += ..` where the function has
  `if`/`loop`/`for`/`match`/`&&`/`||`) — the new value must cross a block boundary.

A non-mutated scalar — even under control flow — stays a dominating SSA register. Where the old rule
slotted *every* local in any function with a single `if`, `fn main() { let k = 40; let c = 1; if c > 0 { .. } .. }` now emits **zero** `memref.alloca`: `k` and `c` are `arith.constant`s used directly across
blocks.

Implementation (`src/hir/flatten.rs`):

- one `analyze_local_uses` pre-pass collects both `address_taken` and `mutated` (assignment targets);
- `has_control_flow` is split out from `memory` (an aggregate param / struct construction still turns
  on the block model, but does *not* force a mutated scalar into a slot in a straight-line function —
  a straight-line reassignment is a pure-SSA rebind);
- `bind_local` applies the rule; aggregates always slot, pointer/tensor locals keep the coarse model.
  Loop induction variables are bound directly as slots by `lower_for` (not via `bind_local`), so they
  are unaffected.

**Why this can't silently miscompile.** A register only ever names a *single-definition* local's value
— always the correct one — because every reassigned local is kept in a slot. The only failure mode is
a register read outside its definition's dominance (e.g. a value defined in one branch read after the
merge), and that is **invalid MLIR** the verifier rejects, so the flat path declines to the AST oracle
rather than emitting a wrong answer. Under-collecting `mutated` therefore costs at most a decline, never
correctness.

**Verified:** `if_else_slots_only_the_mutated_local` (unit: one `Alloca`, not two);
`flat_registers_non_mutated_locals_under_control_flow` / `flat_still_slots_a_mutated_local_under_control_flow`
(flat-vs-AST parity + the memory-traffic assertion); the full flat differential suite (control flow,
loops, value-`if`, match) still matches the AST oracle; and a corpus differential over
`tests/backend/pass` (12 programs on the flat path) shows no regression.

**Deferred:** applying the same per-local rule to *pointer* locals (still on the coarse model — a
missed optimization, not a correctness gap) and §5 places/projections ([#275](https://github.com/hiraditya/Vx/issues/275)).
