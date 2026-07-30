# Design: Scalar References (`&i32`) on the Flat Path

**Status:** design — not implemented
**Relates to:** [#230](https://github.com/hiraditya/Vx/issues/230) (borrows / pointer values, closed for the aggregate subset) · [#197](https://github.com/hiraditya/Vx/issues/197) (flat pipeline epic)
**Companion:** [`hir_flattening.md`](hir_flattening.md) — the SSA/instruction conventions this builds on

______________________________________________________________________

## 1. What declines today

Three related gaps, all around a reference whose pointee is a scalar:

1. **`&i32` as a parameter.** `lower_ty_synth` (`src/hir/flatten.rs`) maps scalars, tensors and
   aggregates; `is_ptr_to_agg` (`src/hir/flatten.rs`) covers `&mut Vec` but not `&i32`. So
   `fn pick(a : &i32, b : &i32) -> &i32` never lowers.
2. **`&x` on a scalar local** explicitly declines (the `Expr::Borrow` arm in `src/hir/flatten.rs`).
   The local is an SSA register with no address; it would need promoting to an `Alloca` slot.
3. **Returning a reference, and `*r`** need the reference threaded through as a pointer value.

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

## 4. Keep `Ptr` opaque; type the load and store instead

The obvious move is `LoweredTy::Ptr(Box<LoweredTy>)` so a deref knows what to load. **Don't.**

- MLIR and LLVM both moved *to* opaque pointers; carrying a pointee re-litigates that.
- A recursive variant complicates every `match` on `LoweredTy`, which is currently flat by design.
- The pointee type is available at every deref site from the HIR anyway.

Instead put the type on the **instruction** — `Load { ty }` / `Store { ty }`, which is what LLVM does
post-opaque-pointers. `LoweredTy` stays flat, and `is_ptr_to_agg` does not get extended: it collapses
to "is this a reference or pointer type," with the aggregate clause **deleted** rather than widened.

## 5. Places and projections — the direction, not now

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
2. **If flat-only is accepted anyway**, JIT verification has to be stronger than "it runs" to
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
