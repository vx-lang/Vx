# Design: Scalar References (`&i32`) on the Flat Path

**Status:** **immutable slice (§9), mutable slice (§10), and the step-2 per-local memory rule (§11) implemented** — `&x` / `&mut x` / `*r` / `*p = v` / `&i32` params + returns lower on the flat path and run (locally and *across a module boundary*), the immutable forms verified against the AST oracle and the mutable forms against the value-semantics equivalent (§6.2); a control-flow function now slots only the locals that need it. Places ([#275](https://github.com/hiraditya/Vx/issues/275)): empty-path (§12, Example A — the alloca that should not exist) and field references (§13, Example B/C substrate) lower on the flat path. Deferred: the §5.4 disjointness→alias-metadata payoff (M2b).
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

### 5.1 What a "place" is

Today a reference is a **pointer value**: an `!llvm.ptr` sitting in an SSA register, which you load
from and store through. A *place* is the other option — a **symbolic description of a location** that
has not been turned into an address yet:

```
Pointer (today):  r = <some register holding a machine address>
Place:            r = (local `x`, projections [ .field(0), .index(7) ])
```

The difference is *when the address gets materialized*. With pointers, the moment you write `&x` an
address must exist, so `x` must live in memory. With places, `&x` only records **which location you
mean**; an actual address is computed at the point of use — and often never, because the use can read
the local directly.

### 5.2 Example A — the alloca that should not exist

This is the motivating case, and it is one §3 cannot fix:

```rust
fn f() -> i32 {
  let x = 5;
  let r = &x;
  return *r;
}
```

Under §3, `x` is in `address_taken`, so it is demoted to a `Slot`:

```
alloca x            ; a stack slot
store 5 -> x
%p = <x's slot>     ; &x
%v = load %p        ; *r
ret %v
```

Under places, `r` is just `(x, [])` — no address is ever needed, because `*r` resolves to "read local
`x`", and `x` stays a register:

```
%c = const 5        ; x
ret %c
```

The address is taken syntactically but never *materialized*. §3 demotes on **syntactic** address-taken
because it cannot tell the difference; places let you demote only on **actual materialization** — when
the reference escapes into a return value, a struct field, or an opaque callee. §3 is the conservative
approximation of that rule.

### 5.3 Example B — nested projections

```rust
struct Inner { v : i32 }
struct Outer { inner : Inner }

let mut o = Outer { inner : Inner { v : 1 } };
let r = &mut o.inner.v;
*r = 42;
```

With pointers, each hop materializes an intermediate address:

```
%a = gep o, 0       ; &o.inner        <- an intermediate pointer value
%b = gep %a, 0      ; &o.inner.v      <- another one
store 42 -> %b
```

With places, `r` is `(o, [.field(inner), .field(v)])` and the whole path resolves at the store. The
intermediate pointers never become IR values, so nothing downstream has to prove they were only used
to reach the final one.

### 5.4 Example C — disjointness the borrow checker already proved

```rust
fn update(p : &mut Point) {
  let bx = &mut p.x;
  let by = &mut p.y;   // legal: x and y are disjoint fields
  *bx = 1;
  *by = 2;
}
```

The borrow checker accepts this via path-overlap analysis — `BorrowRecord` stores
`path: Vec<String>`, so it knows `["x"]` and `["y"]` cannot alias (see
[`borrow_checker_architecture.md`](../borrow_checker_architecture.md)).

**Codegen then throws that away.** Both borrows become opaque `!llvm.ptr` values, and LLVM has to
*re-derive* the disjointness from GEP offsets — work the frontend already did, with a proof the
frontend already had. With places, `(p, [.field(x)])` and `(p, [.field(y)])` carry it into the IR.

That is also why places are the natural fit rather than an arbitrary choice: `BorrowRecord.path` **is
a projection path**. The two layers are computing the same structure in different representations and
never reconciling them.

### 5.5 Why it is still deferred

Two concrete costs, not just effort:

- **Paths are variable-length; the instruction is fixed-width.** `HirInstruction` is
  `{ opcode, operand1, operand2, type_idx, imm }` (see [`hir_flattening.md`](hir_flattening.md)).
  A place needs a base plus an arbitrary-length projection list, which does not fit — it would need a
  side table or an interning scheme for paths, and that is a real addition to the flat representation.
- **§3 and §4 already unblock `&i32`.** The gain from places is *precision* (Example A's missing
  alloca) and *information preservation* (Example C), not capability.

### 5.6 Why §4 is compatible with this

If `LoweredTy::Ptr` carried its pointee (`Ptr<T>`), places would arrive as a *second, parallel* way to
say "reference" — two representations, two sets of load/store forms. Because §4 keeps `Ptr` opaque and
puts the type on the **instruction**, a place lowers to the *same* typed `Load`/`Store`; only the
operand changes from "a register holding an address" to "a place descriptor." The instruction set does
not fork.

That is what "compatible, not a step away" means concretely.

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

### 6.1 What was actually chosen

Resolved **per slice** rather than once, and both branches above ended up used — recorded here so it
is not re-litigated:

| Slice | Oracle | Why |
|---|---|---|
| Immutable (§9) — `&x`, `*r`, `&i32` params/returns | **AST differential** (option 1) | The AST path handles these once the cast resolves, so the invariant holds unmodified. |
| Mutable (§10) — `&mut x`, `*p = v` | **Value-semantics differential** (option 2) | No AST oracle available for the mutating forms; compared against the equivalent program written without references. |

So the invariant in §6 is intact for the immutable surface and consciously relaxed for the mutable
one, with a substitute check rather than "it runs." If the AST path later grows the mutating forms,
§10's cases should move to the stronger oracle.

## 7. Why this matters beyond the flat path

### 7.1 A flat decline is graceful intra-module and fatal cross-module

This is the general property, and it is the reason the work was urgent rather than tidy.

[`hir_flattening.md`](hir_flattening.md) states the keep-green strategy as: *"unsupported functions are
simply un-lowered until their constructs land."* Graceful degradation — the flat emitter declines, the
AST path picks it up, the subset grows by corpus over time.

That has an **unstated precondition: the AST path is available as a fallback.** For an *imported* body
it is not — the consumer has no AST for the library, only the `.vxlib`. So the identical decline
reclassifies:

| | intra-module | cross-module |
|---|---|---|
| flat emitter declines | AST path handles it — harmless, catch it next release | **hard error — this program cannot link** |

Nothing about the decline changes. What changes is whether a fallback exists.

### 7.2 The precedent, and what else it applies to

[#230](https://github.com/hiraditya/Vx/issues/230) is the worked example. It closed borrows and pointer
values *for the aggregate subset*, and that is precisely why structs link across a module boundary
today — [#274](https://github.com/hiraditya/Vx/issues/274) notes that functions and structs "work
end-to-end from a `.vxlib`... because the flat emitter handles them." Closing a flat decline is what
unlocks the cross-module case for that construct.

The same shape is open elsewhere:

- [#273](https://github.com/hiraditya/Vx/issues/273) — scalar references (this document).
- [#274](https://github.com/hiraditya/Vx/issues/274) — an enum value bound from a call.
- [#233](https://github.com/hiraditya/Vx/issues/233) — data-carrying enums.

Each reads as a coverage gap and is in fact a linking blocker.

**Consequence for prioritisation.** "Grow the subset by corpus" gains a second driver: *what do
imported bodies need*. The two rank differently — a construct that is rare in local code but common in
stdlib bodies jumps the queue under the second rule and not the first. Worth applying deliberately
rather than discovering per-issue.

### 7.3 The concrete case this unblocked

[#273](https://github.com/hiraditya/Vx/issues/273): a runnable cross-module return-provenance demo — a
consumer that compiles and JITs a program the *conservative* borrow rule would reject, because the
imported `pick`'s provenance says the result derives from `b` only.

The frontend already worked end to end. With `--link-interface` the consumer type-checks and
borrow-checks against the `.vxlib` with no library source, correctly accepting a mutable reborrow of
the non-aliased local and rejecting the aliased one. **Only codegen of the reference pattern was
missing** — which is what §9/§10 closed.

### 7.4 And the paper example

`fn pick(a : &i32, b : &i32) -> &i32` is one line from the motivating example in
[`borrow_checker_parameter_provenance.md`](../borrow_checker_parameter_provenance.md) §1, which leads
with `fn pick(a : &Map, b : &Map) -> &i32` — the signature rustc cannot compile without a lifetime
annotation.

That example worked **because `&Map` is an aggregate and got a slot for unrelated reasons**; the scalar
variant — the first thing a reader will try — did not lower on either path. It does now (§9–§11), so
the claim is safe to write up with either spelling.

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

## 12. Status: places, M1 — the non-escaping borrow (§5 Example A)

The first slice of the place representation ([#275](https://github.com/hiraditya/Vx/issues/275)), landing
§5 Example A: a `&x` that never escapes binds `x` to a **symbolic place** instead of materializing an
address, so `let r = &x; return *r` compiles to `const 5; ret` — **zero `alloca`**, the alloca that
should not exist, gone.

The key structural choice (from §5.5): a **non-escaping** place is pure lowerer data — a
`Binding::Place { base, path: Vec<Projection>, ty }` in the scope map — so the variable-length
projection path lives in Rust data, not in the fixed-width `HirInstruction`. The side-table §5.5 flags
is only needed if a place must become a first-class HIR *value*, which is exactly the escaping case that
materializes to a pointer anyway. M1 uses **empty** paths (Example A); the `Projection::Field` variant
is defined for the M2 field slice (Example B).

**Escape analysis** (`analyze_local_uses`, two passes, body-local, §3.4-safe): a `let r = &x`
(immutable, plain local) is a place *candidate*; a candidate's base stays a register unless the
reference **escapes** — used as anything but `*r` (a call arg, a return, an rvalue, `f(r)`), or written
through. The old "any `&x` slots x" set (§9) is replaced by a `materialized` set of bases whose address
actually escapes. So `dbl(*r)` (only the *value* escapes) stays a place, while `id(r)` (the reference
escapes) materializes — verified both ways.

**Why it can't miscompile.** If the analysis wrongly keeps a base a register, the escaping `&x` reaches
the `Expr::Borrow` arm with a non-slot base and **declines** to the AST oracle (the same MLIR-verifier
backstop as §11). Reading a place *as a value*, assigning to it by name, or a nested place all decline.
M1 is immutable-only: a `&mut` borrow always materializes (its mutation path is §10), and a written
place falls back to materialization — Example C's mutable disjoint fields wait for a later slice.

**Verified:** `non_escaping_borrow_binds_a_place_with_no_alloca` /
`escaping_borrow_materializes_a_flagged_slot` (unit); `flat_non_escaping_borrow_needs_no_alloca` /
`flat_escaping_borrow_materializes_an_address` (flat-vs-AST parity + the alloca assertion, both
directions); the full suite and the `tests/backend/pass` corpus differential show no regression.

**Next (M2):** field-projection resolution — a non-empty path GEP'd through the base aggregate, landing
§5 Example B (`&mut o.inner.v`), with the AST path as the oracle (aggregates compile on it).

## 13. Status: places, M2a — field references (§5 Example B/C substrate)

The field-projection slice, and the substrate §5.4's disjointness payoff needs: a `let bx = &[mut] p.x`
whose reference does not escape binds `bx` to a **field place** over the borrowed expression, and `*bx`
/ `*bx = v` re-lower it — a `FieldLoad` / `FieldStore` through `p` — with no pointer materialized. §5
Example C (`let bx = &mut p.x; let by = &mut p.y; *bx = 1; *by = 2`) now runs on the flat path, matching
the AST oracle.

The key simplification over a base + interned-path representation: `Binding::Place` stores the
**borrowed place-expression itself** (`Identifier(x)` for `&x`, `MemberAccess(p.x)` for `&p.x`), and
resolution just re-lowers it through the *existing* member-access read (`FieldLoad`) and field-store
(`FieldStore`) paths — the same code that already lowers `p.x` / `p.x = v` directly. So M1 and M2
unified: a deref is `lower_expr(place)`, a deref-store is `lower_assign(place, rhs)` (the `Assign` body
factored out for exactly this re-dispatch). The variable-length projection path §5.5 worried about
lives in the AST expression it already came from — no side table.

Escape analysis (`analyze_local_uses`) generalized: a candidate is now `let r = &[mut] <lvalue>` for any
lvalue rooted at a local (`place_root`), classified `is_field`. A field candidate realizes as a place
whenever the reference does not escape (mutable and written are fine — they resolve to field stores); a
scalar candidate keeps the stricter Example A rule (immutable, read-only, un-materialized base). Escape
still means decline: an escaping or unsupported field borrow (e.g. a two-level `&o.inner.v`, whose
nested `FieldLoad` is an independent flat-emitter gap) falls back to the AST oracle, never a miscompile.

**Verified:** `flat_runs_disjoint_field_borrows_through_places` (Example C, flat-vs-AST parity),
`flat_reads_and_writes_a_field_through_a_place`, `flat_reads_an_immutable_field_through_a_place`
(parity); `field_places_resolve_writes_to_field_stores` (unit: two `FieldStore`s, zero `PtrStore`); full
suite + corpus differential unchanged.

**Next (M2b):** the disjointness payoff — recover the projection paths from the two places, prove
`[x]`/`[y]` disjoint, and emit LLVM alias-scope / `noalias` metadata on the stores so the frontend's
proof is not re-derived (§5.4). No alias-metadata infrastructure exists yet, so this is the larger,
codegen-side half.

## 14. Status: places, M2b-1 — reference-parameter aliasing attributes (§5.4)

Investigating the §5.4 payoff surfaced a correction worth recording: **for struct fields the alias
metadata is redundant.** Example C's two stores lower to `getelementptr %p[0,0]` / `%p[0,1]` — constant
offsets off one base — from which LLVM already proves disjointness. So scope metadata on *those* stores
tells LLVM nothing (the section itself concedes "LLVM re-derives from GEP offsets"). The guarantee LLVM
genuinely **cannot** re-derive is the exclusivity of the reference *itself*: a `&mut` parameter aliases
nothing, but the callee cannot see the caller's borrows.

M2b-1 carries that guarantee into the function signature, exactly as rustc does:

- `&mut T` parameter → `llvm.noalias` (an exclusive borrow — the borrow checker forbids any other
  reference to the same memory for its duration);
- `&T` parameter → `llvm.readonly` (a shared reference cannot be written *through* — type-guaranteed).
  Deliberately **not** `noalias`: two `&T` may alias, so only the no-write guarantee is sound;
- raw `*mut`/`*const` → nothing (no exclusivity).

`param_alias_attrs` (`src/codegen/flat.rs`) renders the attribute after the param type;
`func.func @update(%arg0 : !llvm.ptr {llvm.noalias})` survives lowering to `llvm.func` and translates to
an LLVM IR parameter attribute.

**Verification caveat, made explicit.** An `-O0` differential is *blind* to alias attributes — they
change no result, only what an optimizer may assume. A *wrong* attribute would miscompile only under
optimization, which the oracle never runs. So correctness here rests on the **borrow checker's
soundness**, not a runtime check — hence the conservative split above. Tests assert structural presence
(`flat_marks_mut_ref_param_noalias`, `flat_marks_shared_ref_param_readonly`) plus a parity run proving
the attribute doesn't break translation; the full suite + corpus differential are unchanged.

**Next (M2b-2):** the literal §5.4 field alias-scopes — redundant for struct fields today (LLVM derives
it), but the machinery (recover `[x]`/`[y]` from the places, emit `llvm.alias_scope`/`noalias_scopes`)
generalizes to disjointness LLVM can't derive from a GEP (opaque bases, dynamic indices).
