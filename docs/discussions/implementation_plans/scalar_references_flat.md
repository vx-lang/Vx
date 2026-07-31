# Design: Scalar References (`&i32`) on the Flat Path

**Status:** **immutable slice (§9), mutable slice (§10), and the step-2 per-local memory rule (§11) implemented** — `&x` / `&mut x` / `*r` / `*p = v` / `&i32` params + returns lower on the flat path and run (locally and *across a module boundary*), the immutable forms verified against the AST oracle and the mutable forms against the value-semantics equivalent (§6.2); a control-flow function now slots only the locals that need it. Places ([#275](https://github.com/hiraditya/Vx/issues/275)): empty-path (§12, Example A — the alloca that should not exist) and field references (§13, Example B/C substrate) lower on the flat path, and the §5.4 disjointness→alias-metadata payoff is landed end to end — reference-parameter `noalias`/`readonly` (§14, M2b-1) and field alias-scopes on disjoint place-writes (§15, M2b-2). §16 audits what remains; **M3a (§17) landed Example B (`&mut o.inner.v`) end to end** by fixing the borrow-checker over-rejection ([#276](https://github.com/hiraditya/Vx/issues/276)) and by-value nested-aggregate construction ([#277](https://github.com/hiraditya/Vx/issues/277)).
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
*(The pointer-local half is now done — M3b, §17.)*

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

## 15. Status: places, M2b-2 — field alias-scope metadata, as implemented

The other half of §5.4: attach `alias_scopes`/`noalias_scopes` to the disjoint place-writes, so Example
C's `*bx = 1` / `*by = 2` each declare their own alias scope and name the other as a `noalias` sibling.
As M2b-1 conceded, this is **redundant for constant-offset struct fields** (LLVM already proves `%p[0,0]`
and `%p[0,1]` disjoint); it is built because the *machinery* — carrying a frontend-proved disjointness
onto individual stores — generalizes to disjointness a GEP can't express (opaque bases, dynamic indices).

**Carry the proof, don't re-derive it (§5.4's actual point).** The disjointness is computed in the
lowerer from the *places*, not in codegen from GEP offsets:

- lowering a place-write `*r = v` (`lower_assign`), the borrowed place's `(root, field path)` is recorded
  alongside the `FieldStore`'s stream position (`pending_place_write` → `place_field_stores`). A *direct*
  `p.x = v` (not through a `&mut` place) leaves the tag unset and is never scoped — only
  reference-mediated writes carry metadata;
- after lowering, `reduce_place_alias` interns each distinct `(root, path)` as a **group** and, per
  store, lists the groups that are *disjoint siblings*: same root, non-prefix-disjoint path
  (`paths_may_alias` — `[x]` vs `[y]` disjoint, `[inner]` vs `[inner, v]` overlapping). Stores to the
  *same* field share a group (they alias); to different fields they become mutual `noalias` siblings.
  This structural field-disjointness is sound independent of borrow *liveness*; the frontend contribution
  is knowing the writes came from distinct, borrow-checker-admitted `&mut` borrows;
- the numeric table `(position, group, siblings)` rides to codegen per-function on the worker
  (`local_place_alias_stores`), index-aligned with `funcs` like the string tables.

Codegen (`emit_function_mlir`) numbers each function's groups from a **module-global `distinct[]`
counter** (0 reserved for the shared `alias_scope_domain`), so scopes stay distinct across functions even
after inlining, and `alias_store_attrs` renders `{alias_scopes = [<own>], noalias_scopes = [<siblings>]}`
on the tagged `llvm.store`. A lone place-write gets its own scope but **no** `noalias_scopes` (nothing
proven disjoint from it).

**Verification caveat (same as M2b-1).** An `-O0` differential is blind to alias metadata — it changes no
result, only what an optimizer may assume — so correctness rests on the borrow checker plus the
structural disjointness above, not a runtime oracle. Tests assert the reduction
(`disjoint_place_writes_reduce_to_mutual_noalias_siblings`, `a_lone_place_write_has_no_noalias_sibling`,
`direct_field_assignment_is_not_tagged`) and the emitted metadata
(`flat_tags_disjoint_field_stores_with_alias_scopes`, `flat_tags_a_lone_field_store_without_noalias`),
each paired with a parity run proving the attributes don't break translation or the JIT result. Untagged
stores emit byte-identically (`unwrap_or_default`), so nothing outside the `&mut`-field-place shape moves.

This closes M2b (both halves of §5.4). The place representation now carries disjointness proofs to the
IR; extending them past constant field offsets (where they stop being redundant) is future work.

## 16. What remains, and how to fix it

Audited after M2b-2 (`c625954f`). Recorded here because the per-slice "Next (Mx)" chain ends at §15
with no successor, so the remaining [#275](https://github.com/hiraditya/Vx/issues/275) scope had no
plan attached to it.

> [!NOTE]
> **Resolved by M3a (§17).** Both defects below are fixed: #276 (`0cee9a4`) and #277 (`1886814`).
> Example B now runs end to end on the flat path with AST parity. The diagnosis is kept for the record.

### 16.1 §5 Example B is blocked by two separate defects, neither of them a place bug

§12 named Example B (`&mut o.inner.v`) as M2's target; §13 delivered Example C instead and reassigned
it to "an independent flat-emitter gap." Probing that gap found **two** defects, in this order:

**First — a borrow-checker over-rejection ([#276](https://github.com/hiraditya/Vx/issues/276)).**
Example B never reaches codegen:

```rust
let r = &mut p.x;  *r = 42;  return p.x;   // Error: Cannot access 'p' because it is mutably borrowed
```

One level of projection, so not a depth problem. The discriminator: creating a **new borrow** after
`r`'s last use is accepted, **reading** after it is not —

```rust
let r = &mut p.x;  *r = 42;  let s = &mut p.y;  *s = 7;   // ACCEPTED
```

NLL dead-borrow cleanup runs in `check_borrow_expr` and `track_reference_arg_borrow` (both sweep with
`is_variable_used_after` before testing for a conflict — that sweep is what makes bc5 pass) but the
identifier-access check does not.

> *Fix:* factor the existing sweep out of `track_reference_arg_borrow` and run it in the
> identifier-access arm before iterating `active_borrows`. The change only *removes* diagnostics, so
> it cannot introduce an unsound accept — the same safety argument that made #269 landable.

> [!NOTE]
> The retired `docs/lang/borrow_checker_mapping.md` §6 described this split precisely. It was assessed
> as stale when that file was retired (`0440ead0`) — **incorrectly**: `bc5` only ever exercised the
> borrow-*creation* path, so it never contradicted the note. The rest of that retirement stands. The
> note should be reinstated in [`borrow_checker_architecture.md`](../borrow_checker_architecture.md)
> once #276 is fixed, as a record of what the two checks now share.

**Second — by-value nested-aggregate construction
([#277](https://github.com/hiraditya/Vx/issues/277)).** With the borrow error sidestepped, the flat
emitter produces invalid MLIR:

```mlir
%v2 = llvm.alloca %n2 x !llvm.struct<(i32)>            ; Inner{v:1} -> %v2 is a POINTER
%p5 = llvm.getelementptr %v1[0, 0]                     ; &o.inner
llvm.store %v2, %p5 : !llvm.struct<(i32)>, !llvm.ptr   ; stores the ADDRESS, typed as the VALUE
```

```
error: use of value '%v2' expects different type than prior uses: '!llvm.struct<(i32)>' vs '!llvm.ptr'
```

**The two-level projection itself is correct** — `gep %v1[0,0]` then `gep %v6[0,0]`, with M2b-2's
alias metadata rendering fine on the final store. Only the constructor is wrong. (The nested type
spelling `!llvm.struct<(!llvm.struct<(i32)>)>` is also fine; both prefixed and unprefixed nestings
parse — checked against `mlir-opt`.)

> *Fix, two options:* (1) load the inner aggregate and store the **value** — minimal, one extra
> aggregate copy; (2) **construct in place** — do not allocate the inner slot at all, GEP the outer's
> field and build directly into it. (2) is the better shape *and* composes with places: a place is
> precisely "a destination to build into," so once construction accepts a place, nested construction
> and nested projection share one mechanism. (1) is a fine first cut if destination-passing is too
> large to bundle.

**Severity note.** This surfaces as *invalid MLIR caught by the parser*, not as a clean decline. The
end result today is the same (AST fallback, no miscompile), but it is a different safety net than the
designed one, and the decline predicate does not know about the gap. Per §7.1 that matters
cross-module, where a decline is fatal and one that only appears as a parse failure is harder to
enumerate ahead of time. Worth asserting in the emitter that no emitted module fails to parse, so this
class becomes a decline.

> [!NOTE]
> **Hardened (`build_flat_module`).** When `emit_module_mlir` returns `Some(text)` — the emitter
> *claiming* it handled the module — but that text fails `Module::parse`, that is now a `debug_assert`
> failure: an emitter bug (a construct emitted unparseable text instead of declining in
> `emit_module_mlir`), distinct from the designed `None` decline. Debug/test builds fail hard so the gap
> surfaces where a fallback and the corpus exist; release still falls back gracefully. Verified the whole
> `tests/backend/pass` corpus and the flat differential run clean under the assert (nothing currently
> emits invalid text and silently degrades). The `-O2` differential (§16.3) is the remaining unaddressed
> hardening.

### 16.2 The `BorrowRecord.path` alignment goal is now unaddressed

#275's rationale was that both stages should "speak the same language about which memory a reference
names." M2a instead stores the **borrowed AST expression** and re-lowers it — a good simplification
(§13), and it removed the need for the side table §5.5 anticipated. But there is consequently no
`Vec<Projection>` to align with `BorrowRecord.path`, and M2b-2 had to *recover* `(root, field path)`
ad hoc in `reduce_place_alias` to compute disjointness.

So there are now **two independent path derivations**, which is the situation the alignment goal
existed to remove.

> *Decision needed, not a fix:* either drop the alignment goal explicitly (the M2a simplification is
> worth more than the unification) or schedule a slice that reconciles them. Leaving it implicit means
> the third derivation gets written when reborrows land.

> [!NOTE]
> **Resolved: deferred, not scheduled.** Both derivations are individually sound — the checker's
> `BorrowRecord.path` and the lowerer's `place_base_path`/`paths_may_alias` compute the same fact
> correctly, just twice. The only cost is the ~15 lines of duplicated path logic, not correctness. This
> is not on M3b's critical path, so the alignment goal is dropped as a *blocker*; revisit only if M4
> (reborrows) makes the duplication genuinely painful — at which point the checker's path can be threaded
> down rather than a third copy written.

### 16.3 The M2b verification caveat has no exit

§14 and §15 both rest on "correctness follows from the borrow checker's soundness, not a runtime
check," because an `-O0` differential is blind to alias metadata. Fair once; it is now load-bearing
for two slices with no named way to ever check it.

> *Fix:* an **`-O2` differential** — same program at `-O0` and `-O2`, results must agree. That
> exercises exactly what the attributes license an optimizer to assume, and a wrong `noalias` would
> diverge. It is the only proposed check that tests the thing the caveat waives.

> [!NOTE]
> **Done (`flat_alias_metadata_is_sound_under_o2`).** The lowered Example C module (which carries the
> `noalias` metadata — asserted present, so the check isn't vacuous) is JIT-run at `-O0` and `-O2` and
> both return 12. `execute_mlir` translates the MLIR `alias_scopes`/`noalias_scopes` to LLVM
> `!alias.scope`/`!noalias` via `mlir-translate` and then runs `opt -passes=default<O2>`, so `-O2` is
> where the optimizer actually consumes them; a wrong `noalias` would let it reorder/drop a store and
> diverge from the `-O0` ground truth. The caveat now has its exit.

### 16.4 Suggested milestones for the rest

| | Content | Why grouped | Status |
|---|---|---|---|
| **M3a** | #276 + #277 | Prerequisites for Example B; small, and unblock a deliverable already named in §12 | **Done (§17)** |
| **M3b** | Reference returns as places (§12 treats a return as escape); pointer locals off the coarse model (§11's remainder) | Both extend the existing representation — no new machinery | Open |
| **M4** | Reborrows, nested references (`&&T`), reference-typed struct fields | The genuinely new representational work, and where §16.2's decision must be made | Open (needs §16.2) |
| **Ongoing** | The `-O2` differential (§16.3) | Retires a caveat that otherwise compounds per slice | Open |

The alternative is to close #275 as "places, first cut, delivered" — M1 and M2a landed real value
(Example A's eliminated alloca; Example C running on flat) — and re-file M4 with §16.2 decided up
front. The remaining bullets are arguably different work from what M1/M2 built.

## 17. Status: M3a — Example B end to end, as implemented

Example B (`&mut o.inner.v`, §5.3) — named as M2's target in §12, then reassigned to "an independent
flat-emitter gap" in §13 — now lowers and runs on the flat path with AST parity. §16.1 found it blocked
by two defects in front of the place machinery, neither a place bug; M3a fixed both.

**#276 — borrow-checker over-rejection (`0cee9a4`).** The NLL dead-borrow sweep ran only when a *new*
borrow was created (`track_reference_arg_borrow`), so `let r = &mut p.x; *r = 42; return p.x;` was
rejected: reading `p` while `r`'s (dead) loan was still lexically in scope. Factored the sweep into
`sweep_dead_borrows` and ran it in the identifier- and member-access checks too, so a read after a loan
is dead is accepted exactly as a new borrow after it is. One-directional (removing a dead record can only
withdraw a diagnostic), skipped under `silent` speculation. `borrow_use_after_mut.vx` — a *live* borrow —
still rejects. The retired `borrow_checker_mapping.md` §6 note is reinstated in
[`borrow_checker_architecture.md`](../borrow_checker_architecture.md), updated to record the shared sweep.

**#277 — by-value nested-aggregate construction (`1886814`).** `Outer { inner : Inner { .. } }` lowered
the inner `StructInit` to its construction *slot* (a pointer) and `FieldStore`d that address into the
outer's field, which codegen types as the field's struct *value* — invalid MLIR (a parse failure, caught,
so never a miscompile, but not the designed decline). Load the inner aggregate to a value before the
store, mirroring `lower_assign`'s existing aggregate-RHS fix. This is §16.1 option 1; destination-passing
(option 2) — build directly into the GEP'd field, sharing one mechanism with places — remains the better
follow-up.

Tests: `flat_runs_nested_field_place_example_b` (writes 42 through `*r`),
`flat_constructs_a_nested_aggregate_by_value` (#277 in isolation, reads 7),
`borrow_read_after_dead_field_mut.vx` (accepts). Full flat differential module 96 pass (was 94).

### 17.1 M3b part 1 — pointer locals off the coarse model

§11 left pointer locals on the function-global `memory` flag ("a missed optimization, not a correctness
gap"). The per-local rule now covers them: a `LoweredTy::Ptr` local that is *not* address-taken binds as
an SSA register unless it is reassigned in a control-flow function — identical to the scalar rule, with
the same MLIR-verifier backstop (a register read out of its definition's dominance is invalid MLIR → the
flat path declines, never miscompiles). An *address-taken* pointer (`&p`, a pointer-to-pointer) keeps the
coarse model, since its addressable-slot codegen isn't in the subset.

**Where it fires.** The direct shape — a `let p = &mut x` scalar pointer local dereferenced under control
flow — does *not* lower yet (mutable-through-a-named-local isn't in the subset; only `&mut` *params* in a
callee and `&mut x` as a *call argument* are), so it can't exercise the rule. The real consumer is the
raw-pointer local: `Vec::push`'s `let ptr: *mut i8 = self.data;` inside the grow `if` — a pointer local,
under control flow, read once and never reassigned. Before, it was slotted; now it is a register. The Vec
differential programs still match the AST oracle, so the flip is sound.

Test: `pointer_local_under_control_flow_stays_a_register` (unit) — the distilled `let ptr = b.data`
shape lowers with **zero** `Alloca` (verified to fail — one slot — with the rule reverted, so it genuinely
pins the rule).

### 17.2 M3b part 2 — reference returns (scalar-field address)

§16.4 grouped "reference returns as places" with M3b as "no new machinery." A probe corrected that: a
reference-returning function (`probe(m : &Map) -> &i32 { return &m.slot; }`) returned `flat=None` — it
declined entirely. The cause was *not* the cross-function return-provenance §5 deferred (the borrow
checker already proves a returned reference outlives the callee, #243) but a **bounded emitter gap**: the
`Expr::Borrow` arm addressed only nested-*aggregate* fields (via `lower_agg_base`, which requires
`FieldTy::Nominal`); a `&scalar_field` fell through and declined.

Fixed by addressing a scalar field too: `lower_scalar_field_addr` GEPs the parent aggregate (a `&Map`
param, a local slot, or a nested aggregate — whatever `lower_agg_base` resolves) to the field and yields
a `!llvm.ptr`, tagging the `FieldAddr` with the field's *scalar* GID. Codegen's `FieldAddr` then branches:
a GID in `aggs` is an aggregate slot (a chained access GEPs through it, as before); a scalar GID is a
plain element pointer (`ptr_of`). Existing nested-aggregate receivers are unchanged (their GID is in
`aggs`), so the full differential is unregressed. The safety proof stays the borrow checker's — the flat
path only emits the address the frontend already validated.

Tests: `flat_returns_a_reference_to_a_scalar_field` (`&m.slot`, first field, reads 7),
`flat_returns_a_reference_to_a_non_first_field` (`&m.present`, non-zero offset, reads 9 — guards the
`field_idx`). Full flat differential unregressed. This is the *scalar-field* reference return; a reference
to a nested-aggregate field or to a by-value local (not through a pointer) is follow-up.

**Still open from §16:** M4 (reborrows, `&&T`, reference-typed fields), with §16.2 deferred (both path
derivations are sound; revisit only if M4 makes the duplication painful). §16.1 and §16.3 are done (above).

## 18. Status: M4 — assessment and the reachable slice

M4's three constructs were **classified by AST-oracle availability** (a flat differential needs the AST
path to compile the program) before committing to any of them:

| Construct | AST oracle | Flat (before) | Verdict |
|---|---|---|---|
| reborrow-by-name to a `&param` (`via(m) { read(m) }`) | ✓ = 7 | ✓ = 7 | **already lowers** — the common case is done |
| reference-typed struct field (`struct H { r : &i32 }`) | ✓ = 5 | declined | **closable** — a bounded escape gap (§18.1) |
| nested reference `&&T` (`let rr = &r; **rr`) | unreliable (≠ expected) | declined | **deferred** — no clean oracle, #233-class |

So reborrow-by-passing needs nothing; `&&T` has no trustworthy oracle to differentiate against (the AST
path itself does not produce the expected value), so it is deferred like #233; and the one reachable
slice was reference-typed struct fields.

### 18.1 Reference-typed struct fields

`Holder { r : &x }` / `*h.r` declined at the **construction**, and the cause was a one-line escape-analysis
gap, not a representational one: `BorrowScan::expr` (and `RefUseScan::expr`) had no `Expr::StructInit` /
`Expr::EnumVariant` arm, so a `&x` *inside* an aggregate literal was never seen — `x` stayed a register
and the `&x` declined at lowering (the "missing a fact is safe" invariant, §3, turned it into a decline
rather than a miscompile). Descending into the literal's field/payload values materializes `x`; the store
of the pointer into the `Opaque` field and the `*h.r` load-then-deref then lower on the existing
raw-pointer-field machinery (#242) with no further change.

Tests: `flat_runs_a_reference_typed_struct_field` (reads 5, parity),
`borrow_inside_a_struct_literal_materializes_its_base` (unit: the literal's `&x` produces an addressable
`imm = 1` alloca). Full flat differential unregressed — the scan only descends further, and no existing
program has a borrow inside a literal that was relying on the old under-collection.

**Remaining M4, deferred:** `&&T` / nested references (needs a working AST oracle first — an AST-path
issue, not a flat one) and reborrow-as-place refinements beyond the by-passing case. Reference-typed
fields and reference returns (§17.2) close the reachable reference surface for now.
