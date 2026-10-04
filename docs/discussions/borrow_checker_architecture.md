# The Vx Borrow Checker Architecture

> [!IMPORTANT]
> **Companion document:** `borrow_checker_precision_analysis.md` (no longer in the tree; see git history)
> measures this design against Rust's NLL and Polonius on nine reduced cases
> (tracked in [#243](https://github.com/hiraditya/Vx/issues/243)). Read it alongside
> this one — it does not supersede this document, but it corrects and extends it in
> three places:
>
> 1. **The lexical model described below understates the implementation.** §1 says a
>    borrow is released when its lexical block ends; in practice the loan is released at
>    the *last use* of the borrowing binding, which is NLL-grade behaviour rather than
>    pre-NLL lexical behaviour. See §5.1 of the companion doc for the discriminating case.
> 1. **Two soundness holes were not covered by either half — now closed (#243).** Reborrows
>    through a reference parameter created no borrow record (`active_borrows` is keyed by local
>    variable name), and there was no escape analysis on returned references — so
>    `fn dangle() -> &i32 { let x = 5; return &x; }` used to compile. Both are fixed; see
>    *Reborrows and escapes (#243)* below and §5.2–5.3/§9 of the companion doc.
> 1. **The Region-ID encoding in §2 forecloses Polonius by construction.** A 12-bit
>    Region ID equal to lexical scope depth is a scalar; location sensitivity requires
>    regions to be *sets of program points*. That is a legitimate trade — constant-time
>    subtyping checks in exchange for an NLL-grade precision ceiling — but it should be
>    made knowingly. See §6.
>
> The design intent recorded in this document (avoid whole-program constraint graphs;
> split local aliasing from global subtyping) still stands and is why the encoding looks
> the way it does.

This document provides a comprehensive overview of how the Borrow Checker is implemented in the Vx compiler.

To achieve massive compilation speedups over traditional compilers (like Rust's `rustc`), the Vx Borrow Checker explicitly avoids building heavy, whole-program constraint graphs (like NLL or Polonius). Instead, the algorithm is deliberately **split into two distinct halves**, relying on AST lexical tracking for local rules, and 256-bit hardware-level math for global rules.

______________________________________________________________________

## 1. The Lexical Borrow Checker (Local Scope)

**Location:** `src/hir/borrow_cx.rs` (`BorrowCx`, the borrow table and liveness) and the checks in
`src/hir/check/` (`access.rs`, `escape.rs`, `views.rs`)

The Lexical Borrow Checker is responsible for enforcing **Strict Aliasing** (Shared XOR Mutable) rules within the local body of a function or block. It guarantees memory safety by ensuring you cannot have active mutable and immutable references to the same variable simultaneously.

### How it works:

- **State Tracking:** `BorrowCx` keeps a private `active_borrows: HashMap<Symbol, Vec<BorrowRecord>>`,
  read only through `live_borrows`, which drops dead borrows first.
- **AST Iteration:** As the compiler traverses the AST:
  - When a borrow is created (e.g., `&mut x`), a `BorrowRecord` is pushed onto `active_borrows` for `x`.
  - The compiler checks existing records: if a mutable borrow is requested while an immutable one exists, it throws a compile-time error.
  - The record stores the `scope_depth` at which the borrow was created.
- **Scope Popping:** When an AST block scope ends, `TypeChecker` automatically iterates through `active_borrows` and pops any records where the `scope_depth` matches the exiting block. This releases the borrow.

This approach handles local variable lifetimes without the overhead of tracking complex control-flow graphs.

> **Release is at last *use*, not lexical scope end (NLL-grade).** The description above understated
> the checker. Before a new borrow conflicts, `check_borrow_expr` drops any existing record whose
> borrower is not *used after* this point (`is_variable_used_after`, backed by a per-block liveness
> pre-scan). So `let r = &x; let v = *r; mutate(&mut x);` compiles — the loan `r` is dead at the
> mutation. The lexical `scope_depth`/`pop_scope` machinery is the *outer* bound on a loan's life;
> use-based liveness is the *tighter* one that actually decides conflicts. This is NLL-grade for
> named locals; the fast-path region encoding (§2) means NLL-grade is also the ceiling (no Polonius
> location sensitivity). See `borrow_checker_precision_analysis.md` in git history.

> **The access checks run the same sweep (#276).** The dead-borrow cleanup above once ran *only* on
> the borrow-*creation* path, so reborrowing worked but *reading* the owner while a semantically dead
> loan was still lexically in scope was over-rejected (`let r = &mut p.x; *r = 42; return p.x;` — E4002,
> spuriously). The sweep is now part of `BorrowCx::live_borrows` and runs before **both** the
> identifier access (`check_identifier_expr`) and the field access (member-access arm) tests, so a read
> after a loan's last use is accepted exactly as a new borrow after it was. Removing a dead record can
> only *withdraw* a diagnostic, never admit an unsound access, so the change is one-directional (the
> same safety argument as #269). The sweep is skipped under `silent` speculation, which must not mutate
> borrow state. (This supersedes the old "identifier access does not perform NLL cleanup" limitation
> noted in the retired `borrow_checker_mapping.md` §6.)

### Reborrows and escapes (#243)

Two rules extend the lexical checker beyond `&x`-on-a-named-local:

- **Reborrow through a reference parameter.** Passing an existing reference *by name* to a reference
  parameter (`probe(m)`, not `&m`) reborrows the underlying storage and creates a borrow record
  against it, so a later conflicting use (`insert(m, …)` while the reborrow is live) is caught
  (`E4003`/`E4004`). The record persists past the call only when the callee returns a reference
  (the reborrow escapes into the result); otherwise it lasts only for the call.
- **Return-escape analysis.** Every reference-typed binding carries a *provenance*: `External` (it
  reborrows a reference parameter — safe to return) or `Local` (it roots in a `let`, a by-value
  parameter, or a temporary). Returning a `Local`-provenance reference is a dangling escape
  (`E4005`). Provenance is computed structurally and threads through `let` bindings and
  reference-returning calls.

______________________________________________________________________

## 2. The 256-bit FastPath Bounds Checker (Global Scope)

**Location:** `src/borrow.rs` & `src/gid.rs`

While the Lexical checker handles local variables, it cannot verify lifetimes across function boundaries, struct assignments, or trait bounds (e.g., ensuring `&'a T` satisfies a parameter requiring `&'b T`). This is where the FastPath comes in.

### How it works:

Instead of building a cross-module constraint graph, Vx mathematically compresses lifetime bounds into the 256-bit `TypeId` registry.

- **Lowering:** In `sema.rs`, when a reference is assigned or passed to a function, `lower_to_type_id` dynamically creates a `TypeId`. The Lexical `scope_depth` is assigned as the **Region ID**.
- **Bitpacking:** The Region ID and the Reference Variance are packed into a 16-bit slot inside Word 2 of the 256-bit `TypeId`. Slots 1-3 (the parameters) hold `[ region: 12 | variance: 3 | reserved: 1 ]`; slot 0 (the return) reserves three of those region bits for a provenance code (see below), so it holds `[ region: 9 | prov: 3 | variance: 3 | reserved: 1 ]`.
- **Hardware Math:** When assigning a reference to a function parameter, `verify_subtyping_bounds` (in `borrow.rs`) executes a hardware-level check. It applies a bitwise mask to extract the Region IDs and performs a direct mathematical comparison (`region_a <= region_b`). The mask is *slot-dependent*: slot 0 uses the 9-bit `REGION_MASK_0`, slots 1-3 the 12-bit `REGION_MASK`.

Because Region 0 represents `'static`, a *smaller* Region ID mathematically proves a *longer* lifetime.

### The inline return-provenance field (#265)

The return's slot (slot 0) carries a **3-bit return-provenance code** in the top of its region field
(`FAST_RETURN_PROV_MASK = 0x0E00`, defined in [`src/gid.rs`](../../src/gid.rs); accessed via
`TypeId::set_return_provenance` / `extract_return_provenance`). It names which parameter a returned
reference derives from, inline in the type's identity so the cross-module borrow check can read it
from the `TypeId` with no side table:

- `0` — no provenance (not a reference, or a `NotAReference` return).
- `1..=4` — derives from parameter slot `0..=3`.
- `7` — conservative top: may derive from any parameter (today's all-arguments behaviour). Every
  over-budget case (a multi-parameter union, more than four reference parameters, a `Local`/`unsafe`
  return, an unknown callee) encodes here, so the field **degrades gracefully and locally, never
  unsoundly**. `5`/`6` are reserved and also read as top.

`encode_return_provenance` (in [`src/hir/provenance.rs`](../../src/hir/provenance.rs)) maps the
per-function `ReturnProvenance` summary to this code, and it is a **conservative refinement**: for
every parameter the summary flags as an alias source, the code flags it too (proven by
`inline_code_conservatively_refines_the_summary`). Intra-compilation the summary side table
(`return_provenances`) still drives the per-argument reborrow decision; the inline code is populated
and checked against the summary on every call (a debug-only round-trip assertion in the
`Expr::FunctionCall` arm), and the **cross-module consumer that reads it from a serialised `.vxlib`
is step 7** — deferred, tracked with [#220](https://github.com/hiraditya/Vx/issues/220) /
[#224](https://github.com/hiraditya/Vx/issues/224).

### The reserved "unset" region sentinel (#267)

The **maximum** value of each region field is reserved as an **unset / not-yet-assigned** sentinel. A
parsed reference type carries it until the borrow checker binds a real scope depth
([`src/parser/types.rs`](../../src/parser/types.rs)); it surfaces in generic-deduction diagnostics as
`region_id: 4095`. Because slot 0's region is narrower than the parameter slots (#265), there are two
sentinels, both defined in [`src/borrow.rs`](../../src/borrow.rs):

- `REGION_UNSET` (`= REGION_MASK`, `0x0FFF = 4095`) for the 12-bit parameter slots.
- `REGION_UNSET_0` (`= REGION_MASK_0`, `0x01FF = 511`) for slot 0's 9-bit region.

Two rules keep either from being confused with a real depth:

- **It is a wildcard, not a number.** `verify_subtyping_bounds` checks `region == <sentinel>` *before*
  the numeric `<=`/`==` comparison and skips the region dimension when either side is unset. The
  sentinel's numeric position is never trusted. The comparison masks each slot with its own width
  (`REGION_MASK_0` for slot 0, `REGION_MASK` otherwise), so the provenance code in slot 0's top bits
  can never fold into the lifetime.
- **Real depths never reach it.** `lower_to_type_id` clamps a real scope depth with
  `region_for_depth` (parameters, → `REGION_MAX = 4094`) or `region_for_depth_slot0` (return, →
  `REGION_MAX_0 = 510`), so no genuine region can equal a sentinel; only the deliberate placeholder
  does.

**If you narrow a region field further**, you **must** move that slot's sentinel to the new field's
maximum and clamp its real depths below it, keeping the explicit `== sentinel` recognition and a
slot-width-matched mask in `verify_subtyping_bounds`. Do **not** rely on `4095` being out of range: a
narrowed field truncates it to a legal value and silently corrupts subtyping (`4095 & 0x1FF = 511`,
an ordinary region) — the exact hazard slot 0's 9-bit narrowing had to handle. The `borrow.rs` unit
tests (`unset_region_is_a_wildcard`, `max_real_region_is_distinct_from_the_sentinel`,
`slot0_provenance_bits_do_not_corrupt_region_compare`, `region_for_depth_slot0_maps_sentinel_and_clamps`)
pin this invariant.

______________________________________________________________________

## 3. Out of scope: the Rust stdlib FFI boundary

Neither half of the borrow checker reaches across the C ABI into the Rust core
(`stdlib/rust_core`, built as `libvx_std_core.a`). Non-scalar values cross that boundary as
opaque `*mut c_void`, so ownership there is a **hand-maintained convention**, not a checked
property: Vx cannot observe a Rust `Drop`, and Rust cannot observe a Vx scope exit.

That convention — `Box::into_raw` in constructors, plain references in accessors,
`Box::from_raw` exactly once in destructors — is specified in
**[`docs/lang/abi.md` §2.1 "Opaque-pointer ownership lifecycle"](../lang/abi.md)**, and generated
for the collections surface by `instantiate_vec_ffi!` / `instantiate_hash_map_ffi!` in
[`stdlib/rust_core/src/ffi/macros.rs`](../../stdlib/rust_core/src/ffi/macros.rs).

Read it before adding a stdlib binding: a violation there is a use-after-free or a leak that
*neither* language's checker will catch, and it will not show up in any of the borrow-checker
fixtures.

> [!NOTE]
> This is unrelated to `transfer(x, Memory::X)`. That is a compiler-level memory-space move
> lowered to the `vx.transfer` op (`memref.alloc` + `memref.copy`), not an FFI call — see
> [`docs/topology_representation.md`](../topology_representation.md).

## 4. Views of a tensor, and why the checker must see them (#1041, #1049)

A row `q[i]`, a field `h.t` that holds a tensor, and a view of either share their owner's memory:
writing through one changes the other. Before #1049 the checker treated such a value as an
unrelated tensor, so it accepted using a row after its tensor was moved, writing the tensor under a
row, and returning a row of a tensor the function owns.

### The decision: freeing memory depends on this checker

Vx is moving from MLIR's buffer deallocation, which rebuilds ownership from IR and skips what it
cannot follow, to drops the checker decides (`docs/implementation_plans/drop_semantics.md`). We
decided:

- **An owner that only holds memory** (a tensor, a `Vec` with no `Drop` of its own) **is freed after
  the last use of the owner and of every view of it.** Nothing in the program can observe the free,
  and it keeps peak memory low.

- **A type with a `Drop` implementation is dropped at the end of its block**, in reverse
  declaration order, because its drop has effects whose timing is part of the program: a lock, a
  `RefCell` guard, a file. Nothing borrows a `let _lock = m.lock()` after that line, so a last-use
  rule would release it at once.

- **An owner a raw pointer was taken from** (`as_ptr`, `from_ptr`) also waits for the end of its
  block. No borrow checker sees raw pointers.

- **A view is never copied behind the programmer's back.** Where a tensor of its own is
  needed, a function parameter taken by value for example, a view is refused and the error asks
  for `.clone()`. An automatic copy would hide a copy of possibly large data in an ordinary
  looking call; Rust makes the same choice.

So for memory, **a free that comes too early is a borrow checker bug**, not a rule for the
programmer to remember. That is why views come first: phase 0 of the plan.

### The rules

A view is a borrow of its owner, recorded exactly as `let r = &q` records one, so the existing
checks apply unchanged:

| Rule | Error |
| --- | --- |
| A view declared `let mut` is a `&mut` borrow, and its owner must be `mut`; any other view is `&`. | E4010 |
| The owner cannot be moved while the view is still used. | E4007 |
| The owner cannot be assigned while the view is still used. | E4009 |
| Under a `mut` view, the owner cannot be read or borrowed again. | E4002 to E4004 |
| A view of a local, or of a parameter taken by value, cannot be returned. A view of a parameter taken by reference can. | E4005 |
| A view cannot be stored where a tensor of its own is held: a variable that already exists, or a struct field. `p[i] = q[j]` copies and is fine. | E4011 |
| A view cannot be passed to a function that takes a tensor by value, since the function would own it. Pass `q[i].clone()`, or take `&Tensor`. | E4011 |

A borrow ends at the view's last use, as for `&`.

### How it is built

- **A view is told from an owner by where it came from, not by its type**: both are `Tensor`.
  `view_of` in `src/hir/check/views.rs` recognises an index or field chain whose value is a tensor,
  `t.reshape(..)`, a variable that is already a view, and an `if` or `match` whose value is one
  of these. A separate view type (#400) can replace this later.
- **A view may borrow several tensors.** One chosen by an `if` or a `match` borrows every tensor
  a branch can give, and its provenance is the shortest-lived of theirs. A branch's value is its
  last expression; a `return` in it leaves the function and is not its value. A tensor declared
  inside the branch is gone after it, so a view of it is a block escape (E4005).
- **A view of a view borrows the first owner.** `BorrowCx.views` remembers, for each view
  variable, the tensors it borrows, the path inside each, and whether they are local to the
  function. That is reset for each function, like `ref_provenance`.
- **Through a reference, conflicts are checked on the reference.** For `rq[1]` with `rq = &q`,
  and for a reborrow `&*rq`, the borrow is recorded on `rq` and also on `q` (`BorrowCx::borrowed_by`
  finds what `rq` borrows), so `q` stays borrowed after `rq`'s last use. Conflicts are checked
  only against `rq`: `rq` already holds its own borrow of `q`, and checking `q` would refuse a
  `&mut` reborrow of a `&mut` reference.
- **A closure keeps its views' tensors borrowed.** A closure becomes a struct of the variables it
  uses; when it is bound with `let f = ..`, `f` becomes a borrower of every tensor a captured
  view borrows.
- **A copy carries its borrows.** `let g = f` or `g = f` gives `g` every borrow `f` holds, so a
  reference or a closure copied to another variable keeps what it points at borrowed while
  either name is used. Before this, `let r2 = r; let s = q;` with `r = &q` was accepted.
- **The borrower is the view variable,** so the existing liveness sweep ends the borrow at its last
  use.
- **Indices are not part of a borrow's path** (`places::base_and_path`), so `q[0]` and `q[1]` count
  as the same place. That is sound and sometimes too strict (#1060).

### Not checked yet

| Case | Issue |
| --- | --- |
| A closure that uses a row reads the wrong value on the default code generator. This is a code generation bug, not a borrow rule. | #1080 |
| Two `mut` rows of different indices are refused. | #1060 |

Only a handful of programs in the repository take a view, so a fuzzer generator for them (#1062)
is the main test still missing.

## Summary

If you are modifying the Borrow Checker, always remember this split:

- **Fixing rules about borrowing a variable twice?** Look in `src/hir/borrow_cx.rs` (`active_borrows`) and
  `src/hir/check/access.rs` (`check_borrow_conflicts`).
- **Fixing rules about rows and fields of a tensor?** Look in `src/hir/check/views.rs`, and read §4.
- **Fixing rules about passing references to functions or structs?** Look in `src/borrow.rs` (`verify_subtyping_bounds`).
- **Adding a Rust stdlib FFI shim?** Not the borrow checker's job — follow the ownership contract in [`docs/lang/abi.md` §2.1](../lang/abi.md).
