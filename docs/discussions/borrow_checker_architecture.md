# The Vx Borrow Checker Architecture

> [!IMPORTANT]
> **Companion document:** [`borrow_checker_precision_analysis.md`](borrow_checker_precision_analysis.md)
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

**Location:** `src/sema.rs` (Inside the Semantic Analyzer)

The Lexical Borrow Checker is responsible for enforcing **Strict Aliasing** (Shared XOR Mutable) rules within the local body of a function or block. It guarantees memory safety by ensuring you cannot have active mutable and immutable references to the same variable simultaneously.

### How it works:

- **State Tracking:** The `TypeChecker` struct in `sema.rs` maintains an `active_borrows: HashMap<String, Vec<BorrowRecord>>`.
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
> location sensitivity). See [`borrow_checker_precision_analysis.md`](borrow_checker_precision_analysis.md).

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

## Summary

If you are modifying the Borrow Checker, always remember this split:

- **Fixing rules about borrowing a variable twice?** Look in `src/sema.rs` (`active_borrows`).
- **Fixing rules about passing references to functions or structs?** Look in `src/borrow.rs` (`verify_subtyping_bounds`).
- **Adding a Rust stdlib FFI shim?** Not the borrow checker's job — follow the ownership contract in [`docs/lang/abi.md` §2.1](../lang/abi.md).
