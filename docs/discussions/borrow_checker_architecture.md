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
> 1. **Two soundness holes are not covered by either half.** Reborrows through a
>    reference parameter create no borrow record (`active_borrows` is keyed by local
>    variable name), and there is no escape analysis on returned references — so
>    `fn dangle() -> &i32 { let x = 5; return &x; }` currently compiles. See §5.2–5.3.
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

This approach perfectly handles local variable lifetimes without the overhead of tracking complex control-flow graphs.

______________________________________________________________________

## 2. The 256-bit FastPath Bounds Checker (Global Scope)

**Location:** `src/borrow.rs` & `src/gid.rs`

While the Lexical checker handles local variables, it cannot verify lifetimes across function boundaries, struct assignments, or trait bounds (e.g., ensuring `&'a T` satisfies a parameter requiring `&'b T`). This is where the FastPath comes in.

### How it works:

Instead of building a cross-module constraint graph, Vx mathematically compresses lifetime bounds into the 256-bit `TypeId` registry.

- **Lowering:** In `sema.rs`, when a reference is assigned or passed to a function, `lower_to_type_id` dynamically creates a `TypeId`. The Lexical `scope_depth` is assigned as the **Region ID**.
- **Bitpacking:** The Region ID (12 bits) and the Reference Variance (4 bits) are packed into a 16-bit slot inside Word 2 of the 256-bit `TypeId`.
- **Hardware Math:** When assigning a reference to a function parameter, `verify_subtyping_bounds` (in `borrow.rs`) executes a hardware-level check. It applies a bitwise mask to extract the Region IDs and performs a direct mathematical comparison (`region_a <= region_b`).

Because Region 0 represents `'static`, a *smaller* Region ID mathematically proves a *longer* lifetime.

______________________________________________________________________

## Summary

If you are modifying the Borrow Checker, always remember this split:

- **Fixing rules about borrowing a variable twice?** Look in `src/sema.rs` (`active_borrows`).
- **Fixing rules about passing references to functions or structs?** Look in `src/borrow.rs` (`verify_subtyping_bounds`).
